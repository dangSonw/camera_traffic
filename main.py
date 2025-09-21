import time
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from dataclasses import dataclass
from contextlib import contextmanager

from traffic_monitor.system_utils import (
    setup_cpu_affinity, print_progress_bar,
)
from traffic_monitor.config import build_arg_parser, load_runtime_config

@dataclass
class ProcessingState:
    frame_count: int = 0
    total_inference_time: float = 0.0
    start_time: float = 0.0
    ema_fps: Optional[float] = None
    ema_inf: Optional[float] = None

@dataclass
class CountingStats:
    before_line: int = 0
    after_line: int = 0
    
    def reset(self):
        self.before_line = 0
        self.after_line = 0

@dataclass 
class ModelConfig:
    input_width: int
    input_height: int
    scale_factor: float
    mean_values: List[float]
    swap_rb: bool
    crop: bool
    classes: List[str]

class ROIProcessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.roi_cfg = cfg.get('model', {}).get('roi', {})
        self.enabled = self.roi_cfg.get('enabled', False)
        self.roi_type = self.roi_cfg.get('type', 'rectangle')
        self.pts_px = self._get_roi_points()
        self.color = tuple(self.roi_cfg.get('color', [0, 255, 0]))
        self.thickness = int(self.roi_cfg.get('thickness', 2))
        
        # Load overlay config - SỬA ĐỂ KHỚP VỚI STRUCTURE JSON
        self.overlay_cfg = cfg.get('processing', {}).get('display', {}).get('overlay', {})
        
        # Counting line config (chỉ cho rectangle)
        self.counting_cfg = self.overlay_cfg.get('counting_line', {})
        self.counting_enabled = self.counting_cfg.get('enabled', True) and self.roi_type == 'rectangle'
        self.counting_line_pos = float(self.counting_cfg.get('position', 0.5))  # 0.0 to 1.0
        self.counting_line_color = tuple(self.counting_cfg.get('color', [0, 0, 255]))
        self.counting_line_thickness = int(self.counting_cfg.get('thickness', 3))
        self.counting_extend_factor = float(self.counting_cfg.get('extend_factor', 0.1))
        
        # Calculate counting line coordinates
        self.counting_line_coords = self._calculate_counting_line()
        
    def _get_roi_points(self) -> np.ndarray:
        if not self.enabled:
            return np.array([])
            
        if self.roi_type == 'trapezoid':
            pts = self.roi_cfg.get('trapezoid', {}).get('points', 
                [[300, 300], [1000, 300], [1200, 800], [100, 800]])
        else:  # rectangle
            rect = self.roi_cfg.get('rectangle', {})
            x, y = rect.get('x', 400), rect.get('y', 300)
            w, h = rect.get('width', 800), rect.get('height', 600)
            pts = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        
        return np.array(pts, np.int32)
    
    def _calculate_counting_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Calculate HORIZONTAL counting line coordinates for rectangle ROI"""
        if not self.counting_enabled or self.roi_type != 'rectangle' or len(self.pts_px) < 4:
            return None
            
        # Get rectangle bounds
        rect_cfg = self.roi_cfg.get('rectangle', {})
        x = rect_cfg.get('x', 400)
        y = rect_cfg.get('y', 300)
        w = rect_cfg.get('width', 800)
        h = rect_cfg.get('height', 600)
        
        # Calculate HORIZONTAL line position (along height)
        line_y = int(y + h * self.counting_line_pos)
        
        # Extend line beyond rectangle bounds (horizontally)
        extend_width = int(w * self.counting_extend_factor)
        line_x1 = x - extend_width
        line_x2 = x + w + extend_width
        
        return ((line_x1, line_y), (line_x2, line_y))
    
    def apply_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        
        if not self.enabled:
            return frame.copy(), (0, 0, w, h)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.pts_px], 255)
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        x_min, y_min = np.min(self.pts_px, axis=0)
        x_max, y_max = np.max(self.pts_px, axis=0)
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        cropped = roi_frame[y_min:y_max, x_min:x_max].copy() if x_min < x_max and y_min < y_max else frame.copy()
        
        return cropped, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_overlay(self, frame: np.ndarray) -> None:
        if not self.enabled or len(self.pts_px) == 0: 
            return
            
        # Use config values instead of hardcoded
        overlay_alpha = float(self.overlay_cfg.get('alpha', 0.2))
        background_alpha = float(self.overlay_cfg.get('background_alpha', 0.8))
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.pts_px], self.color)
        cv2.addWeighted(overlay, overlay_alpha, frame, background_alpha, 0, frame)
  
        cv2.polylines(frame, [self.pts_px], True, self.color, self.thickness)
        
        # Draw corner points with config
        corner_cfg = self.overlay_cfg.get('corners', {})
        corner_radius = int(corner_cfg.get('radius', 5))
        corner_color = tuple(corner_cfg.get('color', [0, 0, 255]))
        corner_border_color = tuple(corner_cfg.get('border_color', [255, 255, 255]))
        corner_border_thickness = int(corner_cfg.get('border_thickness', 1))
        
        for i, pt in enumerate(self.pts_px):
            cv2.circle(frame, tuple(pt), corner_radius, corner_color, -1)
            cv2.circle(frame, tuple(pt), corner_radius + 2, corner_border_color, corner_border_thickness)
            
        # Draw ROI label with config
        label_cfg = self.overlay_cfg.get('roi_label', {})
        font = getattr(cv2, label_cfg.get('font', 'FONT_HERSHEY_SIMPLEX'))
        font_scale = float(label_cfg.get('scale', 0.7))
        font_thickness = int(label_cfg.get('thickness', 2))
        label_offset_y = int(label_cfg.get('offset_y', 10))
        
        cv2.putText(frame, f"ROI: {self.roi_type.upper()}", 
                   (self.pts_px[0][0], self.pts_px[0][1] - label_offset_y), 
                   font, font_scale, self.color, font_thickness)
        
        # Draw counting line if enabled
        if self.counting_enabled and self.counting_line_coords:
            pt1, pt2 = self.counting_line_coords
            cv2.line(frame, pt1, pt2, self.counting_line_color, self.counting_line_thickness)
    
    def count_objects_by_line(self, detections: List) -> CountingStats:
        """Count objects ABOVE and BELOW the horizontal counting line"""
        stats = CountingStats()
        
        if not self.counting_enabled or not self.counting_line_coords:
            return stats
            
        line_y = self.counting_line_coords[0][1]  # Y coordinate of horizontal line
        
        for x, y, w, h, conf, class_id in detections:
            # Use center point of detection box
            center_y = y + h // 2
            
            if center_y < line_y:
                stats.before_line += 1  # Above the line
            else:
                stats.after_line += 1   # Below the line
                
        return stats
    
    def point_in_roi(self, point: Tuple[int, int]) -> bool:
        if not self.enabled or len(self.pts_px) == 0:
            return True
        return cv2.pointPolygonTest(self.pts_px, point, False) >= 0

class DetectionProcessor:
    def __init__(self, cfg: Dict[str, Any], model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        # SỬA: Lấy detection config từ đúng path trong JSON
        det_cfg = cfg.get('processing', {}).get('detection', {})
        self.conf_thresh = float(det_cfg.get('conf_thresh', 0.50))
        self.nms_thresh = float(det_cfg.get('nms_threshold', 0.4))
        self.allowed_classes = set(a.lower() for a in det_cfg.get('allowed_classes', []))
    
    def process(self, pred: np.ndarray, roi_bbox: Tuple[int, int, int, int], 
                frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float, int]]:
        h, w = frame_shape
        roi_x, roi_y, roi_w, roi_h = roi_bbox
        
        valid_mask = pred[0, 0, :, 2] > self.conf_thresh
        valid_dets = pred[0, 0, valid_mask]
        
        if len(valid_dets) == 0:  return []
        
        boxes, confidences, class_ids = [], [], []
        
        for det in valid_dets:
            class_id = int(det[1])
            
            if self.allowed_classes and class_id < len(self.model_cfg.classes):
                if self.model_cfg.classes[class_id].lower() not in self.allowed_classes:
                    continue
            box = det[3:7]
            if roi_w > 0 and roi_h > 0:
                x1 = int(box[0] * roi_w + roi_x)
                y1 = int(box[1] * roi_h + roi_y)
                x2 = int(box[2] * roi_w + roi_x)
                y2 = int(box[3] * roi_h + roi_y)
            else:
                x1, y1 = int(box[0] * w), int(box[1] * h)
                x2, y2 = int(box[2] * w), int(box[3] * h)
            
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                boxes.append([x1, y1, bw, bh])
                confidences.append(float(det[2]))
                class_ids.append(class_id)

        if not boxes: return []
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        if len(indices) == 0: return []

        indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 
                 confidences[i], class_ids[i]) for i in indices]

class Visualizer:
    def __init__(self, cfg: Dict[str, Any], show: bool):
        # SỬA: Lấy display config từ đúng path trong JSON
        display_cfg = cfg.get('processing', {}).get('display', {})
        self.show = show
        self.max_w = int(display_cfg.get('max_width', 1280))
        self.max_h = int(display_cfg.get('max_height', 720))
        self.window_name = display_cfg.get('window_name', 'Traffic Monitor')
        
        # Detection box styling from config
        self.box_cfg = display_cfg.get('detection_box', {})
        self.box_color = tuple(self.box_cfg.get('color', [0, 255, 0]))
        self.box_thickness = int(self.box_cfg.get('thickness', 4))
        
        # Label styling from config
        self.label_cfg = display_cfg.get('label', {})
        self.label_font = getattr(cv2, self.label_cfg.get('font', 'FONT_HERSHEY_SIMPLEX'))
        self.label_scale = float(self.label_cfg.get('scale', 1.0))
        self.label_color = tuple(self.label_cfg.get('color', [0, 199, 200]))
        self.label_thickness = int(self.label_cfg.get('thickness', 2))
        self.label_offset_x = int(self.label_cfg.get('offset_x', 2))
        self.label_offset_y = int(self.label_cfg.get('offset_y', 5))
        
        # SỬA: Lấy counting config từ đúng path trong JSON
        self.counting_cfg = cfg.get('processing', {}).get('counting', {})
        self.counting_enabled = self.counting_cfg.get('enabled', True)
        self.display_stats = self.counting_cfg.get('display_stats', True)
        
        if self.display_stats:
            stats_cfg = self.counting_cfg.get('stats_font', {})
            self.stats_pos = self.counting_cfg.get('stats_position', {})
            self.stats_x = int(self.stats_pos.get('x', 50))
            self.stats_y = int(self.stats_pos.get('y', 50))
            self.stats_font = getattr(cv2, stats_cfg.get('font', 'FONT_HERSHEY_SIMPLEX'))
            self.stats_scale = float(stats_cfg.get('scale', 1.2))
            self.stats_color = tuple(stats_cfg.get('color', [255, 255, 255]))
            self.stats_thickness = int(stats_cfg.get('thickness', 2))
            self.stats_line_spacing = int(stats_cfg.get('line_spacing', 35))
            
            # Background config
            bg_cfg = stats_cfg.get('background', {})
            self.stats_bg_enabled = bg_cfg.get('enabled', True)
            self.stats_bg_color = tuple(bg_cfg.get('color', [0, 0, 0]))
            self.stats_bg_alpha = float(bg_cfg.get('alpha', 0.7))
            self.stats_bg_padding = int(bg_cfg.get('padding', 10))
        
        if show:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.max_w, self.max_h)
    
    def draw_detections(self, frame: np.ndarray, detections: List, classes: List[str]) -> None:
        for x, y, w, h, conf, class_id in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)      
            if class_id < len(classes):
                label = f"{classes[class_id]}:{conf:.2f}"
                cv2.putText(frame, label, (x + self.label_offset_x, y - self.label_offset_y),
                           self.label_font, self.label_scale, self.label_color, self.label_thickness)
    
    def draw_counting_stats(self, frame: np.ndarray, stats: CountingStats) -> None:
        """Draw counting statistics on frame"""
        if not self.display_stats or not self.counting_enabled:
            return
            
        # Prepare text lines
        texts = [
            f"Before Line: {stats.before_line}",
            f"After Line: {stats.after_line}",
            f"Total: {stats.before_line + stats.after_line}"
        ]
        
        # Calculate background rectangle if enabled
        if self.stats_bg_enabled:
            # Calculate text dimensions
            text_sizes = [cv2.getTextSize(text, self.stats_font, self.stats_scale, self.stats_thickness)[0] 
                         for text in texts]
            max_width = max(size[0] for size in text_sizes)
            total_height = len(texts) * self.stats_line_spacing
            
            # Draw background
            bg_x1 = self.stats_x - self.stats_bg_padding
            bg_y1 = self.stats_y - self.stats_bg_padding - text_sizes[0][1]
            bg_x2 = self.stats_x + max_width + self.stats_bg_padding
            bg_y2 = self.stats_y + total_height - self.stats_line_spacing + self.stats_bg_padding
            
            # Create overlay for semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.stats_bg_color, -1)
            cv2.addWeighted(overlay, self.stats_bg_alpha, frame, 1 - self.stats_bg_alpha, 0, frame)
        
        # Draw text lines
        for i, text in enumerate(texts):
            y_pos = self.stats_y + i * self.stats_line_spacing
            cv2.putText(frame, text, (self.stats_x, y_pos),
                       self.stats_font, self.stats_scale, self.stats_color, self.stats_thickness)
    
    def display(self, frame: np.ndarray):
        if not self.show:
            return False
        
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1) 

@contextmanager
def video_capture_context(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        yield cap
    finally:
        cap.release()

def load_caffe_model(args, cfg: Dict[str, Any]):
    model = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
    
    # Get backend and target from config instead of hardcoded
    backend_cfg = cfg.get('model', {}).get('backend', {})
    backend_name = backend_cfg.get('name', 'opencv').lower()
    target_name = backend_cfg.get('target', 'cpu').lower()
    use_opencl = backend_cfg.get('use_opencl', True)
    
    # Default to OpenCV backend and CPU target
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU

    # Use OpenCL if available and enabled in config
    if use_opencl and os.environ.get('OPENCV_DNN_OPENCL', '0') == '1':
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            target = cv2.dnn.DNN_TARGET_OPENCL
    
    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    
    return model

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    try:
        cfg = load_runtime_config(args)

        setup_cpu_affinity(args.core)
        
        # Load model config from config file instead of hardcoded defaults
        model_cfg_dict = cfg.get('model', {})
        model_cfg = ModelConfig(
            input_width=int(model_cfg_dict.get('input_width', 300)),
            input_height=int(model_cfg_dict.get('input_height', 300)),
            scale_factor=float(model_cfg_dict.get('scale_factor', 0.00784313725490196)),
            mean_values=model_cfg_dict.get('mean_values', [127.5, 127.5, 127.5]),
            swap_rb=bool(model_cfg_dict.get('swap_rb', False)),
            crop=bool(model_cfg_dict.get('crop', False)),
            classes=model_cfg_dict.get('classes', [])
        )
        
        roi_processor = ROIProcessor(cfg)
        detection_processor = DetectionProcessor(cfg, model_cfg)
        visualizer = Visualizer(cfg, args.show)
        
        # Load Caffe model with config
        model = load_caffe_model(args, cfg)
        
        state = ProcessingState(start_time=time.time())
        
        # Get processing config instead of hardcoded values
        proc_cfg = cfg.get('processing', {})
        progress_interval = int(proc_cfg.get('progress_update_interval', 10))
        ema_alpha = float(proc_cfg.get('ema_alpha', 0.2))
        min_loop_time = float(proc_cfg.get('min_loop_time', 1e-6))
        frame_interval_sleep = bool(proc_cfg.get('frame_interval_sleep', True))
        
        # Get video config
        video_cfg = cfg.get('video', {})
        default_fps_fallback = float(video_cfg.get('default_fps_fallback', 30.0))
        
        # Main processing loop
        with video_capture_context(args.source) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS) or default_fps_fallback
            frame_interval = 1.0 / args.fps if frame_interval_sleep else 0
            
            print(f"Video: {total_frames} frames @ {original_fps:.1f} FPS")
            print(f"Processing at: {args.fps} FPS")
            
            while True:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video")
                    break
                
                state.frame_count += 1
                
                # ROI processing
                roi_frame, roi_bbox = roi_processor.apply_roi(frame)
                roi_processor.draw_overlay(frame)
                
                # Prepare Caffe model input with config values
                input_size = (model_cfg.input_width, model_cfg.input_height)
                resized_roi = cv2.resize(roi_frame, input_size)
                
                # Caffe inference
                inference_start = time.time()
                
                blob = cv2.dnn.blobFromImage(
                    resized_roi,
                    scalefactor=model_cfg.scale_factor,
                    size=input_size,
                    mean=tuple(model_cfg.mean_values),
                    swapRB=model_cfg.swap_rb,
                    crop=model_cfg.crop
                )
                model.setInput(blob)
                pred = model.forward()
                
                inference_time = time.time() - inference_start
                state.total_inference_time += inference_time
                
                # Process detections
                detections = detection_processor.process(pred, roi_bbox, frame.shape[:2])
                
                # Filter by ROI
                filtered_dets = [d for d in detections 
                                if roi_processor.point_in_roi((d[0] + d[2]//2, d[1] + d[3]//2))]
                
                # Count objects by counting line
                counting_stats = roi_processor.count_objects_by_line(filtered_dets)
                
                # Draw detections
                visualizer.draw_detections(frame, filtered_dets, model_cfg.classes)
                
                # Draw counting statistics
                visualizer.draw_counting_stats(frame, counting_stats)
                
                # Calculate metrics with config values
                current_fps = 1.0 / max(time.time() - loop_start, min_loop_time)
                
                # Update EMA with config alpha
                if state.ema_fps is None:
                    state.ema_fps = current_fps
                    state.ema_inf = inference_time * 1000
                else:
                    state.ema_fps = (1 - ema_alpha) * state.ema_fps + ema_alpha * current_fps
                    state.ema_inf = (1 - ema_alpha) * state.ema_inf + ema_alpha * (inference_time * 1000)
                
                if visualizer.display(frame): break

                # Use config for progress update interval
                if state.frame_count % progress_interval == 0:
                    progress = print_progress_bar(state.frame_count, total_frames)
                    print(f"\r{progress} | FPS:{state.ema_fps:.1f} | Objects:{len(filtered_dets)} | Before:{counting_stats.before_line} | After:{counting_stats.after_line}", 
                          end="", flush=True)

                # Frame rate control with config
                if frame_interval_sleep and frame_interval > 0:
                    elapsed = time.time() - loop_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if args.show:
            cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    sys.exit(main())