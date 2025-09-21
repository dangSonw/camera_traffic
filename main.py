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
class ModelConfig:
    input_width: int = 300
    input_height: int = 300
    scale_factor: float = 0.00784313725490196
    mean_values: List[float] = None
    swap_rb: bool = False
    crop: bool = False
    classes: List[str] = None
    
    def __post_init__(self):
        if self.mean_values is None:
            self.mean_values = [127.5, 127.5, 127.5]
        if self.classes is None:
            self.classes = []

class ROIProcessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.roi_cfg = cfg.get('model', {}).get('roi', {})
        self.enabled = self.roi_cfg.get('enabled', False)
        self.roi_type = self.roi_cfg.get('type', 'rectangle')
        self.pts_px = self._get_roi_points()
        self.color = tuple(self.roi_cfg.get('color', [0, 255, 0]))
        self.thickness = int(self.roi_cfg.get('thickness', 2))
        
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
        if not self.enabled or len(self.pts_px) == 0: return
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.pts_px], self.color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
  
        cv2.polylines(frame, [self.pts_px], True, self.color, self.thickness)
        
        for i, pt in enumerate(self.pts_px):
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(pt), 7, (255, 255, 255), 1)
        cv2.putText(frame, f"ROI: {self.roi_type.upper()}", 
                   (self.pts_px[0][0], self.pts_px[0][1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
    
    def point_in_roi(self, point: Tuple[int, int]) -> bool:
        if not self.enabled or len(self.pts_px) == 0:
            return True
        return cv2.pointPolygonTest(self.pts_px, point, False) >= 0

class DetectionProcessor:
    def __init__(self, cfg: Dict[str, Any], model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        det_cfg = cfg.get('detection', {})
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
    def __init__(self, show: bool, max_width: int = 1280, max_height: int = 720):
        self.show = show
        self.max_w = max_width
        self.max_h = max_height
        
        if show:
            cv2.namedWindow('Traffic Monitor', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Traffic Monitor', self.max_w, self.max_h)
    
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List, classes: List[str]) -> None:
        for x, y, w, h, conf, class_id in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)      
            if class_id < len(classes):
                label = f"{classes[class_id]}:{conf:.2f}"
                cv2.putText(frame, label, (x + 2, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 199, 200), 2)
    
    def display(self, frame: np.ndarray):
        if not self.show:
            return False
        
        cv2.imshow('Traffic Monitor', frame)
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

def load_caffe_model(args):
    model = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU

    if os.environ.get('OPENCV_DNN_OPENCL', '0') == '1':
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
        used_cores = setup_cpu_affinity(args.core)

        cfg = load_runtime_config(args)
        
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
        visualizer = Visualizer(args.show, cfg.get('display', {}).get('max_width', 1280), cfg.get('display', {}).get('max_height', 720))
        
        # Load Caffe model
        model = load_caffe_model(args)
        
        state = ProcessingState(start_time=time.time())
        
        # Main processing loop
        with video_capture_context(args.source) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / args.fps
            
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
                
                # Prepare Caffe model input
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
                
                # Draw detections
                Visualizer.draw_detections(frame, filtered_dets, model_cfg.classes)
                
                # Calculate metrics
                current_fps = 1.0 / max(time.time() - loop_start, 1e-6)
                
                # Update EMA (Exponential Moving Average)
                if state.ema_fps is None:
                    state.ema_fps = current_fps
                    state.ema_inf = inference_time * 1000
                else:
                    alpha = 0.2
                    state.ema_fps = (1 - alpha) * state.ema_fps + alpha * current_fps
                    state.ema_inf = (1 - alpha) * state.ema_inf + alpha * (inference_time * 1000)
                
                if visualizer.display(frame): break

                if state.frame_count % 10 == 0:
                    progress = print_progress_bar(state.frame_count, total_frames)
                    print(f"\r{progress} | FPS:{state.ema_fps:.1f} | Objects:{len(filtered_dets)}", 
                          end="", flush=True)

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