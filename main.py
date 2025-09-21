import time
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import logging
import traceback
from typing import List, Tuple, Optional, Dict, Any
import threading
import os
from dataclasses import dataclass
from contextlib import contextmanager

# Import modules
from traffic_monitor.system_utils import (
    start_quit_listener, SystemMonitor, setup_cpu_affinity, print_progress_bar,
)
from traffic_monitor.config import build_arg_parser, load_runtime_config
from traffic_monitor.logging_utils import MetricLogger
from traffic_monitor.counter import CounterState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional Windows keyboard support
try:
    import msvcrt
except ImportError:
    msvcrt = None

@dataclass
class ProcessingState:
    """Trạng thái xử lý video"""
    frame_count: int = 0
    total_inference_time: float = 0.0
    start_time: float = 0.0
    ema_fps: Optional[float] = None
    ema_inf: Optional[float] = None
    last_print_t: float = 0.0

@dataclass 
class ModelConfig:
    """Cấu hình model"""
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
    """Xử lý vùng ROI"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.roi_cfg = cfg.get('model', {}).get('roi', {})
        self.enabled = self.roi_cfg.get('enabled', False)
        self.roi_type = self.roi_cfg.get('type', 'rectangle')
        self.pts_px = self._get_roi_points()
        
    def _get_roi_points(self) -> List[List[int]]:
        """Lấy các điểm ROI"""
        if not self.enabled:
            return []
            
        if self.roi_type == 'trapezoid':
            return self.roi_cfg.get('trapezoid', {}).get('points', 
                [[300, 300], [1000, 300], [1200, 800], [100, 800]])
        else:  # rectangle
            rect = self.roi_cfg.get('rectangle', {})
            x, y = rect.get('x', 400), rect.get('y', 300)
            w, h = rect.get('width', 800), rect.get('height', 600)
            return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    
    def create_roi_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Tạo frame ROI và trả về bounding box"""
        if not self.enabled:
            h, w = frame.shape[:2]
            return frame.copy(), (0, 0, w, h)
            
        h, w = frame.shape[:2]
        
        # Tạo mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.pts_px, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        
        # Áp dụng mask
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Tính bounding box
        x_coords = [pt[0] for pt in self.pts_px]
        y_coords = [pt[1] for pt in self.pts_px]
        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
        
        # Cắt frame
        roi_rect = roi_frame[y_min:y_max, x_min:x_max].copy() if x_min < x_max and y_min < y_max else frame.copy()
        
        return roi_rect, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_roi_overlay(self, frame: np.ndarray) -> None:
        """Vẽ overlay ROI lên frame"""
        if not self.enabled:
            return
            
        pts = np.array(self.pts_px, np.int32).reshape((-1, 1, 2))
        color = self.roi_cfg.get('color', [0, 255, 0])
        thickness = int(self.roi_cfg.get('thickness', 2))
        
        # Tạo overlay với độ trong suốt
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Vẽ đường viền và labels
        cv2.polylines(frame, [pts], True, color, thickness)
        cv2.putText(frame, f"ROI: {self.roi_type.upper()}", 
                   (self.pts_px[0][0], self.pts_px[0][1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Đánh dấu góc
        for i, pt in enumerate(self.pts_px):
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i+1), (pt[0]+5, pt[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def point_in_roi(self, point: Tuple[int, int]) -> bool:
        """Kiểm tra điểm có trong ROI không"""
        if not self.enabled:
            return True
        return self._point_in_polygon(point, self.pts_px)
    
    @staticmethod
    def _point_in_polygon(point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        """Ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

class DetectionProcessor:
    """Xử lý detection"""
    
    def __init__(self, cfg: Dict[str, Any], model_cfg: ModelConfig):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.detection_cfg = cfg.get('detection', {})
        self.conf_thresh = float(self.detection_cfg.get('conf_thresh', 0.50))
        self.nms_thresh = float(self.detection_cfg.get('nms_threshold', 0.4))
        self.allowed_classes = set(a.lower() for a in self.detection_cfg.get('allowed_classes', []))
    
    def process_detections(self, pred: np.ndarray, roi_bbox: Tuple[int, int, int, int], 
                          frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float, int]]:
        """Xử lý kết quả detection"""
        h, w = frame_shape
        roi_x, roi_y, roi_w, roi_h = roi_bbox
        
        boxes_for_nms, confidences, class_ids = [], [], []
        
        for i in range(pred.shape[2]):
            confidence = pred[0, 0, i, 2]
            if confidence <= self.conf_thresh:
                continue
                
            class_id = int(pred[0, 0, i, 1])
            
            # Filter classes
            if self.allowed_classes:
                if class_id >= len(self.model_cfg.classes):
                    continue
                class_name = self.model_cfg.classes[class_id]
                if class_name.lower() not in self.allowed_classes:
                    continue
            
            # Scale coordinates
            box = pred[0, 0, i, 3:7]
            if roi_w > 0 and roi_h > 0:  # ROI enabled
                x1 = int(box[0] * roi_w + roi_x)
                y1 = int(box[1] * roi_h + roi_y)
                x2 = int(box[2] * roi_w + roi_x)
                y2 = int(box[3] * roi_h + roi_y)
            else:  # Full frame
                x1, y1 = int(box[0] * w), int(box[1] * h)
                x2, y2 = int(box[2] * w), int(box[3] * h)
            
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            if bw > 0 and bh > 0:
                boxes_for_nms.append([x1, y1, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        # Apply NMS
        if not boxes_for_nms:
            return []
            
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, self.conf_thresh, self.nms_thresh)
        if len(indices) == 0:
            return []
        
        detections = []
        for idx in (indices.flatten() if hasattr(indices, 'flatten') else indices):
            x, y, bw, bh = boxes_for_nms[idx]
            detections.append((x, y, bw, bh, confidences[idx], class_ids[idx]))
        
        return detections

class ModelLoader:
    """Tải model"""
    
    @staticmethod
    def load_caffe_model(prototxt: str, weights: str) -> cv2.dnn.Net:
        """Tải Caffe model"""
        net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        
        # Set backend/target
        use_opencl = os.environ.get('OPENCV_DNN_OPENCL', '0') == '1'
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CPU
        
        if use_opencl:
            try:
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.haveOpenCL():
                    try:
                        target = cv2.dnn.DNN_TARGET_OPENCL_FP16
                    except Exception:
                        target = cv2.dnn.DNN_TARGET_OPENCL
            except Exception:
                pass
        
        net.setPreferableBackend(backend)
        net.setPreferableTarget(target)
        return net
    
    @staticmethod
    def load_tflite_model(model_path: str):
        """Tải TFLite model"""
        try:
            import importlib
            tflite_runtime = importlib.import_module('tflite_runtime.interpreter')
            TFLiteInterpreter = tflite_runtime.Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
        
        interpreter = TFLiteInterpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

class DisplayManager:
    """Quản lý hiển thị"""
    
    def __init__(self, cfg: Dict[str, Any], show: bool = False):
        self.cfg = cfg
        self.show = show
        self.display_cfg = cfg.get('display', {})
        self.max_w = int(self.display_cfg.get('max_width', 1280))
        self.max_h = int(self.display_cfg.get('max_height', 720))
        
        if show:
            cv2.namedWindow('Traffic Monitor', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Traffic Monitor', self.max_w, self.max_h)
    
    def create_display_panels(self, frame: np.ndarray, roi_frame: np.ndarray, 
                             model_input: np.ndarray, detections: List) -> np.ndarray:
        """Tạo panels hiển thị"""
        if not self.show:
            return frame
        
        col_w = max(2, self.max_w // 2)
        col_h = max(2, self.max_h)
        row_h_top = col_h // 2
        row_h_bottom = col_h - row_h_top
        
        # Fit panels vào kích thước cố định
        left_panel = self._fit_into_box(frame, col_w, col_h)
        tr_panel = self._fit_into_box(roi_frame, col_w, row_h_top)
        br_panel = self._fit_into_box(model_input, col_w, row_h_bottom)
        
        # Tạo combined display
        combined = np.zeros((col_h, col_w * 2, 3), dtype=np.uint8)
        combined[0:col_h, 0:col_w] = left_panel
        combined[0:row_h_top, col_w:col_w + col_w] = tr_panel
        combined[row_h_top:col_h, col_w:col_w + col_w] = br_panel
        
        # Add labels
        h, w = frame.shape[:2]
        cv2.putText(combined, f'Original {w}x{h}', (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f'ROI {roi_frame.shape[1]}x{roi_frame.shape[0]}', 
                   (col_w + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f'Model Input', 
                   (col_w + 10, row_h_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return combined
    
    def _fit_into_box(self, src: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
        """Fit ảnh vào box với tỷ lệ"""
        if src.size == 0:
            return np.zeros((box_h, box_w, 3), dtype=np.uint8)
        
        h, w = src.shape[:2]
        if w == 0 or h == 0:
            return np.zeros((box_h, box_w, 3), dtype=np.uint8)
        
        scale = min(box_w / w, box_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        
        resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        off_x, off_y = (box_w - new_w) // 2, (box_h - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        
        return canvas
    
    def show_frame(self, frame: np.ndarray) -> bool:
        """Hiển thị frame, return True nếu quit"""
        if not self.show:
            return False
        
        cv2.imshow('Traffic Monitor', frame)
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')

@contextmanager
def video_capture_context(source):
    """Context manager cho VideoCapture"""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        yield cap
    finally:
        cap.release()

def validate_args(args) -> bool:
    """Validate arguments"""
    use_tflite = getattr(args, 'backend', 'caffe') == 'tflite'
    
    if use_tflite:
        if not getattr(args, 'tflite', None) or not Path(args.tflite).exists():
            logger.error(f"TFLite model not found: {getattr(args, 'tflite', 'None')}")
            return False
    else:
        weights = getattr(args, 'weights', None)
        prototxt = getattr(args, 'prototxt', None)
        if not weights or not Path(weights).exists():
            logger.error(f"Weights not found: {weights}")
            return False
        if not prototxt or not Path(prototxt).exists():
            logger.error(f"Prototxt not found: {prototxt}")
            return False
    
    source_arg = str(args.source)
    is_url = source_arg.startswith(('rtsp://', 'http://', 'https://'))
    is_cam_index = source_arg.isdigit()
    if not is_url and not is_cam_index and not Path(source_arg).exists():
        logger.error(f"Source not found: {args.source}")
        return False
    
    return True

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    if not validate_args(args):
        return 1
    
    try:
        # Setup
        used_cores = setup_cpu_affinity(args.core)
        cfg = load_runtime_config(args)
        
        # Override configs
        if getattr(args, 'sample_interval', None) is not None:
            cfg['sample_interval_sec'] = max(0.0, float(args.sample_interval))
        
        # Initialize components
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
        display_manager = DisplayManager(cfg, args.show)
        
        # Load model
        use_tflite = getattr(args, 'backend', 'caffe') == 'tflite'
        if use_tflite:
            model = ModelLoader.load_tflite_model(args.tflite)
            logger.info(f"Loaded TFLite model: {Path(args.tflite).name}")
        else:
            model = ModelLoader.load_caffe_model(args.prototxt, args.weights)
            logger.info(f"Loaded Caffe model: {Path(args.weights).name}")
        
        # Setup monitoring
        monitor = SystemMonitor(used_cores)
        metric_logger = MetricLogger()
        metric_logger.open()
        
        state = ProcessingState()
        state.start_time = time.time()
        
        counter_state = CounterState(cfg)
        stop_event = threading.Event()
        quit_thread = start_quit_listener(stop_event)
        
        print("=== STARTING VIDEO PROCESSING ===")
        
        # Main processing loop
        with video_capture_context(args.source) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / args.fps
            
            logger.info(f"Video: {total_frames} frames @ {original_fps:.1f} FPS")
            
            while True:
                if stop_event.is_set():
                    logger.info("Quit requested")
                    break
                
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video")
                    break
                
                state.frame_count += 1
                
                try:
                    # Process frame
                    inference_start = time.time()
                    
                    # ROI processing
                    roi_frame, roi_bbox = roi_processor.create_roi_frame(frame)
                    roi_processor.draw_roi_overlay(frame)
                    
                    # Prepare model input
                    resized_roi = cv2.resize(roi_frame, (model_cfg.input_width, model_cfg.input_height))
                    
                    # Create blob
                    blob = cv2.dnn.blobFromImage(
                        resized_roi,
                        scalefactor=model_cfg.scale_factor,
                        size=(model_cfg.input_width, model_cfg.input_height),
                        mean=tuple(model_cfg.mean_values),
                        swapRB=model_cfg.swap_rb,
                        crop=model_cfg.crop
                    )
                    
                    # Inference
                    if use_tflite:
                        # TFLite inference (simplified)
                        input_details = model.get_input_details()
                        model.set_tensor(input_details[0]['index'], np.expand_dims(resized_roi.astype(np.float32), 0))
                        model.invoke()
                        # Get outputs and convert to Caffe format
                        pred = np.array([[[]]])  # Placeholder - implement TFLite output parsing
                    else:
                        model.setInput(blob)
                        pred = model.forward()
                    
                    inference_time = time.time() - inference_start
                    state.total_inference_time += inference_time
                    
                    # Process detections
                    detections = detection_processor.process_detections(pred, roi_bbox, frame.shape[:2])
                    
                    # Filter detections by ROI
                    filtered_detections = []
                    for det in detections:
                        x, y, w, h = det[:4]
                        cx, cy = x + w // 2, y + h // 2
                        if roi_processor.point_in_roi((cx, cy)):
                            filtered_detections.append(det)
                    
                    # Draw detections
                    for x, y, w, h, conf, class_id in filtered_detections:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if class_id < len(model_cfg.classes):
                            label = f"{model_cfg.classes[class_id]}:{conf:.2f}"
                            cv2.putText(frame, label, (x, max(0, y - 5)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Update display
                    combined_display = display_manager.create_display_panels(frame, roi_frame, resized_roi, filtered_detections)
                    
                    # Add HUD info
                    cv2.putText(combined_display, f"Objects: {len(filtered_detections)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(combined_display, f"Inference: {inference_time*1000:.1f}ms", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if display_manager.show_frame(combined_display):
                        stop_event.set()
                    
                    # Metrics
                    counter_state.window_count = len(filtered_detections)
                    counter_state.total_count += len(filtered_detections)
                    
                except Exception as e:
                    logger.error(f"Frame {state.frame_count} failed: {e}")
                    continue
                
                # Frame rate control
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
                # Progress reporting
                if state.frame_count % 10 == 0 or state.frame_count <= 10:
                    current_fps = 1.0 / max(elapsed, 1e-6)
                    system_cpu, _, ram_usage = monitor.get_metrics()
                    
                    # EMA smoothing
                    if state.ema_fps is None:
                        state.ema_fps = current_fps
                        state.ema_inf = inference_time * 1000
                    else:
                        alpha = 0.2
                        state.ema_fps = (1 - alpha) * state.ema_fps + alpha * current_fps
                        state.ema_inf = (1 - alpha) * state.ema_inf + alpha * (inference_time * 1000)
                    
                    # Progress bar
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if total_frames > 0 else state.frame_count
                    progress_bar = print_progress_bar(current_pos, total_frames)
                    
                    now = time.time()
                    if now - state.last_print_t >= 1.0:
                        state.last_print_t = now
                        print(f"\r{progress_bar} | FPS:{state.ema_fps:.1f} | INF:{state.ema_inf:.0f}ms | CPU:{system_cpu:.0f}% | RAM:{ram_usage:.0f}MB ", 
                              end="", flush=True)
                    
                    # Write metrics
                    metric_logger.write(
                        total_time_ms=elapsed*1000.0,
                        fps=current_fps,
                        system_cpu=system_cpu,
                        ram_mb=ram_usage
                    )
        
        # Final stats
        if state.frame_count > 0:
            total_elapsed = time.time() - state.start_time
            avg_inference = state.total_inference_time / state.frame_count * 1000
            actual_fps = state.frame_count / total_elapsed
            
            print(f"\n{'='*60}")
            print("=== PROCESSING COMPLETE ===")
            print(f"Processed: {state.frame_count} frames")
            print(f"Total time: {total_elapsed:.1f}s")
            print(f"Avg inference: {avg_inference:.1f}ms")
            print(f"Actual FPS: {actual_fps:.2f}")
            print(f"{'='*60}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        return 1
    finally:
        try:
            metric_logger.close()
            if args.show:
                cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Cleanup completed")

if __name__ == "__main__":
    sys.exit(main())