import time
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import datetime
# Serial functionality removed (unused)

from traffic_monitor.system_utils import (
    setup_cpu_affinity,
)
from traffic_monitor.config import build_arg_parser, load_runtime_config

class TrafficLightState(Enum):
    RED = "RED"
    GREEN = "GREEN"

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
    scale_factor: float
    mean_values: List[float]
    swap_rb: bool
    crop: bool
    classes: List[str]

@dataclass
class TrafficMonitorState:
    current_light: TrafficLightState = TrafficLightState.GREEN
    last_detection_time: float = 0.0
    detection_interval: float = 2.0
    density_before: float = 0.0
    density_after: float = 0.0
    # Debug flags and thresholds for display
    before_high: bool = False
    before_low: bool = False
    before_empty: bool = False
    after_high: bool = False
    after_low: bool = False
    after_empty: bool = False
    th_before_high: float = 0.0
    th_before_low: float = 0.0
    th_before_empty: float = 0.0
    th_after_high: float = 0.0
    th_after_low: float = 0.0
    th_after_empty: float = 0.0
    source_info: str = ""
    frames_to_skip: int = 0
    # Performance and explanation
    recent_fps: float = 0.0
    recent_inf_ms: float = 0.0
    decision_reason: str = ""
    # Counts per region (per-frame) and their flags/thresholds
    count_before: int = 0
    count_after: int = 0
    # Only record counts; no longer used for decisions
    # EMA-based smoothing for densities and delta
    ema_alpha: float = 0.2
    ema_before: Optional[float] = None
    ema_after: Optional[float] = None
    delta_raw: float = 0.0
    delta_ema: Optional[float] = None
    # Band levels (VH/H/L/E) for before/after using EMA densities
    level_before: str = ""
    level_after: str = ""
    # Debounce/confirmation for state changes
    confirmed_light: 'TrafficLightState' = TrafficLightState.GREEN
    pending_light: Optional['TrafficLightState'] = None
    pending_seen: int = 0
    pending_match: int = 0
    pending_required: int = 3
    # Recent light results history for consensus
    recent_lights: List['TrafficLightState'] = field(default_factory=list)
    history_length: int = 3

class ROIProcessor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.roi_cfg = cfg.get('model', {}).get('roi', {})
        self.enabled = self.roi_cfg.get('enabled', False)
        # Force use of full_region ROI only
        self.roi_type = 'full_region'
        self.full_region_config = self.roi_cfg.get('full_region', {})
        self.pts_px = self._get_roi_points()
        self.color = tuple(self.roi_cfg.get('color', [0, 255, 0]))
        self.thickness = int(self.roi_cfg.get('thickness', 2))
        self.overlay_cfg = cfg.get('processing', {}).get('display', {}).get('overlay', {})
        self.counting_cfg = self.overlay_cfg.get('counting_line', {})
        # Always enable counting line for full_region to split densities
        self.counting_enabled = self.counting_cfg.get('enabled', True)
        self.counting_line_pos = float(self.counting_cfg.get('position', 0.5))
        self.counting_line_color = tuple(self.counting_cfg.get('color', [0, 0, 255]))
        self.counting_line_thickness = int(self.counting_cfg.get('thickness', 3))
        self.counting_extend_factor = float(self.counting_cfg.get('extend_factor', 0.1))
        
        self.counting_line_coords = self._calculate_counting_line()
    
    def switch_roi_type(self, new_type: str):
        # Obsolete: ROI always full_region now
        return
    
    def _get_roi_points(self) -> np.ndarray:
        return self._get_roi_points_for_type(self.roi_type)
    
    def _get_roi_points_for_type(self, roi_type: str) -> np.ndarray:
        if not self.enabled:
            return np.array([])
        # Only full_region supported
        return np.array(self.full_region_config.get('points', 
            [[300, 300], [1000, 300], [1200, 800], [100, 800]]), np.int32)
    
    def _calculate_counting_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Calculate HORIZONTAL counting line coordinates for the current ROI polygon"""
        if not self.counting_enabled or len(self.pts_px) < 4:
            return None

        pts = self.pts_px
        top_y = min(pts[0][1], pts[3][1])  
        bottom_y = max(pts[1][1], pts[2][1])  
        height = bottom_y - top_y
        
        line_y = int(top_y + height * self.counting_line_pos)

        t_left = (line_y - pts[0][1]) / (pts[1][1] - pts[0][1]) if pts[1][1] != pts[0][1] else 0
        line_x1 = int(pts[0][0] + t_left * (pts[1][0] - pts[0][0]))
        
        t_right = (line_y - pts[3][1]) / (pts[2][1] - pts[3][1]) if pts[2][1] != pts[3][1] else 0
        line_x2 = int(pts[3][0] + t_right * (pts[2][0] - pts[3][0]))

        extend_width = int((line_x2 - line_x1) * self.counting_extend_factor)
        line_x1 -= extend_width
        line_x2 += extend_width
        
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
    
    def draw_overlay(self, frame: np.ndarray, traffic_state: TrafficLightState = None) -> None:
        if not self.enabled or len(self.pts_px) == 0: 
            return
            
        if traffic_state == TrafficLightState.RED:
            overlay_color = (0, 0, 255)  
        elif traffic_state == TrafficLightState.GREEN:
            overlay_color = (0, 255, 0)  
        else:
            overlay_color = self.color 
            
        overlay_alpha = float(self.overlay_cfg.get('alpha', 0.2))
        background_alpha = float(self.overlay_cfg.get('background_alpha', 0.8))
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.pts_px], overlay_color)
        cv2.addWeighted(overlay, overlay_alpha, frame, background_alpha, 0, frame)
  
        cv2.polylines(frame, [self.pts_px], True, overlay_color, self.thickness)

        corner_cfg = self.overlay_cfg.get('corners', {})
        corner_radius = int(corner_cfg.get('radius', 5))
        corner_color = tuple(corner_cfg.get('color', [0, 0, 255]))
        corner_border_color = tuple(corner_cfg.get('border_color', [255, 255, 255]))
        corner_border_thickness = int(corner_cfg.get('border_thickness', 1))
        
        for i, pt in enumerate(self.pts_px):
            cv2.circle(frame, tuple(pt), corner_radius, corner_color, -1)
            cv2.circle(frame, tuple(pt), corner_radius + 2, corner_border_color, corner_border_thickness)

        label_cfg = self.overlay_cfg.get('roi_label', {})
        font = getattr(cv2, label_cfg.get('font', 'FONT_HERSHEY_SIMPLEX'))
        font_scale = float(label_cfg.get('scale', 0.7))
        font_thickness = int(label_cfg.get('thickness', 2))
        label_offset_y = int(label_cfg.get('offset_y', 10))
        
        roi_label = f"ROI: Full Detection"
        if traffic_state:
            roi_label += f" | Light: {traffic_state.value}"
            
        cv2.putText(frame, roi_label, 
                   (self.pts_px[0][0], self.pts_px[0][1] - label_offset_y), 
                   font, font_scale, overlay_color, font_thickness)
        if traffic_state == TrafficLightState.RED and self.roi_type == 'full_region':
            extra_color = (255, 0, 0)  
            extra_pts = self._get_roi_points_for_type('full_region')
            if len(extra_pts) > 0:
                cv2.fillPoly(overlay, [extra_pts], extra_color)
                cv2.addWeighted(overlay, overlay_alpha, frame, background_alpha, 0, frame)
                cv2.polylines(frame, [extra_pts], True, extra_color, self.thickness)

                for pt in extra_pts:
                    cv2.circle(frame, tuple(pt), corner_radius, corner_color, -1)
                    cv2.circle(frame, tuple(pt), corner_radius + 2, corner_border_color, corner_border_thickness)

                extra_label = "Extra ROI: Full Detection"
                cv2.putText(frame, extra_label, 
                            (extra_pts[0][0], extra_pts[0][1] - label_offset_y), 
                            font, font_scale, extra_color, font_thickness)

        if self.counting_enabled and self.counting_line_coords:
            pt1, pt2 = self.counting_line_coords
            cv2.line(frame, pt1, pt2, self.counting_line_color, self.counting_line_thickness)
    
    def count_objects_by_line(self, detections: List) -> CountingStats:
        stats = CountingStats()
        
        if not self.counting_enabled or not self.counting_line_coords:
            return stats
            
        line_y = self.counting_line_coords[0][1] 
        
        for x, y, w, h, conf, class_id in detections:
            center_y = y + h // 2
            
            if center_y < line_y:
                stats.before_line += 1 
            else:
                stats.after_line += 1   
                
        return stats
    
    def count_all_objects(self, detections: List) -> int:
        return len(detections)
    
    def point_in_roi(self, point: Tuple[int, int]) -> bool:
        if not self.enabled or len(self.pts_px) == 0:
            return True
        return cv2.pointPolygonTest(self.pts_px, point, False) >= 0

    def compute_densities(self, detections: List[Tuple[int, int, int, int, float, int]],
                          roi_bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Compute area densities of detections before and after the horizontal counting line.
        Density is approximated as sum(box areas classified by center_y) divided by the
        rectangular ROI area above/below the line.
        """
        if not self.counting_enabled or not self.counting_line_coords:
            return 0.0, 0.0

        roi_x, roi_y, roi_w, roi_h = roi_bbox
        if roi_w <= 0 or roi_h <= 0:
            return 0.0, 0.0

        line_y = self.counting_line_coords[0][1]

        area_before_region = max(0, min(line_y - roi_y, roi_h)) * max(roi_w, 0)
        area_after_region = max(0, (roi_y + roi_h) - line_y) * max(roi_w, 0)
        if area_before_region <= 0 and area_after_region <= 0:
            return 0.0, 0.0

        sum_area_before, sum_area_after = 0.0, 0.0
        for x, y, w, h, conf, class_id in detections:
            box_area = max(w, 0) * max(h, 0)
            center_y = y + h // 2
            if center_y < line_y:
                sum_area_before += box_area
            else:
                sum_area_after += box_area

        density_before = (sum_area_before / area_before_region) if area_before_region > 0 else 0.0
        density_after = (sum_area_after / area_after_region) if area_after_region > 0 else 0.0
        return density_before, density_after

class DetectionProcessor:
    def __init__(self, cfg: Dict[str, Any], model_cfg: ModelConfig):
        self.model_cfg = model_cfg
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
        display_cfg = cfg.get('processing', {}).get('display', {})
        self.show = show
        self.max_w = int(display_cfg.get('max_width', 1280))
        self.max_h = int(display_cfg.get('max_height', 720))
        self.window_name = display_cfg.get('window_name', 'Traffic Monitor')
        
        self.box_cfg = display_cfg.get('detection_box', {})
        self.box_color = tuple(self.box_cfg.get('color', [0, 255, 0]))
        self.box_thickness = int(self.box_cfg.get('thickness', 4))

        self.label_cfg = display_cfg.get('label', {})
        self.label_font = getattr(cv2, self.label_cfg.get('font', 'FONT_HERSHEY_SIMPLEX'))
        self.label_scale = float(self.label_cfg.get('scale', 1.0))
        self.label_color = tuple(self.label_cfg.get('color', [0, 199, 200]))
        self.label_thickness = int(self.label_cfg.get('thickness', 2))
        self.label_offset_x = int(self.label_cfg.get('offset_x', 2))
        self.label_offset_y = int(self.label_cfg.get('offset_y', 5))
        
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
    
    def draw_counting_stats(self, frame: np.ndarray, stats: CountingStats, 
                           traffic_state: TrafficMonitorState = None) -> None:
        if not self.display_stats or not self.counting_enabled:
            return

        texts = [
            f"Before Line: {stats.before_line}",
            f"After Line: {stats.after_line}",
            f"Total: {stats.before_line + stats.after_line}"
        ]

        if traffic_state:
            texts.append(f"Traffic Light: {traffic_state.current_light.value}")
            # Band-based summary (4-state): VH/H/L/E
            texts.append(f"Band Before: {traffic_state.level_before}  (dens: {traffic_state.density_before:.3f})")
            texts.append(f"Band After : {traffic_state.level_after}   (dens: {traffic_state.density_after:.3f})")
            # EMA for stability debugging (delta removed per request)
            if traffic_state.ema_before is not None and traffic_state.ema_after is not None:
                texts.append(f"EMA Bf/Af: {traffic_state.ema_before:.3f}/{traffic_state.ema_after:.3f}")
            texts.append(f"Bf: {traffic_state.th_before_high:.3f}/{traffic_state.th_before_low:.3f}/{traffic_state.th_before_empty:.3f}")
            texts.append(f"Af: {traffic_state.th_after_high:.3f}/{traffic_state.th_after_low:.3f}/{traffic_state.th_after_empty:.3f}")
            
        if self.stats_bg_enabled:
            text_sizes = [cv2.getTextSize(text, self.stats_font, self.stats_scale, self.stats_thickness)[0] 
                         for text in texts]
            max_width = max(size[0] for size in text_sizes)
            total_height = len(texts) * self.stats_line_spacing

            bg_x1 = self.stats_x - self.stats_bg_padding
            bg_y1 = self.stats_y - self.stats_bg_padding - text_sizes[0][1]
            bg_x2 = self.stats_x + max_width + self.stats_bg_padding
            bg_y2 = self.stats_y + total_height - self.stats_line_spacing + self.stats_bg_padding

            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.stats_bg_color, -1)
            cv2.addWeighted(overlay, self.stats_bg_alpha, frame, 1 - self.stats_bg_alpha, 0, frame)
        
        for i, text in enumerate(texts):
            y_pos = self.stats_y + i * self.stats_line_spacing
            color = self.stats_color
            if i == 3 and traffic_state: 
                if traffic_state.current_light == TrafficLightState.RED:
                    color = (0, 0, 255)
                elif traffic_state.current_light == TrafficLightState.GREEN:
                    color = (0, 255, 0)
                    
            cv2.putText(frame, text, (self.stats_x, y_pos),
                       self.stats_font, self.stats_scale, color, self.stats_thickness)
    
    def display(self, frame: np.ndarray):
        if not self.show:
            return False
        try:
            cv2.imshow(self.window_name, frame)
            return cv2.waitKey(1) == 27
        except Exception as e:
            # Fallback: disable display if backend (e.g., Qt/Wayland) is unavailable
            print(f"[WARN] Display disabled due to backend error: {e}")
            self.show = False
            return False

def _normalize_rtsp_url(url: str) -> str:
    """Normalize RTSP URL so that only the first '?' remains and the rest are '&'."""
    if not isinstance(url, str):
        return url
    if url.startswith('rtsp://') and url.count('?') > 1:
        first = url.find('?')
        return url[:first + 1] + url[first + 1:].replace('?', '&')
    return url

@contextmanager
def video_capture_context(source):
    """Create a cv2.VideoCapture with better defaults for RTSP (TCP) and retry."""
    src_str = str(source)
    cap = None
    # Prefer FFMPEG for RTSP and enforce TCP to improve reliability
    if src_str.startswith('rtsp://'):
        # Keep local to this process; won't persist system-wide
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS',
                              'rtsp_transport;tcp|max_delay;5000000|stimeout;5000000')
        norm_url = _normalize_rtsp_url(src_str)
        cap = cv2.VideoCapture(norm_url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
    else:
        # Numeric webcam index
        if src_str.isdigit():
            cap = cv2.VideoCapture(int(src_str))
        else:
            cap = cv2.VideoCapture(src_str)

    # Small retry loop for flaky network inits
    attempts, opened = 5, False
    for _ in range(attempts):
        if cap is not None and cap.isOpened():
            opened = True
            break
        time.sleep(0.4)
    try:
        if not opened:
            raise ValueError(f"Cannot open video: {source}")
        yield cap
    finally:
        if cap is not None:
            cap.release()

def load_caffe_model(args, cfg: Dict[str, Any]):
    model = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
    
    backend_cfg = cfg.get('model', {}).get('backend', {})
    backend_name = backend_cfg.get('name', 'opencv').lower()
    target_name = backend_cfg.get('target', 'cpu').lower()
    use_opencl = backend_cfg.get('use_opencl', True)
    
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU

    if use_opencl and os.environ.get('OPENCV_DNN_OPENCL', '0') == '1':
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            target = cv2.dnn.DNN_TARGET_OPENCL
    
    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    
    return model

def process_density_logic(state, densities, cfg, now=None):
    """
    Decide GREEN/RED based on vehicle density in two regions (before & after the line).
    - densities = (before, after)
    - cfg = {'before': {'empty':..., 'low':..., 'high':...}, 'after': {...}}
    """
    now = now or time.time()
    if now - state.last_detection_time < state.detection_interval:
        return
    state.last_detection_time = now

    d_before, d_after = densities
    state.density_before, state.density_after = d_before, d_after

    # No EMA: use current densities directly
    eb, ea = d_before, d_after

    # Density band classifier
    def level(x, th):
        if x >= th['high']: return "VH"  # Very High
        if x > th['low']:  return "H"   # High
        if x > th['empty']:return "L"   # Low
        return "E"                     # Empty

    bf_lvl = level(eb, cfg['before'])
    af_lvl = level(ea, cfg['after'])
    state.level_before, state.level_after = bf_lvl, af_lvl

    # Update thresholds for UI display
    try:
        state.th_before_high = float(cfg.get('before', {}).get('high', 0.0))
        state.th_before_low = float(cfg.get('before', {}).get('low', 0.0))
        state.th_before_empty = float(cfg.get('before', {}).get('empty', 0.0))
        state.th_after_high = float(cfg.get('after', {}).get('high', 0.0))
        state.th_after_low = float(cfg.get('after', {}).get('low', 0.0))
        state.th_after_empty = float(cfg.get('after', {}).get('empty', 0.0))
    except Exception:
        pass

    # -------------------------------
    # Decision logic (16 combinations simplified)
    # -------------------------------
    if bf_lvl in ("VH", "H") and af_lvl in ("L", "E"):
        result = TrafficLightState.RED
        reason = f"Front heavy ({bf_lvl}), back light ({af_lvl}) -> RED"
    elif bf_lvl in ("L", "E") and af_lvl in ("VH", "H"):
        result = TrafficLightState.GREEN
        reason = f"Front light ({bf_lvl}), back heavy ({af_lvl}) -> GREEN"
    elif bf_lvl in ("VH", "H") and af_lvl in ("VH", "H"):
        result = TrafficLightState.GREEN
        reason = f"Both sides heavy ({bf_lvl}, {af_lvl}) -> GREEN for relief"
    elif bf_lvl in ("L", "E") and af_lvl in ("L", "E"):
        result = TrafficLightState.RED
        reason = f"Both sides light ({bf_lvl}, {af_lvl}) -> RED (hold)"
    elif bf_lvl in ("VH", "H") and af_lvl == "L":
        result = TrafficLightState.RED
        reason = f"Front heavy ({bf_lvl}), back low ({af_lvl}) -> RED"
    elif bf_lvl == "L" and af_lvl in ("VH", "H"):
        result = TrafficLightState.GREEN
        reason = f"Front low ({bf_lvl}), back heavy ({af_lvl}) -> GREEN"
    else:
        # Intermediate cases (near-balanced)
        if eb > ea:
            result = TrafficLightState.RED
            reason = f"Front density > back ({eb:.3f} > {ea:.3f}) -> RED"
        else:
            result = TrafficLightState.GREEN
            reason = f"Back density >= front ({ea:.3f} >= {eb:.3f}) -> GREEN"

    # Consensus: change only when last N frames agree
    state.recent_lights.append(result)
    if len(state.recent_lights) > state.history_length:
        state.recent_lights = state.recent_lights[-state.history_length:]

    if len(state.recent_lights) == state.history_length and all(
        s == state.recent_lights[-1] for s in state.recent_lights
    ):
        state.current_light = state.recent_lights[-1]
    # else: keep current state.current_light
    state.decision_reason = reason


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    try:
        cfg = load_runtime_config(args)
        setup_cpu_affinity(args.core)
        
        model_cfg_dict = cfg.get('model', {})
        model_cfg = ModelConfig(
            scale_factor=float(model_cfg_dict.get('scale_factor', 0.00784313725490196)),
            mean_values=model_cfg_dict.get('mean_values', [127.5, 127.5, 127.5]),
            swap_rb=bool(model_cfg_dict.get('swap_rb', False)),
            crop=bool(model_cfg_dict.get('crop', False)),
            classes=model_cfg_dict.get('classes', [])
        )
        
        density_cfg_root = cfg.get('processing', {}).get('density', {})
        traffic_state = TrafficMonitorState(
            detection_interval=float(density_cfg_root.get('detection_interval', 2.0))
        )
        
        roi_processor = ROIProcessor(cfg)
        detection_processor = DetectionProcessor(cfg, model_cfg)
        # Create visualizer; if GUI backend unavailable (e.g., Qt/Wayland), fall back to headless
        try:
            visualizer = Visualizer(cfg, args.show)
        except Exception as e:
            print(f"[WARN] Failed to initialize display, running headless: {e}")
            visualizer = Visualizer(cfg, False)
        
        model = load_caffe_model(args, cfg)
        state = ProcessingState(start_time=time.time())

        proc_cfg = cfg.get('processing', {})
        ema_alpha = float(proc_cfg.get('ema_alpha', 0.2))
        traffic_state.ema_alpha = ema_alpha
        min_loop_time = float(proc_cfg.get('min_loop_time', 1e-6))

        video_cfg = cfg.get('video', {})
        
        with video_capture_context(args.source) as cap:
            # Determine source type and file skip policy
            source_str = str(args.source)
            is_numeric = source_str.isdigit()
            is_stream = isinstance(args.source, str) and (source_str.startswith('rtsp://') or source_str.startswith('http'))
            is_live = is_numeric or is_stream

            file_cfg = proc_cfg.get('file', {})
            frames_to_skip = 0 if is_live else int(file_cfg.get('skip_frame', 0))

            # Fill source info for debug
            if is_numeric:
                traffic_state.source_info = "webcam"
            elif is_stream:
                traffic_state.source_info = "ethernet"
            else:
                traffic_state.source_info = "file"
            traffic_state.frames_to_skip = frames_to_skip

            frame_idx = -1
            while True:
                loop_start = time.time()
                frame_idx += 1
                ret, frame = cap.read()
                
                if not ret:
                    print("\nEnd of video")
                    break

                state.frame_count += 1

                roi_frame, roi_bbox = roi_processor.apply_roi(frame)
                roi_processor.draw_overlay(frame, traffic_state.current_light)

                # Always use full ROI resolution for detection input
                input_size = (roi_frame.shape[1], roi_frame.shape[0])
                resized_roi = roi_frame.copy()

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

                detections = detection_processor.process(pred, roi_bbox, frame.shape[:2])
                filtered_dets = [d for d in detections
                                if roi_processor.point_in_roi((d[0] + d[2]//2, d[1] + d[3]//2))]

                # Compute counts and densities then decide state
                counting_stats = roi_processor.count_objects_by_line(filtered_dets)
                density_cfg = cfg.get('processing', {}).get('density', {})
                d_before, d_after = roi_processor.compute_densities(filtered_dets, roi_bbox)
                process_density_logic(traffic_state, (d_before, d_after), density_cfg)
                visualizer.draw_detections(frame, filtered_dets, model_cfg.classes)
                visualizer.draw_counting_stats(frame, counting_stats, traffic_state)
                
                # Compute FPS based on loop duration
                current_time = time.time()
                current_fps = 1.0 / max(current_time - loop_start, min_loop_time)

                traffic_state.recent_fps = current_fps
                traffic_state.recent_inf_ms = inference_time * 1000

                if state.ema_fps is None:
                    state.ema_fps = current_fps
                    state.ema_inf = inference_time * 1000
                else:
                    state.ema_fps = (1 - ema_alpha) * state.ema_fps + ema_alpha * current_fps
                    state.ema_inf = (1 - ema_alpha) * state.ema_inf + ema_alpha * (inference_time * 1000)
                
                if visualizer.display(frame):
                    print("\nStopped by user (ESC)")
                    break
                
                # Skip frames for file sources only (video files)
                if not is_live and frames_to_skip > 0:
                    for _ in range(frames_to_skip):
                        cap.grab()
        
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