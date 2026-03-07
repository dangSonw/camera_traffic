import cv2
import numpy as np
from collections import deque
from datetime import datetime

class CustomTracker:
    def __init__(self):
        self.template = None
        self.bbox = None  # (x, y, w, h)
        self.previous_bbox = None  # Lưu lại vị trí cuối cùng để dự đoán hướng

    def init(self, frame, bbox):
        """
        Khởi tạo tracker với frame đầu tiên và bounding box.
        """
        x, y, w, h = map(int, bbox)
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        self.bbox = (x, y, w, h)
        self.previous_bbox = self.bbox  # Khởi tạo vị trí trước đó
        # Lưu lại hình ảnh của đối tượng để so khớp sau này
        self.template = frame[y:y+h, x:x+w].copy()
        return True

    def update(self, frame, predicted_bbox=None):
        """
        Cập nhật vị trí đối tượng trong frame mới.
        predicted_bbox: (x, y, w, h) vị trí dự đoán từ Kalman Filter (giúp giảm vùng tìm kiếm)
        Trả về: (success, bbox)
        """
        if self.template is None or self.bbox is None:
            return False, self.bbox

        x, y, w, h = self.bbox
        h_img, w_img = frame.shape[:2]

        # --- Tối ưu: Sử dụng vị trí dự đoán để thu hẹp vùng tìm kiếm ---
        if predicted_bbox is not None:
            pred_x, pred_y, _, _ = predicted_bbox
            center_x, center_y = pred_x, pred_y
        else:
            # Nếu không có dự đoán, dùng vị trí cũ
            center_x, center_y = x, y

        # Mở rộng vùng tìm kiếm dựa trên tốc độ (độ lớn của dx, dy)
        # Vì đã có dự đoán chính xác hơn, ta có thể giảm padding để tiết kiệm CPU
        base_padding = 20 
        padding_x = base_padding
        padding_y = base_padding

        # Xác định vùng tìm kiếm (Search Window) xung quanh vị trí *dự đoán*
        x_min = max(0, int(center_x - padding_x))
        y_min = max(0, int(center_y - padding_y))
        x_max = min(w_img, int(center_x + w + padding_x))
        y_max = min(h_img, int(center_y + h + padding_y))

        search_region = frame[y_min:y_max, x_min:x_max]

        # Template Matching
        if search_region.shape[0] < h or search_region.shape[1] < w:
             return False, self.bbox

        res = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.4:  # Ngưỡng tin cậy
            return False, self.bbox

        self.previous_bbox = self.bbox  # Cập nhật vị trí cũ *trước khi* gán vị trí mới

        top_left = max_loc
        new_x = x_min + top_left[0]
        new_y = y_min + top_left[1]
        self.bbox = (new_x, new_y, w, h)
        return True, self.bbox

class Track:
    """
    Quản lý trạng thái của một đối tượng được theo dõi:
    - ID, BBox, Class, Score
    - Bộ lọc Kalman (Vận tốc, Vị trí, Hướng)
    - Lịch sử di chuyển
    """
    def __init__(self, track_id, bbox, class_id, score, config, tracker_instance):
        self.id = track_id
        self.bbox = list(map(int, bbox)) # [x1, y1, x2, y2]
        self.class_id = class_id
        self.score = score
        self.tracker = tracker_instance # Instance của OpenCV/Custom Tracker
        
        # Config
        self.alpha = config.get("smoothing_alpha", 0.5)
        self.beta = config.get("smoothing_beta", 0.8)
        self.angle_smooth = config.get("smoothing_angle", 0.3)
        self.history_len = config.get("history_length", 60)
        
        # State (Kalman / Smoothing)
        self.velocity = (0.0, 0.0, 0.0, 0.0) # vx1, vy1, vx2, vy2
        self.angle = 0.0
        self.timestamp = datetime.now()
        self.lost_frames = 0
        
        # History for Forecast
        self.history_pos = deque(maxlen=self.history_len)
        self.history_time = deque(maxlen=self.history_len)
        
        # Forecast results
        self.speed_val = 0.0
        self.direction_text = ""
        self.direction_vector = (0, 0)

    def predict(self, time_delta):
        """Dự đoán vị trí tiếp theo (Prediction Step)"""
        if time_delta <= 0: return self.bbox
        
        x1, y1, x2, y2 = self.bbox
        vx1, vy1, vx2, vy2 = self.velocity
        
        # Pred = Current + Velocity * dt
        pred_x1 = x1 + vx1 * time_delta
        pred_y1 = y1 + vy1 * time_delta
        pred_x2 = x2 + vx2 * time_delta
        pred_y2 = y2 + vy2 * time_delta
        
        w, h = pred_x2 - pred_x1, pred_y2 - pred_y1
        return (int(pred_x1), int(pred_y1), int(w), int(h)) # Trả về xywh cho tracker update

    def correct(self, raw_bbox, score, time_delta):
        """Hiệu chỉnh vị trí dựa trên đo đạc mới (Correction Step)"""
        self.score = score
        self.lost_frames = 0
        
        if time_delta <= 0: 
            self.bbox = list(map(int, raw_bbox))
            return

        # Raw measurement
        rx1, ry1, rx2, ry2 = map(float, raw_bbox)
        
        # Current State
        cx1, cy1, cx2, cy2 = map(float, self.bbox)
        cvx1, cvy1, cvx2, cvy2 = self.velocity

        # Prediction
        px1, py1, px2, py2 = cx1 + cvx1*time_delta, cy1 + cvy1*time_delta, cx2 + cvx2*time_delta, cy2 + cvy2*time_delta

        # Update Position (Alpha filter)
        nx1 = (1 - self.alpha) * px1 + self.alpha * rx1
        ny1 = (1 - self.alpha) * py1 + self.alpha * ry1
        nx2 = (1 - self.alpha) * px2 + self.alpha * rx2
        ny2 = (1 - self.alpha) * py2 + self.alpha * ry2

        # Update Velocity (Beta filter)
        inst_vx1, inst_vy1 = (nx1 - cx1)/time_delta, (ny1 - cy1)/time_delta
        inst_vx2, inst_vy2 = (nx2 - cx2)/time_delta, (ny2 - cy2)/time_delta
        
        nvx1 = (1 - self.beta) * cvx1 + self.beta * inst_vx1
        nvy1 = (1 - self.beta) * cvy1 + self.beta * inst_vy1
        nvx2 = (1 - self.beta) * cvx2 + self.beta * inst_vx2
        nvy2 = (1 - self.beta) * cvy2 + self.beta * inst_vy2

        # Update Angle
        vel_cx, vel_cy = (nvx1 + nvx2)/2, (nvy1 + nvy2)/2
        inst_speed = np.sqrt(vel_cx**2 + vel_cy**2)
        if inst_speed > 2.0:
            inst_angle = np.degrees(np.arctan2(vel_cy, vel_cx))
            diff = (inst_angle - self.angle + 180) % 360 - 180
            self.angle += self.angle_smooth * diff

        self.bbox = [int(nx1), int(ny1), int(nx2), int(ny2)]
        self.velocity = (nvx1, nvy1, nvx2, nvy2)
        self.timestamp = datetime.now()