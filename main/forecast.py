import numpy as np
from collections import deque

class Forecast:
    def __init__(self, history_length=60, min_frames=15, still_threshold=2.0, perspective_factor=0.0):
        """
        Khởi tạo đối tượng Forecast với khả năng lưu trữ lịch sử để làm mượt dữ liệu.
        """
        self.history = {}
        self.history_length = max(1, history_length)
        self.min_frames = min_frames
        self.still_threshold = still_threshold
        self.perspective_factor = perspective_factor
        self.smoothed_speeds = {} # Lưu tốc độ đã làm mượt của frame trước

    def remove_history(self, tracker_id):
        """Xóa lịch sử của một tracker ID cụ thể khi tracker bị xóa."""
        if tracker_id in self.history:
            del self.history[tracker_id]
        if tracker_id in self.smoothed_speeds:
            del self.smoothed_speeds[tracker_id]

    def reset_history(self):
        """Xóa toàn bộ lịch sử, ví dụ khi reset tracking."""
        self.history.clear()
        self.smoothed_speeds.clear()

    def calculate_speed(self, tracker_data):
        """
        Tính toán tốc độ, hướng di chuyển (đã được làm mượt).
        """
        speeds = {}
        directions = {}
        positions = {}

        for data in tracker_data:
            tid = data["id"]
            cx_curr, cy_curr = data["current_pos"]
            time_delta = data["time_delta"]
            kalman_angle = data.get("angle", None)

            # ========== CẬP NHẬT LỊCH SỬ ==========
            if tid not in self.history:
                self.history[tid] = {
                    "positions": deque(maxlen=self.history_length),
                    "times": deque(maxlen=self.history_length)
                }
            
            # Tính thời gian tích lũy cho tracker này
            if len(self.history[tid]["times"]) > 0:
                current_t = self.history[tid]["times"][-1] + time_delta
            else:
                current_t = 0.0

            self.history[tid]["positions"].append((cx_curr, cy_curr))
            self.history[tid]["times"].append(current_t)

            # ========== TÍNH TOÁN SAU KHI LÀM MƯỢT (SMOOTHED) ==========
            if len(self.history[tid]["positions"]) < self.min_frames:
                smoothed_speed = 0.0
                smoothed_dx, smoothed_dy = 0.0, 0.0
                direction_text = "CALC..."
            else:
                pos_start = self.history[tid]["positions"][0]
                pos_end = self.history[tid]["positions"][-1]
                time_start = self.history[tid]["times"][0]
                time_end = self.history[tid]["times"][-1]
                
                dt = time_end - time_start

                if dt > 0:
                    avg_dx = (pos_end[0] - pos_start[0]) / dt
                    avg_dy = (pos_end[1] - pos_start[1]) / dt
                    smoothed_speed = np.sqrt(avg_dx**2 + avg_dy**2)

                    # --- Perspective Correction (Tuyến tính hóa tốc độ theo góc quay) ---
                    # Giả định chiều cao ảnh là 640 (do letterbox). Y=0 (đỉnh) -> scale lớn, Y=640 (đáy) -> scale nhỏ
                    y_norm = max(0, min(1, cx_curr / 640.0)) if False else max(0, min(1, cy_curr / 640.0))
                    # Công thức: Speed_new = Speed_old * (1 + factor * (1 - y_norm))
                    scale = 1.0 + self.perspective_factor * (1.0 - y_norm)
                    smoothed_speed *= scale

                    # --- BỘ LỌC TỐC ĐỘ (EMA) ---
                    # Giúp giảm dao động khi xe ở gần camera (pixel nhảy lớn)
                    # alpha thấp (0.3) giúp tốc độ ổn định, ít bị giật cục
                    prev_s = self.smoothed_speeds.get(tid, smoothed_speed)
                    smoothed_speed = prev_s * 0.7 + smoothed_speed * 0.3
                    self.smoothed_speeds[tid] = smoothed_speed

                    if kalman_angle is not None and smoothed_speed > self.still_threshold:
                        rad = np.radians(kalman_angle)
                        smoothed_dx = np.cos(rad) * smoothed_speed
                        smoothed_dy = np.sin(rad) * smoothed_speed
                    else:
                        smoothed_dx, smoothed_dy = avg_dx, avg_dy

                    direction_text = self._get_direction_text(smoothed_dx, smoothed_dy, smoothed_speed)
                else:
                    smoothed_speed = 0.0
                    smoothed_dx, smoothed_dy = 0.0, 0.0
                    direction_text = "CALC..."

            # ========== LƯU KẾT QUẢ ==========
            speeds[tid] = smoothed_speed
            directions[tid] = {
                "vector": (smoothed_dx, smoothed_dy),
                "text": direction_text
            }
            positions[tid] = (cx_curr, cy_curr)

        return speeds, directions, positions

    def _get_direction_text(self, dx, dy, speed):
        if speed < self.still_threshold:
            return "STILL"

        angle = np.degrees(np.arctan2(-dy, dx))
        # Chuyển góc thành index 0-7: E, NE, N, NW, W, SW, S, SE
        dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        idx = int((angle + 22.5) % 360 / 45)
        return dirs[idx]