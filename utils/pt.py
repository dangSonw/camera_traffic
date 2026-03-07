import cv2
import json
import numpy as np
import os
import sys
import time
import math
from threading import Thread

class ThreadedCapture:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.ret, self.frame = ret, frame
            time.sleep(0.005)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.cap.release()

class ConfigEditor:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()
        
        # Xác định đường dẫn nguồn video/ảnh
        src = self.config['path']['input']
        # Xử lý đường dẫn tương đối so với file config
        config_dir = os.path.dirname(os.path.abspath(config_file))
        
        if src.startswith("rtsp://") or src.isdigit():
            self.video_source = src
            if src.isdigit():
                self.video_source = int(src)
        else:
            # File path
            if not os.path.isabs(src):
                self.video_source = os.path.normpath(os.path.join(config_dir, src))
            else:
                self.video_source = src

        # Sử dụng ThreadedCapture
        self.cap = ThreadedCapture(self.video_source).start()
        
        ret, _ = self.cap.read()
        if not ret:
            # Thử mở như camera index nếu src là số
            try:
                self.video_source = int(src)
                self.cap.release()
                self.cap = ThreadedCapture(self.video_source).start()
            except:
                pass
        
        ret, _ = self.cap.read()
        if not ret:
            print(f"Lỗi: Không thể mở nguồn dữ liệu: {self.video_source}")
            sys.exit(1)

        self.target_size = (640, 640) # Width, Height
        self.window_name = "Config Editor (640x640)"
        
        self.current_rect = 1
        self.current_point_index = 0
        self.edit_mode = "RECT" # "RECT" or "LINE"
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def load_config(self):
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        if 'rect' not in self.config:
            self.config['rect'] = {}

    def save_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu cấu hình vào: {self.config_file}")

    def get_rect_points(self, rect_id):
        key = f"rect{rect_id}"
        if key in self.config['rect']:
            return self.config['rect'][key]
        return None

    def set_rect_points(self, rect_id, points):
        key = f"rect{rect_id}"
        self.config['rect'][key] = points

    def get_line_points(self, rect_id):
        key = f"rect{rect_id}_line"
        if key in self.config['rect']:
            return self.config['rect'][key]
        # Nếu chưa có line, tạo mặc định từ 2 điểm đầu của rect
        rect_pts = self.get_rect_points(rect_id)
        if rect_pts:
            return [rect_pts[1], rect_pts[2]] # Mặc định cạnh đáy
        return [[0,0], [0,0]]

    def set_line_points(self, rect_id, points):
        key = f"rect{rect_id}_line"
        self.config['rect'][key] = points

    def get_closest_point_on_segment(self, p, a, b):
        """Tìm điểm trên đoạn thẳng ab gần điểm p nhất"""
        x, y = p
        x1, y1 = a
        x2, y2 = b
        
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0: return [x1, y1]

        t = ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t)) # Clamp t về đoạn [0, 1]
        
        return [int(x1 + t * dx), int(y1 + t * dy)]

    def snap_to_rect_border(self, point, rect_points):
        """Tìm điểm trên viền rect gần chuột nhất"""
        best_dist = float('inf')
        best_pt = point
        
        for i in range(4):
            p1 = rect_points[i]
            p2 = rect_points[(i + 1) % 4]
            
            proj = self.get_closest_point_on_segment(point, p1, p2)
            dist = (point[0] - proj[0])**2 + (point[1] - proj[1])**2
            
            if dist < best_dist:
                best_dist = dist
                best_pt = proj
        return best_pt

    def mouse_callback(self, event, x, y, flags, param):
        rect_points = self.get_rect_points(self.current_rect)
        if rect_points is None: return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.edit_mode == "RECT":
                # Cập nhật tọa độ điểm hiện tại theo thứ tự vòng lặp
                rect_points[self.current_point_index] = [int(x), int(y)]
                self.set_rect_points(self.current_rect, rect_points)
                # Chuyển sang điểm tiếp theo (0: TL, 1: BL, 2: BR, 3: TR)
                self.current_point_index = (self.current_point_index + 1) % 4
            
            elif self.edit_mode == "LINE":
                # Lấy line hiện tại
                line_points = self.get_line_points(self.current_rect)
                # Snap điểm chuột vào viền rect
                snapped_pt = self.snap_to_rect_border([x, y], rect_points)
                
                line_points[self.current_point_index] = snapped_pt
                self.set_line_points(self.current_rect, line_points)
                self.current_point_index = (self.current_point_index + 1) % 2

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Resize về 640x640 để hiển thị
            frame_disp = self.letterbox(frame, self.target_size)

            # Vẽ các vùng
            for r_id in range(1, 5):
                pts = self.get_rect_points(r_id)
                if pts:
                    disp_pts = np.array(pts, np.int32)
                    disp_pts = disp_pts.reshape((-1, 1, 2))
                    line_pts = self.get_line_points(r_id)
                    
                    color = (0, 255, 0) if r_id == self.current_rect else (0, 0, 255)
                    thickness = 2 if r_id == self.current_rect else 1
                    cv2.polylines(frame_disp, [disp_pts], True, color, thickness)
                    
                    # Vẽ Line (Vạch kẻ)
                    if line_pts:
                        l_pts = np.array(line_pts, np.int32)
                        cv2.line(frame_disp, tuple(l_pts[0]), tuple(l_pts[1]), (255, 0, 0), 2)

                    # Vẽ các điểm tròn hướng dẫn
                    if r_id == self.current_rect:
                        if self.edit_mode == "RECT":
                            for i, p in enumerate(disp_pts):
                                # Highlight điểm sắp được đặt (màu đỏ)
                                c = (0, 0, 255) if i == self.current_point_index else (0, 255, 255)
                                cv2.circle(frame_disp, tuple(p[0]), 5, c, -1)
                        elif self.edit_mode == "LINE":
                            for i, p in enumerate(line_pts):
                                c = (0, 0, 255) if i == self.current_point_index else (255, 255, 0)
                                cv2.circle(frame_disp, tuple(p), 5, c, -1)

            # Hiển thị thông tin
            cv2.putText(frame_disp, f"Editing: Rect {self.current_rect}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_disp, f"Mode: {self.edit_mode} (Press 'M' to switch)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            point_names = ["P1", "P2", "P3", "P4"] if self.edit_mode == "RECT" else ["Line P1", "Line P2"]
            # Đảm bảo index không vượt quá giới hạn khi chuyển mode
            self.current_point_index = self.current_point_index % len(point_names)
            cv2.putText(frame_disp, f"Next Point: {point_names[self.current_point_index]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_disp, "Keys: 1-4 (Select), +/- (Add/Del), S (Save), Q (Quit)", (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(self.window_name, frame_disp)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_config()
            elif key == ord('m'):
                self.edit_mode = "LINE" if self.edit_mode == "RECT" else "RECT"
                self.current_point_index = 0
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self.current_rect = int(chr(key))
                self.current_point_index = 0
            elif key == ord('+') or key == ord('='): # Thêm vùng
                if self.get_rect_points(self.current_rect) is None:
                    # Tạo vùng mặc định ở giữa
                    cx, cy = 320, 320
                    dx, dy = 60, 60
                    new_pts = [[cx-dx, cy-dy], [cx+dx, cy-dy], [cx+dx, cy+dy], [cx-dx, cy+dy]]
                    self.set_rect_points(self.current_rect, new_pts)
            elif key == ord('-') or key == ord('_'): # Xóa vùng
                k = f"rect{self.current_rect}"
                if k in self.config['rect']:
                    del self.config['rect'][k]

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Tự động tìm file config ở thư mục ../main/config.json
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "../main/config.json")
    
    if not os.path.exists(config_path):
        print(f"Không tìm thấy file config tại: {config_path}")
        sys.exit(1)
        
    editor = ConfigEditor(config_path)
    editor.run()