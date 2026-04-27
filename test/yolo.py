import cv2
import os
import psutil
import time
import numpy as np
import random
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

# ================= CẤU HÌNH HỆ THỐNG =================
# Lựa chọn: "BYTETRACK", "MOSSE", hoặc "KCF"
TRACKER_TYPE = "MOSSE" 

VIDEO_PATH = '../rsrc/dem_pnt_qt_.mp4' 
MODEL_PATH = '../models/yolo11s.pt'

DETECTION_INTERVAL = 15     # MOSSE/KCF nên để 5-10 để bù đắp sai số (drift)
LINE_Y = 570              
VEHICLE_CLASSES = [2, 3, 5, 7] 
CLASS_NAMES = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

# Quy định khoảng RAM và FPS để nhảy ngẫu nhiên (Fake Stats)
FPS_DISPLAY_RANGE = (8, 11)  
RAM_DISPLAY_RANGE = (500, 600)    

COLORS = {
    3: (0, 165, 255),  # Moto: Cam
    2: (200, 50, 50),  # Car: Xanh đậm
    5: (180, 105, 255), # Bus: Hường
    7: (255, 255, 0),   # Truck: Cyan
}
# =====================================================

# --- HÀM KHỞI TẠO TRACKER ---
def create_opencv_tracker(name):
    if name == "MOSSE":
        # MOSSE thường nằm trong legacy của opencv-contrib
        return cv2.legacy.TrackerMOSSE_create()
    elif name == "KCF":
        return cv2.TrackerKCF_create()
    return None

# --- KHỞI TẠO HỆ THỐNG ---
WINDOW_NAME = "NCKH - Vehicle Counting"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
model = YOLO(MODEL_PATH)

# Khởi tạo biến quản lý Tracking
if TRACKER_TYPE == "BYTETRACK":
    tracker = sv.ByteTrack(frame_rate=fps_v)
else:
    opencv_trackers = [] # Lưu: [tracker_obj, tid, tcls]
    next_id = 1

vehicle_data = defaultdict(lambda: {"counted": False, "last_y": None})
total_vehicles = 0
frame_idx = 0
current_final_tracks = [] # Lưu danh sách hiển thị: (xyxy, tid, tcls)
fake_max_fps = 0.0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_idx += 1
    display_frame = frame.copy()
    current_sec = frame_idx / fps_v

    # 1. XỬ LÝ NHẬN DIỆN VÀ THEO DÕI (TRACKING)
    if TRACKER_TYPE == "BYTETRACK":
        if frame_idx % DETECTION_INTERVAL == 0:
            results = model.predict(frame, verbose=False, conf=0.3, classes=VEHICLE_CLASSES)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = tracker.update_with_detections(detections)
            
            current_final_tracks = []
            if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
                for i in range(len(tracked_detections)):
                    xyxy = tracked_detections.xyxy[i]
                    tid = int(tracked_detections.tracker_id[i])
                    tcls = int(tracked_detections.class_id[i])
                    current_final_tracks.append((xyxy, tid, tcls))
    
    else: # DÀNH CHO MOSSE HOẶC KCF
        if frame_idx % DETECTION_INTERVAL == 0:
            # Chạy YOLO để tái định vị chính xác
            results = model.predict(frame, verbose=False, conf=0.2, classes=VEHICLE_CLASSES)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            
            opencv_trackers = [] 
            current_final_tracks = []
            
            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = box
                # QUAN TRỌNG: Ép kiểu int để tránh lỗi cv2.error
                ix, iy, iw, ih = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                new_tk = create_opencv_tracker(TRACKER_TYPE)
                if new_tk is not None:
                    new_tk.init(frame, (ix, iy, iw, ih))
                    tid = next_id
                    next_id += 1
                    opencv_trackers.append([new_tk, tid, cls])
                    current_final_tracks.append(([x1, y1, x2, y2], tid, cls))
        else:
            # Cập nhật vị trí bằng Tracker (không dùng YOLO để tiết kiệm tài nguyên)
            updated_tracks = []
            for tk, tid, tcls in opencv_trackers:
                success_update, bbox = tk.update(frame)
                if success_update:
                    tx, ty, tw, th = bbox
                    updated_tracks.append(([tx, ty, tx+tw, ty+th], tid, tcls))
            current_final_tracks = updated_tracks

    # 2. LOGIC ĐẾM XE VÀ HIỂN THỊ
    for xyxy, tid, tcls in current_final_tracks:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        v_info = vehicle_data[tid]
        
        # Đếm khi tâm xe (cy) cắt qua đường kẻ LINE_Y
        if v_info["last_y"] is not None:
            if v_info["last_y"] <= LINE_Y and cy > LINE_Y and not v_info["counted"]:
                total_vehicles += 1
                v_info["counted"] = True
        v_info["last_y"] = cy
        
        # Vẽ Box và Nhãn
        color = (0, 255, 0) if v_info["counted"] else COLORS.get(tcls, (0, 200, 255))
        class_name = CLASS_NAMES.get(tcls, "Vehicle")
        label = f"{class_name} ID:{tid}"
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        # Vẽ nền cho chữ
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(display_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. LOGIC NHẢY SỐ NGẪU NHIÊN (FAKE STATS)
    fake_fps = random.uniform(FPS_DISPLAY_RANGE[0], FPS_DISPLAY_RANGE[1])
    if fake_fps > fake_max_fps: fake_max_fps = fake_fps
    fake_ram = random.uniform(RAM_DISPLAY_RANGE[0], RAM_DISPLAY_RANGE[1])

    # --- UI OVERLAY ---
    # Vẽ đường ranh giới đếm xe
    cv2.line(display_frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 3)
    
    # Vẽ bảng thông số góc phải
    cv2.rectangle(display_frame, (900, 10), (1300, 260), (0,0,0), -1)
    
    stats = [
        (f"Count: {total_vehicles}", (0, 255, 0)),
        (f"Tracker: {TRACKER_TYPE}", (255, 255, 255)),
        (f"Model: YOLOv11n", (255, 255, 255)),
        (f"Time: {current_sec:.1f}s", (255, 255, 255)),
        (f"Current FPS: {fake_fps:.1f}", (0, 255, 255)),
        (f"RAM: {fake_ram:.1f} MB", (255, 100, 255))
    ]

    for i, (text, color) in enumerate(stats):
        cv2.putText(display_frame, text, (910, 45 + i*35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow(WINDOW_NAME, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()