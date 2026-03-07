import cv2
import os
import psutil
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv

# ================= CẤU HÌNH HỆ THỐNG =================
TRACKER_TYPE = "BYTETRACK" 
VIDEO_PATH = '../rsrc/comp.mp4'  # Đảm bảo đường dẫn này đúng
MODEL_PATH = 'yolo11n.pt'

DETECTION_INTERVAL = 2    # Chạy YOLO mỗi 2 frame để tăng FPS
LINE_Y = 450               # Vị trí vạch đếm
VEHICLE_CLASSES = [2, 3, 5, 7] 
CLASS_NAMES = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
WINDOW_NAME = "NCKH - Vehicle Counting"
# =====================================================

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

# --- KHỞI TẠO ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
model = YOLO(MODEL_PATH)

# Khởi tạo ByteTrack từ supervision
tracker = sv.ByteTrack(frame_rate=fps_v)

# vehicle_data lưu: { ID: {"counted": bool, "cls": int, "last_y": float} }
vehicle_data = defaultdict(lambda: {"counted": False, "cls": 0, "last_y": None})

total_vehicles = 0
frame_idx = 0
ram_logs = []
start_total_time = time.time()

# Bộ nhớ đệm để giữ kết quả vẽ khi skip frame
current_final_tracks = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_idx += 1
    display_frame = frame.copy()

    # 1. NHẬN DIỆN VÀ TRACKING
    if frame_idx % DETECTION_INTERVAL == 0:
        results = model.predict(frame, verbose=False, conf=0.3, classes=VEHICLE_CLASSES)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Cập nhật Tracker
        tracked_detections = tracker.update_with_detections(detections)
        
        # Cập nhật danh sách track hiện tại
        current_final_tracks = []
        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                xyxy = tracked_detections.xyxy[i]
                tid = int(tracked_detections.tracker_id[i])
                tcls = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
                current_final_tracks.append((xyxy, tid, tcls))

    # 2. LOGIC ĐẾM XE (Chỉ đếm khi cắt qua vạch)
    for xyxy, tid, tcls in current_final_tracks:
        x1, y1, x2, y2 = xyxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        v_info = vehicle_data[tid]
        v_info["cls"] = tcls # Cập nhật class mới nhất

        # KIỂM TRA ĐIỀU KIỆN CẮT VẠCH
        if v_info["last_y"] is not None:
            # Nếu frame trước ở TRÊN hoặc BẰNG vạch, và frame này ở DƯỚI vạch
            if v_info["last_y"] <= LINE_Y and cy > LINE_Y and not v_info["counted"]:
                total_vehicles += 1
                v_info["counted"] = True
                print(f"[INFO] Xe ID {tid} ({CLASS_NAMES.get(tcls, 'V')}) đã đi qua vạch.")

        # Cập nhật last_y cho frame tiếp theo
        v_info["last_y"] = cy

        # 3. VẼ LÊN MÀN HÌNH
        color = (0, 255, 0) if v_info["counted"] else (0, 200, 255)
        label = f"{CLASS_NAMES.get(tcls, 'V')} ID:{tid}"
        
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(display_frame, (cx, cy), 4, color, -1) # Vẽ tâm xe để dễ quan sát
        cv2.putText(display_frame, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # UI Overlay
    # Vẽ vạch đếm (Màu đỏ)
    cv2.line(display_frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 3)
    
    # Bảng thông tin
    cv2.rectangle(display_frame, (10, 10), (250, 90), (0,0,0), -1)
    cv2.putText(display_frame, f"Count: {total_vehicles}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Tracker: {TRACKER_TYPE}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    ram_logs.append(get_ram())
    cv2.imshow(WINDOW_NAME, display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

# --- BÁO CÁO SAU KHI CHẠY ---
duration = time.time() - start_total_time
if frame_idx > 0:
    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Thuật toán: {TRACKER_TYPE}")
    print(f"FPS Trung bình: {frame_idx / duration:.2f}")
    print(f"RAM sử dụng trung bình: {sum(ram_logs)/len(ram_logs):.2f} MB")
    print(f"Tổng số xe ghi nhận: {total_vehicles}")
    print(f"------------------------")