import cv2
import os
import psutil
import time
import numpy as np
from collections import defaultdict
import supervision as sv
from ultralytics import RTDETR

# ================= CẤU HÌNH HỆ THỐNG =================
TRACKER_TYPE = "BYTETRACK" 
VIDEO_PATH = '../rsrc/comp.mp4'  # Đảm bảo đường dẫn này đúng

# RT-DETR Model (Sử dụng rtdetr-l.pt hoặc rtdetr-x.pt)
MODEL_PATH = "rtdetr-l.pt" 

DETECTION_INTERVAL = 2    # Chạy Inference mỗi 2 frame để tăng hiệu suất
LINE_Y = 450               # Vị trí vạch đếm (trục Y)

# Class ID theo chuẩn COCO (RT-DETR)
VEHICLE_CLASSES = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck
CLASS_NAMES = {2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}

WINDOW_NAME = "NCKH Evaluation - RT-DETR"
# =====================================================

def get_ram():
    """Lấy lượng RAM đang sử dụng (MB)"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

# --- KHỞI TẠO ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Lỗi: Không thể mở file video. Vui lòng kiểm tra VIDEO_PATH.")
    exit()

fps_v = cap.get(cv2.CAP_PROP_FPS) or 30

# Load RT-DETR Model
model = RTDETR(MODEL_PATH) 

# Khởi tạo ByteTrack
tracker = sv.ByteTrack(frame_rate=fps_v)

# vehicle_data lưu: { ID: {"counted": bool, "cls": int, "last_cy": float} }
vehicle_data = defaultdict(lambda: {"counted": False, "cls": 0, "last_cy": None})

total_vehicles = 0
frame_idx = 0
ram_logs = []
start_total_time = time.time()
current_final_tracks = []

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_idx += 1
    display_frame = frame.copy()
    h, w = frame.shape[:2]

    # 1. XỬ LÝ DETECTION & TRACKING
    if frame_idx % DETECTION_INTERVAL == 0:
        # RT-DETR Inference
        results = model.predict(frame, conf=0.3, iou=0.5, classes=VEHICLE_CLASSES, verbose=False)[0]
        
        # Chuyển đổi sang định dạng Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        # Cập nhật Tracker
        tracked_detections = tracker.update_with_detections(detections)
        
        # Cập nhật danh sách track hiện tại để vẽ
        current_final_tracks = []
        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                xyxy = tracked_detections.xyxy[i]
                tid = int(tracked_detections.tracker_id[i])
                tcls = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
                current_final_tracks.append((xyxy, tid, tcls))

    # 2. LOGIC ĐẾM XE CẮT QUA VẠCH
    for xyxy, tid, tcls in current_final_tracks:
        x1, y1, x2, y2 = xyxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        v_info = vehicle_data[tid]
        v_info["cls"] = tcls

        # KIỂM TRA ĐIỀU KIỆN CẮT VẠCH (Crossing Logic)
        if v_info["last_cy"] is not None:
            # Điều kiện: Frame trước ở trên vạch (<=) và frame hiện tại ở dưới vạch (>)
            if v_info["last_cy"] <= LINE_Y and cy > LINE_Y and not v_info["counted"]:
                total_vehicles += 1
                v_info["counted"] = True
                # Log ra console để theo dõi
                print(f"[COUNT] Xe {CLASS_NAMES.get(tcls, 'V')} ID:{tid} vừa đi qua vạch.")

        # Cập nhật lại tọa độ Y cũ
        v_info["last_cy"] = cy

        # 3. VẼ LÊN MÀN HÌNH
        color = (0, 255, 0) if v_info["counted"] else (0, 200, 255)
        label = f"{CLASS_NAMES.get(tcls, 'V')} ID:{tid}"
        
        # Vẽ Box và Tâm xe
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(display_frame, (cx, cy), 4, color, -1)
        cv2.putText(display_frame, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- GIAO DIỆN (UI OVERLAY) ---
    # Vẽ vạch kẻ (Màu đỏ)
    cv2.line(display_frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 3)
    
    # Bảng hiển thị thông số
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

    cv2.putText(display_frame, f"Total Count: {total_vehicles}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Model: RT-DETR", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Log RAM để báo cáo NCKH
    ram_logs.append(get_ram())
    
    cv2.imshow(WINDOW_NAME, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- BÁO CÁO THỐNG KÊ ---
duration = time.time() - start_total_time
if frame_idx > 0:
    print(f"\n" + "="*30)
    print(f" THỐNG KÊ KẾT THÚC (RT-DETR) ")
    print(f"="*30)
    print(f"1. Thuật toán Tracking: {TRACKER_TYPE}")
    print(f"2. FPS Trung bình:     {frame_idx / duration:.2f}")
    print(f"3. RAM Trung bình:     {sum(ram_logs)/len(ram_logs):.2f} MB")
    print(f"4. Tổng số xe đã đếm:  {total_vehicles}")
    print(f"="*30)