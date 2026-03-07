import cv2
import os
import psutil
import time
import numpy as np
from collections import defaultdict
import supervision as sv

# ================= CẤU HÌNH HỆ THỐNG =================
TRACKER_TYPE = "BYTETRACK" 
VIDEO_PATH = '../rsrc/comp.mp4'  # Đảm bảo đường dẫn này đúng

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

DETECTION_INTERVAL = 2    # Chạy Inference mỗi 2 frame để tối ưu FPS
LINE_Y = 450               # Vị trí vạch đếm

# Các class ID tương ứng trong MobileNetSSD
VEHICLE_CLASSES = [6, 7, 14, 15] 
CLASS_NAMES = {6: "Bus", 7: "Car", 14: "Moto", 15: "Person"}

WINDOW_NAME = "NCKH Evaluation - SSD"
# =====================================================

def get_ram():
    """Lấy lượng RAM đang sử dụng (MB)"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

# --- KIỂM TRA FILE MODEL ---
for f in [PROTOTXT, MODEL_PATH]:
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        print(f"Lỗi: File {f} không tồn tại hoặc bị rỗng. Vui lòng kiểm tra lại!")
        exit()

# --- KHỞI TẠO ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Lỗi: Không thể mở video.")
    exit()

fps_v = cap.get(cv2.CAP_PROP_FPS) or 30

# Load Model DNN Caffe
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Khởi tạo ByteTrack
tracker = sv.ByteTrack(frame_rate=fps_v)

# vehicle_data lưu trữ: { ID: {"counted": bool, "cls": int, "last_cy": float} }
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
        # Tạo blob cho SSD (300x300 là kích thước chuẩn của MobileNetSSD)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections_raw = net.forward()
        
        xyxy, confidences, class_ids = [], [], []
        
        if detections_raw is not None:
            for i in range(detections_raw.shape[2]):
                conf = detections_raw[0, 0, i, 2]
                if conf > 0.3: # Ngưỡng tin cậy
                    idx = int(detections_raw[0, 0, i, 1])
                    if idx in VEHICLE_CLASSES:
                        # Convert tọa độ từ [0, 1] sang pixel
                        box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
                        xyxy.append(box)
                        confidences.append(conf)
                        class_ids.append(idx)

        # Chuyển đổi sang định dạng Supervision để Tracking
        detections = sv.Detections(
            xyxy=np.array(xyxy) if xyxy else np.empty((0, 4)),
            confidence=np.array(confidences) if confidences else np.empty(0),
            class_id=np.array(class_ids) if class_ids else np.empty(0)
        )
        
        tracked_detections = tracker.update_with_detections(detections)
        
        # Lưu kết quả track để vẽ và đếm
        current_final_tracks = []
        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i]
                tid = int(tracked_detections.tracker_id[i])
                tcls = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
                current_final_tracks.append((box, tid, tcls))

    # 2. LOGIC ĐẾM XE CẮT QUA VẠCH
    for box, tid, tcls in current_final_tracks:
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        v_info = vehicle_data[tid]
        v_info["cls"] = tcls

        # KIỂM TRA ĐIỀU KIỆN ĐI NGANG QUA VẠCH
        if v_info["last_cy"] is not None:
            # Điều kiện: Trước ở TRÊN hoặc BẰNG vạch, sau ở DƯỚI vạch
            if v_info["last_cy"] <= LINE_Y and cy > LINE_Y and not v_info["counted"]:
                total_vehicles += 1
                v_info["counted"] = True
                print(f"[EVENT] {CLASS_NAMES.get(tcls, 'Object')} ID:{tid} crossed the line.")

        # Cập nhật vị trí Y cũ cho frame tiếp theo
        v_info["last_cy"] = cy

        # 3. VẼ LÊN MÀN HÌNH
        color = (0, 255, 0) if v_info["counted"] else (0, 200, 255)
        label = f"{CLASS_NAMES.get(tcls, 'V')} ID:{tid}"
        
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(display_frame, (cx, cy), 4, color, -1) # Vẽ tâm để kiểm chứng
        cv2.putText(display_frame, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- UI OVERLAY ---
    # Vẽ vạch đếm (Màu đỏ)
    cv2.line(display_frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 3)
    
    # Bảng số liệu
    cv2.putText(display_frame, f"Count: {total_vehicles}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Model: MobileNet-SSD", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    ram_logs.append(get_ram())
    cv2.imshow(WINDOW_NAME, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- BÁO CÁO KẾT QUẢ ---
duration = time.time() - start_total_time
if frame_idx > 0:
    print(f"\n" + "-"*30)
    print(f" THỐNG KÊ KẾT THÚC (SSD) ")
    print(f"-"*30)
    print(f"Thuật toán:     {TRACKER_TYPE}")
    print(f"FPS Trung bình: {frame_idx / duration:.2f}")
    print(f"RAM Trung bình: {sum(ram_logs)/len(ram_logs):.2f} MB")
    print(f"Tổng số xe:     {total_vehicles}")
    print(f"-"*30)