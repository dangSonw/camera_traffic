import cv2
import os
import psutil
import time
import torch
import numpy as np
from collections import defaultdict
import supervision as sv
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

# ================= CẤU HÌNH HỆ THỐNG =================
TRACKER_TYPE = "BYTETRACK" 
VIDEO_PATH = '../rsrc/v10_te2.mp4'

# Tên model trên Hugging Face (Bản ResNet-18 tương đương bản Small)
MODEL_NAME = "PCP-AI/rtdetr-resnet18" # Hoặc "PCP-AI/rtdetrv2-s" nếu có trên hub

DETECTION_INTERVAL = 2    
LINE_Y = 450               

# Class ID của tập COCO (Hugging Face RT-DETR chuẩn)
# 1: person, 3: car, 4: motorcycle, 6: bus, 8: truck
VEHICLE_CLASSES = [1, 3, 4, 6, 8] 
CLASS_NAMES = {1: "Person", 3: "Car", 4: "Moto", 6: "Bus", 8: "Truck"}

WINDOW_NAME = "NCKH Evaluation - RT-DETR (Transformers)"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =====================================================

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

# --- KHỞI TẠO ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_v = cap.get(cv2.CAP_PROP_FPS) or 30

# Load Processor và Model từ Transformers
print(f"Đang tải model {MODEL_NAME} lên {DEVICE}...")
processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
model = RTDetrForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE)

if TRACKER_TYPE == "BYTETRACK":
    tracker = sv.ByteTrack(frame_rate=fps_v)

vehicle_data = defaultdict(lambda: {"counted": False, "cls": 0})
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

    # Xử lý Detection & Tracking
    if frame_idx % DETECTION_INTERVAL == 0:
        # 1. Tiền xử lý (Transformers yêu cầu định dạng RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb_frame, return_tensors="pt").to(DEVICE)
        
        # 2. Inference (Tắt gradient để tiết kiệm RAM)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 3. Hậu xử lý kết quả (Post-processing)
        # Chuyển đổi output về kích thước frame gốc
        results = processor.post_process_object_detection(outputs, target_sizes=[(h, w)], threshold=0.3)[0]
        
        # 4. Chuyển sang định dạng Supervision để Tracking
        detections = sv.Detections(
            xyxy=results["boxes"].cpu().numpy(),
            confidence=results["scores"].cpu().numpy(),
            class_id=results["labels"].cpu().numpy()
        )
        
        # Lọc các class không cần thiết
        mask = np.isin(detections.class_id, VEHICLE_CLASSES)
        detections = detections[mask]
        
        if TRACKER_TYPE == "BYTETRACK":
            tracked_detections = tracker.update_with_detections(detections)
            
            current_final_tracks = []
            if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
                for i in range(len(tracked_detections)):
                    current_final_tracks.append([
                        tracked_detections.xyxy[i][0], tracked_detections.xyxy[i][1],
                        tracked_detections.xyxy[i][2], tracked_detections.xyxy[i][3],
                        tracked_detections.tracker_id[i], 0, 
                        tracked_detections.class_id[i]
                    ])

    # Vẽ và Đếm
    for t in current_final_tracks:
        x1, y1, x2, y2, tid, _, tcls = t
        tid, tcls = int(tid), int(tcls)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        v_info = vehicle_data[tid]
        v_info["cls"] = tcls

        if cy > LINE_Y and not v_info["counted"]:
            total_vehicles += 1
            v_info["counted"] = True

        color = (0, 255, 0) if v_info["counted"] else (0, 200, 255)
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{CLASS_NAMES.get(tcls, 'V')} ID:{tid}"
        cv2.putText(display_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.line(display_frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 2)
    cv2.putText(display_frame, f"Count: {total_vehicles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    ram_logs.append(get_ram())
    cv2.imshow(WINDOW_NAME, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- BÁO CÁO ---
duration = time.time() - start_total_time
print(f"\n--- THỐNG KÊ KẾT THÚC (Transformers) ---")
print(f"Model: {MODEL_NAME}")
print(f"FPS Trung bình: {frame_idx / duration:.2f}")
print(f"RAM Trung bình: {sum(ram_logs)/len(ram_logs):.2f} MB")
print(f"Tổng số xe: {total_vehicles}")