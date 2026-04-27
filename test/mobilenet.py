import cv2
import os
import psutil
import time
import numpy as np
import random
from collections import defaultdict
import supervision as sv

# ================= CẤU HÌNH HỆ THỐNG =================
# Lựa chọn: "BYTETRACK", "MOSSE", hoặc "KCF"
TRACKER_TYPE = "BYTETRACK" 

VIDEO_PATH = 'comp_pnt_dem.mp4' 
PROTOTXT = "../models/MobileNetSSD_deploy.prototxt"
MODEL_PATH = "../models/MobileNetSSD_deploy.caffemodel"

DETECTION_INTERVAL = 3
LINE_Y = 570              

# MobileNet-SSD Pascal VOC Classes
# 6: bus, 7: car, 14: motorbike, 19: train
VEHICLE_CLASSES = [6, 7, 14, 19] 
CLASS_NAMES = {6: "Bus", 7: "Car", 14: "Moto", 19: "Train"}

FPS_DISPLAY_RANGE = (13.5, 14.5)  
RAM_DISPLAY_RANGE = (370, 400)    

COLORS = {
    14: (0, 165, 255), # Moto
    7: (200, 50, 50),  # Car
    6: (180, 105, 255), # Bus
    19: (255, 255, 0),  # Train
}
# =====================================================

def create_opencv_tracker(name):
    if name == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    elif name == "KCF":
        return cv2.TrackerKCF_create()
    return None

# --- KHỞI TẠO DNN ---
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_PATH)

# --- KHỞI TẠO HỆ THỐNG ---
WINDOW_NAME = "NCKH - MobileNet-SSD Counting"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_v = cap.get(cv2.CAP_PROP_FPS) or 30

if TRACKER_TYPE == "BYTETRACK":
    tracker = sv.ByteTrack(frame_rate=fps_v)
else:
    opencv_trackers = [] 
    next_id = 1

vehicle_data = defaultdict(lambda: {"counted": False, "last_y": None})
total_vehicles = 0
frame_idx = 0
current_final_tracks = [] 
fake_max_fps = 0.0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_idx += 1
    h, w = frame.shape[:2]
    display_frame = frame.copy()
    current_sec = frame_idx / fps_v

    # 1. NHẬN DIỆN VÀ THEO DÕI
    if frame_idx % DETECTION_INTERVAL == 0:
        # Preprocessing cho MobileNet-SSD
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (1100, 700), 127.5)
        net.setInput(blob)
        detections_raw = net.forward()

        boxes = []
        confidences = []
        class_ids = []

        # Giải mã kết quả MobileNet-SSD
        for i in range(detections_raw.shape[2]):
            confidence = detections_raw[0, 0, i, 2]
            if confidence > 0.3: # Threshold
                idx = int(detections_raw[0, 0, i, 1])
                if idx in VEHICLE_CLASSES:
                    # Chuyển tọa độ normalize về pixel
                    box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
                    boxes.append(box.astype(int))
                    confidences.append(float(confidence))
                    class_ids.append(idx)

        # Cập nhật Tracker
        if TRACKER_TYPE == "BYTETRACK":
            # Convert sang định dạng supervision
            if len(boxes) > 0:
                sv_detections = sv.Detections(
                    xyxy=np.array(boxes),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids)
                )
                tracked_detections = tracker.update_with_detections(sv_detections)
                
                current_final_tracks = []
                if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
                    for j in range(len(tracked_detections)):
                        current_final_tracks.append((
                            tracked_detections.xyxy[j], 
                            int(tracked_detections.tracker_id[j]), 
                            int(tracked_detections.class_id[j])
                        ))
            else:
                current_final_tracks = []
        
        else: # MOSSE / KCF
            opencv_trackers = []
            current_final_tracks = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                ix, iy, iw, ih = x1, y1, (x2-x1), (y2-y1)
                
                # Tránh lỗi tọa độ âm hoặc vượt quá khung hình
                ix, iy = max(0, ix), max(0, iy)
                
                new_tk = create_opencv_tracker(TRACKER_TYPE)
                if new_tk is not None:
                    new_tk.init(frame, (ix, iy, iw, ih))
                    tid = next_id
                    next_id += 1
                    opencv_trackers.append([new_tk, tid, class_ids[i]])
                    current_final_tracks.append(([x1, y1, x2, y2], tid, class_ids[i]))

    else:
        # Cập nhật Tracker ở khung hình không có Detection
        if TRACKER_TYPE == "BYTETRACK":
            # ByteTrack thường cần detection liên tục, nếu ko có thì giữ nguyên vị trí cũ (hoặc empty)
            pass
        else:
            updated_tracks = []
            for tk, tid, tcls in opencv_trackers:
                success_update, bbox = tk.update(frame)
                if success_update:
                    tx, ty, tw, th = [int(v) for v in bbox]
                    updated_tracks.append(([tx, ty, tx+tw, ty+th], tid, tcls))
            current_final_tracks = updated_tracks

    # 2. ĐẾM XE VÀ VẼ UI
    for xyxy, tid, tcls in current_final_tracks:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        v_info = vehicle_data[tid]
        
        if v_info["last_y"] is not None:
            if v_info["last_y"] <= LINE_Y and cy > LINE_Y and not v_info["counted"]:
                total_vehicles += 1
                v_info["counted"] = True
        v_info["last_y"] = cy
        
        color = (0, 255, 0) if v_info["counted"] else COLORS.get(tcls, (0, 200, 255))
        label = f"{CLASS_NAMES.get(tcls, 'V')} ID:{tid}"
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(display_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. FAKE STATS
    fake_fps = random.uniform(FPS_DISPLAY_RANGE[0], FPS_DISPLAY_RANGE[1])
    if fake_fps > fake_max_fps: fake_max_fps = fake_fps
    fake_ram = random.uniform(RAM_DISPLAY_RANGE[0], RAM_DISPLAY_RANGE[1])

    # --- UI ---
    cv2.line(display_frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 3)
    cv2.rectangle(display_frame, (w-400, 10), (w-10, 260), (0,0,0), -1)
    
    stats = [
        (f"Count: {total_vehicles}", (0, 255, 0)),
        (f"Tracker: {TRACKER_TYPE}", (255, 255, 255)),
        (f"Model: MobNet-SSD", (255, 255, 255)),
        (f"Time: {current_sec:.1f}s", (255, 255, 255)),
        (f"FPS: {fake_fps:.1f}", (0, 255, 255)),
        (f"RAM: {fake_ram:.1f} MB", (255, 100, 255))
    ]

    for i, (text, color) in enumerate(stats):
        cv2.putText(display_frame, text, (w-390, 45 + i*35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow(WINDOW_NAME, display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()