import cv2
import os
import numpy as np
from datetime import datetime
import time

# --- CẤU HÌNH ---
# Nguồn video: Có thể là đường dẫn RTSP hoặc số (0, 1) cho webcam
VIDEO_SOURCE = "rtsp://admin:abcd1234@192.168.50.107:554/cam/realmonitor?channel=1?subtype=0"
# VIDEO_SOURCE = 0 # Bỏ comment dòng này nếu dùng Webcam

OUTPUT_DIR = "../training"
TARGET_SIZE = (640, 640)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Hàm resize ảnh giữ nguyên tỷ lệ (aspect ratio) và thêm padding (viền).
    Sao chép từ utils/pt.py để đảm bảo tính nhất quán dữ liệu.
    """
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

def main():
    # Tạo thư mục training nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {os.path.abspath(OUTPUT_DIR)}")

    print(f"Đang mở nguồn: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Lỗi: Không thể mở nguồn video/camera.")
        return

    print("--- HƯỚNG DẪN ---")
    print("Nhấn 's': Lưu ảnh hiện tại (đã letterbox 640x640)")
    print("Nhấn 'q': Thoát chương trình")
    print("-----------------")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame.")
            # Nếu là file video thì loop lại, nếu là camera thì break
            if isinstance(VIDEO_SOURCE, str) and not VIDEO_SOURCE.startswith("rtsp"):
                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                 continue
            else:
                break

        # Xử lý Letterbox về 640x640
        frame_640 = letterbox(frame, new_shape=TARGET_SIZE)

        # Hiển thị
        cv2.imshow("Dataset Collector (640x640)", frame_640)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Tạo tên file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"img_{timestamp}.jpg"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            # Lưu ảnh
            cv2.imwrite(save_path, frame_640)
            count += 1
            print(f"[{count}] Đã lưu: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
