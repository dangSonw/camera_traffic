import cv2
import os
import time
from datetime import datetime

# --- CẤU HÌNH ---
# Thay đổi đường dẫn RTSP của bạn ở đây
RTSP_URL = "rtsp://admin:abcd1234@192.168.50.107:554/cam/realmonitor?channel=1?subtype=0" 
OUTPUT_DIR = "../rsrc"
FPS_RECORD = 20.0 # FPS mong muốn khi lưu file

def main():
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {os.path.abspath(OUTPUT_DIR)}")

    print(f"Đang kết nối tới: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Lỗi: Không thể mở luồng RTSP.")
        return

    # Lấy thông số video đầu vào
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo tên file dựa trên thời gian hiện tại
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"record_{timestamp}.mp4"
    save_path = os.path.join(OUTPUT_DIR, filename)

    # Khởi tạo VideoWriter (Sử dụng codec mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, FPS_RECORD, (width, height))

    print(f"Đang ghi hình vào: {save_path}")
    print("Nhấn 'q' để dừng và lưu video.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất kết nối hoặc kết thúc luồng.")
            break

        # Ghi frame vào file
        out.write(frame)
        
        # Hiển thị (Resize nhỏ lại để dễ nhìn nếu độ phân giải cao)
        disp_frame = cv2.resize(frame, (640, 360))
        cv2.imshow("RTSP Recording (Press 'q' to stop)", disp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Đã lưu video. Tổng số frame: {frame_count}")

if __name__ == "__main__":
    main()
