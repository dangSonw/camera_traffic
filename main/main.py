import cv2
import time
from threading import Thread
import numpy as np

import config
import log
import draw
import core
import forecast
from data import TranPkg
import communication

class ThreadedCapture:
    """
    Đọc frame từ camera trong một luồng riêng biệt để tăng FPS.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        # Giảm buffer size xuống 1 để đảm bảo lấy frame mới nhất từ driver
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.ret, self.frame = ret, frame
            # Không sleep để đảm bảo thread luôn đọc và xả buffer nhanh nhất có thể

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

def run(show=True, max_loops=None, fps_limit=30, frame_skip=0):
    # 1. Tải cấu hình
    cfg = config.Config("config.json")
    path_cfg = cfg.get("path")
    model_cfg = cfg.get("model")
    
    # 2. Khởi tạo các đối tượng cốt lõi

    logger = log.Log(path_cfg.get("log", "../logs/terminal.txt"))
    drawer = draw.Draw()
    
    # Lấy tham số cấu hình từ file config.json
    min_frames = model_cfg.get("min_frames", 15)
    # Tăng ngưỡng still_threshold lên một chút để tránh nhiễu khi xe rung lắc tại chỗ
    still_threshold = model_cfg.get("still_threshold", 5.0) 
    perspective_factor = model_cfg.get("perspective_factor", 0.0)
    debug_level = model_cfg.get("debug_level", 1)

    forecast_obj = forecast.Forecast(
        min_frames=min_frames, 
        still_threshold=still_threshold,
        perspective_factor=perspective_factor
    )
    core_obj = core.Core(path_cfg, model_cfg, logger, forecast_obj)

    # Khởi tạo giao tiếp Serial
    comm = communication.Communication()
    last_serial_time = time.time()
    serial_status = "Serial: Ready"

    # 3. Bắt đầu chương trình
    src = path_cfg.get("input", "../rsrc/img.png")
    out_path = path_cfg.get("output", "../output/output.jpg")
    logger.print(f"[main.py] Bắt đầu chương trình với FPS limit={fps_limit}")

    cap = None
    static_frame = None
    is_static_image = False
    
    if src.startswith("rtsp://"):
        # TODO: Cân nhắc thêm các tùy chọn cho RTSP như UDP/TCP
        cap = ThreadedCapture(src).start()
        logger.print("[main.py] Nguồn: camera RTSP (Threaded)")
    elif src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        static_frame = cv2.imread(src)
        if static_frame is None:
            logger.print(f"[main.py] Lỗi: Không đọc được ảnh từ {src}")
            return
        logger.print("[main.py] Nguồn: ảnh tĩnh (sẽ loop)")
        is_static_image = True
    elif src.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(src)
        logger.print("[main.py] Nguồn: video file")
    else:
        try:
            # Thử mở như camera index (0, 1, ...)
            device_index = int(src)
            cap = ThreadedCapture(device_index).start()
            if not cap.isOpened(): raise ValueError
            logger.print(f"[main.py] Nguồn: camera device index {device_index} (Threaded)")
        except (ValueError, TypeError):
            logger.print(f"[main.py] Lỗi: Không xác định được nguồn dữ liệu: {src}")
            return

    if cap and not cap.isOpened():
        logger.print(f"[main.py] Lỗi: Không thể mở nguồn video: {src}")
        return

    loop_count = 0
    target_frame_time = 1.0 / fps_limit
    actual_fps = 0.0
    tranpkg = TranPkg()

    # Tối ưu: Khởi tạo Masking trước vòng lặp (vì size cố định 640x640)
    roi_mask = None
    rect_cfg = cfg.get("rect", {})
    active_rects = rect_cfg.get("active", [])
    
    if active_rects:
        roi_mask = np.zeros((640, 640), dtype=np.uint8)
        for r_id in active_rects:
            pts = np.array(rect_cfg.get(f"rect{r_id}", []), np.int32).reshape((-1, 1, 2))
            if pts.size > 0: cv2.fillPoly(roi_mask, [pts], 255)
    
    while True:
        frame_start_time = time.time()
        
        if cap:
            ret, frame = cap.read()
            if not ret:
                # Hết video hoặc mất kết nối
                logger.print("[main.py] Hết frame từ nguồn video.")
                break
        elif is_static_image:
            frame = static_frame.copy()
        else:
            break

        # Xử lý frame chính
        # Truyền biến show vào để Core chỉ vẽ khi cần thiết
        if loop_count % (frame_skip + 1) != 0:
            loop_count += 1
            continue

        # Resize về 640x640 trước vì tọa độ config là trên hệ quy chiếu 640x640
        frame = core_obj.detector.letterbox(frame, 640)

        # Áp dụng Mask: Chỉ giữ lại pixel trong vùng Rect, còn lại bôi đen
        if roi_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

        # Xử lý frame (Detection + Tracking + Forecast)
        # Core trả về frame (đã letterbox), danh sách kết quả tracking, và gói tin
        frame, tracking_results, tranpkg, rect_info = core_obj.process_frame(frame, rect_cfg)

        # --- Gửi dữ liệu Serial (5s/lần) ---
        if time.time() - last_serial_time >= 5.0:
            last_serial_time = time.time()
            # Format: "quantity,timeGreen,timeRed"
            serial_data = f"{tranpkg.quantity},{tranpkg.timeGreen},{tranpkg.timeRed}"
            
            if comm.send_data(serial_data):
                serial_status = f"Serial: Sent ({tranpkg.quantity})"
            else:
                serial_status = "Serial: Error/No Connection"
                if debug_level >= 3:
                    logger.print(f"[main.py] Serial Error: Could not send data to ESP.")


        if show and debug_level > 0:
            # --- DEBUG LEVEL 3: Vẽ chi tiết vùng Rect, Line và số lượng ---
            if debug_level >= 3:
                for r_id in active_rects:
                    # Vẽ vùng Rect
                    pts = rect_cfg.get(f"rect{r_id}", [])
                    if pts:
                        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
                        
                        # Lấy thông tin trạng thái
                        info = rect_info.get(r_id, {"count": 0, "status": "WAIT"})
                        status = info["status"]
                        count = info["count"]
                        
                        # Màu sắc dựa trên trạng thái
                        color = (0, 0, 255) if status == "RED" else ((0, 255, 0) if status == "GREEN" else (0, 255, 255))
                        cv2.polylines(frame, [pts_np], True, color, 2)
                        
                        # Vẽ Line (Vạch kẻ)
                        line_pts = rect_cfg.get(f"rect{r_id}_line", [])
                        if line_pts:
                            l_p1 = tuple(line_pts[0])
                            l_p2 = tuple(line_pts[1])
                            cv2.line(frame, l_p1, l_p2, (0, 0, 255), 2)

                        # Hiển thị số lượng xe trong vùng
                        # Lấy điểm đầu tiên của rect để hiển thị text
                        txt_pos = tuple(pts[0])
                        cv2.putText(frame, f"R{r_id}: {count} [{status}]", txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Hiển thị trạng thái Serial
                cv2.putText(frame, serial_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- DEBUG LEVEL 2: Vẽ Bounding Box và Tracking Info ---
            if debug_level >= 2:
                show_arrow = (debug_level >= 3) # Chỉ hiện mũi tên ở level 3
                drawer.draw_tracking_data(frame, tracking_results, core_obj.class_names, show_arrow=show_arrow)
            
            # --- DEBUG LEVEL 1: Vẽ FPS và Tổng số lượng ---
            if debug_level >= 1:
                drawer.draw_text(frame, f"Quantity: {tranpkg.quantity}", (10, 20), 
                                     color=(0,255,0), font_scale=0.5, thickness=2)
                drawer.draw_text(frame, f"FPS: {actual_fps:.1f}", (10, 40), 
                                     color=(0,255,255), font_scale=0.5, thickness=2)

        elapsed_before_display = time.time() - frame_start_time
        remaining_time = target_frame_time - elapsed_before_display
        
        if show:
            cv2.imshow("Frame", frame)
            
            wait_ms = max(1, int(remaining_time * 1000)) if remaining_time > 0 else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        else:
            if remaining_time > 0:
                time.sleep(remaining_time)
        
        total_frame_time = time.time() - frame_start_time
        actual_fps = 1.0 / max(total_frame_time, 1e-6)
        
        if loop_count % 2 == 0 and debug_level >= 1:
            logger.print(f"[main.py] Quantity={tranpkg.quantity}, FPS={actual_fps:.2f}")

        loop_count += 1
        if max_loops and loop_count >= max_loops:
            break

    if cap:
        cap.release()
    if show:
        cv2.destroyAllWindows()
    logger.print("[main.py] Kết thúc chương trình.")


if __name__ == "__main__":
    run(show=True, max_loops=None, fps_limit=60, frame_skip=0)
