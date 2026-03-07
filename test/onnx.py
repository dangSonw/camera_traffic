from ultralytics import YOLO
import psutil
import os
import logging
import time

# Tắt log thừa để bảng kết quả cuối cùng nổi bật hơn
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def get_ram_usage():
    # Lấy dung lượng RAM mà tiến trình hiện tại đang chiếm dụng (MB)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# 1. Đường dẫn file (Bạn hãy điều chỉnh cho đúng)
onnx_path = '../models/ep_640/yolo11n.onnx'  # File ONNX của bạn
video_path = '../rsrc/comp.mp4'      # Video để test hiệu năng
data_yaml = 'coco8.yaml'             # File cấu hình tập dữ liệu để tính mAP

# 2. Load mô hình ONNX
# Cần task='detect' vì file ONNX không chứa metadata tự động như file .pt
model = YOLO(onnx_path, task='detect')

# --- PHẦN 1: ĐÁNH GIÁ ĐỘ CHÍNH XÁC (VALIDATION) ---
print(f"--- Đang bắt đầu chấm điểm mAP cho {os.path.basename(onnx_path)}... ---")
# imgsz=640 nên khớp với lúc bạn export file ONNX
metrics = model.val(data=data_yaml, verbose=False, imgsz=640)

# --- PHẦN 2: CHẠY TRÊN VIDEO ĐỂ ĐO HIỆU NĂNG ---
print(f"--- Đang xử lý video để đo FPS và RAM... ---")

results = model.predict(
    source=video_path, 
    stream=True, 
    device='cpu', 
    verbose=False, 
    show=False, 
    imgsz=640
)

total_times = {'pre': 0, 'inf': 0, 'post': 0}
frame_count = 0
ram_samples = []

# Vòng lặp xử lý từng khung hình (Generator mode để tiết kiệm RAM)
for r in results:
    s = r.speed 
    total_times['pre'] += s['preprocess']
    total_times['inf'] += s['inference']
    total_times['post'] += s['postprocess']
    frame_count += 1
    
    # Đo RAM thực tế lúc đang inference
    ram_samples.append(get_ram_usage())

# --- PHẦN 3: XUẤT BÁO CÁO TỔNG HỢP ---
print("\n" + "═"*55)
print("             KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ONNX             ")
print("═"*55)

# 3.1. Accuracy Metrics
print(f"1. CHỈ SỐ ĐỘ CHÍNH XÁC (Accuracy):")
print(f"   • mAP50-95:        {metrics.box.map:.4f} (Độ chuẩn quốc tế)")
print(f"   • mAP50:           {metrics.box.map50:.4f}")
print(f"   • Mean Precision:  {metrics.box.mp:.4f}")
print(f"   • Mean Recall:     {metrics.box.mr:.4f}")

# 3.2. Performance Metrics
if frame_count > 0:
    avg_pre = total_times['pre'] / frame_count
    avg_inf = total_times['inf'] / frame_count
    avg_post = total_times['post'] / frame_count
    avg_total = avg_pre + avg_inf + avg_post
    
    print(f"\n2. HIỆU NĂNG HỆ THỐNG (Performance):")
    print(f"   • Tổng số khung hình: {frame_count}")
    print(f"   • Tốc độ trung bình:  {avg_total:.2f} ms/frame")
    print(f"   • FPS trung bình:     {1000/avg_total:.2f} FPS")
    
    print(f"\n3. TÀI NGUYÊN CHIẾM DỤNG (Resources):")
    print(f"   • RAM trung bình:     {sum(ram_samples)/len(ram_samples):.2f} MB")
    print(f"   • RAM đỉnh điểm:      {max(ram_samples):.2f} MB")

# 3.3. File Info
print(f"\n4. THÔNG TIN FILE:")
print(f"   • Định dạng:          ONNX (Open Neural Network Exchange)")
print(f"   • Kích thước file:    {os.path.getsize(onnx_path)/(1024*1024):.2f} MB")

print("═"*55)
print("Lưu ý: FPS thực tế có thể cao hơn nếu bỏ qua bước lưu video.")