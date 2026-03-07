from ultralytics import YOLO
import psutil
import os
import logging

# Tắt các log thừa của hệ thống
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def get_ram_usage():
    # Lấy lượng RAM chiếm dụng (MB)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# 1. Đường dẫn cấu hình
# NCNN thường là một thư mục (ví dụ: yolo11n_ncnn_model/)
ncnn_model_path = '../models/ep_640/yolo11n_ncnn_model/' 
video_path = '../rsrc/comp.mp4'
data_yaml = 'coco8.yaml'

# 2. Load mô hình NCNN
# Cần task='detect' để đảm bảo cấu trúc đầu ra chính xác
model = YOLO(ncnn_model_path, task='detect')

# --- PHẦN 1: ĐÁNH GIÁ ĐỘ CHÍNH XÁC (VALIDATION) ---
print(f"--- Đang đánh giá mAP trên NCNN Engine (Tối ưu ARM)... ---")
# NCNN thường yêu cầu imgsz cố định (ví dụ 640 hoặc 320)
metrics = model.val(data=data_yaml, verbose=False, imgsz=640)

# --- PHẦN 2: CHẠY TRÊN VIDEO ĐỂ ĐO HIỆU NĂNG ---
print(f"--- Đang xử lý video bằng NCNN và đo tài nguyên... ---")

results = model.predict(
    source=video_path, 
    stream=True, 
    device='cpu',  # NCNN được thiết kế tối ưu cho CPU ARM (NEON)
    verbose=False, 
    show=False, 
    imgsz=640
)

total_times = {'pre': 0, 'inf': 0, 'post': 0}
frame_count = 0
ram_samples = []

for r in results:
    s = r.speed 
    total_times['pre'] += s['preprocess']
    total_times['inf'] += s['inference']
    total_times['post'] += s['postprocess']
    frame_count += 1
    ram_samples.append(get_ram_usage())

# --- PHẦN 3: TỔNG HỢP BÁO CÁO CUỐI CÙNG ---
print("\n" + "═"*55)
print("             BÁO CÁO TỔNG HỢP MÔ HÌNH NCNN            ")
print("═"*55)

# 3.1. Accuracy Metrics
print(f"1. CHỈ SỐ ĐỘ CHÍNH XÁC (Accuracy):")
print(f"   • mAP50-95:        {metrics.box.map:.4f}")
print(f"   • mAP50:           {metrics.box.map50:.4f}")
print(f"   • Mean Precision:  {metrics.box.mp:.4f}")
print(f"   • Mean Recall:     {metrics.box.mr:.4f}")

# 3.2. Performance Metrics
if frame_count > 0:
    avg_total = (total_times['pre'] + total_times['inf'] + total_times['post']) / frame_count
    
    print(f"\n2. HIỆU NĂNG HỆ THỐNG (Performance):")
    print(f"   • Tổng số khung hình: {frame_count}")
    print(f"   • Tốc độ trung bình:  {avg_total:.2f} ms/frame")
    print(f"   • FPS trung bình:     {1000/avg_total:.2f} FPS")
    
    print(f"\n3. TÀI NGUYÊN CHIẾM DỤNG (Resources):")
    print(f"   • RAM trung bình:     {sum(ram_samples)/len(ram_samples):.2f} MB")
    print(f"   • RAM đỉnh điểm:      {max(ram_samples):.2f} MB")

# 3.3. Engine Info
print(f"\n4. THÔNG TIN ENGINE:")
print(f"   • Format:             NCNN (Tencent)")
print(f"   • Đặc điểm:           Tối ưu cực thấp mức RAM, dùng NEON trên ARM")

print("═"*55)