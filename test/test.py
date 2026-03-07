from ultralytics import YOLO

# 1. Tải mô hình mặc định (ví dụ YOLOv8 Nano)
model = YOLO('yolo11n.pt') 

# 2. Đánh giá trên tập mẫu chuẩn "coco8"
# Ultralytics sẽ tự tải data này về máy bạn nếu chưa có
metrics = model.val(data='coco8.yaml')

# 3. Xem kết quả
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")