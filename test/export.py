from ultralytics import YOLO

# Load model (pt)
model = YOLO("vehicle.pt")

# Export sang NCNN
model.export(
    format="ncnn",
    imgsz=640,
    half=False,      # True nếu target hỗ trợ FP16
    device="cpu"
)