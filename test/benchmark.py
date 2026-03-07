from ultralytics.utils.benchmarks import benchmark

# benchmark(model="yolo11n.pt",imgsz=640, half=False, device="cpu")
# benchmark(model="yolo11n.pt",imgsz=640, half=True, device="cpu")
# benchmark(model="yolo11n.pt",imgsz=320, half=False, device="cpu")
benchmark(model="yolo11n.pt",imgsz=640, half=False, device='cpu', nms=False, int8=True)