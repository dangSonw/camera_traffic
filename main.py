import time
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
import logging
import traceback
from typing import List, Tuple
import threading
import os
# Removed tracker import
from traffic_monitor.system_utils import (
    start_quit_listener,
    SystemMonitor,
    setup_cpu_affinity,
    print_progress_bar,
)
from traffic_monitor.config import build_arg_parser, load_runtime_config
from traffic_monitor.logging_utils import MetricLogger
from traffic_monitor.counter import CounterState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional Windows-only keyboard support (for 'q' to quit)
try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None


def point_in_polygon(point, polygon):
    """Kiểm tra xem một điểm có nằm trong đa giác hay không."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate required model files
    use_tflite = getattr(args, 'backend', 'caffe') == 'tflite'
    if use_tflite:
        if not getattr(args, 'tflite', None):
            logger.error("--tflite path is required when --backend tflite")
            return 1
        if not Path(args.tflite).exists():
            logger.error(f"TFLite model not found: {args.tflite}")
            return 1
    else:
        if not getattr(args, 'weights', None):
            logger.error("--weights is required when --backend caffe")
            return 1
        if not getattr(args, 'prototxt', None):
            logger.error("--prototxt is required when --backend caffe")
            return 1
        if not Path(args.weights).exists():
            logger.error(f"Caffe model weights not found: {args.weights}")
            return 1
        if not Path(args.prototxt).exists():
            logger.error(f"Caffe prototxt file not found: {args.prototxt}")
            return 1
    # Allow file path, webcam index, or network URL
    source_arg = str(args.source)
    is_url = source_arg.startswith(('rtsp://', 'http://', 'https://'))
    is_cam_index = source_arg.isdigit()
    if not is_url and not is_cam_index and not Path(source_arg).exists():
        logger.error(f"Source not found: {args.source}")
        return 1

    cap = None

    try:
        used_cores = setup_cpu_affinity(args.core)

        # Load runtime configuration (defaults + optional JSON override)
        cfg = load_runtime_config(args)
        # CLI override for sampling interval if provided
        if getattr(args, 'sample_interval', None) is not None:
            try:
                cfg['sample_interval_sec'] = max(0.0, float(args.sample_interval))
            except Exception:
                logger.warning("Invalid --sample-interval; falling back to config/default")
        # Map nested performance.* if present in config.json
        try:
            perf_cfg = cfg.get('performance', {})
            if isinstance(perf_cfg, dict):
                if 'sample_interval_sec' in perf_cfg and 'sample_interval_sec' not in cfg:
                    cfg['sample_interval_sec'] = max(0.0, float(perf_cfg.get('sample_interval_sec', 0.0)))
                if 'target_fps' in perf_cfg and (args.fps == 10.0 or args.fps is None):
                    # Only override if user didn't explicitly change default
                    args.fps = float(perf_cfg.get('target_fps', args.fps))
        except Exception:
            pass

        # Inspect prototxt for declared input size (training size)
        declared_w = None
        declared_h = None
        try:
            with open(args.prototxt, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            # Try input_dim sequence: input_dim: 1\n input_dim: 3\n input_dim: H\n input_dim: W
            import re
            dims = re.findall(r"input_dim\s*:\s*(\d+)", txt)
            if len(dims) >= 4:
                # N, C, H, W
                declared_h = int(dims[2])
                declared_w = int(dims[3])
            else:
                # Try input_shape { dim: 1 dim: 3 dim: H dim: W }
                dims2 = re.findall(r"dim\s*:\s*(\d+)", txt)
                if len(dims2) >= 4:
                    declared_h = int(dims2[2])
                    declared_w = int(dims2[3])
        except Exception:
            pass

        # Load MobileNetSSD model
        tflite_interpreter = None
        if use_tflite:
            import importlib
            try:
                tflite_runtime = importlib.import_module('tflite_runtime.interpreter')
                TFLiteInterpreter = tflite_runtime.Interpreter
            except Exception:
                # fallback to full TF if present
                from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter  # type: ignore
            logger.info(f"Loading TFLite model: {Path(args.tflite).name}")
            tflite_interpreter = TFLiteInterpreter(model_path=args.tflite)
            tflite_interpreter.allocate_tensors()
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            logger.info(f"TFLite inputs: {input_details}")
            logger.info(f"TFLite outputs: {output_details}")
        else:
            logger.info(f"Loading model: {Path(args.weights).name}")
            net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        # Set preferable backend and target (CPU by default)
        use_opencl = os.environ.get('OPENCV_DNN_OPENCL', '0') == '1'
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CPU
        if use_opencl and not use_tflite:
            try:
                # Try OpenCL FP16 first if available
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.haveOpenCL():
                    backend = cv2.dnn.DNN_BACKEND_OPENCV
                    # On Pi OpenCL device often maps to FP16 capable; fallback to OPENCL
                    try:
                        target = cv2.dnn.DNN_TARGET_OPENCL_FP16
                    except Exception:
                        target = cv2.dnn.DNN_TARGET_OPENCL
            except Exception:
                pass
        if not use_tflite:
            net.setPreferableBackend(backend)
            net.setPreferableTarget(target)
        try:
            backend_name = {cv2.dnn.DNN_BACKEND_DEFAULT: 'Default', cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE: 'IE', cv2.dnn.DNN_BACKEND_OPENCV: 'OpenCV'}.get(backend, str(backend))
            target_name = {cv2.dnn.DNN_TARGET_CPU: 'CPU', cv2.dnn.DNN_TARGET_OPENCL: 'OpenCL', cv2.dnn.DNN_TARGET_OPENCL_FP16: 'OpenCLFP16'}.get(target, str(target))
            if use_tflite:
                logger.info("Backend: TFLite")
            else:
                logger.info(f"DNN backend: {backend_name}, target: {target_name}")
        except Exception:
            pass

        monitor = SystemMonitor(used_cores)

        cap = cv2.VideoCapture(int(source_arg) if is_cam_index else source_arg)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {args.source}")
            return 1

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam_index and not is_url else 0
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        try:
            logger.info(f"Video: {total_frames} frames @ {float(original_fps):.1f} FPS")
        except Exception:
            logger.info("Video: live source")

        frame_interval = 1.0 / args.fps
        # Determine sampling step in frames (process one frame every N seconds of video time)
        sample_interval_sec = float(cfg.get('sample_interval_sec', 0.0) or 0.0)
        if original_fps and original_fps > 0:
            sample_step = max(1, int(round(sample_interval_sec * original_fps))) if sample_interval_sec > 0 else 1
        else:
            # fallback if FPS unknown
            sample_step = 1 if sample_interval_sec <= 0 else max(1, int(round(sample_interval_sec * 30.0)))

        # Performance optimization: reduce display update frequency
        display_update_interval = max(1, int(args.fps / 10))  # Update display every N frames
        display_frame_count = 0

        metric_logger = MetricLogger()
        metric_logger.open()

        frame_count = 0  # processed frames count
        total_inference_time = 0
        start_time = time.time()

        # Chỉ sử dụng CounterState để hiển thị thông tin
        state = CounterState(cfg)

        print("\n=== STARTING VIDEO PROCESSING ===")
        print(f"Total frames: {total_frames}")
        print(f"Target FPS: {args.fps}")
        print("=" * 60)
        print("Press 'q' to quit at any time...")
        # Report model IO details
        model_cfg = cfg.get('model', {})
        cfg_w = model_cfg.get('input_width', 300)
        cfg_h = model_cfg.get('input_height', 300)
        if declared_w and declared_h:
            print(f"Model (prototxt) expects: {declared_w}x{declared_h} (WxH)")
        else:
            print("Model (prototxt) expects: unknown (no input_dim found)")
        print(f"Runtime config input: {cfg_w}x{cfg_h} (WxH)")
        classes = model_cfg.get('classes', [])
        if classes:
            print(f"Classes ({len(classes)}): {', '.join(map(str, classes))}")
        else:
            print("Classes: not provided in config.json")
        print("Note: Source/frame size does not change detection input; ROI is resized to model input size.")

        # Start background quit listener
        stop_event = threading.Event()
        _quit_thread = start_quit_listener(stop_event)

        # Prepare display window
        if args.show:
            cv2.namedWindow('Traffic Monitor', cv2.WINDOW_NORMAL)
            try:
                disp_cfg = cfg.get('display', {})
                init_w = int(disp_cfg.get('max_width', 1280))
                init_h = int(disp_cfg.get('max_height', 720))
            except Exception:
                init_w, init_h = 1280, 720
            cv2.resizeWindow('Traffic Monitor', init_w, init_h)

        while True:
            if stop_event.is_set():
                print("\n" + "=" * 60)
                logger.info("Quit requested by user ('q' pressed)")
                break
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("\n" + "=" * 60)
                logger.info("End of video reached")
                break

            frame_count += 1
            show_progress = frame_count <= 10 or frame_count % 10 == 0
            display_frame_count += 1
            should_update_display = args.show and (display_frame_count % display_update_interval == 0)

            try:
                # Preprocess frame according to model requirements
                inference_start = time.time()
                (h, w) = frame.shape[:2]

                # Get model configuration
                model_cfg = cfg.get('model', {})
                roi_cfg = model_cfg.get('roi', {})
                # Ensure config values are respected; fallback to 300 only if missing
                try:
                    input_width = int(model_cfg.get('input_width')) if 'input_width' in model_cfg else 300
                    input_height = int(model_cfg.get('input_height')) if 'input_height' in model_cfg else 300
                except Exception:
                    input_width, input_height = 300, 300

                # Calculate ROI coordinates - hỗ trợ cả hình chữ nhật và hình thang
                if roi_cfg.get('enabled', False):
                    # Xác định loại ROI từ config (rectangle hoặc trapezoid)
                    roi_type = roi_cfg.get('type', 'rectangle')

                    if roi_type == 'trapezoid':
                        # Lấy 4 tọa độ theo thứ tự ngược chiều kim đồng hồ cho hình thang
                        try:
                            trapz_points = roi_cfg.get('trapezoid', {}).get('points', [
                                [300, 300], [1000, 300], [1200, 800], [100, 800]
                            ])
                            # Sử dụng tọa độ pixel trực tiếp vì config.json đã dùng tọa độ pixel
                            roi_pts_px = trapz_points
                            if frame_count <= 3:
                                logger.info(f"Sử dụng ROI hình thang với các điểm: {roi_pts_px}")
                        except Exception as e:
                            logger.warning(f"Lỗi khi lấy tọa độ ROI hình thang: {e}")
                            # Dùng giá trị mặc định từ config.json
                            roi_pts_px = [[300, 300], [1000, 300], [1200, 800], [100, 800]]
                    else:  # rectangle
                        try:
                            # Lấy thông tin hình chữ nhật từ config
                            rect = roi_cfg.get('rectangle', {})
                            x = rect.get('x', 400)
                            y = rect.get('y', 300)
                            width = rect.get('width', 800)
                            height = rect.get('height', 600)

                            # Tạo 4 điểm cho hình chữ nhật
                            roi_pts_px = [
                                [x, y],                  # top-left
                                [x + width, y],          # top-right
                                [x + width, y + height], # bottom-right
                                [x, y + height]          # bottom-left
                            ]

                            if frame_count <= 3:
                                logger.info(f"Sử dụng ROI hình chữ nhật ({x}, {y}, {width}x{height})")
                        except Exception as e:
                            logger.warning(f"Lỗi khi lấy tọa độ ROI hình chữ nhật: {e}")
                            # Dùng giá trị mặc định từ config.json
                            roi_pts_px = [
                                [400, 300],                    # top-left
                                [400 + 800, 300],             # top-right
                                [400 + 800, 300 + 600],       # bottom-right
                                [400, 300 + 600]              # bottom-left
                            ]

                    # Tạo mask cho vùng ROI
                    mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(roi_pts_px, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)

                    # Áp dụng mask cho frame để tạo ROI
                    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

                    # Tìm bounding box của ROI để cắt frame
                    x_min = min(pt[0] for pt in roi_pts_px)
                    y_min = min(pt[1] for pt in roi_pts_px)
                    x_max = max(pt[0] for pt in roi_pts_px)
                    y_max = max(pt[1] for pt in roi_pts_px)

                    # Đảm bảo tọa độ không vượt quá kích thước frame
                    y_min = max(0, min(y_min, h-1))
                    y_max = max(0, min(y_max, h))
                    x_min = max(0, min(x_min, w-1))
                    x_max = max(0, min(x_max, w))

                    # Kiểm tra kích thước ROI hợp lệ
                    if x_min < x_max and y_min < y_max:
                        # Cắt frame theo bounding box
                        roi_rect = roi_frame[y_min:y_max, x_min:x_max].copy()
                        if frame_count <= 3:
                            logger.info(f"ROI rect có kích thước: {roi_rect.shape}")
                    else:
                        logger.warning(f"Kích thước ROI không hợp lệ: ({x_min},{y_min}) đến ({x_max},{y_max})")
                        roi_rect = frame.copy()  # Dùng frame gốc nếu kích thước không hợp lệ

                    # Lưu lại tọa độ bounding box để dùng sau này
                    roi_x, roi_y = x_min, y_min
                    roi_w, roi_h = x_max - x_min, y_max - y_min

                    # Vẽ đường biên ROI (hình chữ nhật hoặc hình thang) trên frame gốc
                    color = roi_cfg.get('color', [0, 255, 0])  # Default green
                    thickness = int(roi_cfg.get('thickness', 2))

                    # Tạo overlay với độ trong suốt để hiển thị vùng ROI rõ hơn
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    # Áp dụng độ trong suốt 20%
                    alpha = 0.2
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Vẽ đường viền ROI
                    cv2.polylines(frame, [pts], True, color, thickness)

                    # Thêm nhãn loại ROI
                    roi_label = f"ROI: {roi_type.upper()}"
                    cv2.putText(frame, roi_label, (roi_pts_px[0][0], roi_pts_px[0][1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Đánh dấu các góc của ROI
                    for i, pt in enumerate(roi_pts_px):
                        cv2.circle(frame, (pt[0], pt[1]), 5, (0, 0, 255), -1)
                        cv2.putText(frame, str(i+1), (pt[0]+5, pt[1]+5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                else:
                    # Use full frame if ROI is disabled
                    roi_frame = frame.copy()
                    roi_rect = frame.copy()
                    roi_x, roi_y = 0, 0
                    roi_w, roi_h = w, h
                    # Tạo mask cho toàn bộ frame
                    mask = np.ones((h, w), dtype=np.uint8) * 255

                # Resize ROI to model input size
                # Kiểm tra roi_rect có hợp lệ không
                if 'roi_rect' in locals() and roi_rect is not None and roi_rect.size > 0 and len(roi_rect.shape) == 3:
                    try:
                        # Resize roi_rect để đưa vào model
                        resized_roi = cv2.resize(roi_rect, (input_width, input_height))
                        if frame_count <= 3:
                            logger.info(f"Đã resize ROI rect thành {resized_roi.shape}")
                    except Exception as e:
                        logger.warning(f"Lỗi khi resize ROI: {e}")
                        # Fallback sử dụng frame gốc nếu có lỗi
                        resized_roi = cv2.resize(frame, (input_width, input_height))
                else:
                    # Fallback sử dụng frame gốc
                    logger.debug("Không thể sử dụng roi_rect, dùng frame gốc thay thế")
                    resized_roi = cv2.resize(frame, (input_width, input_height))

                # Tạo bản sao của resized_roi để vẽ hình thang lên đó (đầu vào model)
                model_input_vis = resized_roi.copy()

                # Vẽ hình thang lên ảnh đã resize
                if roi_cfg.get('enabled', False) and 'roi_pts_px' in locals():
                    # Tính lại tỷ lệ từ kích thước ROI sang kích thước model
                    scale_x = input_width / roi_w if roi_w > 0 else 0
                    scale_y = input_height / roi_h if roi_h > 0 else 0

                    if scale_x > 0 and scale_y > 0:
                        model_pts = []
                        for pt in roi_pts_px:
                            model_x = int((pt[0] - roi_x) * scale_x)
                            model_y = int((pt[1] - roi_y) * scale_y)
                            # Đảm bảo tọa độ nằm trong phạm vi ảnh
                            model_x = max(0, min(model_x, input_width - 1))
                            model_y = max(0, min(model_y, input_height - 1))
                            model_pts.append([model_x, model_y])

                        # Vẽ đa giác lên ảnh model_input_vis
                        model_pts_array = np.array(model_pts, np.int32)
                        model_pts_array = model_pts_array.reshape((-1, 1, 2))
                        cv2.polylines(model_input_vis, [model_pts_array], True, (0, 0, 255), 2)

                        # Đánh dấu các góc
                        for i, pt in enumerate(model_pts):
                            cv2.circle(model_input_vis, (pt[0], pt[1]), 3, (255, 0, 0), -1)
                            cv2.putText(model_input_vis, str(i+1), (pt[0]+3, pt[1]+3), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # MobileNetSSD Caffe expects BGR input, scale=0.007843..., mean=127.5, no RGB swap
                scale_val = float(model_cfg.get('scale_factor', 0.00784313725490196))
                mean_vals = model_cfg.get('mean_values', [127.5, 127.5, 127.5])
                swap_rb = bool(model_cfg.get('swap_rb', False))

                blob = cv2.dnn.blobFromImage(
                    resized_roi,
                    scalefactor=scale_val,
                    size=(input_width, input_height),
                    mean=tuple(mean_vals),
                    swapRB=swap_rb,
                    crop=bool(model_cfg.get('crop', False))
                )

                # Prepare debug visualizations in a single window (1 window, 3 panels) - only when needed
                if should_update_display:
                    # Panels in native form
                    panel_left = frame.copy()
                    if roi_cfg.get('enabled', False):
                        cv2.rectangle(panel_left,
                                      (roi_x, roi_y),
                                      (roi_x + roi_w, roi_y + roi_h),
                                      (0, 255, 0),
                                      int(cfg.get('display', {}).get('line_thickness', 2)))

                    # Tính toán các giá trị purple_y và blue_y (nhưng không vẽ chúng)
                    try:
                        disp_cfg_local = cfg.get('display', {})
                        H_full, W_full = panel_left.shape[:2]
                        purple_y = int(float(disp_cfg_local.get('purple_ratio', 0.5)) * H_full)
                        blue_y = int(float(disp_cfg_local.get('blue_ratio', 0.875)) * H_full)
                        # Giữ lại để debug nhưng không vẽ các đường
                        if frame_count <= 5:
                            logger.info(f"Reference points: purple_y={purple_y}, blue_y={blue_y}, H={H_full}")
                    except Exception:
                        pass

                    # Panel trên bên phải hiển thị ROI với hình thang
                    if 'roi_rect' in locals() and roi_rect is not None and roi_rect.size > 0:
                        panel_tr = roi_rect.copy()
                    else:
                        panel_tr = roi_frame.copy()
                    # Bottom-right luôn hiển thị model input (đầu vào của model)
                    panel_br = model_input_vis.copy()
                    # Thêm chú thích về kích thước đầu vào model
                    cv2.putText(panel_br, f"Model Input: {input_width}x{input_height}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Target display grid: two columns of equal width; right column split into two equal rows
                    disp_cfg = cfg.get('display', {})
                    max_w = int(disp_cfg.get('max_width', 1280))
                    max_h = int(disp_cfg.get('max_height', 720))
                    col_w = max(2, max_w // 2)
                    col_h = max(2, max_h)
                    row_h_top = col_h // 2
                    row_h_bottom = col_h - row_h_top

                    def fit_into_box(src, box_w, box_h):
                        h, w = src.shape[:2]
                        if w == 0 or h == 0:
                            return np.zeros((box_h, box_w, 3), dtype=np.uint8)
                        scale = min(box_w / w, box_h / h)
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
                        off_x = (box_w - new_w) // 2
                        off_y = (box_h - new_h) // 2
                        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
                        return canvas

                    left_cell = fit_into_box(panel_left, col_w, col_h)
                    tr_cell = fit_into_box(panel_tr, col_w, row_h_top)
                    br_cell = fit_into_box(panel_br, col_w, row_h_bottom)

                    combined_disp = np.zeros((col_h, col_w * 2, 3), dtype=np.uint8)
                    combined_disp[0:col_h, 0:col_w] = left_cell
                    combined_disp[0:row_h_top, col_w:col_w + col_w] = tr_cell
                    combined_disp[row_h_top:col_h, col_w:col_w + col_w] = br_cell

                    # Labels with true sizes
                    H_left, W_left = frame.shape[:2]
                    H_tr, W_tr = roi_frame.shape[:2]
                    cv2.putText(combined_disp, f'Original {W_left}x{H_left}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(combined_disp, f'ROI {W_tr}x{H_tr}', (col_w + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(combined_disp, f'Model Input {input_width}x{input_height}', (col_w + 10, row_h_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.imshow('Traffic Monitor', combined_disp)

                # Set input and run forward pass
                if use_tflite and tflite_interpreter is not None:
                    # TFLite expects NHWC uint8/float32 depending on model
                    inp = resized_roi
                    input_details = tflite_interpreter.get_input_details()
                    id0 = input_details[0]
                    tensor = inp
                    if id0["dtype"].__name__ == 'float32':
                        tensor = (tensor.astype(np.float32) - np.array(mean_vals, dtype=np.float32)) * (scale_val if scale_val != 0 else 1.0)
                    elif id0["dtype"].__name__ == 'uint8':
                        tensor = tensor.astype(np.uint8)
                    tensor = np.expand_dims(tensor, axis=0)
                    tflite_interpreter.set_tensor(id0['index'], tensor)
                    tflite_interpreter.invoke()
                    # Common SSD MobileNet tflite outputs: boxes [1, N, 4], classes [1,N], scores [1,N], num [1]
                    try:
                        boxes = tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[0]['index'])[0]
                        classes_out = tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[1]['index'])[0]
                        scores = tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[2]['index'])[0]
                        num = int(tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[3]['index'])[0])
                    except Exception:
                        # Fallback: single output case (convert to similar format)
                        out0 = tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[0]['index'])
                        # Not standardized; bail out
                        boxes, classes_out, scores, num = np.empty((0,4)), np.empty((0,)), np.empty((0,)), 0
                    # Build pred-like structure: [batch, 1, N, 7] as in Caffe (id, class, score, x1, y1, x2, y2)
                    pred_list = []
                    for i_ in range(num):
                        score = float(scores[i_])
                        class_id = int(classes_out[i_])
                        y1, x1, y2, x2 = boxes[i_]
                        pred_list.append([0.0, class_id, score, x1, y1, x2, y2])
                    pred = np.array([[pred_list]], dtype=np.float32)
                else:
                    net.setInput(blob)
                    pred = net.forward()


                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Prepare display canvas (used for overlays/tracking on original)
                canvas = frame.copy()

                # Parse detections, filter to vehicle-like classes and reasonable confidence
                rects: List[Tuple[int, int, int, int]] = []
                boxes_for_nms: List[List[int]] = []
                confidences: List[float] = []
                class_ids: List[int] = []
                (h, w) = frame.shape[:2]
                for i in range(0, pred.shape[2]):
                    confidence = pred[0, 0, i, 2]
                    detection_cfg = cfg.get('detection', {})
                    if confidence > float(detection_cfg.get("conf_thresh", 0.50)):
                        class_id = int(pred[0, 0, i, 1])
                        # Filter classes if needed
                        detection_cfg = cfg.get('detection', {})
                        allowed = set(a.lower() for a in detection_cfg.get("allowed_classes", []))
                        if allowed:
                            # Get class name from configuration
                            classes = model_cfg.get('classes', [])
                            if class_id < len(classes):
                                name = classes[class_id]
                                if name.lower() not in allowed:
                                    continue
                            else:
                                continue  # Skip if class ID is out of range
                        # Scale detection box from ROI coordinates to original frame
                        box = pred[0, 0, i, 3:7]  # Get relative coordinates [0-1]

                        if roi_cfg.get('enabled', False):
                            # Scale from ROI size to original frame size
                            x1 = int(box[0] * roi_w + roi_x)
                            y1 = int(box[1] * roi_h + roi_y)
                            x2 = int(box[2] * roi_w + roi_x)
                            y2 = int(box[3] * roi_h + roi_y)
                        else:
                            # Scale to full frame size if no ROI
                            x1 = int(box[0] * w)
                            y1 = int(box[1] * h)
                            x2 = int(box[2] * w)
                            y2 = int(box[3] * h)
                        bw = max(0, x2 - x1)
                        bh = max(0, y2 - y1)
                        if bw <= 0 or bh <= 0:
                            continue
                        boxes_for_nms.append([x1, y1, bw, bh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                # Apply NMS if configured
                detections_kept: List[Tuple[int, int, int, int, float, int]] = []  # x, y, w, h, conf, class_id
                if boxes_for_nms:
                    nms_thresh = float(cfg.get('detection', {}).get('nms_threshold', 0.4))
                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, float(cfg.get('detection', {}).get('conf_thresh', 0.50)), nms_thresh)
                    if len(indices) > 0:
                        for idx in (indices.flatten() if hasattr(indices, 'flatten') else indices):
                            x, y, bw, bh = boxes_for_nms[idx]
                            rects.append((int(x), int(y), int(bw), int(bh)))
                            detections_kept.append((int(x), int(y), int(bw), int(bh), float(confidences[idx]), int(class_ids[idx]) if idx < len(class_ids) else -1))

                # Get display configuration
                display_cfg = cfg.get('display', {})
                H, W = canvas.shape[:2]

                # Lọc rects dựa trên vùng ROI hình thang
                if rects and roi_cfg.get('enabled', False):
                    filtered_rects: List[Tuple[int, int, int, int]] = []

                    for (x, y, w, h) in rects:
                        # Tính tâm của đối tượng
                        cx = x + w // 2
                        cy = y + h // 2

                        # Kiểm tra xem tâm có nằm trong đa giác ROI không
                        if point_in_polygon((cx, cy), roi_pts_px):
                            filtered_rects.append((x, y, w, h))

                    rects = filtered_rects
                    logger.debug(f"Detected {len(rects)} objects in ROI out of {len(detections_kept)}")

                # Vẽ các đối tượng được phát hiện trực tiếp
                object_count = 0
                for (x, y, w, h) in rects:
                    # Vẽ khung cho đối tượng
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Hiển thị thông tin lớp và độ tin cậy
                    for det in detections_kept:
                        if det[0] == x and det[1] == y and det[2] == w and det[3] == h:
                            conf = det[4]
                            class_id = det[5]
                            classes_list = model_cfg.get('classes', [])
                            label = f"{classes_list[class_id] if 0 <= class_id < len(classes_list) else class_id}:{conf:.2f}"
                            cv2.putText(canvas, label, (x, max(0, y - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            break
                    object_count += 1

                # Cập nhật số lượng đối tượng cho trạng thái
                state.window_count = object_count
                state.total_count += object_count

                # Hiển thị tổng số đối tượng phát hiện được
                cv2.putText(canvas, f"Objects: {object_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                # Build ROI visualization with detections for bottom-right panel
                detections_exist = len(detections_kept) > 0

                # Tạo ảnh hiển thị ROI với vùng hình thang
                if 'roi_rect' in locals() and roi_rect is not None and roi_rect.size > 0:
                    roi_vis = roi_rect.copy()
                else:
                    roi_vis = roi_frame.copy()

                # Vẽ đa giác ROI trên ảnh roi_vis nếu ROI được kích hoạt
                if roi_cfg.get('enabled', False) and 'roi_pts_px' in locals():
                    # Tính lại tọa độ cho ảnh đã cắt
                    local_pts = []
                    for pt in roi_pts_px:
                        local_pts.append([pt[0] - roi_x, pt[1] - roi_y])
                    local_pts_array = np.array(local_pts, np.int32)
                    local_pts_array = local_pts_array.reshape((-1, 1, 2))
                    cv2.polylines(roi_vis, [local_pts_array], True, (0, 255, 255), 2)

                if detections_exist:
                    classes_list = model_cfg.get('classes', [])
                    for x, y, bw, bh, conf, cid in detections_kept:
                        # Map to ROI-local coordinates if ROI is enabled
                        if roi_cfg.get('enabled', False):
                            lx = x - roi_x
                            ly = y - roi_y
                        else:
                            # No ROI: use full-frame; bottom-right will show full frame resized
                            lx = x
                            ly = y
                        # Clip to roi_vis bounds
                        lx2 = lx + bw
                        ly2 = ly + bh
                        lx = max(0, min(lx, roi_vis.shape[1] - 1))
                        ly = max(0, min(ly, roi_vis.shape[0] - 1))
                        lx2 = max(0, min(lx2, roi_vis.shape[1] - 1))
                        ly2 = max(0, min(ly2, roi_vis.shape[0] - 1))
                        if lx2 > lx and ly2 > ly:
                            cv2.rectangle(roi_vis, (lx, ly), (lx2, ly2), (0, 255, 0), 2)
                            label = f"{classes_list[cid] if 0 <= cid < len(classes_list) else cid}:{conf:.2f}"
                            cv2.putText(roi_vis, label, (lx, max(0, ly - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                # Vẽ HUD đơn giản không sử dụng purple_y và blue_y
                # Hiển thị số đối tượng và thời gian xử lý
                cv2.putText(canvas, f"Objects: {len(rects)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"Inference: {inference_time*1000:.1f}ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Handle 'q' on the single window (only when displaying)
                if should_update_display:
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            stop_event.set()
                    except Exception:
                        pass

                # Release tensor and clear cache
                if 'im' in locals():
                    del im
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Frame {frame_count} processing failed: {e}")
                continue

            elapsed = time.time() - loop_start
            # Only throttle by target FPS when not using sampling-by-time
            if sample_step == 1 and elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            total_time = time.time() - loop_start
            current_fps = 1.0 / max(total_time, 1e-6)
            system_cpu, affinity_cpu, ram_usage = monitor.get_metrics()

            # Write metrics via helper
            metric_logger.write(total_time_ms=total_time*1000.0,
                                fps=current_fps,
                                system_cpu=system_cpu,
                                ram_mb=ram_usage)

            if show_progress:
                # Use current position in the stream for progress (accounts for skipped frames)
                try:
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                except Exception:
                    current_pos = frame_count
                progress_bar = print_progress_bar(current_pos, total_frames)
                avg_inference = total_inference_time / frame_count * 1000

                cpu_line = f"SysCPU: {system_cpu:.0f}%"

                # concise, once-per-second style reporting using EMA
                if 'ema_fps' not in locals():
                    ema_fps = current_fps
                    ema_inf = avg_inference
                    last_print_t = 0.0
                alpha = 0.2
                ema_fps = (1 - alpha) * ema_fps + alpha * current_fps
                ema_inf = (1 - alpha) * ema_inf + alpha * (avg_inference)
                now = time.time()
                if now - last_print_t >= 1.0:
                    last_print_t = now
                    print(f"\r{progress_bar} | FPS:{ema_fps:.1f} | INF:{ema_inf:.0f}ms | CPU:{system_cpu:.0f}% | RAM:{ram_usage:.0f}MB ", end="", flush=True)

            # Density window logging - đơn giản hóa
            if frame_count % int(cfg.get('density_window', 10)) == 0:
                density = len(rects)
                print(f"\n[Density] {density} objects detected in current frame (frame: {frame_count})")

            if frame_count % 100 == 0:
                avg_inference = total_inference_time / frame_count * 1000
                if affinity_cpu is not None:
                    print(f"\n[Frame {frame_count}] Avg inference: {avg_inference:.1f}ms, Current FPS: {current_fps:.1f}, System CPU: {system_cpu:.1f}%, Affinity CPU: {affinity_cpu:.1f}% (cores {used_cores})")
                else:
                    print(f"\n[Frame {frame_count}] Avg inference: {avg_inference:.1f}ms, Current FPS: {current_fps:.1f}, System CPU: {system_cpu:.1f}%")

            # Skip intermediate frames if sampling is enabled (process one frame every N seconds)
            if sample_step > 1:
                to_skip = sample_step - 1
                skipped = 0
                while skipped < to_skip:
                    grabbed = cap.grab()
                    if not grabbed:
                        break
                    skipped += 1

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        logger.info("Processing interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        return 1
    finally:
        if cap:
            cap.release()
        # if processor:
        #     processor.cleanup()
        try:
            if 'args' in locals() and getattr(args, 'show', False):
                cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            metric_logger.close()
        except Exception:
            pass
        logger.info("Cleanup completed")

    if frame_count > 0:
        avg_inference = total_inference_time / frame_count * 1000
        total_elapsed = time.time() - start_time
        actual_fps = frame_count / total_elapsed
        print("\n" + "=" * 60)
        print("=== PROCESSING COMPLETE ===")
        print(f"Total frames processed: {frame_count}/{total_frames}")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Average inference time: {avg_inference:.1f}ms")
        print(f"Actual processing FPS: {actual_fps:.2f}")
        print(f"Target FPS was: {args.fps}")
        print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())