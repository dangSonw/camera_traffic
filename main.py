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
from tracker import Tracker
from traffic_monitor.system_utils import (
    start_quit_listener,
    SystemMonitor,
    setup_cpu_affinity,
    write_init_log,
    print_progress_bar,
)
from traffic_monitor.config import build_arg_parser, load_runtime_config
from traffic_monitor.logging_utils import MetricLogger
from traffic_monitor.counter import CounterState
from traffic_monitor.overlay import draw_tracks, draw_hud

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional Windows-only keyboard support (for 'q' to quit)
try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate required model files
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
        logger.info(f"Loading model: {Path(args.weights).name}")
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        
        # Set preferable backend and target (CPU by default)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        try:
            backend_name = 'OpenCV'
            target_name = 'CPU'
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

        # Tracker and counting state (simplified)
        tracker = Tracker()
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
                
                # Calculate ROI coordinates
                if roi_cfg.get('enabled', False):
                    # Calculate ROI in pixels
                    roi_x = int(roi_cfg['x'] * w)
                    roi_y = int(roi_cfg['y'] * h)
                    roi_w = int(roi_cfg['width'] * w)
                    roi_h = int(roi_cfg['height'] * h)
                    
                    # Extract ROI from frame
                    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    # Draw ROI rectangle on the original frame
                    color = roi_cfg.get('color', [0, 255, 0])  # Default green
                    thickness = roi_cfg.get('thickness', 2)
                    cv2.rectangle(frame, (roi_x, roi_y), 
                                (roi_x + roi_w, roi_y + roi_h), 
                                color, thickness)
                else:
                    # Use full frame if ROI is disabled
                    roi_frame = frame
                    roi_x, roi_y = 0, 0
                
                # Resize ROI to model input size
                resized_roi = cv2.resize(roi_frame, (input_width, input_height))

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

                    # Draw purple/blue horizontal lines on the left panel (with shadow + labels)
                    try:
                        disp_cfg_local = cfg.get('display', {})
                        H_full, W_full = panel_left.shape[:2]
                        purple_y = int(float(disp_cfg_local.get('purple_ratio', 0.5)) * H_full)
                        blue_y = int(float(disp_cfg_local.get('blue_ratio', 0.875)) * H_full)
                        line_th = max(2, int(disp_cfg_local.get('line_thickness', 2)))
                        # Shadow lines for visibility
                        cv2.line(panel_left, (0, purple_y+1), (W_full - 1, purple_y+1), (0, 0, 0), line_th + 2)
                        cv2.line(panel_left, (0, blue_y+1), (W_full - 1, blue_y+1), (0, 0, 0), line_th + 2)
                        # Colored lines
                        cv2.line(panel_left, (0, purple_y), (W_full - 1, purple_y), (255, 0, 255), line_th)
                        cv2.line(panel_left, (0, blue_y), (W_full - 1, blue_y), (255, 0, 0), line_th)
                        # Labels
                        cv2.putText(panel_left, f"Purple y={purple_y}", (10, max(20, purple_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(panel_left, f"Blue y={blue_y}", (10, min(H_full - 10, blue_y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        # Debug print for first few frames
                        if frame_count <= 5:
                            logger.info(f"Lines: purple_y={purple_y}, blue_y={blue_y}, thickness={line_th}, H={H_full}")
                    except Exception:
                        pass

                    panel_tr = roi_frame.copy()
                    # Bottom-right shows detections drawn on ROI if any; otherwise blank
                    if 'roi_vis' in locals() and len(detections_kept) > 0:
                        panel_br = roi_vis
                    else:
                        panel_br = np.zeros_like(roi_frame)

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
                purple_y = int(display_cfg.get("purple_ratio", 0.5) * H)
                blue_y = int(display_cfg.get("blue_ratio", 0.875) * H)
                line_margin = max(2, int(H * display_cfg.get("line_margin_factor", 0.01)))
                min_delta = max(2, int(H * display_cfg.get("min_delta_factor", 0.0033)))

                # Filter rects to ROI (center below purple_y)
                if rects:
                    filtered_rects: List[Tuple[int, int, int, int]] = []
                    for (x, y, w, h) in rects:
                        cy = (y + y + h) // 2
                        if cy >= purple_y:
                            filtered_rects.append((x, y, w, h))
                    rects = filtered_rects

                # Update tracker and counting state
                tracked = tracker.update(rects)
                id_colors = state.update(tracked, H, blue_y, line_margin, min_delta)

                # Draw overlays
                draw_tracks(canvas, tracked, id_colors, state.histories, cfg)
                # Build ROI visualization with detections for bottom-right panel
                detections_exist = len(detections_kept) > 0
                roi_vis = roi_frame.copy()
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
                draw_hud(canvas, purple_y, blue_y, cfg, state.total_count, state.window_count)

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

                print(f"\r{progress_bar} | Inf: {avg_inference:.0f}ms | FPS: {current_fps:.1f} | {cpu_line} | RAM: {ram_usage:.0f}MB ", end="", flush=True)

            # Density window logging
            maybe_density = state.on_frame_end()
            if maybe_density is not None:
                print(f"\n[Density] {maybe_density} objects/{int(cfg.get('density_window', 10))} frames (cumulative count: {state.total_count})")

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