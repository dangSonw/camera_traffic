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
    if not Path(args.source).exists():
        logger.error(f"Source video not found: {args.source}")
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

        # Load MobileNetSSD model
        logger.info(f"Loading model: {Path(args.weights).name}")
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        
        # Set preferable backend and target (CPU by default)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        monitor = SystemMonitor(used_cores)

        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {args.source}")
            return 1

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video: {total_frames} frames @ {original_fps:.1f} FPS")

        frame_interval = 1.0 / args.fps
        # Determine sampling step in frames (process one frame every N seconds of video time)
        sample_interval_sec = float(cfg.get('sample_interval_sec', 0.0) or 0.0)
        if original_fps and original_fps > 0:
            sample_step = max(1, int(round(sample_interval_sec * original_fps))) if sample_interval_sec > 0 else 1
        else:
            # fallback if FPS unknown
            sample_step = 1 if sample_interval_sec <= 0 else max(1, int(round(sample_interval_sec * 30.0)))

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

        # Start background quit listener
        stop_event = threading.Event()
        _quit_thread = start_quit_listener(stop_event)

        # Prepare display window
        if args.show:
            cv2.namedWindow('Traffic Monitor', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Traffic Monitor', 1600, 1200)  # Initial size, can be resized later

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

            try:
                # Preprocess frame according to model requirements
                inference_start = time.time()
                (h, w) = frame.shape[:2]
                
                # Get model configuration
                model_cfg = cfg.get('model', {})
                roi_cfg = model_cfg.get('roi', {})
                input_width = model_cfg.get('input_width', 300)
                input_height = model_cfg.get('input_height', 300)
                
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
                
                # Create blob from ROI and prepare debug images
                blob = cv2.dnn.blobFromImage(
                    resized_roi,
                    scalefactor=model_cfg.get('scale_factor', 0.00784313725490196),
                    size=(input_width, input_height),
                    mean=tuple(model_cfg.get('mean_values', [127.5, 127.5, 127.5])),
                    swapRB=model_cfg.get('swap_rb', True),
                    crop=False
                )
                
                # Create debug visualizations in a single window
                if args.show:
                    # Create a copy of the frame with ROI rectangle
                    debug_original = frame.copy()
                    if roi_cfg.get('enabled', False):
                        cv2.rectangle(debug_original, 
                                    (roi_x, roi_y), 
                                    (roi_x + roi_w, roi_y + roi_h), 
                                    (0, 255, 0), 2)
                    
                    # Get ROI region
                    debug_roi = roi_frame.copy()
                    
                    # Get model input (resized ROI)
                    debug_model_input = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)  # Convert back to BGR for display
                    
                    # Create a blank canvas for the combined view
                    h, w = debug_original.shape[:2]
                    combined_h = h * 2  # 2 rows
                    combined_w = w + max(debug_roi.shape[1], debug_model_input.shape[1])  # Original + max of ROI/Model
                    
                    # Create a black canvas
                    combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
                    
                    # Place the original frame (top-left)
                    combined[0:h, 0:w] = debug_original
                    
                    # Place the ROI region (bottom-left)
                    roi_h, roi_w = debug_roi.shape[:2]
                    combined[h:h+roi_h, 0:roi_w] = debug_roi
                    
                    # Place the model input (top-right)
                    model_h, model_w = debug_model_input.shape[:2]
                    combined[0:model_h, w:w+model_w] = cv2.resize(debug_model_input, (model_w, model_h))
                    
                    # Add labels
                    cv2.putText(combined, 'Original Frame', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, f'ROI Region ({roi_w}x{roi_h})', (10, h + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined, f'Model Input ({input_width}x{input_height})', (w + 10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display the combined window
                    cv2.imshow('Traffic Monitor', combined)
                
                # Set input and run forward pass
                net.setInput(blob)
                pred = net.forward()



                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Prepare display canvas: keep original frame size (no resizing)
                canvas = frame.copy()

                # Parse detections, filter to vehicle-like classes and reasonable confidence
                rects: List[Tuple[int, int, int, int]] = []
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
                            
                        rects.append((x1, y1, x2 - x1, y2 - y1))

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
                draw_hud(canvas, purple_y, blue_y, cfg, state.total_count, state.window_count)

                # Show window and handle 'q'
                if args.show:
                    try:
                        cv2.imshow('Detections', canvas)
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