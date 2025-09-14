import time
import sys
from pathlib import Path
import torch
import cv2
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
from traffic_monitor.yolo_processor import YOLOProcessor
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

# Ensure workspace root and package root are on sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORKSPACE_ROOT = ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not Path(args.weights).exists():
        logger.error(f"Weights file not found: {args.weights}")
        return 1
    if not Path(args.source).exists():
        logger.error(f"Source video not found: {args.source}")
        return 1

    processor = None
    cap = None

    try:
        used_cores = setup_cpu_affinity(args.core)

        # Load runtime configuration (defaults + optional JSON override)
        cfg = load_runtime_config(args)
        processor = YOLOProcessor(args.weights, 'cpu')
        # Configure preprocessing behavior
        if getattr(args, 'letterbox', False):
            setattr(processor, 'use_letterbox', True)
        write_init_log(args, used_cores, processor)
        monitor = SystemMonitor(used_cores)

        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {args.source}")
            return 1

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video: {total_frames} frames @ {original_fps:.1f} FPS")

        frame_interval = 1.0 / args.fps

        metric_logger = MetricLogger()
        metric_logger.open()

        frame_count = 0
        total_inference_time = 0
        start_time = time.time()

        # Tracker and counting state (configurable)
        try:
            tracker = Tracker(
                iou_threshold=float(cfg.get("tracker_iou_threshold", 0.2)),
                max_distance=float(cfg.get("tracker_max_distance", 60.0)),
                max_missed=int(cfg.get("tracker_max_missed", 10)),
                smooth_alpha=float(cfg.get("tracker_smooth_alpha", 0.25)),
                use_prediction=bool(cfg.get("tracker_use_prediction", True)),
                velocity_alpha=float(cfg.get("tracker_velocity_alpha", 0.5)),
                max_speed=float(cfg.get("tracker_max_speed", 120.0)),
            )
        except TypeError as e:
            logger.warning(f"Tracker init with config args failed ({e}); falling back to default Tracker(). Consider updating tracker.py.")
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

        # Prepare display window if requested
        if args.show:
            try:
                cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Detections', 1000, 800)
            except Exception:
                pass

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
                # Compute source-space purple ROI line
                srcH, srcW = frame.shape[:2]
                purple_src_y = int(float(cfg.get("purple_ratio", 0.5)) * srcH)
                # Prepare masked frame for model input (only bottom region is detectable)
                frame_for_model = frame.copy()
                frame_for_model[:purple_src_y, :] = 0  # black out above purple line for model

                inference_start = time.time()
                im = processor.preprocess_frame(frame_for_model, imgsz=args.imgsz)
                in_shape = im.shape[2:]  # (h, w) of network input
                pred = processor.inference(im)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # Prepare display canvas: full original frame, resized to match model input size for consistent coordinates
                if processor.last_input_bgr is not None:
                    disp_h, disp_w = processor.last_input_bgr.shape[:2]
                    canvas = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                else:
                    canvas = frame.copy()

                # Parse detections, filter to vehicle-like classes and reasonable confidence
                rects: List[Tuple[int, int, int, int]] = []
                det = pred[0] if pred and len(pred) > 0 else None
                if det is not None and len(det):
                    det_np = det.detach().cpu().numpy()
                    allowed = set(a.lower() for a in cfg.get("allowed_classes", []))
                    for *xyxy, conf, cls_id in det_np:
                        cls_id = int(cls_id)
                        name = processor.names[cls_id] if processor.names and cls_id < len(processor.names) else str(cls_id)
                        if allowed and name.lower() not in allowed:
                            continue
                        if conf < float(cfg.get("conf_thresh", 0.30)):
                            continue
                        x1, y1, x2, y2 = map(int, xyxy)
                        w = max(0, x2 - x1)
                        h = max(0, y2 - y1)
                        if w <= 2 or h <= 2:
                            continue
                        rects.append((x1, y1, w, h))

                # Lines on canvas space
                H, W = canvas.shape[:2]
                purple_y = int(float(cfg.get("purple_ratio", 0.5)) * H)
                blue_y = int(float(cfg.get("blue_ratio", 0.875)) * H)
                line_margin = max(2, int(H * float(cfg.get("line_margin_factor", 0.01))))
                min_delta = max(2, int(H * float(cfg.get("min_delta_factor", 0.0033))))

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
                del im
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Frame {frame_count} processing failed: {e}")
                continue

            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
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
                progress_bar = print_progress_bar(frame_count, total_frames)
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
        if processor:
            processor.cleanup()
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