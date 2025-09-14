import importlib
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import cv2
import psutil
import torch

logger = logging.getLogger(__name__)


@contextmanager
def managed_files(*files):
    try:
        yield files
    finally:
        for file in files:
            if file and not file.closed:
                file.close()


def start_quit_listener(stop_event: threading.Event):
    """Start a daemon thread that sets stop_event when user presses 'q'."""
    try:
        import msvcrt  # type: ignore
    except Exception:
        msvcrt = None  # type: ignore

    def _worker():
        if msvcrt is not None:
            while not stop_event.is_set():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch() if hasattr(msvcrt, 'getwch') else msvcrt.getch().decode(errors='ignore')
                        if str(ch).lower() == 'q':
                            stop_event.set()
                            return
                except Exception:
                    pass
                time.sleep(0.05)
        else:
            try:
                while not stop_event.is_set():
                    line = sys.stdin.readline()
                    if not line:
                        break
                    if 'q' in line.strip().lower():
                        stop_event.set()
                        return
            except Exception:
                pass

    t = threading.Thread(target=_worker, name="QuitListener", daemon=True)
    t.start()
    return t


class SystemMonitor:
    def __init__(self, used_cores: Optional[List[int]] = None):
        self.process = psutil.Process(os.getpid())
        self.used_cores = used_cores or []
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def get_metrics(self) -> Tuple[float, Optional[float], float]:
        try:
            system_cpu = psutil.cpu_percent(interval=None)
        except Exception:
            system_cpu = 0.0
        affinity_cpu = None
        try:
            if self.used_cores:
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                selected = [per_core[i] for i in self.used_cores if 0 <= i < len(per_core)]
                if selected:
                    affinity_cpu = float(sum(selected) / len(selected))
        except Exception:
            affinity_cpu = None
        ram_usage = self.process.memory_info().rss / (1024 * 1024)
        return system_cpu, affinity_cpu, ram_usage


def setup_cpu_affinity(core_count: Optional[int]) -> List[int]:
    if core_count is not None:
        try:
            p = psutil.Process()
            num_cores = psutil.cpu_count(logical=True)
            if core_count > num_cores:
                logger.warning(f"Requested {core_count} cores but only {num_cores} available")
                core_count = num_cores
            core_ids = list(range(core_count))
            p.cpu_affinity(core_ids)
            logger.info(f"Set CPU affinity to cores: {core_ids}")
            return core_ids
        except Exception as e:
            logger.error(f"Failed to set CPU affinity: {e}")
            return list(range(psutil.cpu_count(logical=True)))
    else:
        return list(range(psutil.cpu_count(logical=True)))


def write_init_log(args, used_cores: List[int], processor):
    try:
        with open("log/init_log.txt", "w") as f:
            f.write("=== ENHANCED INIT CONFIG ===\n")
            f.write(f"Source: {args.source}\n")
            f.write(f"Weights: {args.weights}\n")
            f.write(f"Target FPS: {args.fps}\n")
            f.write(f"Image size: {args.imgsz}\n")
            f.write(f"CPU physical cores: {psutil.cpu_count(logical=False)}\n")
            f.write(f"CPU logical threads: {psutil.cpu_count(logical=True)}\n")
            f.write(f"Used cores: {used_cores}\n")
            f.write(f"Model stride: {processor.stride}\n")
            f.write(f"Model classes: {len(processor.names)}\n")
            f.write(f"Device: {processor.device}\n")
            f.write(f"PyTorch version: {torch.__version__}\n")
            try:
                f.write(f"OpenCV version: {cv2.__version__}\n")
            except AttributeError:
                f.write("OpenCV version: unknown\n")
            try:
                ultralytics = importlib.import_module('ultralytics')
                f.write(f"Ultralytics version: {ultralytics.__version__}\n")
            except (ImportError, AttributeError):
                f.write("Ultralytics: not available\n")
            f.write(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB\n")
            try:
                freq = psutil.cpu_freq()
                f.write(f"CPU frequency: {freq.current:.0f} MHz\n")
            except Exception:
                pass
        logger.info("Initialization log written successfully")
    except Exception as e:
        logger.error(f"Failed to write init log: {e}")


def print_progress_bar(current, total, length=50):
    percent = float(current) / total
    hashes = '#' * int(round(percent * length))
    spaces = ' ' * (length - len(hashes))
    return f"[{hashes}{spaces}] {percent:.1%} ({current}/{total})"
