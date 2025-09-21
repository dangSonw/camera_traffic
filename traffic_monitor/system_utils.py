import importlib
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import psutil

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

def print_progress_bar(current, total, length=50):
    percent = float(current) / total
    hashes = '#' * int(round(percent * length))
    spaces = ' ' * (length - len(hashes))
    return f"[{hashes}{spaces}] {percent:.1%} ({current}/{total})"
