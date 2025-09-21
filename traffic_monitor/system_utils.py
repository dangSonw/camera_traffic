from contextlib import contextmanager
from typing import List, Optional

import psutil

@contextmanager
def managed_files(*files):
    try:
        yield files
    finally:
        for file in files:
            if file and not file.closed:
                file.close()

def setup_cpu_affinity(core_count: Optional[int]) -> List[int]:
    if core_count is not None:
        try:
            p = psutil.Process()
            num_cores = psutil.cpu_count(logical=True)
            if core_count > num_cores:
                core_count = num_cores
            core_ids = list(range(core_count))
            p.cpu_affinity(core_ids)
            return core_ids
        except Exception as e:
            return list(range(psutil.cpu_count(logical=True)))
    else:
        return list(range(psutil.cpu_count(logical=True)))

def print_progress_bar(current, total, length=50):
    percent = float(current) / total
    hashes = '#' * int(round(percent * length))
    spaces = ' ' * (length - len(hashes))
    return f"[{hashes}{spaces}] {percent:.1%} ({current}/{total})"
