from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MetricLogger:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self._time = None
        self._fps = None
        self._cpu = None
        self._ram = None

    def open(self) -> None:
        try:
            self._time = open("log/time_log.txt", "w")
            self._fps = open("log/fps_log.txt", "w")
            self._cpu = open("log/system_cpu_log.txt", "w")
            self._ram = open("log/ram_log.txt", "w")
        except Exception as e:
            logger.error(f"Failed to open metric logs: {e}")

    def write(self, total_time_ms: float, fps: float, system_cpu: float, ram_mb: float) -> None:
        try:
            if self._time:
                self._time.write(f"{total_time_ms:.2f}\n"); self._time.flush()
            if self._fps:
                self._fps.write(f"{fps:.2f}\n"); self._fps.flush()
            if self._cpu:
                self._cpu.write(f"{system_cpu:.1f}\n"); self._cpu.flush()
            if self._ram:
                self._ram.write(f"{ram_mb:.2f}\n"); self._ram.flush()
        except Exception as e:
            logger.error(f"Failed to write logs: {e}")

    def close(self) -> None:
        for fh in (self._time, self._fps, self._cpu, self._ram):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
        self._time = self._fps = self._cpu = self._ram = None
