from __future__ import annotations

from typing import Dict, List, Tuple, Set

Color = Tuple[int, int, int]


class CounterState:
    def __init__(self, cfg: Dict[str, object]):
        self.cfg = cfg
        self.histories: Dict[int, List[Tuple[int, int]]] = {}
        self.last_y_positions: Dict[int, int] = {}
        self.counted_ids: Set[int] = set()
        self.wrong_way_ids: Set[int] = set()
        self.total_count: int = 0
        self.window_count: int = 0
        self.window_frames: int = 0
        self._density_window: int = int(cfg.get('density_window', 10))

    def update(self,
               tracked: List[List[int]],
               frame_height: int,
               blue_y: int,
               line_margin: int,
               min_delta: int) -> Dict[int, Color]:
        colors: Dict[int, Color] = {}
        max_len = int(self.cfg.get("trail_max_len", 50))

        for x, y, w, h, tid in tracked:
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Update path history
            path = self.histories.get(tid, [])
            path.append((cx, cy))
            if len(path) > max_len:
                path = path[-max_len:]
            self.histories[tid] = path

            # Trajectory-based crossing detection
            if tid not in self.counted_ids:
                crossed_down = False
                crossed_up = False
                start_idx = max(0, len(path) - 3)
                for i in range(start_idx, len(path) - 1):
                    y0 = path[i][1]
                    y1 = path[i + 1][1]
                    if abs(y1 - y0) < min_delta:
                        continue
                    if (y0 <= (blue_y - line_margin)) and (y1 >= (blue_y + line_margin)):
                        crossed_down = True
                        break
                    if (y0 >= (blue_y + line_margin)) and (y1 <= (blue_y - line_margin)):
                        crossed_up = True
                        break
                    if (y0 - blue_y) * (y1 - blue_y) <= 0 and abs(y1 - y0) >= min_delta:
                        if y1 > y0:
                            crossed_down = True
                        elif y1 < y0:
                            crossed_up = True
                        break

                if crossed_down:
                    self.total_count += 1
                    self.window_count += 1
                    self.counted_ids.add(tid)
                elif crossed_up:
                    self.wrong_way_ids.add(tid)

            self.last_y_positions[tid] = cy

            # Assign colors for drawing
            if tid in self.counted_ids:
                colors[tid] = (0, 255, 0)
            elif tid in self.wrong_way_ids:
                colors[tid] = (0, 0, 255)
            else:
                colors[tid] = (0, 255, 255)

        return colors

    def on_frame_end(self) -> int | None:
        self.window_frames += 1
        if self.window_frames >= self._density_window:
            val = self.window_count
            self.window_count = 0
            self.window_frames = 0
            return val
        return None
