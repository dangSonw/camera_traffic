from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]


def draw_tracks(canvas,
                tracked: List[List[int]],
                id_colors: Dict[int, Color],
                histories: Dict[int, List[Tuple[int, int]]],
                cfg: Dict[str, object]) -> None:
    for x, y, w, h, tid in tracked:
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        color = id_colors.get(tid, (0, 255, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, int(cfg.get("box_thickness", 1)))
        cv2.circle(canvas, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"ID {tid}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(cfg.get("text_scale", 0.4)),
            (255, 255, 255),
            int(cfg.get("text_thickness", 1)),
            cv2.LINE_AA,
        )
        # Trails and direction arrow
        path_draw = histories.get(tid, [])
        if len(path_draw) >= 2:
            pts = np.array(path_draw, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=1)
            x0, y0 = path_draw[-2]
            x1_, y1_ = path_draw[-1]
            cv2.arrowedLine(canvas, (int(x0), int(y0)), (int(x1_), int(y1_)), color, max(1, int(cfg.get("line_thickness", 1))), tipLength=0.4)


def draw_hud(canvas,
             purple_y: int,
             blue_y: int,
             cfg: Dict[str, object],
             total_count: int,
             window_count: int) -> None:
    H, W = canvas.shape[:2]
    cv2.line(canvas, (0, purple_y), (W - 1, purple_y), (255, 0, 255), int(cfg.get("line_thickness", 1)))
    cv2.line(canvas, (0, blue_y), (W - 1, blue_y), (255, 0, 0), int(cfg.get("line_thickness", 1)))

    cv2.putText(
        canvas,
        f"Count: {total_count}",
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(cfg.get("text_scale", 0.4)),
        (255, 0, 0),
        int(cfg.get("text_thickness", 1)),
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Window({int(cfg.get('density_window', 10))}f): {window_count}",
        (2, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(cfg.get("text_scale", 0.4)),
        (255, 0, 0),
        int(cfg.get("text_thickness", 1)),
        cv2.LINE_AA,
    )
