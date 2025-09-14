import math
from typing import List, Tuple, Dict


def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2]) * max(0, boxA[3])
    areaB = max(0, boxB[2]) * max(0, boxB[3])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


class Tracker:
    """
    Enhanced centroid tracker with:
    - IoU-based association (greedy) with fallback to distance
    - Exponential smoothing for bbox and center to reduce jitter
    - Missed-frame handling to keep IDs stable over short occlusions
    Public interface remains: update(objects_rect) -> List[[x,y,w,h,id]]
    """

    def __init__(self,
                 iou_threshold: float = 0.2,
                 max_distance: float = 60.0,
                 max_missed: int = 10,
                 smooth_alpha: float = 0.25,
                 use_prediction: bool = True,
                 velocity_alpha: float = 0.5,
                 max_speed: float = 120.0):
        self.id_count = 0
        # tracks: id -> dict(x,y,w,h,cx,cy,vx,vy,missed)
        self.tracks: Dict[int, Dict[str, float]] = {}
        self.missed: Dict[int, int] = {}
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.alpha = smooth_alpha
        self.use_pred = use_prediction
        self.vel_alpha = velocity_alpha
        self.max_speed = max_speed

    def _center(self, x: int, y: int, w: int, h: int) -> Tuple[float, float]:
        return (x + w / 2.0, y + h / 2.0)

    def _dist(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, objects_rect: List[Tuple[int, int, int, int]]):
        # Prepare detections
        dets = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx, cy = self._center(x, y, w, h)
            dets.append({'bbox': (x, y, w, h), 'center': (cx, cy), 'matched': False})

        # Build current track list
        track_ids = list(self.tracks.keys())

        # Predict next positions (constant velocity)
        predicted: Dict[int, Dict[str, float]] = {}
        for tid in track_ids:
            t = self.tracks[tid]
            cx = t['cx']
            cy = t['cy']
            vx = t.get('vx', 0.0)
            vy = t.get('vy', 0.0)
            if self.use_pred:
                cx_p = cx + vx
                cy_p = cy + vy
            else:
                cx_p = cx
                cy_p = cy
            w = t['w']
            h = t['h']
            x_p = cx_p - w / 2.0
            y_p = cy_p - h / 2.0
            predicted[tid] = {'cx': cx_p, 'cy': cy_p, 'x': x_p, 'y': y_p, 'w': w, 'h': h}

        # 1) IoU-based greedy matching (using predicted boxes)
        matches = []  # (tid, det_idx)
        if track_ids and dets:
            # compute all IoUs
            candidates = []
            for tid in track_ids:
                tpred = predicted[tid]
                tbox = (int(round(tpred['x'])), int(round(tpred['y'])), int(round(tpred['w'])), int(round(tpred['h'])))
                for di, d in enumerate(dets):
                    iou = _iou(tbox, d['bbox'])
                    if iou >= self.iou_threshold:
                        candidates.append((iou, tid, di))
            # pick best pairs greedily by IoU
            candidates.sort(reverse=True)
            used_t, used_d = set(), set()
            for iou, tid, di in candidates:
                if tid in used_t or di in used_d:
                    continue
                matches.append((tid, di))
                used_t.add(tid)
                used_d.add(di)
                dets[di]['matched'] = True

        # 2) Fallback distance matching for unmatched tracks/detections (using predicted centers)
        unmatched_tracks = [tid for tid in track_ids if tid not in [m[0] for m in matches]]
        unmatched_dets = [i for i, d in enumerate(dets) if not d['matched']]
        if unmatched_tracks and unmatched_dets:
            cand = []
            for tid in unmatched_tracks:
                tpred = predicted[tid]
                tcenter = (tpred['cx'], tpred['cy'])
                for di in unmatched_dets:
                    dcenter = dets[di]['center']
                    dist = self._dist(tcenter, dcenter)
                    if dist <= self.max_distance and dist <= self.max_speed:
                        cand.append((dist, tid, di))
            cand.sort()
            used_t, used_d = set(), set()
            for dist, tid, di in cand:
                if tid in used_t or di in used_d:
                    continue
                matches.append((tid, di))
                used_t.add(tid)
                used_d.add(di)
                dets[di]['matched'] = True

        # Update matched tracks with smoothing and velocity
        updated_ids = set()
        for tid, di in matches:
            d = dets[di]
            x, y, w, h = d['bbox']
            cx, cy = d['center']
            t = self.tracks.get(tid)
            if t is None:
                # new track safety
                self.tracks[tid] = {'x': x, 'y': y, 'w': w, 'h': h, 'cx': cx, 'cy': cy, 'vx': 0.0, 'vy': 0.0}
                self.missed[tid] = 0
            else:
                a = self.alpha
                # Smooth center and size
                prev_cx = t['cx']
                prev_cy = t['cy']
                t['cx'] = a * cx + (1 - a) * prev_cx
                t['cy'] = a * cy + (1 - a) * prev_cy
                t['w'] = a * w + (1 - a) * t['w']
                t['h'] = a * h + (1 - a) * t['h']
                # Anchor bbox top-left to smoothed center to reduce drift
                t['x'] = t['cx'] - t['w'] / 2.0
                t['y'] = t['cy'] - t['h'] / 2.0
                # Update velocity with smoothing
                vx_meas = cx - prev_cx
                vy_meas = cy - prev_cy
                t['vx'] = self.vel_alpha * vx_meas + (1 - self.vel_alpha) * t.get('vx', 0.0)
                t['vy'] = self.vel_alpha * vy_meas + (1 - self.vel_alpha) * t.get('vy', 0.0)
                self.missed[tid] = 0
            updated_ids.add(tid)

        # Increment missed for unmatched tracks; slightly decay velocity
        still_unmatched_tracks = [tid for tid in track_ids if tid not in [m[0] for m in matches]]
        for tid in still_unmatched_tracks:
            self.missed[tid] = self.missed.get(tid, 0) + 1
            t = self.tracks.get(tid)
            if t is not None:
                t['vx'] *= 0.9
                t['vy'] *= 0.9
        # Remove tracks that exceeded miss limit
        for tid in list(self.tracks.keys()):
            if self.missed.get(tid, 0) > self.max_missed:
                self.tracks.pop(tid, None)
                self.missed.pop(tid, None)

        # Create new tracks for unmatched detections
        for i, d in enumerate(dets):
            if not d['matched']:
                x, y, w, h = d['bbox']
                cx, cy = d['center']
                self.tracks[self.id_count] = {'x': x, 'y': y, 'w': float(w), 'h': float(h), 'cx': cx, 'cy': cy, 'vx': 0.0, 'vy': 0.0}
                self.missed[self.id_count] = 0
                self.id_count += 1

        # Prepare output: one bbox per active track updated this frame (rounded)
        objects_bbs_ids: List[List[int]] = []
        for tid in updated_ids:
            t = self.tracks[tid]
            x = int(round(t['x']))
            y = int(round(t['y']))
            w = int(round(t['w']))
            h = int(round(t['h']))
            objects_bbs_ids.append([x, y, w, h, tid])

        return objects_bbs_ids