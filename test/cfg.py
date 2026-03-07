import cv2
import sys
import os
import json
import argparse
import numpy as np


def _load_points(cfg_path):
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pts = (
            data.get('model', {})
            .get('roi', {})
            .get('full_region', {})
            .get('points', [])
        )
        if isinstance(pts, list) and len(pts) == 4 and all(isinstance(p, list) and len(p) == 2 for p in pts):
            return pts, data
        return [[300, 300], [1000, 300], [1200, 800], [100, 800]], data
    except Exception:
        return [[300, 300], [1000, 300], [1200, 800], [100, 800]], {}


def _save_point(cfg_path, data, idx, x, y):
    if 'model' not in data:
        data['model'] = {}
    if 'roi' not in data['model']:
        data['model']['roi'] = {}
    if 'full_region' not in data['model']['roi']:
        data['model']['roi']['full_region'] = {}
    pts = data['model']['roi']['full_region'].get('points')
    if not isinstance(pts, list) or len(pts) != 4:
        pts = [[300, 300], [1000, 300], [1200, 800], [100, 800]]
    pts[idx] = [int(x), int(y)]
    data['model']['roi']['full_region']['points'] = pts
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
    return pts


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    ctx = param
    idx = ctx['idx']
    data = ctx['data']
    cfg_path = ctx['cfg_path']
    pts = _save_point(cfg_path, data, idx, x, y)
    ctx['points'] = pts
    ctx['idx'] = (idx + 1) % 4
    print(f"Updated point {idx + 1} -> ({x}, {y}) saved to {cfg_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='videos/GT2.MP4')
    ap.add_argument('--config', default='config.json')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(1)

    points, data = _load_points(args.config)
    ctx = {
        'idx': 0,
        'points': points,
        'data': data,
        'cfg_path': args.config,
    }

    cv2.namedWindow("Viewer")
    cv2.setMouseCallback("Viewer", mouse_callback, ctx)

    is_image = os.path.isfile(args.source) and args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if is_image:
        ret, frame = cap.read()
        if ret:
            while True:
                disp = frame.copy()
                pts = ctx['points']
                if len(pts) == 4:
                    pts_np = np.array(pts, dtype=np.int32)
                    cv2.polylines(disp, [pts_np], True, (0, 255, 0), 2)
                    for idx, p in enumerate(pts):
                        cv2.circle(disp, tuple(map(int, p)), 5, (0, 0, 255), -1)
                        cv2.putText(disp, f"P{idx+1}", (int(p[0])+6, int(p[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(disp, f"Click to set P{ctx['idx']+1} (ESC to exit)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("Viewer", disp)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            pts = ctx['points']
            if len(pts) == 4:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(disp, [pts_np], True, (0, 255, 0), 2)
                for idx, p in enumerate(pts):
                    cv2.circle(disp, tuple(map(int, p)), 5, (0, 0, 255), -1)
                    cv2.putText(disp, f"P{idx+1}", (int(p[0])+6, int(p[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(disp, f"Click to set P{ctx['idx']+1} (ESC to exit)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Viewer", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
