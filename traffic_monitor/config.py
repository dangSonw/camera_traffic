from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


DEFAULT_CFG: Dict[str, object] = {
    "purple_ratio": 0.5,            # vạch tím (ROI) as fraction of height
    "blue_ratio": 0.875,            # vạch xanh (count) as fraction of height
    "conf_thresh": 0.30,            # detection confidence threshold
    "allowed_classes": [
        "car", "truck", "bus", "motorbike", "motorcycle", "bicycle",
    ],
    "line_margin_factor": 0.01,     # hysteresis band around blue line (fraction of H)
    "min_delta_factor": 0.0033,     # minimum movement per frame (fraction of H)
    "box_thickness": 1,
    "line_thickness": 1,
    "text_scale": 0.4,
    "text_thickness": 1,
    "trail_max_len": 50,
    "density_window": 10,
    # sampling: process only one frame every N seconds (video-time). 0 disables sampling.
    "sample_interval_sec": 1.0,
    # tracker defaults
    "tracker_iou_threshold": 0.2,
    "tracker_max_distance": 60.0,
    "tracker_max_missed": 10,
    "tracker_smooth_alpha": 0.25,
    "tracker_use_prediction": True,
    "tracker_velocity_alpha": 0.5,
    "tracker_max_speed": 120.0,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhanced YOLO video processing")
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Path to model weights (.pt)')
    parser.add_argument('--source', type=str, default='../traffic_1080_1920_30fps.mp4', help='Path to video or image')
    parser.add_argument('--fps', type=float, default=10.0, help='Target FPS')
    parser.add_argument('--core', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (pixels, square)')
    parser.add_argument('--show', action='store_true', help='Show a GUI window with detection results')
    parser.add_argument('--letterbox', action='store_true', help='Use YOLO letterbox (keep aspect ratio with padding) for preprocessing')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file for runtime parameters')
    # If provided, overrides config key 'sample_interval_sec'. Example: --sample-interval 1.0 -> process 1 frame per second
    parser.add_argument('--sample-interval', type=float, default=None, help='Seconds between processed frames (video-time). 0 or None disables sampling')
    parser.add_argument("--prototxt", required=False,help="path to Caffe 'deploy' prototxt file")
    return parser


def load_runtime_config(args) -> Dict[str, object]:
    cfg: Dict[str, object] = dict(DEFAULT_CFG)
    if getattr(args, 'config', None):
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            logger.error(f"Config file not found: {cfg_path}")
        else:
            try:
                with cfg_path.open('r', encoding='utf-8') as f:
                    user_cfg = json.load(f)
                if isinstance(user_cfg, dict):
                    cfg.update(user_cfg)
                    logger.info(f"Loaded config from {cfg_path}")
                else:
                    logger.warning("Config file is not a JSON object; using defaults")
            except Exception as e:
                logger.error(f"Failed to read config file {cfg_path}: {e}")
    return cfg
