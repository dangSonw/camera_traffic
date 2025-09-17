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
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic Monitoring with MobileNetSSD")
    # Model arguments
    parser.add_argument('--weights', '--model', dest='weights', type=str, required=True, help='Path to Caffe model weights (.caffemodel)')
    parser.add_argument('--prototxt', type=str, required=True, help="Path to Caffe 'deploy' prototxt file")
    
    # Input/Output arguments
    parser.add_argument('--source', '--video', '--input', dest='source', type=str, default='../traffic_1080_1920_30fps.mp4', 
                       help='Path to video/image, webcam index (e.g. 0), or RTSP/HTTP URL')
    parser.add_argument('--fps', type=float, default=10.0, 
                       help='Target processing FPS (lower values reduce CPU usage)')
    
    # Display options
    parser.add_argument('--show', action='store_true', 
                       help='Show detection results in a window')
    
    # Performance options
    parser.add_argument('--core', type=int, default=None, 
                       help='Number of CPU cores to use (default: all available)')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to JSON config file for runtime parameters')
    parser.add_argument('--sample-interval', type=float, default=None, 
                       help='Process one frame every N seconds (0 = process all frames)')
    
    return parser


def load_runtime_config(args) -> Dict[str, object]:
    cfg: Dict[str, object] = dict(DEFAULT_CFG)
    cfg_path: Path | None = None
    if getattr(args, 'config', None):
        cfg_path = Path(args.config)
    else:
        # Auto-load local config.json if present
        default_path = Path('config.json')
        if default_path.exists():
            cfg_path = default_path
    if cfg_path is not None:
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
