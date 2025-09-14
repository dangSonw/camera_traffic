import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class YOLOProcessor:
    def __init__(self, weights_path: str, device_type: str = 'cpu'):
        self.device = None
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        # For visualization of exact model input
        self.last_input_bgr: Optional[np.ndarray] = None
        self.last_input_shape: Optional[Tuple[int, int]] = None  # (h, w)

        try:
            # YOLOv8 via ultralytics only
            try:
                from ultralytics import YOLO as YOLOv8
            except ImportError:
                logger.error("ultralytics package not found. Please install ultralytics to use YOLOv8 backend.")
                raise
            self.device = torch.device('cuda' if (device_type != 'cpu' and torch.cuda.is_available()) else 'cpu')
            logger.info(f"Using device: {self.device}")
            self.model = YOLOv8(weights_path)
            self.names = self.model.names
            # Best effort stride retrieval
            try:
                self.stride = int(max(self.model.model.stride))
            except Exception:
                self.stride = 32
            self.pt = True
            logger.info(f"YOLOv8 model loaded successfully: {weights_path}")
            logger.info(f"Model stride: {self.stride}, Classes: {len(self.names)}")
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def preprocess_frame(self, frame: np.ndarray, imgsz: Optional[int] = None) -> torch.Tensor:
        try:
            h0, w0 = frame.shape[:2]
            use_letterbox = getattr(self, 'use_letterbox', False)
            if use_letterbox and imgsz:
                # Keep aspect ratio with padding to imgsz square and stride multiple
                # The original code referred to letterbox_fn but not implemented; here we simply resize square
                lb_img = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                self.last_input_bgr = lb_img.copy()
                self.last_input_shape = lb_img.shape[:2]
                img_bgr = lb_img
            else:
                if imgsz:
                    new_h = new_w = imgsz
                else:
                    new_h = int(np.ceil(h0 / self.stride) * self.stride)
                    new_w = int(np.ceil(w0 / self.stride) * self.stride)
                if (h0 != new_h) or (w0 != new_w):
                    # Choose interpolation based on scaling direction
                    interp = cv2.INTER_AREA if (new_w < w0 or new_h < h0) else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
                # Save exact model input (BGR) for optional visualization
                self.last_input_bgr = frame.copy()
                self.last_input_shape = (frame.shape[0], frame.shape[1])
                img_bgr = frame
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(self.device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)
            return im
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            raise

    def inference(self, im: torch.Tensor) -> List:
        try:
            # Use ultralytics YOLOv8, run on exact BGR input we prepared
            if self.last_input_bgr is None:
                raise RuntimeError("No preprocessed image available for YOLOv8 inference")
            results = self.model(self.last_input_bgr, verbose=False)
            if not results:
                return [torch.empty((0, 6))]
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return [torch.empty((0, 6))]
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            conf = r.boxes.conf.detach().cpu().numpy().reshape(-1, 1)
            cls = r.boxes.cls.detach().cpu().numpy().reshape(-1, 1)
            arr = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
            return [torch.from_numpy(arr)]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def cleanup(self):
        if hasattr(self, 'model') and self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.last_input_bgr = None
        self.last_input_shape = None
