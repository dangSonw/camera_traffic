import ncnn
import numpy as np
import cv2

class NCNNDetector:
    def __init__(self, model_path, model_cfg):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False # True nếu có GPU hỗ trợ Vulkan
        self.net.opt.num_threads = 4
        
        # Load model
        self.net.load_param(model_path.get("param"))
        self.net.load_model(model_path.get("bin"))
        
        self.threshold = model_cfg.get("threshold", 0.2)
        self.nms_threshold = model_cfg.get("nms_threshold", 0.4)
        self.class_names = model_cfg.get("class", [])
        self.target_size = 640

    @staticmethod
    def letterbox(img, size=640):
        h, w = img.shape[:2]
        if h == size and w == size:
            return img
        r = min(size / h, size / w)
        new_w, new_h = int(w * r), int(h * r)
        
        if new_w != w or new_h != h:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img
        
        dw, dh = size - new_w, size - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return out

    def nms(self, boxes, scores, iou_thres):
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.clip(xx2 - xx1, 0, None)
            h = np.clip(yy2 - yy1, 0, None)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def detect(self, frame):
        # Preprocess
        img_in = self.letterbox(frame, self.target_size)
        
        mat_in = ncnn.Mat.from_pixels(
            img_in.tobytes(),
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.target_size, self.target_size
        )
        mat_in.substract_mean_normalize((0, 0, 0), (1/255.0, 1/255.0, 1/255.0))
        
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        out = ncnn.Mat()
        ex.extract("out0", out)
        
        out_np = np.array(out, dtype=np.float32)
        if out_np.shape[0] == 84: out_np = out_np.T
        
        boxes = out_np[:, 0:4]
        scores = out_np[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]
        
        mask = confidences >= self.threshold
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
        
        # Convert cxcywh to xyxy
        boxes[:, 0] -= boxes[:, 2] / 2  # x1
        boxes[:, 1] -= boxes[:, 3] / 2  # y1
        boxes[:, 2] += boxes[:, 0]      # x2
        boxes[:, 3] += boxes[:, 1]      # y2
        
        return boxes, confidences, class_ids