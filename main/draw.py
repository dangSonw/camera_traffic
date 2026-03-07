import cv2
import numpy as np

class Draw:
    def __init__(self):
        pass

    @staticmethod
    def draw_bbox(img, bbox, color=(0,255,0), thickness=2, label=None):
        """
        Vẽ bounding box lên ảnh
        bbox: [x1, y1, x2, y2]
        label: chuỗi text hiển thị trên box
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def draw_polygon(img, points, color=(0,0,255), thickness=2):
        """
        Vẽ polygon lên ảnh
        points: list các điểm [[x1, y1], [x2, y2], ...]
        """
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        return img

    @staticmethod
    def draw_text(img, text, pos, color=(255,255,255), font_scale=1, thickness=2):
        """
        Vẽ text lên ảnh
        pos: (x, y)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
        return img

    @staticmethod
    def draw_tracking_data(img, tracking_results, class_names, show_arrow=True):
        """
        Vẽ thông tin tracking lên frame
        tracking_results: List các dict chứa info (bbox, speed, direction, ...)
        """
        for res in tracking_results:
            x1, y1, x2, y2 = res["bbox"]
            tid = res["id"]
            cls_id = res["class_id"]
            conf = res["confidence"]
            speed = res["speed"]
            direction_text = res["direction"]
            
            # Chọn màu dựa trên tốc độ
            if direction_text == "CALC...": color = (200, 200, 200)
            elif speed < 30: color = (128, 128, 128)
            elif speed < 45: color = (0, 255, 0)
            elif speed < 90: color = (0, 255, 255)
            else: color = (0, 0, 255)

            # Vẽ BBox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ Label
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            label = f"ID{tid} {cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Vẽ Speed & Direction
            y_offset = y2 + 15
            if speed > 0:
                cv2.putText(img, f"Speed: {speed:.1f}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15
            if direction_text:
                cv2.putText(img, f"Dir: {direction_text}", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Vẽ Mũi tên
            if show_arrow and "centroid" in res and "vector" in res and speed > 2.0:
                cx, cy = res["centroid"]
                dx, dy = res["vector"]
                icx, icy = int(cx), int(cy)
                
                arrow_length = np.clip(speed * 0.8, 20, 60)
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    end_x = int(cx + (dx / dist) * arrow_length)
                    end_y = int(cy + (dy / dist) * arrow_length)
                    cv2.arrowedLine(img, (icx, icy), (end_x, end_y), color, 2, tipLength=0.3)
                    cv2.circle(img, (icx, icy), 3, color, -1)
        
        return img