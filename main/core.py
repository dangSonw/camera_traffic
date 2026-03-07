import sys
import os
import cv2
import csv
import numpy as np
from datetime import datetime
from collections import deque
from data import TranPkg
from concurrent.futures import ThreadPoolExecutor
import tracker
from detector import NCNNDetector

class Core:
    def __init__(self, path_cfg, model_cfg, logger, forecast_obj):
        self.path_cfg = path_cfg
        self.model_cfg = model_cfg
        self.logger = logger
        self.forecast = forecast_obj

        model_path = self.path_cfg["model"]
        
        # Khởi tạo Detector riêng biệt
        self.detector = NCNNDetector(model_path, self.model_cfg)
        
        # Tối ưu: Sử dụng ThreadPool để cập nhật các tracker song song
        # Số luồng có thể bằng số luồng của NCNN hoặc số lõi CPU
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Các tham số cho bộ lọc Kalman đơn giản (làm mượt vị trí và vận tốc)
        # GIẢM alpha/beta để tăng độ mượt (giảm rung) cho xe ở gần
        self.smoothing_alpha = self.model_cfg.get("smoothing_alpha", 0.3) 
        self.smoothing_beta = self.model_cfg.get("smoothing_beta", 0.4)
        self.smoothing_size = self.model_cfg.get("smoothing_size", 0.3) # Trọng số làm mượt kích thước
        self.smoothing_angle = self.model_cfg.get("smoothing_angle", 0.3) # Trọng số làm mượt góc
        self.class_names = self.model_cfg.get("class", [])
        self.tracker_type = self.model_cfg.get("tracker_type", "MOSSE")
        self.byte_tracker = None
        
        self.trackers = {}
        self.tracker_info = {}
        self.next_id = 0
        self.tracker_history = {} 

        self.redetect_interval = self.model_cfg.get("redetect_interval", 10)
        self.frame_count = 0
        self.numtracker = self.model_cfg.get("numtracker", 100)
        
        self.last_time_red = datetime.now()
        self.last_time_green = datetime.now()
        
        self.csv_path = "../output/traffic_data.csv"
        self.roi_lifecycle = {} # Lưu trạng thái chu kỳ đèn cho từng ROI

        self.rect_buffers = {} # Buffer trạng thái cho từng rect: {id: deque}

        self.debug_level = self.model_cfg.get("debug_level", 1)
        self.traffic_light_buffer = self.model_cfg.get("traffic_light_buffer", 5)
        self._init_tracker()
    
    
    def _init_tracker(self):
        try:
            if self.tracker_type == "BYTE-TRACK":
                return self._init_byte_track()
            if hasattr(cv2, 'legacy'):
                test_tracker = cv2.legacy.TrackerMOSSE_create()
            else:
                test_tracker = cv2.TrackerMOSSE_create()
            self.use_tracking = True
            self.logger.print(f"[Core] Tracker {self.tracker_type} available")
        except:
            self.use_tracking = False
            self.logger.print(f"[Core] Tracker not available. Install: pip install opencv-contrib-python")

    def _init_byte_track(self):
        try:
            # Thêm folder build vào path để import thư viện
            build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build"))
            if build_path not in sys.path:
                sys.path.append(build_path)
            
            # Import ByteTracker (giả định tên file/module là bytetrack)
            from bytetrack import ByteTracker

            class Args:
                def __init__(self):
                    self.track_thresh = 0.5
                    self.track_buffer = 30
                    self.match_thresh = 0.8
                    self.mot20 = False
            
            args = Args()
            # Cho phép override từ config.json nếu có
            args.track_thresh = self.model_cfg.get("track_thresh", 0.5)
            args.track_buffer = self.model_cfg.get("track_buffer", 30)
            args.match_thresh = self.model_cfg.get("match_thresh", 0.8)
            
            self.byte_tracker = ByteTracker(args)
            self.use_tracking = True
            self.logger.print(f"[Core] ByteTrack initialized")
        except Exception as e:
            self.use_tracking = False
            self.byte_tracker = None
            self.logger.print(f"[Core] Error initializing ByteTrack: {e}")
    
    
    def _create_tracker(self):
        if not self.use_tracking:
            return None
        
        t_type = self.tracker_type.upper()
        if t_type == 'CUSTOM':
            return tracker.CustomTracker()

        # Mapping tên tracker sang tên hàm tạo của OpenCV
        map_name = {
            'CSRT': 'TrackerCSRT_create', 'KCF': 'TrackerKCF_create',
            'MOSSE': 'TrackerMOSSE_create', 'MIL': 'TrackerMIL_create',
            'BOOSTING': 'TrackerBoosting_create', 'MEDIANFLOW': 'TrackerMedianFlow_create',
            'TLD': 'TrackerTLD_create'
        }
        
        if t_type not in map_name: return None
        
        try:
            factory = cv2.legacy if hasattr(cv2, 'legacy') else cv2
            creator = getattr(factory, map_name[t_type], None)
            return creator() if creator else None
        except:
            return None
    
    
    def _get_next_id(self):
        tid = self.next_id
        self.next_id += 1
        if self.next_id >= 1000:
            self.next_id = 0
            self.logger.print("[Core] Tracker ID reset to 0")
        return tid
    
    def _remove_tracker(self, tracker_id):
        """Hàm phụ trợ để xóa một tracker khỏi tất cả các dictionary."""
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]
        if tracker_id in self.tracker_info:
            del self.tracker_info[tracker_id]
        if tracker_id in self.tracker_history:
            del self.tracker_history[tracker_id]
        # THÊM: Xóa lịch sử tốc độ/hướng trong đối tượng forecast
        self.forecast.remove_history(tracker_id)
        if self.debug_level >= 1:
            self.logger.print(f"[Core] Removed tracker ID={tracker_id}")

    def iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        
        return inter / (union + 1e-9)
    
    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def check_line_crossing_direction(self, A, B, C, D):
        """
        Kiểm tra cắt vạch CÓ HƯỚNG.
        A, B: Điểm đầu/cuối của đoạn 1 (Line vạch kẻ)
        C, D: Điểm đầu/cuối của đoạn 2 (Vector di chuyển của xe)
        Trả về: 0 (không cắt), 1 (cắt xuôi), -1 (cắt ngược)
        """
        # Hàm tính tích có hướng (Cross Product)
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Kiểm tra cắt nhau cơ bản
        if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
            # Kiểm tra hướng dựa trên dấu của tích có hướng
            # Nếu CP(A, B, C) > 0 và CP(A, B, D) < 0 -> Cắt từ bên này sang bên kia
            cp1 = cross_product(A, B, C)
            cp2 = cross_product(A, B, D)
            
            if cp1 > 0 and cp2 < 0:
                return 1  # Chiều thuận
            elif cp1 < 0 and cp2 > 0:
                return -1 # Chiều nghịch
        
        return 0

    def match_detections(self, detections, iou_thres=0.3):
        if len(detections) == 0 or len(self.trackers) == 0:
            return [], list(range(len(detections))), list(self.trackers.keys())
        
        tracker_ids = list(self.trackers.keys())
        iou_matrix = np.zeros((len(detections), len(tracker_ids)))
        
        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(tracker_ids):
                iou_matrix[d_idx, t_idx] = self.iou(det["bbox"], self.tracker_info[tid]["bbox"])
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracker_ids)))
        
        while unmatched_dets and unmatched_trks:
            max_iou = 0
            max_d, max_t = -1, -1
            
            for d in unmatched_dets:
                for t in unmatched_trks:
                    if iou_matrix[d, t] > max_iou:
                        max_iou = iou_matrix[d, t]
                        max_d, max_t = d, t
            
            if max_iou >= iou_thres:
                matched.append((max_d, tracker_ids[max_t]))
                unmatched_dets.remove(max_d)
                unmatched_trks.remove(max_t)
            else:
                break
        
        unmatched_tracker_ids = [tracker_ids[i] for i in unmatched_trks]
        
        return matched, unmatched_dets, unmatched_tracker_ids
    

    def _update_time_tracking(self, detections):
        current_time = datetime.now()
        
        for det in detections:
            class_id = det["class_id"]
            class_name = self.class_names[class_id].lower() if class_id < len(self.class_names) else ""
            
            if "red" in class_name or class_id == 0:
                self.last_time_red = current_time
            
            if "green" in class_name or class_id == 1:
                self.last_time_green = current_time
    
    def _check_traffic_light_status(self, tracking_results, tracker_data_list, rect_cfg):
        """
        Kiểm tra trạng thái đèn cho từng Rect dựa trên config được truyền vào.
        Đếm xe đi qua vạch trong chu kỳ đèn và ghi log CSV.
        """
        if not rect_cfg:
            return {}

        active_rects = rect_cfg.get("active", [])
        still_thresh = self.model_cfg.get("still_threshold", 2.0)
        rect_info = {} 
        current_time = datetime.now()

        # Khởi tạo file CSV nếu chưa tồn tại
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "RectID", "TotalCount", "RedStart", "GreenStart", "Duration(s)"])
            except Exception as e:
                self.logger.print(f"[Core] Error creating CSV: {e}")

        for r_id in active_rects:
            if r_id not in self.rect_buffers:
                self.rect_buffers[r_id] = deque(maxlen=self.traffic_light_buffer)
            
            # Khởi tạo lifecycle cho ROI nếu chưa có
            if r_id not in self.roi_lifecycle:
                self.roi_lifecycle[r_id] = {
                    "status": "RED",        # Trạng thái hiện tại (ổn định)
                    "count": 0,             # Số xe đếm được trong chu kỳ
                    "counted_ids": set(),   # Các ID đã đếm trong chu kỳ này
                    "time_red": current_time,   # Thời điểm bắt đầu đèn đỏ
                    "time_green": None,         # Thời điểm bắt đầu đèn xanh
                    "recent_crossings": deque(maxlen=45) # Tăng lên ~1.5s để nối liền các khoảng ngắt quãng của dòng xe
                }
            poly_pts = rect_cfg.get(f"rect{r_id}", [])
            line_pts = rect_cfg.get(f"rect{r_id}_line", [])
            
            if not poly_pts or not line_pts: continue
            
            poly_np = np.array(poly_pts, np.int32)
            line_p1 = np.array(line_pts[0])
            line_p2 = np.array(line_pts[1])

            # --- BƯỚC 1: PHÂN TÍCH HÀNH VI ĐỐI TƯỢNG (Behavior Analysis) ---
            # Thay vì chỉ dùng tốc độ tức thời, ta dùng lịch sử vị trí để xem xe có thực sự di chuyển không
            
            vehicles_in_roi = 0
            effectively_stopped_count = 0 # Xe dừng hoặc chỉ nhích nhẹ (rung lắc)
            moving_count = 0

            # Lấy lịch sử vị trí từ forecast object
            history_data = self.forecast.history

            for res in tracking_results:
                tid = res["id"]
                cx, cy = res["centroid"]
                
                # Chỉ xét xe nằm trong vùng Polygon
                if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                    vehicles_in_roi += 1
                    
                    # Kiểm tra hành vi dừng:
                    # 1. Tốc độ tức thời thấp
                    is_instant_stop = res["speed"] < still_thresh
                    
                    # 2. Kiểm tra độ dịch chuyển thực tế trong quá khứ (Displacement)
                    # Để loại bỏ trường hợp xe rung lắc tại chỗ (tốc độ > 0 nhưng không đi đâu)
                    is_long_term_stop = False
                    if tid in history_data and len(history_data[tid]["positions"]) > 10:
                        # Lấy vị trí cách đây khoảng 10-15 frame (~0.5s)
                        past_pos = history_data[tid]["positions"][0] # Vị trí cũ nhất trong buffer
                        curr_pos = history_data[tid]["positions"][-1]
                        
                        # Tính khoảng cách Euclide giữa điểm đầu và điểm cuối buffer
                        displacement = np.linalg.norm(np.array(curr_pos) - np.array(past_pos))
                        
                        # Nếu trong khoảng thời gian qua xe di chuyển < 30 pixel -> Coi như đứng yên
                        if displacement < 30.0:
                            is_long_term_stop = True
                    
                    if is_instant_stop or is_long_term_stop:
                        effectively_stopped_count += 1
                    else:
                        moving_count += 1

            # --- BƯỚC 2: ĐẾM SỐ XE CẮT VẠCH (FLOW) ---
            current_frame_crossing_count = 0
            for trk in tracker_data_list:
                tid = trk["id"]
                prev_p = trk["previous_pos"]
                curr_p = trk["current_pos"]
                
                # Kiểm tra cắt vạch (bất kể chiều nào, miễn là cắt qua là có Flow)
                # Hàm check_line_crossing_direction trả về 1 hoặc -1 nếu cắt
                if self.check_line_crossing_direction(line_p1, line_p2, prev_p, curr_p) != 0:
                    current_frame_crossing_count += 1
                    if tid not in self.roi_lifecycle[r_id]["counted_ids"]:
                        self.roi_lifecycle[r_id]["count"] += 1
                        self.roi_lifecycle[r_id]["counted_ids"].add(tid)

            self.roi_lifecycle[r_id]["recent_crossings"].append(current_frame_crossing_count)
            total_recent_flow = sum(self.roi_lifecycle[r_id]["recent_crossings"])

            # --- BƯỚC 3: QUYẾT ĐỊNH TRẠNG THÁI (LOGIC MỚI - STATE MACHINE) ---
            # Sử dụng logic có độ trễ (Hysteresis) để tăng độ chính xác tuyệt đối
            
            current_stable_status = self.roi_lifecycle[r_id]["status"]
            frame_status = "WAIT"

            has_flow = total_recent_flow > 0
            
            # Tính tỷ lệ xe dừng trên tổng số xe
            total_vehs = effectively_stopped_count + moving_count
            stop_ratio = effectively_stopped_count / total_vehs if total_vehs > 0 else 0.0

            if current_stable_status == "RED":
                # Đang ĐỎ, muốn chuyển sang XANH thì cần điều kiện rất chặt để tránh nhiễu:
                # 1. Có dòng xe cắt qua vạch (Flow) VÀ tỷ lệ xe dừng thấp (tránh xe vượt đèn đỏ)
                # 2. HOẶC: Tất cả xe đều đang di chuyển (không ai dừng) VÀ có ít nhất 2 xe (tránh 1 xe vượt đèn đỏ)
                # Nếu có Flow nhưng phần lớn xe vẫn dừng (stop_ratio >= 0.5) -> Coi là vượt đèn đỏ -> Giữ ĐỎ
                if has_flow and stop_ratio < 0.5:
                    frame_status = "GREEN"
                elif moving_count >= 2 and effectively_stopped_count == 0:
                    frame_status = "GREEN"
                else:
                    # Nếu có bất kỳ xe nào dừng, hoặc chỉ 1 xe di chuyển, giữ ĐỎ
                    frame_status = "RED"
            
            elif current_stable_status == "GREEN":
                # Đang XANH, muốn chuyển sang ĐỎ:
                # 1. Không còn dòng xe cắt vạch (Flow = 0)
                # 2. VÀ: Tỷ lệ xe dừng cao (> 40%). Nếu xe chỉ đi chậm hoặc thưa thớt thì vẫn giữ XANH.
                if not has_flow and stop_ratio >= 0.4:
                    frame_status = "RED"
                else:
                    frame_status = "GREEN"
            
            else: # Trạng thái ban đầu (chưa xác định)
                if has_flow: frame_status = "GREEN"
                elif effectively_stopped_count > 0: frame_status = "RED"
                elif moving_count > 0: frame_status = "GREEN"
            
            # Nếu không có xe nào trong vùng (Empty), giữ nguyên trạng thái ổn định cũ
            if vehicles_in_roi == 0:
                frame_status = current_stable_status

            self.rect_buffers[r_id].append(frame_status)

            # --- LOGIC XÁC ĐỊNH TRẠNG THÁI ĐÈN (Majority vote) ---
            detected_status = "WAIT"
            if len(self.rect_buffers[r_id]) == self.traffic_light_buffer:
                rc = self.rect_buffers[r_id].count('RED')
                gc = self.rect_buffers[r_id].count('GREEN')
                threshold = (self.traffic_light_buffer // 2) + 1
                detected_status = "RED" if rc >= threshold else ("GREEN" if gc >= threshold else "WAIT")
            
            # --- LOGIC CHU KỲ ĐÈN VÀ ĐẾM XE ---
            lifecycle = self.roi_lifecycle[r_id]
            
            # THÊM: Kiểm tra thời gian giữ trạng thái tối thiểu (5s)
            # Nếu trạng thái hiện tại chưa giữ được 5s thì không cho phép chuyển đổi
            min_duration = 5.0
            current_state_start = lifecycle["time_red"] if lifecycle["status"] == "RED" else lifecycle["time_green"]
            can_switch = True
            if current_state_start is not None:
                if (current_time - current_state_start).total_seconds() < min_duration:
                    can_switch = False

            # Chỉ xử lý chuyển đổi trạng thái nếu không phải WAIT và đủ thời gian giữ
            if can_switch and detected_status != "WAIT" and detected_status != lifecycle["status"]:
                
                # Chuyển từ RED -> GREEN: Bắt đầu pha xanh
                if lifecycle["status"] == "RED" and detected_status == "GREEN":
                    lifecycle["status"] = "GREEN"
                    lifecycle["time_green"] = current_time
                
                # Chuyển từ GREEN -> RED: Kết thúc chu kỳ -> Ghi Log -> Reset
                elif lifecycle["status"] == "GREEN" and detected_status == "RED":
                    # Tính toán thời gian chu kỳ
                    start_red = lifecycle["time_red"]
                    start_green = lifecycle["time_green"]
                    duration = (current_time - start_red).total_seconds()
                    
                    # Ghi CSV
                    try:
                        with open(self.csv_path, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                r_id,
                                lifecycle["count"],
                                start_red.strftime("%H:%M:%S.%f"),
                                start_green.strftime("%H:%M:%S.%f") if start_green else "N/A",
                                round(duration, 2)
                            ])
                        self.logger.print(f"[Core] Cycle Logged Rect {r_id}: Count={lifecycle['count']}, Duration={duration:.1f}s")
                    except Exception as e:
                        self.logger.print(f"[Core] Error writing CSV: {e}")

                    # Reset cho chu kỳ mới
                    lifecycle["status"] = "RED"
                    lifecycle["count"] = 0
                    lifecycle["counted_ids"] = set()
                    lifecycle["time_red"] = current_time
                    lifecycle["time_green"] = None

            if lifecycle["status"] == "RED":
                self.last_time_red = datetime.now()
            elif lifecycle["status"] == "GREEN":
                self.last_time_green = datetime.now()
            
            # Trả về count tích lũy thay vì count tức thời
            rect_info[r_id] = {"count": lifecycle["count"], "status": lifecycle["status"]}

        return rect_info

    def _process_detections(self, frame_640):
        """Helper để gọi detector và format kết quả"""
        boxes, confs, cls_ids = self.detector.detect(frame_640)
        results = []
        
        # Gom nhóm theo class để chạy NMS
        for c in np.unique(cls_ids):
            if c >= len(self.class_names): continue
            
            inds = np.where(cls_ids == c)[0]
            c_boxes = boxes[inds]
            c_confs = confs[inds]
            
            keep = self.detector.nms(c_boxes, c_confs, self.detector.nms_threshold)
            
            for k in keep:
                i = inds[k]
                x1, y1, x2, y2 = boxes[i]
                results.append({
                    "bbox": [int(np.clip(x1, 0, 639)), int(np.clip(y1, 0, 639)), 
                             int(np.clip(x2, 0, 639)), int(np.clip(y2, 0, 639))],
                    "class_id": int(c),
                    "confidence": float(confs[i])
                })
        return results
    
    def process_frame(self, frame, rect_cfg=None):
        if frame is None:
            return (
                frame, 
                [], 
                TranPkg(
                    quantity=0,
                    timeRed=self.last_time_red.isoformat(),
                    timeGreen=self.last_time_green.isoformat()
                ),
                {},
            )

        frame_640 = self.detector.letterbox(frame, 640)
        current_time = datetime.now()  # THÊM: Timestamp hiện tại

        if not self.use_tracking:
            detections = self._process_detections(frame_640)
            self._update_time_tracking(detections)
            if self.debug_level >= 1:
                self.logger.print(f"[Core] Detected {len(detections)} objects")
            
            # Format lại detections để giống output của tracking
            tracking_results = []
            for det in detections:
                tracking_results.append({
                    "id": -1, # No ID
                    "bbox": det["bbox"],
                    "class_id": det["class_id"],
                    "confidence": det["confidence"],
                    "speed": 0,
                    "direction": "",
                    "vector": (0,0)
                })

            return (
                frame_640, 
                tracking_results, 
                TranPkg(
                    quantity=len(detections),
                    timeRed=self.last_time_red.isoformat(),
                    timeGreen=self.last_time_green.isoformat()
                    ),
                {}
            )

        # --- Xử lý riêng cho BYTE-TRACK (MOT) ---
        if self.tracker_type == "BYTE-TRACK" and self.byte_tracker:
            detections = self._process_detections(frame_640)
            self._update_time_tracking(detections)
            
            # Chuyển đổi detections sang format [x1, y1, x2, y2, score] cho ByteTrack
            dets_list = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                score = det["confidence"]
                dets_list.append([x1, y1, x2, y2, score])
            
            dets_np = np.array(dets_list, dtype=float) if dets_list else np.empty((0, 5))
            
            # Cập nhật tracker
            online_targets = self.byte_tracker.update(dets_np, [640, 640], (640, 640))
            
            current_tids = set()
            for t in online_targets:
                tid = t.track_id
                tlwh = t.tlwh
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                
                # Cố gắng khớp class_id từ detections gốc (dựa trên IoU lớn nhất)
                cls_id = 0
                best_iou = 0
                t_box = [x1, y1, x2, y2]
                for d in detections:
                    iou_val = self.iou(t_box, d["bbox"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        cls_id = d["class_id"]

                self.tracker_info[tid] = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": cls_id,
                    "confidence": t.score
                }
                self.trackers[tid] = True # Dummy value để duy trì count
                current_tids.add(tid)
            
            # Xóa các tracker đã mất dấu
            for tid in list(self.tracker_info.keys()):
                if tid not in current_tids:
                    self._remove_tracker(tid)
        
        # --- Logic cũ cho SOT (MOSSE, KCF, CUSTOM...) ---
        else:
            self.frame_count += 1
            should_detect = (self.frame_count % self.redetect_interval == 1) or (len(self.trackers) == 0)

            if should_detect:
                detections = self._process_detections(frame_640)
                self._update_time_tracking(detections)
                matched, unmatched_dets, unmatched_trks = self.match_detections(detections)

                for det_idx, tracker_id in matched:
                    det = detections[det_idx]
                    x1, y1, x2, y2 = det["bbox"]

                    new_tracker = self._create_tracker()
                    if new_tracker:
                        new_tracker.init(frame_640, (x1, y1, x2-x1, y2-y1))
                        self.trackers[tracker_id] = new_tracker
                        self.tracker_info[tracker_id] = {
                            "bbox": det["bbox"],
                            "class_id": det["class_id"],
                        "confidence": det["confidence"],
                        "lost_frames": 0
                        }

                for det_idx in unmatched_dets:
                    if len(self.trackers) >= self.numtracker:
                        continue

                    det = detections[det_idx]
                    x1, y1, x2, y2 = det["bbox"]

                    new_tracker = self._create_tracker()
                    if new_tracker:
                        new_tracker.init(frame_640, (x1, y1, x2-x1, y2-y1))
                        tid = self._get_next_id()
                        self.trackers[tid] = new_tracker
                        self.tracker_info[tid] = {
                            "bbox": det["bbox"],
                            "class_id": det["class_id"],
                        "confidence": det["confidence"],
                        "lost_frames": 0
                        }
                        if self.debug_level >= 1:
                            self.logger.print(f"[Core] New tracker ID={tid}")

                for tracker_id in unmatched_trks:
                    self._remove_tracker(tracker_id)

            else:
                # --- Tối ưu: Cập nhật tracker đa luồng ---
                tids_to_update = list(self.trackers.keys())
                
                # Chuẩn bị dữ liệu dự đoán cho từng tracker để gửi vào thread
                predictions = {}
                for tid in tids_to_update:
                    if tid in self.tracker_history:
                        # Lấy vị trí dự đoán từ history (đã tính ở frame trước hoặc ước lượng)
                        predictions[tid] = self.tracker_history[tid].get("predicted_bbox", None)

                # Hàm phụ để chạy trong mỗi luồng
                def update_job(tid):
                    tracker_obj = self.trackers.get(tid)
                    pred_bbox = predictions.get(tid, None)
                    
                    if tracker_obj:
                        # Nếu là CustomTracker, truyền thêm predicted_bbox
                        if hasattr(tracker_obj, 'update') and tracker_obj.__class__.__name__ == 'CustomTracker':
                             success, bbox = tracker_obj.update(frame_640, predicted_bbox=pred_bbox)
                        else:
                             success, bbox = tracker_obj.update(frame_640)
                        return tid, success, bbox
                    return tid, False, None

                # Gửi các tác vụ update vào thread pool và nhận kết quả
                results = self.executor.map(update_job, tids_to_update)

                failed_tids = []
                for tid, success, bbox in results:
                    if tid not in self.tracker_info: continue # Bỏ qua nếu tracker đã bị xóa trong lúc xử lý

                    if success:
                        x, y, w, h = bbox
                        self.tracker_info[tid]["bbox"] = [int(x), int(y), int(x+w), int(y+h)]
                        self.tracker_info[tid]["lost_frames"] = 0
                    else:
                        # Tăng bộ đếm bị mất dấu. Nếu vượt ngưỡng thì mới xóa.
                        self.tracker_info[tid]["lost_frames"] += 1
                        max_lost_frames = 5  # Giảm xuống mức thấp để loại bỏ nhanh nếu mất dấu
                        
                        if self.tracker_info[tid]["lost_frames"] > max_lost_frames:
                            failed_tids.append(tid)

                # Xóa các tracker đã thất bại
                for tid in failed_tids:
                    self._remove_tracker(tid)

        # ========== CHUẨN BỊ DỮ LIỆU CHO HÀM TÍNH TỐC ĐỘ ==========
        tracker_data_list = []

        for tid, info in self.tracker_info.items():
            x1, y1, x2, y2 = info["bbox"]
            class_id = info["class_id"]
            conf = info["confidence"]

            if class_id >= len(self.class_names):
                continue

            # Lấy dữ liệu thô từ detection/tracker (Raw measurement)
            raw_x1, raw_y1, raw_x2, raw_y2 = float(x1), float(y1), float(x2), float(y2)

            # --- Áp dụng bộ lọc Kalman đơn giản (g-h filter) cho 4 cạnh ---
            if tid in self.tracker_history:
                # Lấy trạng thái từ frame trước
                prev_state = self.tracker_history[tid]
                prev_bbox = prev_state["bbox"] # (x1, y1, x2, y2)
                prev_vel = prev_state["velocity"] # (vx1, vy1, vx2, vy2)
                prev_angle = prev_state.get("angle", 0.0) # Góc quay trước đó
                prev_time = prev_state["timestamp"]
                time_delta = (current_time - prev_time).total_seconds()

                # Tính tâm cũ để phục vụ tính toán forecast
                prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                prev_pos = (prev_cx, prev_cy)

                if time_delta > 0:
                    # 1. Prediction step: Dự đoán vị trí 4 cạnh tiếp theo dựa trên vận tốc cũ
                    pred_x1 = prev_bbox[0] + prev_vel[0] * time_delta
                    pred_y1 = prev_bbox[1] + prev_vel[1] * time_delta
                    pred_x2 = prev_bbox[2] + prev_vel[2] * time_delta
                    pred_y2 = prev_bbox[3] + prev_vel[3] * time_delta

                    # 2. Correction step:
                    if info["lost_frames"] > 0:
                        # Chỉ dự đoán, không cập nhật từ đo đạc
                        current_bbox = (pred_x1, pred_y1, pred_x2, pred_y2)
                        current_vel = prev_vel # Giữ nguyên vận tốc cũ
                        current_angle = prev_angle
                    else:
                        # Kết hợp vị trí dự đoán và vị trí đo đạc mới cho từng cạnh
                        curr_x1 = (1 - self.smoothing_alpha) * pred_x1 + self.smoothing_alpha * raw_x1
                        curr_y1 = (1 - self.smoothing_alpha) * pred_y1 + self.smoothing_alpha * raw_y1
                        curr_x2 = (1 - self.smoothing_alpha) * pred_x2 + self.smoothing_alpha * raw_x2
                        curr_y2 = (1 - self.smoothing_alpha) * pred_y2 + self.smoothing_alpha * raw_y2
                        
                        current_bbox = (curr_x1, curr_y1, curr_x2, curr_y2)

                        # Tính vận tốc tức thời cho từng cạnh
                        inst_vx1 = (curr_x1 - prev_bbox[0]) / time_delta
                        inst_vy1 = (curr_y1 - prev_bbox[1]) / time_delta
                        inst_vx2 = (curr_x2 - prev_bbox[2]) / time_delta
                        inst_vy2 = (curr_y2 - prev_bbox[3]) / time_delta

                        # Làm mượt vận tốc (EMA) cho từng cạnh
                        curr_vx1 = (1 - self.smoothing_beta) * prev_vel[0] + self.smoothing_beta * inst_vx1
                        curr_vy1 = (1 - self.smoothing_beta) * prev_vel[1] + self.smoothing_beta * inst_vy1
                        curr_vx2 = (1 - self.smoothing_beta) * prev_vel[2] + self.smoothing_beta * inst_vx2
                        curr_vy2 = (1 - self.smoothing_beta) * prev_vel[3] + self.smoothing_beta * inst_vy2

                        current_vel = (curr_vx1, curr_vy1, curr_vx2, curr_vy2)

                        # --- Kalman cho Hướng (Angle Smoothing) ---
                        # Tính vận tốc tâm để xác định hướng
                        vel_cx = (curr_vx1 + curr_vx2) / 2
                        vel_cy = (curr_vy1 + curr_vy2) / 2
                        
                        inst_speed = np.sqrt(vel_cx**2 + vel_cy**2)
                        if inst_speed > 2.0: # Chỉ cập nhật góc khi có chuyển động đáng kể
                            # Tính góc tức thời (Image coordinates: y is down)
                            inst_angle = np.degrees(np.arctan2(vel_cy, vel_cx))
                            
                            # Tính độ lệch góc ngắn nhất (-180 đến 180)
                            diff_angle = inst_angle - prev_angle
                            while diff_angle <= -180: diff_angle += 360
                            while diff_angle > 180: diff_angle -= 360
                            
                            current_angle = prev_angle + self.smoothing_angle * diff_angle
                        else:
                            current_angle = prev_angle

                    # Tính lại tâm từ 4 cạnh đã làm mượt
                    current_cx = (current_bbox[0] + current_bbox[2]) / 2
                    current_cy = (current_bbox[1] + current_bbox[3]) / 2

                    # Cập nhật lại bounding box hiển thị
                    info["bbox"] = [int(current_bbox[0]), int(current_bbox[1]), int(current_bbox[2]), int(current_bbox[3])]

                    # Thêm dữ liệu vào danh sách để gửi cho forecast
                    tracker_data_list.append({
                        "id": tid,
                        "current_pos": (current_cx, current_cy), # Gửi đi tọa độ float để tính toán chính xác hơn
                        "previous_pos": prev_pos, # Vị trí đã làm mượt của frame trước
                        "time_delta": time_delta,
                        "bbox": [x1, y1, x2, y2],
                        "class_name": self.class_names[class_id],
                        "confidence": conf,
                        "angle": current_angle # Truyền góc đã làm mượt sang forecast
                    })
                else: # time_delta is 0, không thể tính vận tốc, giữ nguyên trạng thái
                    current_bbox = prev_bbox
                    current_vel = prev_state["velocity"]
                    current_angle = prev_state.get("angle", 0.0)
            else: # Tracker mới, chưa có trong history, khởi tạo trạng thái
                current_bbox = (raw_x1, raw_y1, raw_x2, raw_y2)
                current_vel = (0.0, 0.0, 0.0, 0.0) # vx1, vy1, vx2, vy2
                current_angle = 0.0

            # Tính toán vị trí dự đoán cho frame tiếp theo (để dùng cho tracker.update)
            # Giả định frame tới cũng có time_delta tương tự (hoặc khoảng 0.03s cho 30fps)
            pred_next_x1 = current_bbox[0] + current_vel[0] * 0.03
            pred_next_y1 = current_bbox[1] + current_vel[1] * 0.03
            pred_next_x2 = current_bbox[2] + current_vel[2] * 0.03
            pred_next_y2 = current_bbox[3] + current_vel[3] * 0.03
            
            pred_w = pred_next_x2 - pred_next_x1
            pred_h = pred_next_y2 - pred_next_y1

            # Cập nhật history với trạng thái mới (vị trí và vận tốc đã được làm mượt)
            self.tracker_history[tid] = {
                "bbox": current_bbox, # Lưu tọa độ 4 cạnh float
                "velocity": current_vel,
                "angle": current_angle,
                "timestamp": current_time,
                "predicted_bbox": (int(pred_next_x1), int(pred_next_y1), int(pred_w), int(pred_h))
            }

        # ========== GỌI HÀM TÍNH TOÁN TỐC ĐỘ (KHÔNG VẼ) ==========
        final_tracking_results = []
        if len(tracker_data_list) > 0:
            speeds, directions, positions = self.forecast.calculate_speed(tracker_data_list)

            # Tổng hợp kết quả cuối cùng
            for tid, info in self.tracker_info.items():
                if tid in speeds:
                    final_tracking_results.append({
                        "id": tid,
                        "bbox": info["bbox"],
                        "class_id": info["class_id"],
                        "confidence": info["confidence"],
                        "speed": speeds[tid],
                        "direction": directions[tid]["text"],
                        "vector": directions[tid]["vector"],
                        "centroid": positions[tid]
                    })

        # --- THÊM: Logic kiểm tra đèn giao thông ---
        rect_info = self._check_traffic_light_status(final_tracking_results, tracker_data_list, rect_cfg)

        if self.debug_level >= 1:
            self.logger.print(f"[Core] Frame {self.frame_count}: {len(self.trackers)}/{self.numtracker} tracked objects")

        return (
            frame_640, 
            final_tracking_results, 
            TranPkg(
                quantity=len(self.trackers),
                timeRed=self.last_time_red.isoformat(),
                timeGreen=self.last_time_green.isoformat()
            ),
            rect_info
        )

    
    def reset_tracking(self):
        self.trackers.clear()
        self.tracker_info.clear()
        self.tracker_history.clear()
        self.next_id = 0
        self.rect_buffers.clear()
        self.roi_lifecycle.clear() # Reset trạng thái chu kỳ
        # THÊM: Reset lịch sử trong đối tượng forecast
        self.forecast.reset_history()
        self.frame_count = 0
        self.last_time_red = datetime.now()
        self.last_time_green = datetime.now()
        if self.tracker_type == "BYTE-TRACK":
            self._init_byte_track()
        self.logger.print("[Core] Tracking reset")
