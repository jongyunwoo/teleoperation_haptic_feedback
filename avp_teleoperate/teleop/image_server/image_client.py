import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from hapticfeedback.visfeedback import HandObjectDepthAssessor, LineOverlayMerger
from hapticfeedback.soundfeedback import ObjectDepthSameSound
from hapticfeedback.code.leftrightsplit import RobotHandSideResolver
from collections import deque
from typing import Iterable, Optional, Dict, Any, Tuple, List
import supervision as sv

# MAX_DEPTH_MM = 4000.0
ROBOT_HAND_CID = 7                 # 단일 로봇손 클래스 (모델 출력)
LEFT_ID, RIGHT_ID = 7, 8           # 파이프라인 호환용 합성 클래스
HAND_CLASSES = (LEFT_ID, RIGHT_ID)
BUCKET_OBJECT = 100
max_len = 3
buffer_size = 5
min_classes = 1
annotator = sv.MaskAnnotator()

class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None,
                 wrist_img_shape = None, wrist_img_shm_name = None, wrist_depth_img_shape = None, wrist_depth_img_shm_name = None,
                image_show = False, server_address = "192.168.123.164", port = 5555, Unit_Test = False):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.

        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.

        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.

        wrist_img_shm_name: Shared memory is used to easily transfer images.
        
        image_show: Whether to display received images in real time.

        server_address: The ip address to execute the image server script.

        port: The port number to bind to. It should be the same as the image server.

        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self.tv_buffer = deque(maxlen=max_len)
        self.wrist_buffer = deque(maxlen=max_len)
        self.model = None #추가
        self._need_load_model = True #추가
        self.depth_assessor = HandObjectDepthAssessor(left_id=7, right_id=8, k=2, tol_mm=15, hysteresis_mm=10)
        self.line_merger = LineOverlayMerger(exclude_class_ids=(7,8), iou_thresh=0.25,
                                     dist_thresh=50.0, angle_thresh_deg=15.0,
                                     depth_thresh_mm=60.0, morph_dilate=2, morph_close=3,
                                     extend_ratio=1.1, color=(0,255,0), thickness=2)        
        self.side_resolver = RobotHandSideResolver(iou_track_thresh=0.5,mirror=False)  # 헤드/손목 카메라가 좌우 반전되면 True
        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape
        # self.tv_depth_img_shape = tv_depth_img_shape
        self.wrist_depth_img_shape = wrist_depth_img_shape
        self.align_sound_path = "/home/scilab/teleoperation/avp_teleoperate/hapticfeedback/sounddata/bell-notification-337658.mp3" 
        self._ods = None 
        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = self.tv_image_shm.buf)
            self.tv_enable_shm = True
            
        # self.tv_depth_enable_shm = False
        # if self.tv_depth_img_shape is not None and tv_depth_img_shm_name is not None:
        #     self.tv_depth_image_shm = shared_memory.SharedMemory(name=tv_depth_img_shm_name)    
        #     self.tv_depth_img_array = np.ndarray(tv_depth_img_shape, dtype = np.uint16, buffer = self.tv_depth_image_shm.buf)
        #     self.tv_depth_enable_shm = True
        
        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        self.wrist_depth_enable_shm = False
        if self.wrist_depth_img_shape is not None and wrist_depth_img_shm_name is not None:
            self.wrist_depth_image_shm = shared_memory.SharedMemory(name = wrist_depth_img_shm_name)
            self.wrist_depth_img_array = np.ndarray(wrist_depth_img_shape, dtype = np.uint16, buffer=self.wrist_depth_image_shm.buf)
            self.wrist_depth_enable_shm = True

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()
          
          
        # ★ fallback 저장소
        self._prev_head: Dict[str, Any] = {"masks_xy": None, "classes": None, "boxes_xyxy": None}
        self._prev_wrist: Dict[str, Any] = {"masks_xy": None, "classes": None, "boxes_xyxy": None}
    def _safe_extract(self, preds):
        masks_xy = None
        classes = None
        boxes_xyxy = None
        try:
            # Boxes
            if getattr(preds, "boxes", None) is not None and preds.boxes is not None:
                # .cpu() 필요할 수 있음 (CUDA 텐서 대비)
                boxes_xyxy = preds.boxes.xyxy
                if hasattr(boxes_xyxy, "cpu"):
                    boxes_xyxy = boxes_xyxy.cpu().numpy()
                else:
                    boxes_xyxy = np.asarray(boxes_xyxy)
                if getattr(preds.boxes, "cls", None) is not None:
                    classes = preds.boxes.cls
                    if hasattr(classes, "cpu"):
                        classes = classes.cpu().numpy().astype(int).tolist()
                    else:
                        classes = list(map(int, np.asarray(classes).tolist()))

            # Masks (polygon)
            if getattr(preds, "masks", None) is not None and preds.masks is not None:
                # preds.masks.xy: list of [N_i x 2] float arrays (CPU)
                xy = getattr(preds.masks, "xy", None)
                if xy is not None:
                    masks_xy = [np.asarray(p, dtype=np.float32) for p in xy]
        except Exception as e:
            print(f"[_safe_extract] error: {e}")

        return masks_xy, classes, boxes_xyxy


    @staticmethod
    def _good_enough(masks_xy, classes, min_classes_count: int):
        """마스크가 존재하고, 클래스도 존재하며, 최소 분류 수 조건 통과"""
        if masks_xy is None or classes is None:
            return False
        if len(masks_xy) == 0 or len(classes) == 0:
            return False
        return len(set(classes)) >= min_classes_count

    @staticmethod
    def _best_idx_by_iou(poly, polys, H, W):
        """poly와 polys 간 IoU 최댓값 인덱스 반환"""
        if poly is None or len(polys) == 0:
            return None
        base = np.zeros((H, W), np.uint8)
        cv2.fillPoly(base, [poly.astype(np.int32)], 1)
        best_i, best_iou = None, 0.0
        for j, pj in enumerate(polys):
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [pj.astype(np.int32)], 1)
            inter = np.logical_and(base, m).sum()
            union = np.logical_or(base, m).sum() + 1e-6
            iou = float(inter / union)
            if iou > best_iou:
                best_i, best_iou = j, iou
        return best_i

    def _assign_lr_classes(self, masks_xy, classes, img_w, img_h):
        """
        단일 로봇손 클래스(ROBOT_HAND_CID) 인스턴스들을 좌/우로 분할하여
        classes_lr (합성 클래스: 7/8)와 좌/우 글로벌 인덱스를 반환.
        """
        if masks_xy is None or classes is None:
            return classes, None, None

        hand_idxs = [i for i, c in enumerate(classes) if c == ROBOT_HAND_CID]
        if not hand_idxs:
            return classes, None, None

        hand_polys = [masks_xy[i] for i in hand_idxs]
        lr = self.side_resolver.update(hand_polys, image_w=img_w, now_ts=time.time())
        left_local = self._best_idx_by_iou(lr["left"], hand_polys, img_h, img_w) if lr["left"] is not None else None
        right_local = self._best_idx_by_iou(lr["right"], hand_polys, img_h, img_w) if lr["right"] is not None else None

        left_idx = hand_idxs[left_local] if left_local is not None else None
        right_idx = hand_idxs[right_local] if right_local is not None else None

        classes_lr = list(classes)
        if left_idx is not None:
            classes_lr[left_idx] = LEFT_ID
        if right_idx is not None:
            classes_lr[right_idx] = RIGHT_ID

        return classes_lr, left_idx, right_idx
    #===================segmentation model load===================#
    def _lazy_load_model(self):
        if not self._need_load_model:
            return
        print("[ImageClient] Loading YOLO11 segmentation model …")
        from ultralytics import YOLO
        self.model = YOLO("/home/scilab/teleoperation/best.pt")
        self._need_load_model = False
        print("[ImageClient] YOLO11n-seg ready (cuda)")

    #===================segmentation model load===================#
    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _close(self):
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    
    def receive_process(self):
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.RCVTIMEO = 1000  # 1초

        print("\nImage client has started, waiting to receive data...")
        try:
            while self.running:
                # Receive message
                message = self._socket.recv_multipart()
                receive_time = time.time()
            
                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        # Attempt to extract header and image data
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    # No header, entire message is image data
                    #jpg_bytes, depth_bytes, wrist_bytes = message
                    jpg_bytes, wrist_jpg_bytes, wrist_depth_bytes = message
                #  Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                wrist_np_img = np.frombuffer(wrist_jpg_bytes, dtype=np.uint8)
                #raw_depth = np.frombuffer(depth_bytes, dtype=np.uint16) if depth_bytes else None
                wrist_raw_depth = np.frombuffer(wrist_depth_bytes, dtype=np.uint16) if wrist_depth_bytes else None
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                wrist_image = cv2.imdecode(wrist_np_img, cv2.IMREAD_COLOR)
                
                if wrist_raw_depth.ndim == 1:
                    wrist_raw_depth = wrist_raw_depth.reshape(wrist_image.shape[0], wrist_image.shape[1])
                
                if current_image is None:
                    print("[Image Client] Failed to decode Image.")
                    continue
                
                # if raw_depth is None:
                #     print("[Image Client] Failed to decode Depth Image")
                #     continue
                
                if wrist_image is None:
                    print("[Image Client] Failed to decode Image.")
                    continue
                
                if wrist_raw_depth is None:
                   print("[Image Client] Failed to decode Wrist Depth Image")
                   continue
                
                self._lazy_load_model()
                head_disp  = current_image.copy()
                wrist_disp = wrist_image.copy()

                imgs = [current_image, wrist_image]
                results = self.model.predict(imgs, imgsz=480, device=0, half=True, verbose=False)[0:2]
                head_preds, wrist_preds = results
                
                 # 안전 추출
                head_masks_xy_now, head_classes_now, head_boxes_xyxy_now   = self._safe_extract(head_preds)
                wrist_masks_xy_now, wrist_classes_now, wrist_boxes_xyxy_now = self._safe_extract(wrist_preds)

                # fallback 결정
                if self._good_enough(head_masks_xy_now, head_classes_now, min_classes):
                    head_masks_xy, head_classes, head_boxes_xyxy = head_masks_xy_now, head_classes_now, head_boxes_xyxy_now
                    self._prev_head = {"masks_xy": head_masks_xy, "classes": head_classes, "boxes_xyxy": head_boxes_xyxy}
                else:
                    head_masks_xy, head_classes, head_boxes_xyxy = self._prev_head["masks_xy"], self._prev_head["classes"], self._prev_head["boxes_xyxy"]

                if self._good_enough(wrist_masks_xy_now, wrist_classes_now, min_classes):
                    wrist_masks_xy, wrist_classes, wrist_boxes_xyxy = wrist_masks_xy_now, wrist_classes_now, wrist_boxes_xyxy_now
                    self._prev_wrist = {"masks_xy": wrist_masks_xy, "classes": wrist_classes, "boxes_xyxy": wrist_boxes_xyxy}
                else:
                    wrist_masks_xy, wrist_classes, wrist_boxes_xyxy = self._prev_wrist["masks_xy"], self._prev_wrist["classes"], self._prev_wrist["boxes_xyxy"]
                
                # ===== 좌/우 합성 클래스 만들기 (wrist) =====
                Wh, Ww = wrist_image.shape[0], wrist_image.shape[1]
                wrist_classes_lr, wrist_left_idx, wrist_right_idx = self._assign_lr_classes(
                    wrist_masks_xy, wrist_classes, img_w=Ww, img_h=Wh
                )

                # ===== 좌/우 합성 클래스 만들기 (head) =====
                Hh, Hw = head_image_h, head_image_w = head_disp.shape[0], head_disp.shape[1]
                head_classes_lr, head_left_idx, head_right_idx = self._assign_lr_classes(
                    head_masks_xy, head_classes, img_w=Hw, img_h=Hh
                )

                # ① 손-물체 깊이 비교 (wrist)
                depth_out = {"left": {"verdict": "no_data"}, "right": {"verdict": "no_data"}}
                if (wrist_raw_depth is not None and wrist_raw_depth.ndim == 2 and
                        wrist_boxes_xyxy is not None and len(wrist_boxes_xyxy) > 0 and
                        wrist_classes_lr is not None):
                    wrist_disp, depth_out = self.depth_assessor.process(
                        image_bgr=wrist_disp,
                        depth_mm=wrist_raw_depth,
                        boxes_xyxy=wrist_boxes_xyxy,
                        classes=np.array(wrist_classes_lr, dtype=int)
                    )

                # ② Head 손 마스크 그라데이션 오버레이 (합성 클래스 사용)
                if head_masks_xy is not None and head_classes_lr is not None:
                    head_disp = self.depth_assessor.colorize_hands_on_head_grad(
                        head_image_bgr=head_disp,
                        head_masks_xy=head_masks_xy,
                        head_classes=list(map(int, head_classes_lr)),
                        left_info=depth_out.get("left", {}),
                        right_info=depth_out.get("right", {}),
                        min_mm=10.0, max_mm=100.0, alpha=0.30
                    )

                # ③ Head 선 하나(객체당) 오버레이
                if head_masks_xy is not None and head_classes_lr is not None:
                    try:
                        # LineOverlayMerger는 exclude=(7,8)라 손에는 선이 그려지지 않음
                        class _MockMasks:
                            def __init__(self, xy): self.xy = xy
                        head_disp = self.line_merger.draw_single_line_per_object(
                            image=head_disp,
                            masks=_MockMasks(head_masks_xy),
                            classes=list(map(int, head_classes_lr)),
                            depth_mm=None
                        )
                    except Exception as e:
                        print(f"[image_show][head line-merge] error: {e}")

                # ④ 사운드 피드백(손 제외하고 물체끼리 버킷팅 → 동일깊이 음성)
                if wrist_masks_xy is not None and wrist_classes is not None and wrist_raw_depth is not None:
                    try:
                        w_masks = wrist_masks_xy
                        w_cls = list(map(int, wrist_classes))  # 원본 클래스 기준
                        w_cls_bucketed = [(c if c in HAND_CLASSES else BUCKET_OBJECT) for c in w_cls]
                        object_masks_only = [m for (m, c) in zip(w_masks, w_cls_bucketed) if c == BUCKET_OBJECT]

                        if self._ods is None:
                            self._ods = ObjectDepthSameSound(
                                depth=wrist_raw_depth,
                                masks=object_masks_only,
                                align_sound_path=self.align_sound_path,
                                k=4, tolerance_mm=20, cooldown_s=0.8, release_mm=40,
                                dwell_s=0.5, key_grid_px=120, stale_s=10.0,
                                suppress_s=100.0, outside_tol_clear_s=1.2
                            )
                        else:
                            self._ods.depth = wrist_raw_depth
                            self._ods.masks = object_masks_only

                        _ = self._ods.sound_depth_same_between_objects()
                    except Exception as e:
                        print(f"[image_show][ODS] error: {e}")
                label_annot = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
                detections = sv.Detections.from_ultralytics(wrist_preds)
                wrist_disp = annotator.annotate(wrist_disp, detections)
                wrist_disp = label_annot.annotate(
                    wrist_disp, detections,
                    labels=[f"{int(c)}:{conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
                )
                if self.tv_enable_shm:
                    # 크기 맞춰 복사
                    H, W = self.tv_img_shape[0], self.tv_img_shape[1]
                    if (head_disp.shape[0], head_disp.shape[1]) != (H, W):
                        head_to_copy = cv2.resize(head_disp, (W, H))
                    else:
                        head_to_copy = head_disp
                    np.copyto(self.tv_img_array, head_to_copy[:, :W])

                if self.wrist_enable_shm:
                    Hw, Ww = self.wrist_img_shape[0], self.wrist_img_shape[1]
                    wrist_to_copy = wrist_disp
                    if (wrist_disp.shape[0], wrist_disp.shape[1]) != (Hw, Ww):
                        wrist_to_copy = cv2.resize(wrist_disp, (Ww, Hw))
                    np.copyto(self.wrist_img_array, wrist_to_copy[:, :Ww])

                if self.wrist_depth_enable_shm:
                    wrist_raw_depth = wrist_raw_depth.reshape(self.wrist_depth_img_shape[0], self.wrist_depth_img_shape[1])
                    np.copyto(self.wrist_depth_img_array, wrist_raw_depth)

                # Local show
                if self._image_show:
                    print("imshow shape:", head_disp.shape, head_disp.dtype)

                    cv2.imshow('Head (overlay)', head_disp)
                    print(2)
                    # cv2.imshow('Wrist (seg)', wrist_disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                    print("success")
                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)


        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

if __name__ == "__main__":
    # example1
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()

    # example2
    # Initialize the client with performance evaluation enabled
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) # local test
    client = ImageClient(image_show = True, server_address='192.168.123.164', Unit_Test=False) # deployment test
    client.receive_process()