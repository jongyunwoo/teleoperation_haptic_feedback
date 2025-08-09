import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from hapticfeedback.visfeedback import draw_alignment, DistanceOverlay
from hapticfeedback.soundfeedback import ObjectDepthSameSound
from collections import deque
import supervision as sv

# MAX_DEPTH_MM = 4000.0
buffer_size = 5
min_classes = 1
annotator = sv.MaskAnnotator()


class ImageClient:
    def __init__(self, tv_img_shape = None, tv_img_shm_name = None,
                 wrist_img_shape = None, wrist_img_shm_name = None, wrist_depth_img_shape = None, wrist_depth_img_shm_name = None, dual_hand_touch_array = None,
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
        self.tv_buffer = deque()
        self.wrist_buffer = deque()
        self.model = None #추가
        self._need_load_model = True #추가
        self.dist_util = None
        
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
        cv2.namedWindow('Image Client Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Client Stream', 960, 720)

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
                imgs = [current_image, wrist_image]  # 두 장 크기를 같게 맞추면 더 좋아요
                results = self.model.predict(imgs, imgsz=640, device=0, half=True, verbose=False)[0:2]
                head_preds, wrist_preds = results
                self.tv_buffer.append(head_preds.masks) 
                head_class_id = head_preds.boxes.cls.tolist()
                self.wrist_buffer.append(wrist_preds.masks)
                wrist_class_id = wrist_preds.boxes.cls.tolist()  

                # if  self.tv_enable_shm:
                #     print(5)
                #     # self._lazy_load_model()
                #     # preds = self.model.predict(current_image,
                #     #                            confidence=0.5)   # Results object
                #     # if preds[0].masks is None:
                #     #     print("[Image Client] No masks found in predictions.")
                #     #     np.copyto(self.tv_img_array, current_image[:, :self.tv_img_shape[1]])
                #     #     continue
                #     # tactile_sensor = self.dual_hand_touch_array
                #     # left_tactile_sensor = tactile_sensor[:1062]
                #     # right_tactile_sensor = tactile_sensor[-1062:]
                #     # current_image = overlay(current_image, left_tactile_sensor, right_tactile_sensor)
                #     np.copyto(self.tv_img_array, current_image[:, :self.tv_img_shape[1]])                    
                    
                #     if head_preds.masks is None or len(set(head_class_id)) < min_classes:
                #         print("[Image Client] No masks found in predictions.")
                #         if self.tv_buffer:
                #             fb_mask = self.tv_buffer[-1] 
                #             current_image = draw_alignment(current_image, fb_mask, head_class_id)    
                #             np.copyto(self.tv_img_array, current_image[:, :self.tv_img_shape[1]])
                #         else:
                #             np.copyto(self.tv_img_array, current_image[:, :self.tv_img_shape[1]])
                    
                #     else:
                #         self.tv_buffer.append(head_preds.masks)
                #         current_image = draw_alignment(current_image, head_preds.masks, head_class_id)
                #         depth = wrist_raw_depth  # uint16 mm

                #         robot_masks_w = []
                #         object_masks_w = []
                #         for m, cid in zip(wrist_preds.masks.xy, wrist_class_id):
                #             if cid in [6,7]:         # 왼/오른손 클래스
                #                 robot_masks_w.append(m)
                #             else:
                #                 object_masks_w.append(m)
                                
                #         head_robot_masks = {6: [], 7: []}
                #         for m, cid in zip(head_preds.masks.xy, head_class_id):
                #             if cid in (6, 7):
                #                 head_robot_masks[cid].append(m)

                #         head_int = {"fx":615,"fy":615,
                #                     "cx":self.tv_img_shape[1]/2,
                #                     "cy":self.tv_img_shape[0]/2}
                #         wrist_int= {"fx":615,"fy":615,
                #                     "cx":self.wrist_img_shape[1]/2,
                #                     "cy":self.wrist_img_shape[0]/2}
                #         T_hw = np.eye(4)
                #         dist_util = DistanceOverlay(head_int, wrist_int, T_hw,
                #                                     min_dist=0.02, max_dist=0.20)

                #         dist_by_cid = {}  
                #         for cid, rmask_w in robot_masks_w:
                #             c_r = dist_util.compute_mask_centroid(rmask_w)
                #             if c_r is None:
                #                 continue
                            
                #             dists = []
                #             for omask_w in object_masks_w:
                #                 c_o = dist_util.compute_mask_centroid(omask_w)
                #                 if c_o is None:
                #                     continue
                #                 d = dist_util.compute_object_distance_simple(c_r, c_o, depth)  # wrist depth 사용
                #                 if d is not None:
                #                     dists.append(d)

                #             if dists:
                #                 dmin = min(dists)
                #                 dist_by_cid[cid] = min(dmin, dist_by_cid.get(cid, float("inf")))

                #             for cid, dist_m in dist_by_cid.items():
                #                 for hmask in head_robot_masks.get(cid, []):
                #                     current_image = dist_util.overlay_mask_with_color(current_image, hmask, dist_m)

                #             np.copyto(self.tv_img_array, current_image[:, :self.tv_img_shape[1]])                    

                # if self.wrist_enable_shm:
                #     self._lazy_load_model()
                #     preds_wrist = self.model.predict(wrist_image)[0]
                #     ods = ObjectDepthSameSound(depth = wrist_raw_depth, masks=preds_wrist.masks,
                #                                k=2, tolerance_mm=10, cooldown_s=0.5, release_mm=15)
                #     if preds_wrist.masks is None or len(set(preds_wrist.boxes.cls.tolist())) < min_classes:
                #         print("[Image Client] No masks found in predictions.")
                #         if self.wrist_buffer:
                #             wrist_fb_mask = self.wrist_buffer[-1]
                #             np.copyto(self.wrist_img_array, np.array(wrist_image[:, :self.wrist_img_shape[1]]))
                #         else:
                #             np.copyto(self.wrist_img_array, np.array(wrist_image[:, :self.wrist_img_shape[1]]))
                #     else:
                #         np.copyto(self.wrist_img_array, np.array(wrist_image[:, :self.wrist_img_shape[1]]))
                        
                #         try:
                #             classes_w = list(map(int, preds_wrist.boxes.cls.tolist()))
                #             masks_w   = preds_wrist.masks.xy  # [np.ndarray(Nx2), ...]

                #             # 로봇손 제외하고 "물체"만 남기기 (왼손=4, 오른손=5 가정)
                #             object_masks = [m for m, c in zip(masks_w, classes_w) if c not in (4, 5)]

                #             # ODS lazy init (혹은 위 __init__에서 이미 생성해둔 self._ods 사용)
                #             if self._ods is None and self.align_sound_path is not None:
                #                 self._ods = ObjectDepthSameSound(
                #                     depth=self.wrist_depth_img_array,
                #                     masks=object_masks,
                #                     align_sound_path=self.align_sound_path,
                #                     k=2, tolerance_mm=10, cooldown_s=0.5, release_mm=15
                #                 )

                #             if self._ods is not None:
                #                 # 최신 depth & masks 반영
                #                 self._ods.depth = self.wrist_depth_img_array
                #                 self._ods.masks = object_masks

                #                 # 이벤트 트리거 (같은 깊이 쌍이 있으면 내부 쿨다운 고려하여 재생)
                #                 res = self._ods.sound_depth_same_between_objects()
                #                 # 디버깅 로그 (원하면)
                #                 # if res["aligned"]:
                #                 #     print("[ODS] aligned pairs:", res["pairs"])
                #         except Exception as e:
                #             print(f"[ODS] runtime error: {e}")
                                
                                
                # # if self.tv_depth_enable_shm:
                # #     raw_depth = raw_depth.reshape(self.tv_depth_img_shape[0], self.tv_depth_img_shape[1])
                # #     np.copyto(self.tv_depth_img_array, raw_depth)
                
                # if self.wrist_depth_enable_shm:
                #     wrist_raw_depth = wrist_raw_depth.reshape(self.wrist_depth_img_shape[0], self.wrist_depth_img_shape[1])
#            
                if self._image_show:
                    # height, width = current_image.shape[:2]
                    # wrist_height, wrist_width = wrist_image.shape[:2]
                    # resized_image = cv2.resize(current_image, (width, height))
                    # wrist_resized_image = cv2.resize(wrist_image, (wrist_width, wrist_height))
                    # cv2.imshow('Head Camera', head_preds.plot())
                    # cv2.imshow('Wrist Camera', wrist_preds.plot())
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.running = False
                    HAND_CLASSES = (4, 5)

                    # DistanceOverlay init once (use runtime frame sizes)
                    if self.dist_util is None and wrist_raw_depth is not None and wrist_raw_depth.ndim == 2:
                        hH, wH = current_image.shape[:2]
                        hW, wW = wrist_image.shape[:2]
                        head_int = {"fx": 615, "fy": 615, "cx": wH / 2.0, "cy": hH / 2.0}
                        wrist_int = {"fx": 615, "fy": 615, "cx": wW / 2.0, "cy": hW / 2.0}
                        self.dist_util = DistanceOverlay(head_int, wrist_int, np.eye(4),
                                                         min_dist=0.02, max_dist=0.20)

                    # 1) wrist side: compute min distance by hand class (if depth 2D available)
                    dist_by_cid = {}
                    if (self.dist_util is not None and
                        wrist_preds is not None and wrist_preds.masks is not None and
                        wrist_raw_depth is not None and wrist_raw_depth.ndim == 2):

                        w_masks = wrist_preds.masks.xy
                        w_cls = list(map(int, wrist_class_id))
                        robot_w = [(c, m) for m, c in zip(w_masks, w_cls) if c in HAND_CLASSES]
                        objs_w = [m for m, c in zip(w_masks, w_cls) if c not in HAND_CLASSES]
                        print(f"[viz] wrist hands: {len(robot_w)}, wrist objects: {len(objs_w)}")

                        for cid, rmask in robot_w:
                            cr = self.dist_util.compute_mask_centroid(rmask)
                            if cr is None:
                                continue
                            dlist = []
                            for om in objs_w:
                                co = self.dist_util.compute_mask_centroid(om)
                                if co is None:
                                    continue
                                d = self.dist_util.compute_object_distance_simple(cr, co, wrist_raw_depth)
                                if d is not None:
                                    dlist.append(d)
                            if dlist:
                                dmin = min(dlist)
                                dist_by_cid[cid] = min(dmin, dist_by_cid.get(cid, float("inf")))

                    # 2) head overlay (colorize hands by min distance)
                    head_disp = current_image.copy()
                    if (self.dist_util is not None and
                        head_preds is not None and head_preds.masks is not None and dist_by_cid):

                        h_masks = head_preds.masks.xy
                        h_cls = list(map(int, head_class_id))
                        head_robot_masks = {cid: [] for cid in HAND_CLASSES}
                        for m, c in zip(h_masks, h_cls):
                            if c in HAND_CLASSES:
                                head_robot_masks[c].append(m)

                        for cid, dist_m in dist_by_cid.items():
                            for hmask in head_robot_masks.get(cid, []):
                                head_disp = self.dist_util.overlay_mask_with_color(head_disp, hmask, dist_m)

                    # 3) sound feedback
                    if wrist_preds is not None and wrist_preds.masks is not None and wrist_raw_depth is not None:
                        try:
                            w_masks = wrist_preds.masks.xy
                            w_cls = list(map(int, wrist_class_id))
                            obj_masks = [m for m, c in zip(w_masks, w_cls) if c not in HAND_CLASSES]
                            if self._ods is None:
                                self._ods = ObjectDepthSameSound(
                                    depth=wrist_raw_depth,
                                    masks=obj_masks,
                                    align_sound_path=self.align_sound_path,
                                    k=2, tolerance_mm=10, cooldown_s=0.5, release_mm=15
                                )
                            else:
                                self._ods.depth = wrist_raw_depth
                                self._ods.masks = obj_masks
                            _ = self._ods.sound_depth_same_between_objects()
                        except Exception as e:
                            print(f"[image_show][ODS] error: {e}")

                    # 4) wrist display (seg plot if masks exist)
                    if wrist_preds is not None and wrist_preds.masks is not None:
                        wrist_disp = wrist_preds.plot()
                    else:
                        wrist_disp = wrist_image.copy()
                    
                    # 5) show
                    hH, wH = head_disp.shape[:2]
                    hW, wW = wrist_disp.shape[:2]
                    cv2.imshow('Head (overlay)', cv2.resize(head_disp, (wH, hH)))
                    cv2.imshow('Wrist (seg)', cv2.resize(wrist_disp, (wW, hW)))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        
                    # cv2.imshow('Image Client Stream', result.plot())
                    
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.running = False

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