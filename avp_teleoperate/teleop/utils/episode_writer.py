import os
import cv2
import json
import datetime
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread

MAX_DEPTH_MM = 4000.0

class EpisodeWriter():
    def __init__(self, task_dir, frequency=30, image_size=[848, 480], rerun_log = True):
        """
        image_size: [width, height]
        """
        print("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            print("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            print("==> RerunLogger initializing ok.\n")
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            print(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            print(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(maxsize=100)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        print("==> EpisodeWriter initialized successfully.\n")

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                # "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                'wrist_image' : {"width" : 480, 'height' : 640, 'fps' : self.frequency},
                'wrist_depth' : {"width" : 480, 'height' : 640, 'fps' : self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   ['kLeftShoulderPitch' ,'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow', 'kLeftWristRoll', 'kLeftWristPitch', 'kLeftWristyaw'],
                    "left_hand":  ["kLeftHandPinky","kLeftHandRing","kLeftHandMiddle","kLeftHandIndex","kLeftHandThumbBend","kLeftHandThumbRotation"],
                    "right_arm":  ["kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw"],
                    "right_hand": ["kRightHandPinky","kRightHandRing","kRightHandMiddle","kRightHandIndex","kRightHandThumbBend","kRightHandThumbRotation"],
                    "body":       [],
                },

                "tactile_names": {
                    "left_hand": ["left_fingerone_tip_touch","left_fingerone_top_touch","left_fingerone_palm_touch",
                                  "left_fingertwo_tip_touch","left_fingertwo_top_touch","left_fingertwo_palm_touch",
                                  "left_fingerthree_tip_touch","left_fingerthree_top_touch","left_fingerthree_palm_touch",
                                  "left_fingerfour_tip_touch","left_fingerfour_top_touch","left_fingerfour_palm_touch","left_fingerfive_tip_touch",
                                  "left_fingerfive_top_touch","left_fingerfive_middle_touch","left_fingerfive_palm_touch","left_palm_touch"],
                    
                    "right_hand":["right_fingerone_tip_touch","right_fingerone_top_touch","right_fingerone_palm_touch",
                                  "right_fingertwo_tip_touch","right_fingertwo_top_touch","right_fingertwo_palm_touch",
                                  "right_fingerthree_tip_touch","right_fingerthree_top_touch","right_fingerthree_palm_touch",
                                  "right_fingerfour_tip_touch","right_fingerfour_top_touch","right_fingerfour_palm_touch","right_fingerfive_tip_touch",
                                  "right_fingerfive_top_touch","right_fingerfive_middle_touch","right_fingerfive_palm_touch","right_palm_touch"],
                }, 
            }
    def text_desc(self):
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "Pick up the cup from the table and place it in another position. The operation should be smooth and the water in the cup should not spill out",
            "steps":"step1: searching for cups. step2: go to the target location. step3: pick up the cup",
        }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            print("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.wrist_color_dir = os.path.join(self.episode_dir, 'wrist_colors') #추가
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.wrist_depth_dir = os.path.join(self.episode_dir, 'wrist_depths') #추가
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        #tactile 데이터 저장용 디렉토리 생성
        self.tactile_dir = os.path.join(self.episode_dir, 'tactiles') #추가
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.wrist_color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.wrist_depth_dir, exist_ok=True) #추가
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.tactile_dir, exist_ok=True) #추가
        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        print(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, wrist_colors, depths, wrist_depths, states=None, actions=None, tactiles=None, audios=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'wrist_colors' : wrist_colors,
            'depths': depths,
            'wrist_depths' : wrist_depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles, 
            'audios': audios,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    print(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        wrist_colors = item_data.get('wrist_colors', {}) #추가
        depths = item_data.get('depths', {})
        wrist_depths = item_data.get('wrist_depths', {}) #추가
        audios = item_data.get('audios', {})
        tactiles = item_data.get('tactiles', {}) #추가

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    print(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # Save depths
        # if depths:
        #     for idx_depth, (depth_key, depth) in enumerate(depths.items()):
        #         depth_name = f'{str(idx).zfill(6)}_{depth_key}.png'
        #         if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
        #             print(f"Failed to save depth image.")
        #         item_data['depths'][depth_key] = os.path.join('depths', depth_name)
        #         depth_clipped = np.clip(depth.astype(np.float32), 0, MAX_DEPTH_MM)
        #         depth_8u = (depth_clipped / MAX_DEPTH_MM * 255).astype(np.uint8)
        #         preview = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)        # 색상 맵
        #         prev_name = f'{idx:06d}_{depth_key}_vis.jpg'
        #         cv2.imwrite(os.path.join(self.depth_dir, prev_name), preview)
        
        
        if wrist_colors:
            for idx_wrist_color, (wrist_color_key, wrist_color) in enumerate(wrist_colors.items()):
                wrist_color_name = f'{str(idx).zfill(6)}_{wrist_color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.wrist_color_dir, wrist_color_name), wrist_color):
                    print(f"Failed to save color image.")
                item_data['wrist_colors'][wrist_color_key] = os.path.join('colors', wrist_color_name)
        
        if depths:
            for depth_key, depth_arr in depths.items():
                fname = f'{idx:06d}_{depth_key}.npy'
                path  = os.path.join(self.depth_dir, fname)
                # raw uint16 numpy 배열로 저장
                np.save(path, depth_arr)
                item_data['depths'][depth_key] = os.path.join('depths', fname)                                
       
        if wrist_depths:
            for wrist_depth_key, wrist_depth_arr in wrist_depths.items():
                fname = f'{idx:06d}_{wrist_depth_key}.npy'
                path  = os.path.join(self.wrist_depth_dir, fname)
                # raw uint16 numpy 배열로 저장
                np.save(path, wrist_depth_arr)
                item_data['wrist_depths'][wrist_depth_key] = os.path.join('wrist_depths', fname)     
        
        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)
        
        #Save tactile 추가
        if tactiles:
            for key, tactile in tactiles.items():
                fname = f'{str(idx).zfill(6)}_{key}.npy'
                path = os.path.join(self.tactile_dir, fname)
                np.save(path, np.array(tactile, dtype=np.float32))
                item_data['tactiles'][key] = os.path.join('tactiles', fname)
                
        # Update episode data
        self.episode_data.append(item_data)

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            print(f"==> episode_id:{self.episode_id}  item_id:{self.item_id}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        print(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        with open(self.json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        print(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()