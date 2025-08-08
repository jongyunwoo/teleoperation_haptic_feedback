from pydub import AudioSegment
from pydub.playback import play
import threading
from bhaptics import better_haptic_player as player
import numpy as np
import cv2
from visfeedback import DistanceOverlay

haptics_paused = False

class StereoSoundFeedbackManager:
    def __init__(self, grip_sound_path):
        self.prev_grip_state = {"left": False, "right": False}
        self.grip_sound = AudioSegment.from_file(grip_sound_path)

    def play_sound(self, hand="right"):
        if hand == "right":
            sound = sound.pan(1.0)   # 오른쪽 출력
        elif hand == "left":
            sound = sound.pan(-1.0)  # 왼쪽 출력
        threading.Thread(target=play, args=(sound,), daemon=True).start()

    def update(self, grip_detected, hand="right"):
        global haptics_paused
        if grip_detected and not self.prev_grip_state[hand]:
            self.play_sound(hand)
            player.stop_all()        # 모든 haptic 중단
            haptics_paused = True    # Haptic 루프에서 중단 플래그 확인
        
        # Grip 해제 → Haptic 재개
        elif not grip_detected and self.prev_grip_state[hand]:
            haptics_paused = False   # 다시 Haptic 재개 가능

        # 상태 갱신
        self.prev_grip_state[hand] = grip_detected
        
Hand_Classes = {6, 7}
     
class ObjectDepthSameSound:
    def __init__(self, depth, masks, align_sound_path='/home/scilab/Documents/teleoperation/avp_teleoperate/hapticfeedback/sounddata/beep-125033.mp3',
                 k=2, tolerance_mm=10, cooldown_s=0.5, release_mm=15, stop_overlap: bool = False):
        self.depth = depth
        self.masks = masks
        self.k = k
        self.tol = tolerance_mm
        self.release = release_mm
        self.cooldown = cooldown_s
        self.stop_overlab = stop_overlab

        self.align_sound = AudioSegment.from_file(align_sound_path)
        self._last_play_t = 0.0
        self._aligned_pairs = set() 
    
    def robust_depth_at(self, cx, cy):
        h, w = self.depth.shape
        k = self.k
        x0, x1 = max(0, cx - k), min(w, cx + k + 1)
        y0, y1 = max(0, cy - k), min(h, cy + k + 1)
        patch = self.depth[y0:y1, x0:x1].astype(float)
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            return np.nan
        return float(np.median(patch))
    
    def _play_nonblocking(self, audio: AudioSegment):
        with self._plock:
            if self._stop_overlap and self._last_play_obj is not None:
                try:
                    if self._last_play_obj.is_playing():
                        self._last_play_obj.stop()
                except Exception:
                    pass
            self._last_play_obj = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )

    def _play_once(self):
        now = time.time()
        if now - self._last_play_t >= self.cooldown:
            self._play_nonblocking(self.align_sound)
            self._last_play_t = now

def sound_depth_same_between_objects(self):
    
        # 1) 각 객체 중심 및 robust depth(mm) 계산
        centers = []
        for poly in self.masks:
            c = DistanceOverlay.compute_mask_centroid(poly)
            if c is None:
                centers.append((None, None, np.nan))
                continue
            d_mm = self.robust_depth_at(c[0], c[1])
            centers.append((c[0], c[1], d_mm))

        # 2) 모든 쌍 비교
        aligned_pairs = []
        newly_aligned = False
        n = len(self.masks)
        for i in range(n):
            ci = centers[i]
            if not np.isfinite(ci[2]):  # depth 없음
                continue
            for j in range(i+1, n):
                cj = centers[j]
                if not np.isfinite(cj[2]):
                    continue

                diff = abs(ci[2] - cj[2])  # mm
                key = (min(i, j), max(i, j))

                # 정렬 조건: diff <= tol
                if diff <= self.tol:
                    aligned_pairs.append((i, j, ci[2], cj[2], diff))
                    # 새로 정렬된 경우에만 사운드
                    if key not in self._aligned_pairs:
                        newly_aligned = True
                        self._aligned_pairs.add(key)
                else:
                    # 히스테리시스: 차이가 크게 벌어지면 정렬 해제
                    if key in self._aligned_pairs and diff >= self.release:
                        self._aligned_pairs.remove(key)

        # 3) 사운드 트리거 (여러 페어가 있어도 딱 한 번만)
        if newly_aligned and aligned_pairs:
            self._play_once()