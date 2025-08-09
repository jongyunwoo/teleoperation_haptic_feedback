import threading
import time
import numpy as np
import cv2
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play

# 손 클래스 (필요하면 밖에서 재사용)
HAND_CLASSES = {4, 5}

class StereoSoundFeedbackManager:
    def __init__(self, grip_sound_path):
        self.prev_grip_state = {"left": False, "right": False}
        self.grip_sound = AudioSegment.from_file(grip_sound_path)

    def play_sound(self, hand="right"):
        sound = self.grip_sound
        if hand == "right":
            sound = sound.pan(1.0)   # 오른쪽 출력
        elif hand == "left":
            sound = sound.pan(-1.0)  # 왼쪽 출력
        threading.Thread(target=play, args=(sound,), daemon=True).start()

    def update(self, grip_detected, hand="right"):
        # 필요시 외부에서 haptics 제어
        if grip_detected and not self.prev_grip_state[hand]:
            self.play_sound(hand)
        self.prev_grip_state[hand] = grip_detected


class ObjectDepthSameSound:
    """
    - depth: HxW uint16 (mm)
    - masks: [np.ndarray(Nx2), ...]  각 객체의 폴리곤(픽셀 좌표)
    """
    def __init__(
        self,
        depth,
        masks,
        align_sound_path='/home/scilab/teleoperation/avp_teleoperate/hapticfeedback/sounddata/beep-125033.mp3',
        k=2,
        tolerance_mm=10,
        cooldown_s=0.5,
        release_mm=15,
        stop_overlap: bool = False
    ):
        self.depth = depth
        self.masks = masks
        self.k = k
        self.tol = tolerance_mm
        self.release = release_mm
        self.cooldown = cooldown_s
        self._stop_overlap = stop_overlap

        self.align_sound = AudioSegment.from_file(align_sound_path)

        # 내부 상태
        self._last_play_t = 0.0
        self._aligned_pairs = set()
        self._plock = threading.Lock()
        self._last_play_obj = None

    @staticmethod
    def _centroid_of_polygon(poly: np.ndarray):
        """poly: (N,2) float/int -> (cx,cy) 또는 None"""
        cnt = np.asarray(poly, dtype=np.int32)
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            return None
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def robust_depth_at(self, cx, cy):
        h, w = self.depth.shape
        k = self.k
        x0, x1 = max(0, cx - k), min(w, cx + k + 1)
        y0, y1 = max(0, cy - k), min(h, cy + k + 1)
        patch = self.depth[y0:y1, x0:x1].astype(float)
        # 유효(depth>0) 값만
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
        """
        마스크 쌍 중 깊이가 tol(mm) 이내로 같은 쌍이 새로 생기면 사운드 1회 재생.
        release(mm) 이상 벌어지면 쌍 해제.
        """
        if self.depth is None or not isinstance(self.depth, np.ndarray) or self.depth.ndim != 2:
            return {"aligned": False, "pairs": []}
        if self.masks is None or len(self.masks) == 0:
            return {"aligned": False, "pairs": []}

        # 1) 각 객체 중심 및 robust depth(mm)
        centers = []
        for poly in self.masks:
            c = self._centroid_of_polygon(poly)
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
            if not np.isfinite(ci[2]):
                continue
            for j in range(i + 1, n):
                cj = centers[j]
                if not np.isfinite(cj[2]):
                    continue

                diff = abs(ci[2] - cj[2])  # mm
                key = (min(i, j), max(i, j))

                if diff <= self.tol:
                    aligned_pairs.append((i, j, ci[2], cj[2], diff))
                    if key not in self._aligned_pairs:
                        newly_aligned = True
                        self._aligned_pairs.add(key)
                else:
                    if key in self._aligned_pairs and diff >= self.release:
                        self._aligned_pairs.remove(key)

        # 3) 사운드 트리거
        if newly_aligned and aligned_pairs:
            self._play_once()

        return {"aligned": newly_aligned, "pairs": aligned_pairs}
