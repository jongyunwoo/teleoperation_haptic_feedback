import threading
import time
import numpy as np
import cv2
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play

# 손 클래스 (필요하면 밖에서 재사용)
HAND_CLASSES = {7}

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
    def __init__(self, depth, masks,
                 align_sound_path='/home/scilab/teleoperation/avp_teleoperate/hapticfeedback/sounddata/bell-notification-337658.mp3',
                 k=4, tolerance_mm=60, cooldown_s=0.8, release_mm=60,
                 stop_overlap=False,
                 dwell_s=0.5,        
                 key_grid_px=60,      
                 stale_s = 10.0,
                 suppress_s = 15.0,
                 outside_tol_clear_s = 1.2
                 ):
        self.depth = depth
        self.masks = masks
        self.k = k
        self.tol = tolerance_mm
        self.release = release_mm
        self.cooldown = cooldown_s
        self._stop_overlap = stop_overlap

        self.dwell_s = dwell_s
        self.key_grid = int(key_grid_px)
        self.stale_s = float(stale_s)
        self.suppress_s = float(suppress_s)
        self.outside_tol_clear_s = float(outside_tol_clear_s)
        self._pair_state = {}
        self.align_sound = AudioSegment.from_file(align_sound_path)
        self._last_play_t = 0.0

        # 인덱스 대신 공간기반 키의 상태를 저장
        # key -> {"first_in_tol": t, "last_seen": t, "beeped_at": t or None}
        self._pair_state = {}

        self._plock = threading.Lock()
        self._last_play_obj = None

    @staticmethod
    def _centroid_of_polygon(poly: np.ndarray):
        cnt = np.asarray(poly, dtype=np.int32)
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            return None
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

    def robust_depth_at(self, cx, cy):
        h, w = self.depth.shape
        k = self.k
        x0, x1 = max(0, cx - k), min(w, cx + k + 1)
        y0, y1 = max(0, cy - k), min(h, cy + k + 1)
        patch = self.depth[y0:y1, x0:x1].astype(float)
        patch = patch[(patch > 0)]  # 필요하면 (patch>min_mm)&(patch<max_mm) 추가
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

    def _pair_key_from_centers(self, ci, cj):
        # 센트로이드(픽셀)를 격자에 맞춰 양자화해서 키 생성 (프레임 간 인덱스 변동에 강함)
        q = self.key_grid
        def qpt(p):
            return (int(round(p[0]/q)), int(round(p[1]/q)))
        a, b = qpt(ci), qpt(cj)
        return tuple(sorted((a, b)))  # 순서 무관

    def sound_depth_same_between_objects(self):
        if self.depth is None or not isinstance(self.depth, np.ndarray) or self.depth.ndim != 2:
            return {"aligned": False, "pairs": []}
        if self.masks is None or len(self.masks) == 0:
            return {"aligned": False, "pairs": []}

        # 1) 각 객체 중심 및 깊이(mm)
        centers = []
        for poly in self.masks:
            c = self._centroid_of_polygon(poly)
            if c is None:
                centers.append((None, None, np.nan))
                continue
            d_mm = self.robust_depth_at(c[0], c[1])
            centers.append((c[0], c[1], d_mm))

        now = time.time()
        aligned_pairs = []
        newly_aligned = False

        # 2) 모든 쌍 비교
        n = len(self.masks)
        for i in range(n):
            ci = centers[i]
            if not np.isfinite(ci[2]):
                continue
            for j in range(i+1, n):
                cj = centers[j]
                if not np.isfinite(cj[2]):
                    continue

                diff = abs(ci[2] - cj[2])  # mm
                key = self._pair_key_from_centers((ci[0], ci[1]), (cj[0], cj[1]))
                state = self._pair_state.get(key)
                if diff <= self.tol:
                    aligned_pairs.append((i, j, ci[2], cj[2], diff))
                    if state is None:
                        # 처음 tol 안으로 들어옴: 타이머 시작
                        self._pair_state[key] = {"first_in_tol": now, "last_seen": now, "beeped_at": None, "out_tol_sinece": None}
                    else:
                        state["last_seen"] = now
                        state["out_tol_since"] = None
                        # dwell 시간 충족 + 글로발 쿨다운 + 해당 key 최근 재생 체크
                        if ((state["beeped_at"] is None) or (now - state["beeped_at"] >= self.suppress_s))\
                            and (now - state["first_in_tol"] >= self.dwell_s):
                            newly_aligned = True
                            self._play_once()
                            state["beeped_at"] = now
                else:
                    if state is not None:
                        state["last_seen"] = now
                        if state.get("out_tol_since") is None:
                            state["out_tol_since"] = now
                        if now - state["out_tol_since"] >= self.outside_tol_clear_s:
                            self._pair_state.pop(key, None)

        # 3) 오래 안 보인 키 정리(유령 상태 방지)
        stale_keys = [k for k, v in self._pair_state.items() if now - v["last_seen"] > 2.0]
        for k in stale_keys:
            self._pair_state.pop(k, None)

        return {"aligned": newly_aligned, "pairs": aligned_pairs}