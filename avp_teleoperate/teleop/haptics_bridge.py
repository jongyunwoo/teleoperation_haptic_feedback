"""
haptics_bridge.py
────────────────────────────────────────────────────────
Inspire Hand tactile (1062×2 공유배열) → bHaptics TactGlove dot-feedback
• 1 초마다 연결 상태 로깅
• 0-값 센서 프레임 스킵
• 자동 스케일링 + 최소 강도 보정
"""
import os, sys

# 1) 이 파일이 있는 디렉터리
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  
# 2) 한 단계 위 → avp_teleoperate
ONE_UP   = os.path.dirname(THIS_DIR)                 
# 3) 또 한 단계 위 → g1_tele
TWO_UP   = os.path.dirname(ONE_UP)                    
# 4) 거기에 있는 tact-python
TACT_PYTHON_DIR = os.path.join(TWO_UP, "tact-python")

# 5) 검색 경로에 추가
if not os.path.isdir(TACT_PYTHON_DIR):
    raise RuntimeError(f"tact-python 폴더를 찾을 수 없습니다: {TACT_PYTHON_DIR}")
sys.path.insert(0, TACT_PYTHON_DIR)


import time
import threading
import numpy as np
from typing import Optional, List, Dict
from bhaptics import better_haptic_player as player
from bhaptics.better_haptic_player import BhapticsPosition


# ──────────────────────────────────────────────────────
# 1. 플레이어 초기화 & 상태 점검
# ──────────────────────────────────────────────────────
def init_player(ws_addr: Optional[str] = None, verbose: bool = True) -> None:
    """
    bHaptics Player WebSocket 초기화
    • ws_addr = "ws://IP:PORT"  (None → 기본 127.0.0.1:15881)
    """
    player.initialize()
    player.is_playing()

    # if verbose:
    #     for pos in (BhapticsPosition.GloveL, BhapticsPosition.GloveR):
    #         print(f"[bHaptics] {pos.name:6} connected:",
    #               player.is_device_connected(pos.value))


# ──────────────────────────────────────────────────────
# 2. 1062-채널 → 5 개 손톱 영역 max → dot 리스트
# ──────────────────────────────────────────────────────
_SEGMENTS = [
    (9, 105), (194, 290), (379, 475), (564, 660), (749, 845)
]  # little, ring, middle, index, thumb


def tactile_to_dotpoints(frame: np.ndarray,
                          min_intensity: int = 50) -> List[Dict[str, int]]:
    """
    frame : (1062,) float or int (0~4095)
    반환    : [{"index": 0-4, "intensity": 0-100}, …]
    - 세기가 min_intensity 미만이면 0 으로 클리핑
    """
    if frame.ndim != 1 or frame.size != 1062:
        raise ValueError("frame shape must be (1062,)")

    fmax = frame.max()
    if fmax == 0:       # 모두 0 → 아무 진동도 안 보냄
        return []

    # scale = 100.0 / fmax
    dots = []
    for idx, (start, end) in enumerate(_SEGMENTS):
        intensity = float(frame[start:end].max() * 100 / 4095)
        if intensity < min_intensity:
            intensity = 0
        dots.append({"index": idx, "intensity": intensity})
    return dots


# ──────────────────────────────────────────────────────
# 3. 공유배열 스트리머
# ──────────────────────────────────────────────────────
def start_haptics_stream(shared_array, hz: int = 30,
                          duration_ms: int = 100) -> threading.Thread:
    """
    shared_array : multiprocessing.Array('d', 1062*2) 등의 버퍼
    hz           : 초당 전송 빈도
    duration_ms  : 모터 진동 지속 시간 (33-100 사이 권장)
    """
    interval = 1.0 / hz
    buf_np = np.frombuffer(shared_array.get_obj()  # type: ignore[attr-defined]
                           if hasattr(shared_array, "get_obj") else shared_array,
                           dtype=np.float64)

    def _loop():
        last_log = 0.0
        while True:
            left = buf_np[:1062].copy()
            right = buf_np[1062:].copy()

            # 값이 전부 0이면 skip (센서 or 컨트롤러 미동작)
            if left.max() or right.max():
                l_dots = tactile_to_dotpoints(left)
                r_dots = tactile_to_dotpoints(right)
                if l_dots:
                    player.submit_dot(
                        "L_touch", BhapticsPosition.GloveL.value,
                        l_dots, duration_millis=duration_ms)
                if r_dots:
                    player.submit_dot(
                        "R_touch", BhapticsPosition.GloveR.value,
                        r_dots, duration_millis=duration_ms)

            # 5초 주기로 디버그 출력
            if time.time() - last_log > 5:
                print(f"[haptics] max L/R = {left.max():.2f} / {right.max():.2f}")
                last_log = time.time()

            time.sleep(interval)

    th = threading.Thread(target=_loop, daemon=True, name="Tactile→Glove")
    th.start()
    return th


# ──────────────────────────────────────────────────────
# 4. 단독 실행 테스트 (python haptics_bridge.py 로 실행)
# ──────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import multiprocessing as mp

#     # (A) 가짜 센서 데이터 → 삼각파로 주기적 진동 확인
#     arr = mp.Array('d', 1062 * 2)
#     buf = np.frombuffer(arr.get_obj(), dtype=np.float64)

#     def fake_sensor():
#         t = 0.0
#         while True:
#             val = (np.sin(t) * 0.5 + 0.5) * 4095
#             buf[:] = val
#             t += 0.1
#             time.sleep(0.05)

#     mp.Process(target=fake_sensor, daemon=True).start()

#     # (B) Player 연결 + 스트리머
#     init_player()
#     start_haptics_stream(arr, hz=30, duration_ms=100)

#     print("Press Ctrl-C to quit.")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         pass
