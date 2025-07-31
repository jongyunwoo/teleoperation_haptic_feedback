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


def init_player(ws_addr: Optional[str] = None, verbose: bool = True) -> None:
    """
    bHaptics Player WebSocket 초기화
    • ws_addr = "ws://IP:PORT"  (None → 기본 127.0.0.1:15881)
    """
    player.initialize()
    player.is_playing()

_SEGMENTS = [
    (749, 925), (564, 740), (379, 555), (194, 370), (9, 185),
]  # thumb, index, middle, ring, little

def tactile_to_dotpoints(
    frame: np.ndarray,
    *,
    key: str,                 # "L" 또는 "R" 손 구분
    thresh: float = 50.0,     # 노이즈 컷(이하 값은 0 취급)
    p_low: float = 1.0,      # 하위 퍼센타일(0%에 대응)
    p_high: float = 99.0,     # 상위 퍼센타일(100%에 대응)
    alpha: float = 0.2,       # EMA 추적 속도(0.1~0.3 권장)
    min_ref_span: float = 300.0,  # 상하 스케일 최소 간격(너무 붙으면 포화)
    min_contact: float = 100.0,    # 전체 접촉 에너지 게이트(낮으면 무시)
    gamma: float = 1.2,       # 감마>1: 빨강 도달 늦춤
) -> list[dict]:

    if frame.ndim != 1 or frame.size != 1062:
        raise ValueError("frame shape must be (1062,)")

    f = frame.astype(np.float32, copy=False)
    seg_vals = []
    for (s, e) in _SEGMENTS:
        seg = f[s:e]
        # 노이즈 컷
        seg = seg[seg > thresh]
        v = 0.0 if seg.size == 0 else float(np.percentile(seg, 90))
        seg_vals.append(v)
    seg_vals = np.array(seg_vals, dtype=np.float32)

    # 접촉이 거의 없으면 모두 0
    if np.max(seg_vals) < min_contact:
        return []

    # 전역 퍼센타일 타깃 계산
    pos = seg_vals[seg_vals > 0]
    lo_t = float(np.percentile(pos, p_low))  if pos.size else 0.0
    hi_t = float(np.percentile(pos, p_high)) if pos.size else min_contact + min_ref_span
    if hi_t - lo_t < min_ref_span:
        hi_t = lo_t + min_ref_span

    # 손별 EMA 상태 저장/갱신
    if not hasattr(tactile_to_dotpoints, "_state"):
        tactile_to_dotpoints._state = {}
    state = tactile_to_dotpoints._state.setdefault(key, {"lo": lo_t, "hi": hi_t})

    lo = (1.0 - alpha) * state["lo"] + alpha * lo_t
    hi = (1.0 - alpha) * state["hi"] + alpha * hi_t
    if hi - lo < min_ref_span:
        hi = lo + min_ref_span

    state["lo"], state["hi"] = lo, hi

    # 0~100 매핑
    dots: list[dict] = []
    for idx, v in enumerate(seg_vals):
        t = (v - lo) / (hi - lo)
        t = 0.0 if t < 0.0 else (t ** gamma)
        if t > 1.0:
            t = 1.0
        intensity = int(round(t * 100.0))
        dots.append({"index": idx, "intensity": intensity})
    return dots

def start_haptics_stream(shared_array,
                         lb: np.ndarray | None = None,
                         rb: np.ndarray | None = None,
                         hz: int = 10,
                         duration_ms: int = 100) -> threading.Thread:

    interval = 1.0 / hz
    buf_np = np.frombuffer(shared_array.get_obj()
                           if hasattr(shared_array, "get_obj") else shared_array,
                           dtype=np.float64)

    # 안전 가드: baseline 모양 확인
    if lb is not None:
        assert lb.shape == (1062,), f"lb shape {lb.shape} must be (1062,)"
    if rb is not None:
        assert rb.shape == (1062,), f"rb shape {rb.shape} must be (1062,)"

    # 파라미터(필요시 조정)
    THRESH = 30.0
    P_LOW, P_HIGH = 10.0, 95.0
    ALPHA = 0.20
    MIN_REF_SPAN = 200.0
    MIN_CONTACT = 50.0
    GAMMA = 1.2

    def _loop():
        last_log = 0.0
        while True:
            # 스냅샷
            left  = np.array(buf_np[:1062],  dtype=np.float32, copy=True)
            right = np.array(buf_np[-1062:], dtype=np.float32, copy=True)

            # 베이스라인 보정
            if lb is not None: left  -= lb.astype(np.float32, copy=False)
            if rb is not None: right -= rb.astype(np.float32, copy=False)

            # 임계값 초과가 하나라도 있으면 전송
            if (left > THRESH).any() or (right > THRESH).any():
                l_dots = tactile_to_dotpoints(
                    left,  key="L", thresh=THRESH, p_low=P_LOW, p_high=P_HIGH,
                    alpha=ALPHA, min_ref_span=MIN_REF_SPAN, min_contact=MIN_CONTACT,
                    gamma=GAMMA
                )
                r_dots = tactile_to_dotpoints(
                    right, key="R", thresh=THRESH, p_low=P_LOW, p_high=P_HIGH,
                    alpha=ALPHA, min_ref_span=MIN_REF_SPAN, min_contact=MIN_CONTACT,
                    gamma=GAMMA
                )

                if l_dots:
                    player.submit_dot("L_touch", BhapticsPosition.GloveL.value,
                                      l_dots, duration_millis=duration_ms)
                if r_dots:
                    player.submit_dot("R_touch", BhapticsPosition.GloveR.value,
                                      r_dots, duration_millis=duration_ms)

            # 1초 주기 디버그
            now = time.time()
            if now - last_log >= 1.0:
                # 강도 리스트 확인용(선택)
                # print("L intens:", [d["intensity"] for d in l_dots] if 'l_dots' in locals() else [])
                # print("R intens:", [d["intensity"] for d in r_dots] if 'r_dots' in locals() else [])
                print(f"[haptics] max L/R = {left.max():.1f} / {right.max():.1f}")
                last_log = now

            time.sleep(interval)

    th = threading.Thread(target=_loop, daemon=True, name="Tactile→Glove")
    th.start()
    return th
# ──────────────────────────────────────────────────────
# 4. 단독 실행 테스트 (python haptics_bridge.py 로 실행)
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp

    # (A) 가짜 센서 데이터 → 삼각파로 주기적 진동 확인
    arr = mp.Array('d', 1062 * 2)
    buf = np.frombuffer(arr.get_obj(), dtype=np.float64)

    def fake_sensor():
        t = 0.0
        while True:
            base = (np.sin(t) * 0.5 + 0.5) * 4095
            # 채널마다 위상 차이를 줌
            vals = base * (1.0 + 0.2*np.sin(t + np.linspace(0, 2*np.pi, 1062*2)))
            buf[:] = vals
            t += 0.1
            time.sleep(0.05)
    mp.Process(target=fake_sensor, daemon=True).start()

    # (B) Player 연결 + 스트리머
    init_player()
    start_haptics_stream(arr, hz=30, duration_ms=100)

    print("Press Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
