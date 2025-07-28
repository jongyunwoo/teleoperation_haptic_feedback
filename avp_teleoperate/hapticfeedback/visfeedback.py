import cv2
import numpy as np

# ─────────────────────────────────────────────────────
# 150×150 기준 영역별 센서 개수
# ─────────────────────────────────────────────────────
touch_dict = {
    "fingerone_tip_touch":    9,   "fingerone_top_touch":    96,  "fingerone_palm_touch":   80,
    "fingertwo_tip_touch":    9,   "fingertwo_top_touch":    96,  "fingertwo_palm_touch":   80,
    "fingerthree_tip_touch":  9,   "fingerthree_top_touch":  96,  "fingerthree_palm_touch": 80,
    "fingerfour_tip_touch":   9,   "fingerfour_top_touch":   96,  "fingerfour_palm_touch":  80,
    "fingerfive_tip_touch":   9,   "fingerfive_top_touch":   96,  "fingerfive_middle_touch": 9,
    "fingerfive_palm_touch":  96,  "palm_touch":            112
}

# ─────────────────────────────────────────────────────
# 화면 상 박스 위치
# ─────────────────────────────────────────────────────
x_offset = 250
y_offset = 30
boxes = {
    "fingerone_tip_touch":    (114+x_offset,50+y_offset,116+x_offset,52+y_offset),
    "fingerone_top_touch":    (112+x_offset,55+y_offset,119+x_offset,66+y_offset),
    "fingerone_palm_touch":   (108+x_offset,75+y_offset,115+x_offset,84+y_offset),

    "fingertwo_tip_touch":    (96+x_offset,38+y_offset,98+x_offset,40+y_offset),
    "fingertwo_top_touch":    (93+x_offset,45+y_offset,100+x_offset,56+y_offset),
    "fingertwo_palm_touch":   (90+x_offset,75+y_offset,97+x_offset,84+y_offset),

    "fingerthree_tip_touch":  (73+x_offset,36+y_offset,75+x_offset,38+y_offset),
    "fingerthree_top_touch":  (71+x_offset,41+y_offset,78+x_offset,52+y_offset),
    "fingerthree_palm_touch": (73+x_offset,75+y_offset,80+x_offset,84+y_offset),

    "fingerfour_tip_touch":   (52+x_offset,38+y_offset,54+x_offset,40+y_offset),
    "fingerfour_top_touch":   (50+x_offset,44+y_offset,57+x_offset,55+y_offset),
    "fingerfour_palm_touch":  (53+x_offset,75+y_offset,60+x_offset,84+y_offset),

    "fingerfive_tip_touch":   (31+x_offset,64+y_offset,33+x_offset,66+y_offset),
    "fingerfive_top_touch":   (30+x_offset,69+y_offset,37+x_offset,80+y_offset),
    "fingerfive_middle_touch":(35+x_offset,90+y_offset,37+x_offset,92+y_offset),
    "fingerfive_palm_touch":  (33+x_offset,100+y_offset,40+x_offset,111+y_offset),

    "palm_touch":             (69+x_offset,94+y_offset,82+x_offset,111+y_offset),
}

# ─────────────────────────────────────────────────────
# 영역 → 센서 인덱스 슬라이스
# ─────────────────────────────────────────────────────
_region_slices = {}
_start = 0
for region, cnt in touch_dict.items():
    _region_slices[region] = slice(_start, _start + cnt)
    _start += cnt

# ─────────────────────────────────────────────────────
# 센서수 → 격자 크기(가로 nx, 세로 ny) 매핑
# 실제 센서 배치와 다르면 nx, ny를 바꾸거나 snake=True 사용
# ─────────────────────────────────────────────────────
GRID_BY_COUNT = {
    9:   (3, 3),
    96:  (12, 8),   # 12×8 = 96
    80:  (10, 8),   # 10×8 = 80
    112: (14, 8),   # 14×8 = 112
}

def _grid_for_count(n: int) -> tuple[int, int]:
    if n in GRID_BY_COUNT:
        return GRID_BY_COUNT[n]
    # fallback: 정사각에 가깝게
    nx = int(np.ceil(np.sqrt(n)))
    ny = int(np.ceil(n / nx))
    return nx, ny

# ─────────────────────────────────────────────────────
# 색상 보간(Blue → Green → Yellow → Orange → Red)
# ─────────────────────────────────────────────────────
_STOPS  = np.array([0.0, 0.25, 0.50, 0.75, 1.00], dtype=float)
_COLORS = np.array([
    [255,   0,   0],  # Blue   (BGR)
    [  0, 255,   0],  # Green
    [  0, 255, 255],  # Yellow
    [  0, 165, 255],  # Orange
    [  0,   0, 255],  # Red
], dtype=float)

def _bgr_from_t(t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    i = np.searchsorted(_STOPS, t, side="right") - 1
    i = np.clip(i, 0, len(_STOPS) - 2)
    t0, t1 = _STOPS[i], _STOPS[i + 1]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    c = (1.0 - a) * _COLORS[i] + a * _COLORS[i + 1]
    return int(c[0]), int(c[1]), int(c[2])

# ─────────────────────────────────────────────────────
# 한 박스를 nx×ny 격자로 나눠 센서별로 칠하기
# snake=True 이면 행마다 지그재그(배선이 왕복형일 때)
# vmax=None 이면 해당 박스 값의 95% 퍼센타일을 기준 스케일로 사용
# ─────────────────────────────────────────────────────
def _draw_grid(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    vals: np.ndarray,
    nx: int, ny: int,
    vmax: float | None,
    gamma: float,
    gain: float,
    t_min: float,
    snake: bool,
):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    n = nx * ny

    vals = np.asarray(vals, dtype=float)
    if vals.size < n:
        vals = np.pad(vals, (0, n - vals.size), mode='constant')
    vals = vals[:n]

    grid = vals.reshape(ny, nx)  # 행우선 가정
    if snake:
        for j in range(ny):
            if j % 2 == 1:
                grid[j] = grid[j][::-1]

    # 자동 스케일
    if vmax is None:
        pos = grid[grid > 0]
        ref = np.percentile(pos, 95) if pos.size else 1.0
    else:
        ref = float(vmax)
    if ref <= 1e-9:
        ref = 1.0

    for j in range(ny):
        yj0 = y0 + int(j * h / ny)
        yj1 = y0 + int((j + 1) * h / ny)
        for i in range(nx):
            xi0 = x0 + int(i * w / nx)
            xi1 = x0 + int((i + 1) * w / nx)
            v = grid[j, i]
            t = (v / ref)
            t = (t ** gamma) * gain
            t = max(t, t_min)
            t = np.clip(t, 0.0, 1.0)
            color = _bgr_from_t(t)
            cv2.rectangle(img, (xi0, yj0), (xi1, yj1), color, thickness=-1)

# ─────────────────────────────────────────────────────
# 메인 오버레이 함수
# - 센서별(격자)로 칠함
# - gamma<1, gain>1, t_min>0 으로 옅은 색 문제 해결
# - vmax=None: 자동 스케일(95% 퍼센타일)
# ─────────────────────────────────────────────────────
def overlay(
    img: np.ndarray,
    left_tactile: np.ndarray,
    right_tactile: np.ndarray,
    *,
    vmax: float | None = None,   # 전체 스케일 고정값. 자동 스케일 쓰려면 None
    gamma: float = 0.6,          # <1 이면 약한 값 부스트
    gain: float = 1.25,          # 전체 증폭
    t_min: float = 0.06,         # 최소 표시 강도
    snake: bool = False,         # 센서 행 지그재그 배치면 True
) -> np.ndarray:
    h, w = img.shape[:2]
    # 오른손 박스 좌우 반전
    right_boxes = {r: (w - 1 - x1, y0, w - 1 - x0, y1)
                   for r, (x0, y0, x1, y1) in boxes.items()}

    # 왼손
    for name, box in boxes.items():
        sl = _region_slices[name]
        cnt = sl.stop - sl.start
        nx, ny = _grid_for_count(cnt)
        _draw_grid(img, box, left_tactile[sl], nx, ny, vmax, gamma, gain, t_min, snake)

    # 오른손
    for name, box in right_boxes.items():
        sl = _region_slices[name]
        cnt = sl.stop - sl.start
        nx, ny = _grid_for_count(cnt)
        _draw_grid(img, box, right_tactile[sl], nx, ny, vmax, gamma, gain, t_min, snake)

    return img
