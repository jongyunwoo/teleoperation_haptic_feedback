import cv2
import numpy as np
import json
import os
#손가락 tactile 센서 개수
touch_dict = {
    "fingerone_tip_touch":    9,   "fingerone_top_touch":    96,  "fingerone_palm_touch":   80,
    "fingertwo_tip_touch":    9,   "fingertwo_top_touch":    96,  "fingertwo_palm_touch":   80,
    "fingerthree_tip_touch":  9,   "fingerthree_top_touch":  96,  "fingerthree_palm_touch": 80,
    "fingerfour_tip_touch":   9,   "fingerfour_top_touch":   96,  "fingerfour_palm_touch":  80,
    "fingerfive_tip_touch":   9,   "fingerfive_top_touch":   96,  "fingerfive_middle_touch": 9,
    "fingerfive_palm_touch":  96,  "palm_touch":            112
}

# 클릭한 좌표 저장
left_json_path = "left_hand_outline.json"
right_json_path = "right_hand_outline.json"

points = []       # 현재 클릭 중인 좌표
current_hand = "left"  # 기본은 왼손(left)
# 오버레이 위치 보정(이미지 상의 위치 좌표)
x_offset = 250
y_offset = 30

#tactile 센서 이미지 위치 지정
#센서-픽셀 1:1 매핑
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

    "palm_touch":             (69+x_offset,94+y_offset, 82+x_offset,101+y_offset),
}

#tactile sensor 별 index slicing 생성
region_slice = {}
start = 0
for region, cnt in touch_dict.items():
    region_slice[region] = slice(start, start + cnt) 
    start += cnt

#tactile sensor row & col
GRID_BY_COUNT = {
    9:   (3, 3),
    96:  (12, 8),   
    80:  (10, 8),   
    112: (14, 8),   
}

#센서 개수 n개 받아서 직사각형 형태로 grid 
def grid_for_count(n: int) -> tuple[int, int]:
    if n in GRID_BY_COUNT:
        return GRID_BY_COUNT[n] # ex) 9 --> nx = 3, ny =3, 96 --> nx = 12, ny = 8
    
    #정의되어 있지 않은 경우 정사각형 형태로 반환
    nx = int(np.ceil(np.sqrt(n))) # n의 제곱근을 구해서 가로(nx) 칸수를 정함
    ny = int(np.ceil(n / nx)) # 세로(ny)는 전체 센서 개수를 가로 칸수로 나눠서 결정 
    return nx, ny


STOP  = np.array([0.0, 0.25, 0.50, 0.75, 1.00], dtype=float) # 색상 매핑 기준점 (강도 구간)

COLORS = np.array([
    [255,   0,   0],  # Blue   (BGR)
    [  0, 255,   0],  # Green
    [  0, 255, 255],  # Yellow
    [  0, 165, 255],  # Orange
    [  0,   0, 255],  # Red
], dtype=float)

def bgr_from_t(t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0)) #tactile sensor 값을 0.0~1.0으로 cliping
    i = np.searchsorted(STOP, t, side="right") - 1 #t : tactile sensor값이 속하는 지점을 찾음. ex) t가 0.4일 때 (0.4 초과 첫 값은 0.50)-1 해서 i=1 (즉, 0.25~0.50 구간에 속한다고 판정)
    i = np.clip(i, 0, len(STOP) - 2) #인덱스 i가 범위를 벗어나지 않도록 제한. (예: t=1.0일 경우 인덱스가 배열 끝을 넘어가지 않게 처리)
    t0, t1 = STOP[i], STOP[i + 1] 
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0) #t가 현재 구간에서 얼마나 진행됐는지 비율(보간 계수 a) 계산
    c = (1.0 - a) * COLORS[i] + a * COLORS[i + 1] # 현재 구간의 색상(COLORS[i])과 다음 구간 색상(COLORS[i+1]) 사이를 선형 보간. 만약 a=0.6이면 두 색을 40:60으로 섞어서 중간 색상 생성
    return int(c[0]), int(c[1]), int(c[2]) # BGR (정수형) 튜플로 변환해 반환

def draw_grid(
    img: np.ndarray, #overlay할 이미지
    box: tuple[int, int, int, int], # (x0, y0, x1, y1) 형태의 사각형 좌표 
    vals: np.ndarray, # 촉각 센서 값 배열 (1D array)
    nx: int, ny: int, # 그리드를 가로(nx) × 세로(ny) 몇 개로 나눌지
    vmax: float | None, # 색상 스케일의 최대 기준값 (None이면 자동으로 95% 상위값 사용)
    gamma: float, # 보정 값. tactile sensor 값이 약할 경우에도 눈에 잘 띄게
    gain: float, # 밝기 값. tactile sensor 값들을 색으로 변경하기 전에 전반적으로 밝기(강도)를 조정
    t_min: float, # 계산된 값이 t_min보다 작으면 강제로 t_min로 올려서 표시.
    snake: bool, # 행(row)마다 배선 방식 때문에 지그재그 형태로 데이터가 들어옴. True일 시 반전 시켜 이를 보정 
):
    x0, y0, x1, y1 = box # tactile sensor 이미지 상의 픽셀 좌표 범위 
    w = x1 - x0 # 넓이(가로)
    h = y1 - y0 # 높이(세로)
    n = nx * ny # sensor 값 개수

    vals = np.asarray(vals, dtype=float)
    if vals.size < n:
        vals = np.pad(vals, (0, n - vals.size), mode='constant') # 센서 개수가 이미지상에 그릴 좌표 개수보다 많을 경우 padding 
    vals = vals[:n] # 만약 센서 값이 격자 셀 수보다 많으면 초과분은 잘라냄.
    grid = vals.reshape(ny, nx)  # 행우선
    if snake:
        for j in range(ny):
            if j % 2 == 1:
                grid[j] = grid[j][::-1]

    # 자동 스케일
    if vmax is None:
        pos = grid[grid > 0] # tactile sensor 값이 0이상인 것만 취급
        ref = np.percentile(pos, 95) if pos.size else 1.0 # pos중 95% 분위수(percentile)를 기준으로 사용. 이상치로 인해 scale이 깨지는 것을 방지
    else:
        ref = float(vmax) # vmax가 정의 되었을 경우 vmax 그대로 기준으로 사용
    if ref <= 1e-9: # 너무 작을 경우 0으로 나누는 것을 방지하기 위해 
        ref = 1.0 # 기준 1.0으로 설정 

    for j in range(ny):  
        yj0 = y0 + int(j * h / ny) # 전체 높이(h)를 ny로 나누어 j번째 행 시작 위치 산출
        yj1 = y0 + int((j + 1) * h / ny) # 현재 행(j)의 끝 y좌표 계산: 다음 행 시작 위치가 곧 현재 행의 끝
        for i in range(nx): 
            xi0 = x0 + int(i * w / nx) # 현재 열(i)의 시작 x좌표 계산: 전체 폭(w)을 nx로 나누어 i번째 열 시작 위치 산출
            xi1 = x0 + int((i + 1) * w / nx) # 현재 열(i)의 끝 x좌표 계산: 다음 열 시작 위치가 곧 현재 열의 끝 
            v = grid[j, i] # 현재 셀 (j행, i열)의 센서 값 가져오기
            t = (v / ref)
            t = (t ** gamma) * gain
            t = max(t, t_min)
            t = np.clip(t, 0.0, 1.0)
            color = bgr_from_t(t)
            cv2.rectangle(img, (xi0, yj0), (xi1, yj1), color, thickness=-1)

def mouse_callback(event, x, y, flags, param):
    global points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 클릭 시
        points.append((x, y))
        print(f"[{current_hand}] 점 추가: {x}, {y}")

    # 실시간 표시
    temp_img = display_img.copy()
    for p in points:
        cv2.circle(temp_img, p, 3, (0, 255, 0), -1)
    if len(points) > 1:
        cv2.polylines(temp_img, [np.array(points, np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
    cv2.imshow("Draw Outline", temp_img)

def save_points(points, path):
    with open(path, "w") as f:
        json.dump(points, f)
    print(f"[저장 완료] {path}")

def load_points(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            pts = json.load(f)
        print(f"[불러오기 완료] {path}")
        return [(int(x), int(y)) for x, y in pts]
    return []

def draw_hand_outline(img: np.ndarray, points: list[tuple[int,int]], color=(255,255,255)):
    """ 저장된 점으로 윤곽선을 그림 """
    if not points: return img
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    return img


def overlay(
    img: np.ndarray,
    left_tactile: np.ndarray,
    right_tactile: np.ndarray,
    *,
    vmax: float | None = None,   
    gamma: float = 0.6,          
    gain: float = 1.25,          
    t_min: float = 0.06,         
    snake: bool = False,        
) -> np.ndarray:
    h, w = img.shape[:2]
    # 오른손 박스 좌우 반전
    right_boxes = {r: (w - 1 - x1, y0, w - 1 - x0, y1)
                   for r, (x0, y0, x1, y1) in boxes.items()}

    # 왼손
    for name, box in boxes.items():
        sl = region_slice[name]
        cnt = sl.stop - sl.start
        nx, ny = grid_for_count(cnt)
        draw_grid(img, box, left_tactile[sl], nx, ny, vmax, gamma, gain, t_min, snake)

    # 오른손
    for name, box in right_boxes.items():
        sl = region_slice[name]
        cnt = sl.stop - sl.start
        nx, ny = grid_for_count(cnt)
        draw_grid(img, box, right_tactile[sl], nx, ny, vmax, gamma, gain, t_min, snake)

    return img

def draw_hand_outline(img: np.ndarray, points: list[tuple[int,int]], color=(255,255,255)):
    """ 저장된 점으로 윤곽선을 그림 """
    if not points: return img
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    return img

if __name__ == "__main__":
    # 배경 이미지 (실제는 overlay 결과 이미지 사용 가능)
    img = cv2.imread('/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0004/colors/000000_color_0.jpg')
    display_img = img.copy()

    # 좌/우 윤곽선 불러오기
    left_points = load_points(left_json_path)
    right_points = load_points(right_json_path)

    if left_points:
        display_img = draw_hand_outline(display_img, left_points, color=(0,255,0))  # 왼손은 초록색
    if right_points:
        display_img = draw_hand_outline(display_img, right_points, color=(255,0,0))  # 오른손은 파란색

    cv2.namedWindow("Draw Outline")
    cv2.setMouseCallback("Draw Outline", mouse_callback)

    print("키 설명: [l]왼손 / [r]오른손 선택, [Enter]저장, [c]초기화, [ESC]종료")

    while True:
        cv2.imshow("Draw Outline", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):  # 왼손 선택
            current_hand = "left"
            points = []
            print("왼손 윤곽선 지정 시작")

        elif key == ord('r'):  # 오른손 선택
            current_hand = "right"
            points = []
            print("오른손 윤곽선 지정 시작")

        elif key == 13:  # Enter: 현재 손 윤곽선 저장
            if points:
                if current_hand == "left":
                    save_points(points, left_json_path)
                    left_points = points.copy()
                else:
                    save_points(points, right_json_path)
                    right_points = points.copy()
                display_img = img.copy()
                if left_points: display_img = draw_hand_outline(display_img, left_points, (0,255,0))
                if right_points: display_img = draw_hand_outline(display_img, right_points, (255,0,0))
                points = []

        elif key == ord('c'):  # 현재 점 초기화
            points = []
            print("현재 손 윤곽선 점 초기화")

        elif key == 27:  # ESC 종료
            break

    cv2.destroyAllWindows()