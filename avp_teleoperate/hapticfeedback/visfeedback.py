import cv2
import numpy as np

# 150×150 기준 영역별 센서 개수 & 박스 정의
touch_dict = {
    "fingerone_tip_touch":    9,   "fingerone_top_touch":    96,  "fingerone_palm_touch":   80,
    "fingertwo_tip_touch":    9,   "fingertwo_top_touch":    96,  "fingertwo_palm_touch":   80,
    "fingerthree_tip_touch":  9,   "fingerthree_top_touch":  96,  "fingerthree_palm_touch": 80,
    "fingerfour_tip_touch":   9,   "fingerfour_top_touch":   96,  "fingerfour_palm_touch":  80,
    "fingerfive_tip_touch":   9,   "fingerfive_top_touch":   96,  "fingerfive_middle_touch": 9,
    "fingerfive_palm_touch":  96,  "palm_touch":            112
}
offset = 200
boxes = {
    "fingerone_tip_touch":    (114+offset,50,116+offset,52),
    "fingerone_top_touch":    (112+offset,55,119+offset,66),
    "fingerone_palm_touch":   (108+offset,75,115+offset,84),
    "fingertwo_tip_touch":    (96+offset,38,98+offset,40),
    "fingertwo_top_touch":    (93+offset,45,100+offset,56),
    "fingertwo_palm_touch":   (90+offset,75,97+offset,84),
    "fingerthree_tip_touch":  (73+offset,36,75+offset,38),
    "fingerthree_top_touch":  (71+offset,41,78+offset,52),
    "fingerthree_palm_touch": (73+offset,75,80+offset,84),
    "fingerfour_tip_touch":   (52+offset,38,54+offset,40),
    "fingerfour_top_touch":   (50+offset,44,57+offset,55),
    "fingerfour_palm_touch":  (53+offset,75,60+offset,84),
    "fingerfive_tip_touch":   (31+offset,64,33+offset,66),
    "fingerfive_top_touch":   (30+offset,69,37+offset,80),
    "fingerfive_middle_touch":(35+offset,90,37+offset,92),
    "fingerfive_palm_touch":  (33+offset,100,40+offset,111),
    "palm_touch":             (69+offset,94,82+offset,101),
}

# 각 region → tactile_array 인덱스 슬라이스 계산
_region_slices = {}
_start = 0
for region, cnt in touch_dict.items():
    _region_slices[region] = slice(_start, _start+cnt)
    _start += cnt

def overlay(img, left_tactile, right_tactile, max_val=1e-10):
    """
    img           : BGR image
    left_tactile  : numpy array shape (1062,)
    right_tactile : numpy array shape (1062,)
    """
    h,w = img.shape[:2]

    # 오른손 박스 좌우 반전 계산
    right_boxes = {r:(w-1-x1, y0, w-1-x0, y1)
                   for r,(x0,y0,x1,y1) in boxes.items()}

    # 각 박스마다 센서 값 꺼내서 색 구하기
    for side, tactile, boxmap in (
        ("left",  left_tactile,  boxes),
        ("right", right_tactile, right_boxes)
    ):
        for name, box in boxmap.items():
            x0,y0,x1,y1 = box
            sl = _region_slices[name]
            v = float(np.max(tactile[sl]))
            t = np.clip(v/max_val, 0.0, 1.0)
            # blue→red 보간
            color = (int((1-t)*255), 0, int(t*255))
            overlayimg=cv2.rectangle(img, (x0,y0), (x1,y1), color, thickness=-1)
        
    return overlayimg
