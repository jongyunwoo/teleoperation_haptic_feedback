import cv2
import numpy as np
import os.path as osp
import os

import matplotlib.pyplot as plt

def yolo_seg_to_mask(txt_path, image_size, save_path=None):
    """
    YOLO-SEG txt 파일을 읽어 세그멘테이션 마스크를 생성합니다.

    Args:
        txt_path (str): YOLO-SEG 포맷의 .txt 파일 경로
        image_size (tuple): (height, width) — 마스크 이미지 크기
        save_path (str, optional): 저장할 경로 지정 시, PNG 이미지로 저장
    Returns:
        mask (np.ndarray): 3채널 마스크 이미지 (255로 채워진 폴리곤 영역)
    """
    h, w = image_size
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) <= 5:
            continue  # 중심좌표 + 2개 점 이하인 경우 무시

        # polygon 좌표만 추출 (x1 y1 x2 y2 ...)
        poly_coords = list(map(float, parts[5:]))
        points = []

        for i in range(0, len(poly_coords), 2):
            x = int(float(poly_coords[i]) * w)
            y = int(float(poly_coords[i+1]) * h)
            points.append([x, y])

        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts, (255, 255, 255))

    if save_path:
        cv2.imwrite(save_path, mask)

    return mask

def get_obb_from_yoloseg(txt_path, image_size):
    """
    YOLO-SEG 포맷에서 OBB(Oriented Bounding Box) 추출

    Args:
        txt_path (str): YOLO-SEG 텍스트 경로
        image_size (tuple): (height, width) — 이미지 크기

    Returns:
        obbs (list): 각 OBB의 꼭짓점 4개 좌표 (np.ndarray, shape=(4,2))
    """
    h, w = image_size
    obbs = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) <= 5:
            continue

        coords = list(map(float, parts[5:]))
        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])

        cnt = np.array(points, dtype=np.int32)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        obbs.append(box)

    return obbs


## parameter 
arrow_length = 100

root_path = "/home/smg/autodistill/dataset/previous/outputs_block/valid"
rgb_img_path = osp.join(root_path, 'images', 'input_img_0001.jpg')
mask_txt_path = osp.join(root_path, 'labels', 'input_img_0001.txt')

rgb_img = cv2.cvtColor(cv2.imread(rgb_img_path), cv2.COLOR_BGR2RGB)
h, w = rgb_img.shape[0], rgb_img.shape[1]

mask = yolo_seg_to_mask(mask_txt_path, (h,w))

obbs = get_obb_from_yoloseg(mask_txt_path, (h,w))

## vis overlay obb 
obb_img = rgb_img.copy()

for box in obbs:
    cv2.drawContours(obb_img, [box], 0 , color=(255,0,0), thickness=2)


## vis direction 
direct_img = rgb_img.copy()

for box in obbs:
    center_x = int(np.mean(box[:,0]))
    center_y = int(np.mean(box[:,1]))
    center = (center_x, center_y)
    
    tip = (center_x, center_y - arrow_length)
    
    cv2.arrowedLine(direct_img, center, tip, color=(255,0,0), thickness=2, tipLength=0.3)
    
plt.subplot(2,2,1)
plt.imshow(rgb_img)
plt.subplot(2,2,2)
plt.imshow(mask)
plt.subplot(2,2,3)
plt.imshow(obb_img)
plt.subplot(2,2,4)
plt.imshow(direct_img)
plt.tight_layout()
plt.show()