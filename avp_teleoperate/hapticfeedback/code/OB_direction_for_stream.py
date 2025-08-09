import cv2
import numpy as np
import os
import os.path as osp
from glob import glob
import natsort

def get_obb_from_yoloseg(txt_path, image_size):
    h, w = image_size
    obbs = []
    centers = []
    angles = []

    if not osp.exists(txt_path):
        return obbs, centers, angles

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
        (cx, cy), (rw, rh), angle = rect

        if rw < rh:
            angle += 90

        box = cv2.boxPoints(rect).astype(np.int32)
        obbs.append(box)
        centers.append((int(cx), int(cy)))
        angles.append(angle)

    return obbs, centers, angles

# ======================== #
# ===== MAIN SCRIPT ===== #
# ======================== #

arrow_length = 100
root_path = "/home/smg/autodistill/dataset/previous/outputs_block/valid"
img_dir = osp.join(root_path, "images")
label_dir = osp.join(root_path, "labels")
output_dir = osp.join(root_path, "output_drawn")
os.makedirs(output_dir, exist_ok=True)

img_paths = natsort.natsorted(glob(osp.join(img_dir, "*.jpg")))

# 직전 프레임 angle 저장용 (객체 1개만 처리하는 경우를 기본으로)
previous_angle = None

for idx, img_path in enumerate(img_paths):
    filename = osp.basename(img_path)
    name_wo_ext = osp.splitext(filename)[0]
    label_path = osp.join(label_dir, name_wo_ext + ".txt")

    rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = rgb_img.shape[:2]

    obbs, centers, angles = get_obb_from_yoloseg(label_path, (h, w))
    draw_img = rgb_img.copy()

    for box, center, angle in zip(obbs, centers, angles):
        # unwrap 각도 보정 (단일 객체 기준)
        if previous_angle is not None:
            diff = angle - previous_angle
            if diff > 90:
                angle -= 180
            elif diff < -90:
                angle += 180

        previous_angle = angle  # 현재 각도를 다음 프레임용으로 저장

        # 각도 → 라디안
        theta = np.deg2rad(angle)

        # 중심 기준으로 angle 반대 방향 (물체 위쪽)
        tip_x = int(center[0] - arrow_length * np.cos(theta))
        tip_y = int(center[1] - arrow_length * np.sin(theta))
        tip = (tip_x, tip_y)

        # OBB 및 화살표 그리기
        cv2.drawContours(draw_img, [box], 0, color=(255, 0, 0), thickness=2)
        cv2.arrowedLine(draw_img, center, tip, color=(255, 0, 0), thickness=2, tipLength=0.3)

    # 저장
    save_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    out_path = osp.join(output_dir, f"direction_obb_image_{idx:04d}.jpg")
    cv2.imwrite(out_path, save_img)

    print(f"Saved: {out_path}")
