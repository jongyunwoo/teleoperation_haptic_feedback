# inference.py

from autodistill_yolov8 import YOLOv8
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO  # 또는 YOLOv8 래퍼

def main():
    img = cv2.imread("/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0004/colors/000593_color_0.jpg")
    img = cv2.resize(img, (640, 640))
    # 1) 모델 로드
    model = YOLO(
    "/home/scilab/Documents/teleoperation/runs/segment/train/weights/best.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.model = model.model.to(device)
    start = time.time()
    # 2) 예측 (기본 옵션 그대로)
    pred = model.predict(
        img
    )
    end = time.time()
    print(end-start)
    pred_list = list(pred)
    
    res = pred_list[0]

    # 3) 원본 처리 이미지와 마스크 가져오기
    img_proc = res.orig_img                # (H_proc, W_proc, 3), BGR
    masks_tensor = res.masks.data          # (N, H_proc, W_proc)

    # 4) 강제 리사이즈: (H_proc, W_proc) → (480, 848)
    TARGET_H, TARGET_W = 480, 848

    # 4-1) 이미지 리사이즈 (BGR)
    img_resized = cv2.resize(
        img_proc,
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_LINEAR
    )

    # 4-2) 마스크 리사이즈
    #    torch.Tensor이면 numpy로 변환
    if isinstance(masks_tensor, torch.Tensor):
        mask_np = masks_tensor.cpu().numpy()
        print(1)
    else:
        mask_np = masks_tensor
        print(2)
    resized_masks = []
    for m in mask_np:
        # bool → uint8 0/255 배열
        m_uint8 = (m.astype(np.uint8) * 255)
        # nearest neighbor 리사이즈
        m_rs = cv2.resize(
            m_uint8,
            (TARGET_W, TARGET_H),
            interpolation=cv2.INTER_NEAREST
        )
        resized_masks.append(m_rs > 0)

    # (N, 480, 848)
    resized_masks = np.stack(resized_masks, axis=0)

    #5) 합쳐서 2D boolean mask
    combined_mask = resized_masks.any(axis=0)  # (480, 848)

    # 6) 가장 위쪽(y 최소) 픽셀 좌표 구하기
    ys, xs = np.where(combined_mask)
    if len(ys) == 0:
        print("검출된 마스크가 없습니다.")
        return

    min_y = int(ys.min())
    xs_at_min_y = xs[ys == min_y]
    mean_x = int(xs_at_min_y.mean())

    print(f"Topmost pixel → x: {mean_x}, y: {min_y}")
    
    
    mean_y = int(ys.mean())
    mean_x = int(xs.mean())
    print(f"Centroid → x: {mean_x}, y: {mean_y}")
    
    
    # 7) 시각화: matplotlib에 RGB로 표시
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 5))
    plt.imshow(img_rgb)
    plt.scatter(mean_x, min_y, s=150, c='red', marker='x')
    plt.scatter(mean_x, mean_y, s=150, c='red', marker='x')
    plt.axis('off')
    plt.show()
    # hand_info = []  # [(idx, mean_x, mean_y, side), ...]
    # img_center_x = TARGET_W / 2

    # for idx, m in enumerate(resized_masks):
    #     ys, xs = np.where(m)
    #     if ys.size == 0:
    #         continue

    #     mean_y = int(ys.mean())
    #     mean_x = int(xs.mean())
    #     side = "left" if mean_x < img_center_x else "right"
    #     hand_info.append((idx, mean_x, mean_y, side))

    # if not hand_info:
    #     print("검출된 마스크가 없습니다.")
    #     return

    # # 7) 결과 출력
    # for idx, mx, my, side in hand_info:
    #     print(f"Mask #{idx} → side: {side}, centroid x: {mx}, y: {my}")

    # # 8) 시각화: 모든 손 위치에 표시
    # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(8, 5))
    # plt.imshow(img_rgb)
    # for idx, mx, my, side in hand_info:
    #     color = 'blue' if side=='left' else 'green'
    #     plt.scatter(mx, my, s=150, marker='x', label=f"{side} hand")
    # plt.legend()
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    main()
