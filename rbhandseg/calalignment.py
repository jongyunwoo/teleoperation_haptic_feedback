from autodistill_yolov8 import YOLOv8
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

def main():
    img = cv2.imread("/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0004/colors/000593_color_0.jpg")
    img = cv2.resize(img, (640, 640))

    # 1) 모델 로드
    model = YOLO("/home/scilab/Documents/teleoperation/runs/detect/train/weights/best.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.model = model.model.to(device)

    start = time.time()
    pred = model.predict(source=img)
    end = time.time()
    print(f"Inference Time: {end-start:.3f}s")

    res = pred[0]

    # 원본 이미지 및 마스크
    img_proc = res.orig_img
    masks_tensor = res.masks.data  # (N, H, W)

    # 타겟 리사이즈
    TARGET_H, TARGET_W = 480, 848
    img_resized = cv2.resize(img_proc, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # 마스크 변환 및 리사이즈
    if isinstance(masks_tensor, torch.Tensor):
        mask_np = masks_tensor.cpu().numpy()
    else:
        mask_np = masks_tensor

    resized_masks = []
    for m in mask_np:
        m_uint8 = (m.astype(np.uint8) * 255)
        m_rs = cv2.resize(m_uint8, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        resized_masks.append(m_rs > 0)
    resized_masks = np.stack(resized_masks, axis=0)

    # 기울기 계산 및 시각화
    for idx, m in enumerate(resized_masks):
        m_uint8 = (m.astype(np.uint8) * 255)

        # 외곽선 추출
        contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        # 가장 큰 contour 선택
        contour = max(contours, key=cv2.contourArea)

        # 최소 외접 사각형 계산 (중심, 크기, 회전각)
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        # angle 보정: OpenCV는 -90~0 범위
        if w < h:
            angle = angle
        else:
            angle = angle + 90

        print(f"[Mask #{idx}] 중심: ({int(cx)}, {int(cy)}), 기울기: {angle:.2f}°")

        # 회전 박스 시각화
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_resized, [box], 0, (0, 255, 0), 2)

        # 중심점 및 각도 표시
        cv2.circle(img_resized, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(img_resized, f"{angle:.1f} deg", (int(cx)-30, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 시각화 출력
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
