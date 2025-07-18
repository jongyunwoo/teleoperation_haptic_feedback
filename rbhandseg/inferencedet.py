from autodistill_yolov8 import YOLOv8
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

img = cv2.imread("/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0002/colors/001000_color_0.jpg")
img = cv2.resize(img, (480, 848))

# 1) 모델 로드
model = YOLOv8(
"/home/scilab/Documents/teleoperation/runs/detect/train/weights/best.pt")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.model = model.model.to(device)
start = time.time()
# 2) 예측 (기본 옵션 그대로)
pred = model.predict(
    '/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0002/colors/001000_color_0.jpg'
)
end = time.time()
print(end-start)
pred_list = list(pred)


res = pred_list[0]  # Results 객체

# 4) 시각화: plot()이 그려준 BGR 이미지 반환
annotated_bgr = res.plot()  

# 5) matplotlib에 RGB로 표시
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6,6))
plt.imshow(annotated_rgb)
plt.axis('off')
plt.show()

