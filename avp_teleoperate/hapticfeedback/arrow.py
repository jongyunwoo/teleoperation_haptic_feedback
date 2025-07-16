import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) 이미지 로드 (BGR)
img_bgr = cv2.imread("/home/scilab/Documents/teleoperation/images/6.jpg")
# (필요하다면 리사이즈)
# img_bgr = cv2.resize(img_bgr, (848, 480))

# 2) 좌표 예시 (실제값으로 바꿔주세요)
cx, cy = 400, 200    # 손 중심
tx, ty = 420, 150    # 손 끝
ox, oy = 300, 300    # 물체

# 3) 화살표 그리기
vis = img_bgr.copy()
#   손 중심 → 손 끝 (파란색)
cv2.arrowedLine(vis, (cx, cy), (tx, ty),
                color=(255, 0, 0),    # BGR: 파랑
                thickness=2,
                tipLength=0.2)
#   손 중심 → 물체 (초록색)
cv2.arrowedLine(vis, (cx, cy), (ox, oy),
                color=(0, 255, 0),    # BGR: 초록
                thickness=2,
                tipLength=0.2)

# 4) Matplotlib으로 RGB로 표시
img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()