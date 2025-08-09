import cv2
import numpy as np
import os
import os.path as osp
from glob import glob

# 경로 설정
root = "/home/scilab/Documents/teleoperation/yolo/output_robot_hand"
img_dir = osp.join(root, "valid/images")
lbl_dir = osp.join(root, "valid/labels")
vis_dir = osp.join(root, "vis")
os.makedirs(vis_dir, exist_ok=True)

# 컬러와 알파(반투명) 설정
overlay_color = (0, 255, 0)  # 녹색 마스크
alpha = 0.4                 # 반투명도

# 클래스 리스트 (YOLO data.yaml 과 동일 순서)
class_names = ["left robot hand", "right robot hand"]

def draw_segmentation(img, txt_path):
    h, w = img.shape[:2]
    # 빈 마스크 레이어
    overlay = img.copy()
    # 다각형 레이블 읽기
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
                            for i in range(0, len(coords), 2)], np.int32)
            if pts.shape[0] < 3:
                continue
            # 채우기
            cv2.fillPoly(overlay, [pts], overlay_color)
            # 외곽선
            cv2.polylines(img, [pts], True, (0,0,255), 2)
            # 클래스 이름 표시 (다각형 첫점 위)
            x0, y0 = pts[0]
            cv2.putText(img, class_names[cls_id], (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    # 반투명 합성
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img

# 실행
for img_path in sorted(glob(osp.join(img_dir, "*.jpg"))):
    name = osp.splitext(osp.basename(img_path))[0]
    lbl_path = osp.join(lbl_dir, name + ".txt")
    img = cv2.imread(img_path)
    vis = draw_segmentation(img, lbl_path)
    # 화면에 표시
    cv2.imshow("Segmentation Check", vis)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    # 저장 (원하면)
    cv2.imwrite(osp.join(vis_dir, f"{name}_vis.jpg"), vis)

cv2.destroyAllWindows()


# import os
# from glob import glob

# LABEL_DIRS = [
#     "/home/scilab/Documents/teleoperation/yolo/datasets/peg_in_hole/labels/train",
#     "/home/scilab/Documents/teleoperation/yolo/datasets/peg_in_hole/labels/val"
# ]

# for lbl_dir in LABEL_DIRS:
#     for txt_path in glob(os.path.join(lbl_dir, "*.txt")):
#         lines = []
#         with open(txt_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 cls = int(parts[0]) - 1           # ID 재매핑: 1→0, 2→1
#                 coords = parts[1:]
#                 # 클래스 ID가 유효범위(0~nc-1)를 벗어나면 경고하고 건너뜁니다
#                 if cls < 0 or cls > 1:
#                     print(f"⚠️  {txt_path}: 클래스 {cls+1} → 재매핑 불가능, 스킵")
#                     continue
#                 lines.append(" ".join([str(cls)] + coords))
#         # 덮어쓰기
#         with open(txt_path, 'w') as f:
#             f.write("\n".join(lines))
#         print(f"✅  Fixed labels in {txt_path}")
