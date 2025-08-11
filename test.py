# import cv2
# import numpy as np
# import os
# import os.path as osp
# from glob import glob

# # 경로 설정
# root = "/home/scilab/Documents/teleoperation/yolo/output_robot_hand"
# img_dir = osp.join(root, "valid/images")
# lbl_dir = osp.join(root, "valid/labels")
# vis_dir = osp.join(root, "vis")
# os.makedirs(vis_dir, exist_ok=True)

# # 컬러와 알파(반투명) 설정
# overlay_color = (0, 255, 0)  # 녹색 마스크
# alpha = 0.4                 # 반투명도

# # 클래스 리스트 (YOLO data.yaml 과 동일 순서)
# class_names = ["left robot hand", "right robot hand"]

# def draw_segmentation(img, txt_path):
#     h, w = img.shape[:2]
#     # 빈 마스크 레이어
#     overlay = img.copy()
#     # 다각형 레이블 읽기
#     with open(txt_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             cls_id = int(parts[0])
#             coords = list(map(float, parts[1:]))
#             pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
#                             for i in range(0, len(coords), 2)], np.int32)
#             if pts.shape[0] < 3:
#                 continue
#             # 채우기
#             cv2.fillPoly(overlay, [pts], overlay_color)
#             # 외곽선
#             cv2.polylines(img, [pts], True, (0,0,255), 2)
#             # 클래스 이름 표시 (다각형 첫점 위)
#             x0, y0 = pts[0]
#             cv2.putText(img, class_names[cls_id], (x0, y0-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
#     # 반투명 합성
#     cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
#     return img

# # 실행
# for img_path in sorted(glob(osp.join(img_dir, "*.jpg"))):
#     name = osp.splitext(osp.basename(img_path))[0]
#     lbl_path = osp.join(lbl_dir, name + ".txt")
#     img = cv2.imread(img_path)
#     vis = draw_segmentation(img, lbl_path)
#     # 화면에 표시
#     cv2.imshow("Segmentation Check", vis)
#     key = cv2.waitKey(0) & 0xFF
#     if key == ord('q'):
#         break
#     # 저장 (원하면)
#     cv2.imwrite(osp.join(vis_dir, f"{name}_vis.jpg"), vis)

# cv2.destroyAllWindows()


# # import os
# # from glob import glob

# # LABEL_DIRS = [
# #     "/home/scilab/Documents/teleoperation/yolo/datasets/peg_in_hole/labels/train",
# #     "/home/scilab/Documents/teleoperation/yolo/datasets/peg_in_hole/labels/val"
# # ]

# # for lbl_dir in LABEL_DIRS:
# #     for txt_path in glob(os.path.join(lbl_dir, "*.txt")):
# #         lines = []
# #         with open(txt_path, 'r') as f:
# #             for line in f:
# #                 parts = line.strip().split()
# #                 cls = int(parts[0]) - 1           # ID 재매핑: 1→0, 2→1
# #                 coords = parts[1:]
# #                 # 클래스 ID가 유효범위(0~nc-1)를 벗어나면 경고하고 건너뜁니다
# #                 if cls < 0 or cls > 1:
# #                     print(f"⚠️  {txt_path}: 클래스 {cls+1} → 재매핑 불가능, 스킵")
# #                     continue
# #                 lines.append(" ".join([str(cls)] + coords))
# #         # 덮어쓰기
# #         with open(txt_path, 'w') as f:
# #             f.write("\n".join(lines))
# #         print(f"✅  Fixed labels in {txt_path}")

# import os, glob, json, collections, re

# JSON_DIR = "/home/scilab/Documents/teleoperation/yolo/datasets/raw_total/labelme/labeljson"

# # 당신의 기준 이름들
# CANON = [
#  "orange block","green block","purple block","pegging black block","Hole box",
#  "small plastic cup","large plastic cup","left robot hand","right robot hand"
# ]

# def norm(s):
#     # 공백/대소문자/밑줄/중복스페이스 정규화(라벨 편차 방어)
#     s = s.replace('_',' ').strip()
#     s = re.sub(r'\s+',' ', s)
#     return s

# canon_norm = {norm(x):x for x in CANON}

# seen = collections.Counter()
# unknown = collections.Counter()

# for p in glob.glob(os.path.join(JSON_DIR, "**/*.json"), recursive=True):
#     try:
#         with open(p, "r") as f: data = json.load(f)
#     except: continue
#     for sh in data.get("shapes", []):
#         raw = sh.get("label","")
#         n = norm(raw)
#         if n in canon_norm:
#             seen[canon_norm[n]] += 1
#         else:
#             unknown[raw] += 1

# print("[OK] counts:", dict(seen))
# print("[WARN] unknown labels:", dict(unknown))
# import os

# label_dir = "/home/scilab/Documents/teleoperation/yolo/datasets/raw_total/labels/total"

# for fname in os.listdir(label_dir):
#     if not fname.endswith(".txt"):
#         continue
    
#     fpath = os.path.join(label_dir, fname)
#     with open(fpath, "r") as f:
#         lines = f.readlines()

#     new_lines = []
#     changed = False
#     for line in lines:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         # YOLO format: class x y w h ...
#         if parts[0] == "8":
#             parts[0] = "7"
#             changed = True
#         new_lines.append(" ".join(parts) + "\n")

#     if changed:
#         with open(fpath, "w") as f:
#             f.writelines(new_lines)
#         print(f"Updated: {fname}")

# print("변경 완료!")

import os
import shutil

src_dir = "/home/scilab/Documents/teleoperation/yolo/datasets/raw_total/labels/total"
train_dir = os.path.join(os.path.dirname(src_dir), "train")
val_dir = os.path.join(os.path.dirname(src_dir), "val")

# 폴더 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# val 범위 목록
val_ranges = [
    range(81, 101),
    range(181, 201),
    range(281, 301),
    range(381, 401),
    range(681, 701),
    range(781, 801)
]

def in_val_set(num):
    return any(num in r for r in val_ranges)

for fname in os.listdir(src_dir):
    if not fname.endswith(".txt"):
        continue
    
    # 파일 이름에서 숫자 추출
    name_no_ext = os.path.splitext(fname)[0]
    try:
        num = int(name_no_ext)
    except ValueError:
        print(f"Skip: {fname} (숫자 아님)")
        continue
    
    # 이동
    if in_val_set(num):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(val_dir, fname))
    else:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(train_dir, fname))

print("train / val 분리 완료!")
