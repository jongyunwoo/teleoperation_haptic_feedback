import numpy as np
import cv2
import os.path as osp
import matplotlib.pyplot as plt



# def load_yolo_seg_labels(label_path):
#     polygons = []
#     class_ids = []
#     with open(label_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             # if len(parts) < 5:  # cls + 최소 3점(x,y)
#             #     continue
#             cls = int(parts[0])
#             coords = list(map(float, parts[1:]))
#             points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
#             polygons.append(points)
#             class_ids.append(cls)
#     return polygons, class_ids

# def group_polygons_by_class(polygons, class_ids, image_shape):
#     """
#     YOLO-Seg polygon 정보에서 cls_id별 마스크 그룹을 생성
#     반환: dict[int, List[np.ndarray]]  # 클래스별 마스크 리스트
#     """
#     h, w = image_shape
#     cls2masks = {}

#     for poly, cid in zip(polygons, class_ids):
#         # polygon 좌표를 denormalize
#         pts = []
#         for x, y in poly:
#             px = int(round(np.clip(x, 0.0, 1.0) * (w - 1)))
#             py = int(round(np.clip(y, 0.0, 1.0) * (h - 1)))
#             pts.append([px, py])
#         contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

#         # 빈 마스크에 채워서 저장
#         mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.fillPoly(mask, [contour], 1)
#         cls2masks.setdefault(cid, []).append(mask)

#     return cls2masks


# ## 두 마스크 연결 (with GPT) ##
# def denorm_polygon(poly_xy01, img):
#     # YOLO-Seg 좌표는 [0,1], 이를 픽셀 좌표로 변환
#     img_h, img_w = img.shape[:2]
#     pts = []
#     for x, y in poly_xy01:
#         px = int(round(np.clip(x, 0.0, 1.0) * (img_w - 1)))
#         py = int(round(np.clip(y, 0.0, 1.0) * (img_h - 1)))
#         pts.append([px, py])
#     # fillPoly는 (N,1,2) int32 형태 권장
#     return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)

# def build_class_union_masks(polygons, class_ids, img, do_close=True, close_kernel=15):
#     """
#     반환: dict[int, np.ndarray]  # 클래스별 union된 바이너리 마스크 (0/1, uint8)
#     """
#     # 클래스별 컨투어(폴리곤) 목록 모으기
#     img_h, img_w = img.shape[:2]
#     cls2contours = {}
#     for poly, cid in zip(polygons, class_ids):
#         cnt = denorm_polygon(poly, img)
#         cls2contours.setdefault(cid, []).append(cnt)

#     cls2mask = {}
#     for cid, contours in cls2contours.items():
#         mask = np.zeros((img_h, img_w), dtype=np.uint8)
#         # 여러 폴리곤을 한 번에 채우면 곧바로 union 효과
#         cv2.fillPoly(mask, contours, 1)   # value=1로 채움(바이너리)
#         if do_close and close_kernel > 0:
#             k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
#         cls2mask[cid] = mask
#     return cls2mask

# def connect_disjoint_class_regions(cls2contours, img_shape, distance_thresh=50):
#     """
#     각 클래스의 분리된 조각들 사이를 연결선으로 이어주는 마스크 생성기
#     """
#     h, w = img_shape
#     cls2mask = {}
    
#     for cid, contours in cls2contours.items():
#         mask = np.zeros((h, w), dtype=np.uint8)
#         centers = []

#         # 기존 폴리곤들 마스크화
#         for cnt in contours:
#             cv2.fillPoly(mask, [cnt], 1)
#             M = cv2.moments(cnt)
#             if M["m00"] > 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 centers.append((cx, cy))

#         # 조각들 간 거리 비교 → 가까우면 선 연결
#         for i in range(len(centers)):
#             for j in range(i + 1, len(centers)):
#                 pt1, pt2 = centers[i], centers[j]
#                 dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
#                 if dist < distance_thresh:
#                     cv2.line(mask, pt1, pt2, 1, thickness=5)  # 선 두께 조정 가능

#         cls2mask[cid] = mask
#     return cls2mask

# def soft_merge_regions(cls2contours, img_shape, distance_thresh=50,
#                        area_ratio_thresh=0.5, thickness=5, dilation_kernel=25):
#     h, w = img_shape
#     cls2mask = {}

#     for cid, contours in cls2contours.items():
#         base_mask = np.zeros((h, w), dtype=np.uint8)
#         line_mask = np.zeros((h, w), dtype=np.uint8)
#         regions = []

#         for cnt in contours:
#             cv2.fillPoly(base_mask, [cnt], 1)
#             M = cv2.moments(cnt)
#             if M["m00"] == 0:
#                 continue
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             area = cv2.contourArea(cnt)
#             regions.append({"center": (cx, cy), "area": area})

#         # 선 연결
#         for i in range(len(regions)):
#             for j in range(i + 1, len(regions)):
#                 r1, r2 = regions[i], regions[j]
#                 dist = np.linalg.norm(np.array(r1["center"]) - np.array(r2["center"]))
#                 if dist > distance_thresh:
#                     continue
#                 area_ratio = min(r1["area"], r2["area"]) / max(r1["area"], r2["area"])
#                 if area_ratio < area_ratio_thresh:
#                     continue
#                 cv2.line(line_mask, r1["center"], r2["center"], 1, thickness=thickness)

#         combined_mask = np.clip(base_mask + line_mask, 0, 1).astype(np.uint8)
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
#         dilated_mask = cv2.dilate(combined_mask, k)

#         contours_filled, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         filled_mask = np.zeros_like(dilated_mask)
#         cv2.fillPoly(filled_mask, contours_filled, 1)

#         cls2mask[cid] = filled_mask

#     return cls2mask

# def merge_masks_with_convex_hull(cls2masks):
#     """
#     다수의 마스크를 convex hull로 하나로 연결
#     """
#     merged_masks = {}
#     for cls_id, masks in cls2masks.items():
#         if len(masks) >= 2:
#             combined = np.sum(masks, axis=0)
#             combined = (combined > 0).astype(np.uint8)
#             contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if len(contours) == 0:
#                 return combined
#             all_points = np.concatenate(contours)
#             hull = cv2.convexHull(all_points)
#             result = np.zeros_like(combined)
#             cv2.fillConvexPoly(result, hull, 1)
#             merged_masks[cls_id] = result
            
#         else:
#             merged_masks[cls_id] = masks[0]
            
#     return merged_masks

# def merge_masks_with_distance_blending(masks, blend_width=20):
#     """
#     다수 마스크를 distance transform 기반으로 부드럽게 연결
#     """
#     base = np.sum(masks, axis=0)
#     base = (base > 0).astype(np.uint8)

#     # 각 개별 마스크 distance transform 후, 평균
#     blended_zone = np.zeros_like(base, dtype=np.float32)
#     for m in masks:
#         dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
#         blended_zone += dist

#     blended_zone /= len(masks)
#     blended_mask = (blended_zone < blend_width).astype(np.uint8)

#     final = np.clip(base + blended_mask, 0, 1).astype(np.uint8)
#     return final

## main ##
# root_path = r"C:\Users\smg\Desktop\kiat\Multimodal_teleopertaion\data\aug_test"
# img_path = osp.join(root_path, "test.jpg")
# lbl_path = osp.join(root_path, "test.txt")

# img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
# polygons, cls_ids = load_yolo_seg_labels(lbl_path)

# cls2contours = {}
# for poly, cid in zip(polygons, cls_ids):
#     cnt = denorm_polygon(poly, img)
#     cls2contours.setdefault(cid, []).append(cnt)
    
# # cls2masks = build_class_union_masks(polygons, cls_ids, img, True)
# # connected_masks = connect_disjoint_class_regions(polygons, img.shape[:2], distance_thresh=50)

# cls2masks = group_polygons_by_class(polygons, cls_ids, img.shape[:2])
# merged_masks = merge_masks_with_convex_hull(cls2masks)
# # merged = merge_masks_with_distance_blending(cls2masks[0], blend_width=15)


# print()
# import glob
# images_dir = "/home/scilab/Documents/teleoperation/yolo/datasets/raw_total/images/total"
# labels_dir = '/home/scilab/Documents/teleoperation/yolo/datasets/raw_total/labels/total'  # YOLO 구조라면 labels 폴더에 txt 있음

# # 이미지 파일 목록
# img_files = sorted(glob.glob(osp.join(images_dir, "*.jpg")))

# for img_path in img_files:
#     # 라벨 파일 경로
#     base_name = osp.splitext(osp.basename(img_path))[0]
#     lbl_path = osp.join(labels_dir, base_name + ".txt")
    
#     if not osp.exists(lbl_path):
#         print(f"[WARN] 라벨 없음: {lbl_path}")
#         continue

#     # 이미지 & 라벨 로드
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     polygons, cls_ids = load_yolo_seg_labels(lbl_path)

#     # 클래스별 마스크 만들기 → convex hull 병합
#     cls2masks = group_polygons_by_class(polygons, cls_ids, img.shape[:2])
#     merged_masks = merge_masks_with_convex_hull(cls2masks)

#     # 오버레이 만들기
#     overlay = img.copy()
#     colors = plt.cm.get_cmap('tab10', len(merged_masks))
#     for idx, (cid, mask) in enumerate(merged_masks.items()):
#         color = np.array(colors(idx)[:3]) * 255
#         overlay[mask > 0] = overlay[mask > 0] * 0.4 + color * 0.6

#     # 시각화
#     plt.figure(figsize=(8, 8))
#     plt.imshow(overlay)
#     plt.title(f"{base_name} - merged masks")
#     plt.axis("off")
#     plt.show()
# input("Press Enter for next...")