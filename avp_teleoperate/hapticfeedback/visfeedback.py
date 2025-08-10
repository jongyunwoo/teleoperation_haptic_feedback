import cv2
import numpy as np
import os
import sys



def draw_alignment(image, masks, classes, arrow_length=80, color=(0, 255, 0)):
    
    exclude_class_ids = [4, 5]
    h, w = image.shape[:2]
    draw_img = image.copy()

    for idx, (mask, class_id) in enumerate(zip(masks.xy, classes)):
        if int(class_id) in exclude_class_ids:
            continue  # 로봇 손 무시

        cnt = np.array(mask, dtype=np.int32)
        if cnt.shape[0] < 3:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        if rw < rh:
            angle += 90

        center = (int(cx), int(cy))
        theta = np.deg2rad(angle)

        tip = (
            int(cx - arrow_length * np.cos(theta)),
            int(cy - arrow_length * np.sin(theta))
        )

        cv2.arrowedLine(draw_img, center, tip, color=color, thickness=2, tipLength=0.3)

    return draw_img

# class DistanceOverlay:
#     """
#     유연한 생성자:
#       1) 구버전(단순 초점거리): DistanceOverlay(focal_len_px=615.0, min_dist=0.01, max_dist=0.10)
#       2) 신버전(내/외부 파라미터): DistanceOverlay(head_int:dict, wrist_int:dict, T_hw:np.ndarray,
#                                                min_dist=0.01, max_dist=0.10)
#          - head_int, wrist_int는 {"fx","fy","cx","cy"} 키를 가정
#          - T_hw는 head<-wrist 변환(여기선 사용 안함, 호환 목적)
#     """

#     def __init__(self, *args, **kwargs):
#         # 공통 파라미터
#         self.min_dist = float(kwargs.get("min_dist", 0.01))
#         self.max_dist = float(kwargs.get("max_dist", 0.10))

#         # 모드1: (focal_len_px, ...)
#         if len(args) == 0 or isinstance(args[0], (int, float)):
#             focal_len_px = float(args[0]) if len(args) >= 1 else 615.0
#             self.focal_len_px = focal_len_px
#             self._mode = 1
#             # 호환을 위한 더미 값
#             self.head_int = None
#             self.wrist_int = None
#             self.T_hw = None
#             return

#         # 모드2: (head_int, wrist_int, T_hw, ...)
#         if len(args) >= 3 and isinstance(args[0], dict) and isinstance(args[1], dict):
#             self.head_int  = args[0]
#             self.wrist_int = args[1]
#             self.T_hw      = args[2]
#             # 깊이/픽셀 거리 변환에 사용할 초점거리는 wrist 기준 fx 사용(상황에 맞게 조정 가능)
#             self.focal_len_px = float(self.wrist_int.get("fx", 615.0))
#             self._mode = 2
#             return

#         raise TypeError(
#             "DistanceOverlay 생성자 인수가 올바르지 않습니다. "
#             "예) DistanceOverlay(615.0, min_dist=0.02, max_dist=0.20) "
#             "또는 DistanceOverlay(head_int, wrist_int, T_hw, min_dist=0.02, max_dist=0.20)"
#         )

#     @staticmethod
#     def compute_mask_centroid(mask: np.ndarray):
#         """다각형( Nx2 ) 마스크의 무게중심 (cx, cy) 반환. 실패 시 None."""
#         cnt = mask.astype(np.int32)
#         if cnt.ndim != 2 or cnt.shape[1] != 2 or cnt.shape[0] < 3:
#             return None
#         M = cv2.moments(cnt)
#         if M["m00"] == 0:
#             return None
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#         return (cx, cy)

#     def compute_object_distance_simple(self, center1, center2, depth_image: np.ndarray):
#         """
#         두 픽셀 center1, center2 사이의 유클리드 픽셀 거리(d_pixel)를
#         초점거리와 평균 깊이로 간단히 m 단위로 환산.
#         """
#         if center1 is None or center2 is None:
#             return None
#         (u1, v1), (u2, v2) = center1, center2

#         h, w = depth_image.shape[:2]
#         if not (0 <= u1 < w and 0 <= v1 < h and 0 <= u2 < w and 0 <= v2 < h):
#             return None

#         z1 = int(depth_image[v1, u1])
#         z2 = int(depth_image[v2, u2])
#         if z1 <= 0 or z2 <= 0:
#             return None

#         z1_m = z1 / 1000.0
#         z2_m = z2 / 1000.0
#         z_avg = (z1_m + z2_m) / 2.0

#         d_pixel = float(np.hypot(u2 - u1, v2 - v1))
#         return (d_pixel * z_avg) / float(self.focal_len_px)

#     def overlay_mask_with_color(self, image: np.ndarray, mask: np.ndarray, distance: float) -> np.ndarray:
#         """
#         거리값을 min_dist~max_dist 구간에 색상으로 매핑해서 마스크 영역에 오버레이.
#         가까움(초록) ↔ 멀어짐(빨강), 극단적인 색 대비
#         """
#         if distance is None:
#             return image

#         # 0~1 정규화
#         denom = max(self.max_dist - self.min_dist, 1e-6)
#         norm = float(np.clip((distance - self.min_dist) / denom, 0.0, 1.0))

#         # 감마 보정으로 색 대비 강화 (0.5 미만은 더 어둡게, 0.5 이상은 더 밝게)
#         gamma = 0.3  # 0.3~0.7 정도로 조절 가능
#         norm_gamma = norm ** gamma

#         # 가까울수록 초록(0,255,0), 멀수록 빨강(0,0,255)
#         inv = 1.0 - norm_gamma
#         color = (0, int(inv * 255), int(norm_gamma * 255))  # B, G, R

#         # 마스크 영역에 색 적용
#         mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
#         cv2.fillPoly(mask_img, [mask.astype(np.int32)], 255)
#         overlay = np.zeros_like(image)
#         overlay[mask_img == 255] = color

#         return cv2.addWeighted(image, 1.0, overlay, 0.7, 0.0)
    
from typing import Optional, Tuple, Dict, Any, List

class HandObjectDepthAssessor:
    """
    깊이 이미지(정렬된, mm 단위)와 YOLO 박스/클래스를 입력으로 받아
    - 왼손 vs 주변 물체
    - 오른손 vs 주변 물체
    를 비교하고, 결과를 포인트/마스크로 시각화합니다.
    그라데이션(멀수록 빨강 → 주황 → 노랑 → 초록)과 EMA 스무딩을 지원합니다.
    """

    def __init__(self,
                 left_id: int = 7,
                 right_id: int = 8,
                 k: int = 2,
                 tol_mm: int = 15,
                 min_valid_ratio: float = 0.3,
                 hysteresis_mm: int = 10,
                 smooth_beta: float = 0.2):  # ← EMA 계수(0.1~0.3 권장)
        self.LEFT_HAND_ID = int(left_id)
        self.RIGHT_HAND_ID = int(right_id)
        self.HAND_CLASSES = (self.LEFT_HAND_ID, self.RIGHT_HAND_ID)

        # 비교 파라미터
        self.k = int(k)
        self.tol_mm = int(tol_mm)
        self.min_valid_ratio = float(min_valid_ratio)
        self.hysteresis_mm = int(hysteresis_mm)

        # 프레임 간 판정 안정화
        self._last_verdict: Dict[str, str] = {}   # pair_key -> verdict

        # 그라데이션/스무딩
        self.smooth_beta = float(smooth_beta)
        self._smooth_norm: Dict[str, float] = {   # 'left'/'right' → 0..1 (0=멀다, 1=가깝다)
            "left": 0.5,
            "right": 0.5
        }

    # ---------- 저수준 유틸 ----------

    @staticmethod
    def box_center_xyxy(box: np.ndarray) -> Tuple[int, int]:
        x1, y1, x2, y2 = map(float, box)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        return cx, cy

    def robust_depth_at(self, depth_img: np.ndarray, uv: Tuple[int, int]) -> Optional[float]:
        """
        depth_img: HxW, uint16(mm). uv: (u,v)
        k: 패치 반경 (2 => 5x5)
        min_valid_ratio: 유효(>0) 픽셀 비율
        """
        assert depth_img.ndim == 2, "depth_img must be 2D (HxW), uint16(mm)"
        h, w = depth_img.shape[:2]
        u, v = uv
        if not (0 <= u < w and 0 <= v < h):
            return None

        k = self.k
        u0, u1 = max(0, u-k), min(w, u+k+1)
        v0, v1 = max(0, v-k), min(h, v+k+1)
        patch = depth_img[v0:v1, u0:u1].reshape(-1)

        valid = patch[patch > 0]
        need = ((2*k+1)*(2*k+1)) * self.min_valid_ratio
        if len(valid) < need:
            return None
        return float(np.median(valid))  # mm

    def _compare_core(self, dA: Optional[float], dB: Optional[float], tol_mm: int) -> str:
        if dA is None or dB is None:
            return "unknown"
        diff = dA - dB
        if abs(diff) <= tol_mm:
            return "same"
        return "A_closer" if diff < 0 else "B_closer"

    def compare_two(self,
                    depth_img: np.ndarray,
                    boxA_xyxy: np.ndarray,
                    boxB_xyxy: np.ndarray,
                    pair_key: Optional[str] = None) -> Dict[str, Any]:
        """
        A(대상1) vs B(대상2) 비교. 히스테리시스 적용 가능.
        반환: {A_uv, B_uv, A_mm, B_mm, verdict}
        """
        cA = self.box_center_xyxy(boxA_xyxy)
        cB = self.box_center_xyxy(boxB_xyxy)

        dA = self.robust_depth_at(depth_img, cA)  # mm
        dB = self.robust_depth_at(depth_img, cB)  # mm

        verdict = self._compare_core(dA, dB, self.tol_mm)
        if pair_key is not None and verdict != "unknown":
            last = self._last_verdict.get(pair_key)
            if last and last != verdict and last != "same":
                # 바뀌려면 여분 mm(hysteresis) 이상 차이 필요
                core_now = self._compare_core(dA, dB, self.tol_mm + self.hysteresis_mm)
                if core_now != verdict:
                    verdict = last  # 유지
            self._last_verdict[pair_key] = verdict

        return {"A_uv": cA, "B_uv": cB, "A_mm": dA, "B_mm": dB, "verdict": verdict}

    def annotate_points(self,
                        image_bgr: np.ndarray,
                        results: Dict[str, Any],
                        labelA: str = "A",
                        labelB: str = "B") -> np.ndarray:
        """
        결과를 이미지에 간단히 표시.
        """
        out = image_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        def draw_point(uv, text):
            if uv is None: return
            u, v = uv
            cv2.circle(out, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(out, text, (u+6, v-6), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

        dA = results.get("A_mm")
        dB = results.get("B_mm")
        draw_point(results.get("A_uv"), f"{labelA}:{'--' if dA is None else int(dA)}mm")
        draw_point(results.get("B_uv"), f"{labelB}:{'--' if dB is None else int(dB)}mm")

        verdict = results.get("verdict", "?")
        cv2.putText(out, f"verdict: {verdict}", (10, 24), font, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, f"verdict: {verdict}", (10, 24), font, 0.7, (0,255,0), 1, cv2.LINE_AA)
        return out

    # ---------- (NEW) 폴리곤 오버레이 & 그라데이션 유틸 ----------

    @staticmethod
    def overlay_polys(image_bgr: np.ndarray,
                      polys: List[np.ndarray],
                      color_bgr: Tuple[int,int,int],
                      alpha: float = 0.6) -> np.ndarray:
        """polys(list of Nx2)를 color로 채워 반투명 오버레이"""
        if not polys:
            return image_bgr
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            if poly is None:
                continue
            p = np.asarray(poly, dtype=np.int32)
            if p.ndim == 2 and p.shape[0] >= 3:
                cv2.fillPoly(mask, [p], 255)
        overlay = image_bgr.copy()
        overlay[mask == 255] = color_bgr
        return cv2.addWeighted(image_bgr, 1.0 - alpha, overlay, alpha, 0.0)

    @staticmethod
    def _lerp_color(c1, c2, t):
        t = float(np.clip(t, 0.0, 1.0))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _grad_color_far_to_near(self, near_norm: float) -> tuple:
        """
        near_norm: 0=멀다(빨강) → 1=가깝다(초록).
        빨강 → 주황 → 노랑 → 초록으로 보간.
        """
        red    = (0, 0, 255)
        orange = (0, 128, 255)
        yellow = (0, 255, 255)
        green  = (0, 255, 0)
        t = float(np.clip(near_norm, 0.0, 1.0))
        if t < 1/3:
            return self._lerp_color(red, orange, t / (1/3))
        elif t < 2/3:
            return self._lerp_color(orange, yellow, (t - 1/3) / (1/3))
        else:
            return self._lerp_color(yellow, green, (t - 2/3) / (1/3))

    def _ema_norm(self, side_key: str, new_norm: Optional[float]) -> float:
        prev = self._smooth_norm.get(side_key, 0.5)
        if new_norm is None:
            return prev
        beta = self.smooth_beta
        smoothed = prev * (1.0 - beta) + float(new_norm) * beta
        self._smooth_norm[side_key] = smoothed
        return smoothed

    def _near_norm_from_pair(self, A_mm: Optional[float], B_mm: Optional[float],
                             min_mm: float, max_mm: float) -> Optional[float]:
        """
        A=손, B=최근접 물체. |A-B|로 '가까움 정규화' 계산.
        near_norm: 0=멀다, 1=가깝다
        """
        if A_mm is None or B_mm is None:
            return None
        dist_mm = abs(A_mm - B_mm)
        far_norm = np.clip((dist_mm - min_mm) / max(1e-6, (max_mm - min_mm)), 0.0, 1.0)  # 0=가까움, 1=멀다
        near_norm = 1.0 - far_norm
        return float(near_norm)

    def colorize_hands_on_head_grad(self,
                                    head_image_bgr: np.ndarray,
                                    head_masks_xy: List[np.ndarray],
                                    head_classes: List[int],
                                    left_info: Dict[str, Any],
                                    right_info: Dict[str, Any],
                                    min_mm: float = 50.0,
                                    max_mm: float = 300.0,
                                    alpha: float = 0.55) -> np.ndarray:
        """
        left_info/right_info: process()의 out['left'], out['right'] 그대로 사용.
        min_mm/max_mm: '같다'로 보는 근접 기준/스케일.
        """
        LEFT_ID, RIGHT_ID = self.HAND_CLASSES
        left_masks  = [m for m, c in zip(head_masks_xy, head_classes) if c == LEFT_ID]
        right_masks = [m for m, c in zip(head_masks_xy, head_classes) if c == RIGHT_ID]

        l_near = self._near_norm_from_pair(left_info.get("A_mm"),  left_info.get("B_mm"),  min_mm, max_mm)
        r_near = self._near_norm_from_pair(right_info.get("A_mm"), right_info.get("B_mm"), min_mm, max_mm)

        l_sm = self._ema_norm("left",  l_near)
        r_sm = self._ema_norm("right", r_near)

        l_color = self._grad_color_far_to_near(l_sm)
        r_color = self._grad_color_far_to_near(r_sm)

        out = head_image_bgr.copy()
        out = self.overlay_polys(out, left_masks,  l_color, alpha=alpha)
        out = self.overlay_polys(out, right_masks, r_color, alpha=alpha)
        return out

    # ---------- 손별(왼/오) 최근접 물체 찾기 ----------

    def _nearest_object_to_box(self,
                               depth_mm: np.ndarray,
                               boxes_xyxy: np.ndarray,
                               classes: np.ndarray,
                               target_idx: int) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        h_center = self.box_center_xyxy(boxes_xyxy[target_idx])
        h_mm = self.robust_depth_at(depth_mm, h_center)

        nearest_idx, nearest_mm = None, None
        for i, cls in enumerate(classes):
            if i == target_idx:
                continue
            if cls in self.HAND_CLASSES:  # 다른 손 제외
                continue
            o_center = self.box_center_xyxy(boxes_xyxy[i])
            o_mm = self.robust_depth_at(depth_mm, o_center)
            if o_mm is None:
                continue
            if nearest_mm is None or o_mm < nearest_mm:
                nearest_idx, nearest_mm = i, o_mm

        return nearest_idx, h_mm, nearest_mm

    def _compare_hand_side(self,
                           side_label: str,
                           depth_mm: np.ndarray,
                           boxes_xyxy: np.ndarray,
                           classes: np.ndarray,
                           image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        side_label: 'left' 또는 'right'
        해당 손과 '가장 가까운 물체' 비교 + 포인트 주석.
        """
        if side_label == "left":
            hand_ids = [i for i, c in enumerate(classes) if c == self.LEFT_HAND_ID]
            pair_key = "left_vs_nearest"
            a_label = "left_hand"
        else:
            hand_ids = [i for i, c in enumerate(classes) if c == self.RIGHT_HAND_ID]
            pair_key = "right_vs_nearest"
            a_label = "right_hand"

        if len(hand_ids) == 0:
            return image_bgr, {"verdict": "no_hand"}

        h_idx = hand_ids[0]
        nearest_idx, h_mm, n_mm = self._nearest_object_to_box(depth_mm, boxes_xyxy, classes, h_idx)
        if nearest_idx is None:
            tmp = {
                "A_uv": self.box_center_xyxy(boxes_xyxy[h_idx]),
                "A_mm": h_mm, "B_uv": None, "B_mm": None, "verdict": "no_object"
            }
            disp = self.annotate_points(image_bgr, tmp, labelA=a_label, labelB="obj?")
            return disp, tmp

        res = self.compare_two(
            depth_img=depth_mm,
            boxA_xyxy=boxes_xyxy[h_idx],                 # A=손
            boxB_xyxy=boxes_xyxy[nearest_idx],           # B=최근접 물체
            pair_key=pair_key
        )
        disp = self.annotate_points(image_bgr, res, labelA=a_label, labelB=f"obj{nearest_idx}")
        res.update({"hand_idx": h_idx, "obj_idx": nearest_idx})
        return disp, res

    # ---------- 한 번에 처리 ----------

    def process(self,
                image_bgr: np.ndarray,
                depth_mm: np.ndarray,
                boxes_xyxy: Optional[np.ndarray],
                classes: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        한 프레임 처리:
          - 왼손 vs 주변 물체
          - 오른손 vs 주변 물체
        를 각각 수행하고 이미지에 모두 표시.

        반환:
          disp: 시각화된 이미지
          out : {"left": {...}, "right": {...}}
        """
        disp = image_bgr.copy()
        out: Dict[str, Any] = {"left": {"verdict": "no_data"}, "right": {"verdict": "no_data"}}

        if depth_mm is None or depth_mm.ndim != 2 or boxes_xyxy is None or len(boxes_xyxy) == 0:
            return disp, out

        # 왼손
        disp, left_res = self._compare_hand_side("left", depth_mm, boxes_xyxy, classes, disp)
        out["left"] = left_res

        # 오른손 (왼손 결과가 그려진 disp 위에 추가로 그리기)
        disp, right_res = self._compare_hand_side("right", depth_mm, boxes_xyxy, classes, disp)
        out["right"] = right_res

        return disp, out

    
    