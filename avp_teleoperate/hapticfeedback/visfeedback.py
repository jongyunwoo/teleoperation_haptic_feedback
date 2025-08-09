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

class DistanceOverlay:
    """
    유연한 생성자:
      1) 구버전(단순 초점거리): DistanceOverlay(focal_len_px=615.0, min_dist=0.01, max_dist=0.10)
      2) 신버전(내/외부 파라미터): DistanceOverlay(head_int:dict, wrist_int:dict, T_hw:np.ndarray,
                                               min_dist=0.01, max_dist=0.10)
         - head_int, wrist_int는 {"fx","fy","cx","cy"} 키를 가정
         - T_hw는 head<-wrist 변환(여기선 사용 안함, 호환 목적)
    """

    def __init__(self, *args, **kwargs):
        # 공통 파라미터
        self.min_dist = float(kwargs.get("min_dist", 0.01))
        self.max_dist = float(kwargs.get("max_dist", 0.10))

        # 모드1: (focal_len_px, ...)
        if len(args) == 0 or isinstance(args[0], (int, float)):
            focal_len_px = float(args[0]) if len(args) >= 1 else 615.0
            self.focal_len_px = focal_len_px
            self._mode = 1
            # 호환을 위한 더미 값
            self.head_int = None
            self.wrist_int = None
            self.T_hw = None
            return

        # 모드2: (head_int, wrist_int, T_hw, ...)
        if len(args) >= 3 and isinstance(args[0], dict) and isinstance(args[1], dict):
            self.head_int  = args[0]
            self.wrist_int = args[1]
            self.T_hw      = args[2]
            # 깊이/픽셀 거리 변환에 사용할 초점거리는 wrist 기준 fx 사용(상황에 맞게 조정 가능)
            self.focal_len_px = float(self.wrist_int.get("fx", 615.0))
            self._mode = 2
            return

        raise TypeError(
            "DistanceOverlay 생성자 인수가 올바르지 않습니다. "
            "예) DistanceOverlay(615.0, min_dist=0.02, max_dist=0.20) "
            "또는 DistanceOverlay(head_int, wrist_int, T_hw, min_dist=0.02, max_dist=0.20)"
        )

    @staticmethod
    def compute_mask_centroid(mask: np.ndarray):
        """다각형( Nx2 ) 마스크의 무게중심 (cx, cy) 반환. 실패 시 None."""
        cnt = mask.astype(np.int32)
        if cnt.ndim != 2 or cnt.shape[1] != 2 or cnt.shape[0] < 3:
            return None
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def compute_object_distance_simple(self, center1, center2, depth_image: np.ndarray):
        """
        두 픽셀 center1, center2 사이의 유클리드 픽셀 거리(d_pixel)를
        초점거리와 평균 깊이로 간단히 m 단위로 환산.
        """
        if center1 is None or center2 is None:
            return None
        (u1, v1), (u2, v2) = center1, center2

        h, w = depth_image.shape[:2]
        if not (0 <= u1 < w and 0 <= v1 < h and 0 <= u2 < w and 0 <= v2 < h):
            return None

        z1 = int(depth_image[v1, u1])
        z2 = int(depth_image[v2, u2])
        if z1 <= 0 or z2 <= 0:
            return None

        z1_m = z1 / 1000.0
        z2_m = z2 / 1000.0
        z_avg = (z1_m + z2_m) / 2.0

        d_pixel = float(np.hypot(u2 - u1, v2 - v1))
        return (d_pixel * z_avg) / float(self.focal_len_px)

    def overlay_mask_with_color(self, image: np.ndarray, mask: np.ndarray, distance: float) -> np.ndarray:
        """
        거리값을 min_dist~max_dist 구간에 색상으로 매핑해서 마스크 영역에 오버레이.
        가까움(초록) ↔ 멀어짐(빨강), 극단적인 색 대비
        """
        if distance is None:
            return image

        # 0~1 정규화
        denom = max(self.max_dist - self.min_dist, 1e-6)
        norm = float(np.clip((distance - self.min_dist) / denom, 0.0, 1.0))

        # 감마 보정으로 색 대비 강화 (0.5 미만은 더 어둡게, 0.5 이상은 더 밝게)
        gamma = 0.3  # 0.3~0.7 정도로 조절 가능
        norm_gamma = norm ** gamma

        # 가까울수록 초록(0,255,0), 멀수록 빨강(0,0,255)
        inv = 1.0 - norm_gamma
        color = (0, int(inv * 255), int(norm_gamma * 255))  # B, G, R

        # 마스크 영역에 색 적용
        mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [mask.astype(np.int32)], 255)
        overlay = np.zeros_like(image)
        overlay[mask_img == 255] = color

        return cv2.addWeighted(image, 1.0, overlay, 0.7, 0.0)