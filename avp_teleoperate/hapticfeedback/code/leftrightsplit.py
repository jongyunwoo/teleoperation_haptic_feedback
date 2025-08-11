import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

class RobotHandSideResolver:
    """
    Seg만 있는 상황에서 '로봇손'(단일 클래스) 인스턴스들을 left/right로 라벨링.
    - 프레임 내: x-중심 정렬로 좌우 할당
    - 프레임 간: IoU로 추적해 안정화, 실패 시 x-정렬
    - 하나만 보이면: 이전 할당 유지 or 규칙 기반 임시 할당
    """

    def __init__(self,
                 iou_track_thresh: float = 0.3,
                 mirror: bool = False,        # True면 좌우 뒤집어서 판단
                 keep_ms: float = 0.6):       # 최근 관측 유효 시간(초) — 필요하면 사용
        self.iou_track_thresh = float(iou_track_thresh)
        self.mirror = bool(mirror)
        self.keep_ms = float(keep_ms)
        self._last = { "left": None, "right": None }     # {"poly": Nx2 np.float32, "t": timestamp}
    
    @staticmethod
    def _poly_to_bbox(poly: np.ndarray) -> np.ndarray:
        x, y = poly[:,0], poly[:,1]
        return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)

    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        ub = max(0.0, (bx2 - bx1) * (by2 - by1))
        return float(inter / (ua + ub - inter + 1e-6))

    def _cx(self, poly: np.ndarray) -> float:
        bb = self._poly_to_bbox(poly)
        return float((bb[0] + bb[2]) * 0.5)

    def _match_by_iou(self, polys: List[np.ndarray], now_ts: float) -> Dict[int, str]:
        """
        지난 프레임(left/right)과 IoU로 매칭. 반환: {index: 'left'/'right'}
        """
        assign: Dict[int, str] = {}
        cand = list(range(len(polys)))
        # left 우선
        if self._last["left"] is not None:
            lbb = self._poly_to_bbox(self._last["left"]["poly"])
            best_iou, best_idx = 0.0, None
            for i in cand:
                iou = self._iou_xyxy(self._poly_to_bbox(polys[i]), lbb)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx is not None and best_iou >= self.iou_track_thresh:
                assign[best_idx] = "left"
                cand.remove(best_idx)
        # right 다음
        if self._last["right"] is not None and cand:
            rbb = self._poly_to_bbox(self._last["right"]["poly"])
            best_iou, best_idx = 0.0, None
            for i in cand:
                iou = self._iou_xyxy(self._poly_to_bbox(polys[i]), rbb)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx is not None and best_iou >= self.iou_track_thresh:
                assign[best_idx] = "right"
                cand.remove(best_idx)
        return assign

    def update(self, masks_xy: List[np.ndarray], image_w: int, now_ts: float) -> Dict[str, Optional[np.ndarray]]:
        """
        입력: YOLO masks.xy (각 Nx2), 이미지 폭, 현재 timestamp(초)
        출력: {"left": poly or None, "right": poly or None}
        """
        # 0) 폴리 전처리
        polys = []
        for m in masks_xy or []:
            p = np.asarray(m, dtype=np.float32)
            if p.ndim == 2 and p.shape[0] >= 3:
                polys.append(p)
        if len(polys) == 0:
            return {"left": None, "right": None}

        # 1) 우선 IoU로 지난 프레임과 매칭
        assign = self._match_by_iou(polys, now_ts)

        # 2) 남은 것들은 x-중심 정렬로 좌/우 할당
        rest = [i for i in range(len(polys)) if i not in assign]
        if rest:
            rest_sorted = sorted(rest, key=lambda i: self._cx(polys[i]), reverse=self.mirror)
            if len(rest_sorted) == 1:
                i = rest_sorted[0]
                # 빈쪽에 우선 할당
                if "left" not in assign.values():
                    assign[i] = "left"
                elif "right" not in assign.values():
                    assign[i] = "right"
                else:
                    # 둘 다 이미 배정됐다면 더 가까운 쪽 기준으로(드문 케이스)
                    assign[i] = "left" if self._cx(polys[i]) < (image_w * 0.5) ^ self.mirror else "right"
            else:
                # 두 개 이상이면 좌/우 순서대로 채우기
                if "left" not in assign.values():
                    assign[rest_sorted[0]] = "left"
                if len(rest_sorted) >= 2 and "right" not in assign.values():
                    assign[rest_sorted[-1]] = "right"

        # 3) 최종 폴리 저장
        out = {"left": None, "right": None}
        for i, side in assign.items():
            out[side] = polys[i].copy()
            self._last[side] = {"poly": polys[i].copy(), "t": now_ts}

        return out
