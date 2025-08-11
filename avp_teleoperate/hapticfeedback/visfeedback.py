import cv2
import numpy as np
import os
import sys
from typing import Optional, Tuple, Dict, Any, List, Iterable

class LineOverlayMerger:

    def __init__(self,
                 exclude_class_ids: Iterable[int] = (7,8),
                 iou_thresh: float = 0.25,
                 dist_thresh: float = 50.0,
                 angle_thresh_deg: float = 15.0,
                 depth_thresh_mm: float = 60.0,
                 morph_dilate: int = 2,
                 morph_close: int = 3,
                 extend_ratio: float = 1.0,
                 color: Tuple[int,int,int] = (0,255,0),
                 thickness: int = 2):
        self.exclude_class_ids = set(int(x) for x in exclude_class_ids)
        self.iou_thresh = float(iou_thresh)
        self.dist_thresh = float(dist_thresh)
        self.angle_thresh_deg = float(angle_thresh_deg)
        self.depth_thresh_mm = float(depth_thresh_mm)
        self.morph_dilate = int(morph_dilate)
        self.morph_close = int(morph_close)
        self.extend_ratio = float(extend_ratio)
        self.color = tuple(int(c) for c in color)
        self.thickness = int(thickness)

    # ---------- 내부 유틸 ----------

    @staticmethod
    def _line_intersection(p0, v, a, b):
        w = b - a
        M = np.array([[v[0], -w[0]],[v[1], -w[1]]], dtype=np.float32)
        rhs = (a - p0).astype(np.float32)
        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        if abs(det) < 1e-8:
            return None
        invM = (1.0/det)*np.array([[ M[1,1], -M[0,1]],[-M[1,0],  M[0,0]]], dtype=np.float32)
        t,u = invM @ rhs
        if 0.0 <= u <= 1.0:
            pt = p0 + t*v
            return pt, float(t), float(u)
        return None

    def _draw_fitline_over_minarearect(self, image, mask_xy):
        draw = image
        cnt = np.asarray(mask_xy, dtype=np.float32)
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            return draw
        vx,vy,x0,y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        v  = np.array([vx,vy], dtype=np.float32)
        p0 = np.array([x0,y0], dtype=np.float32)

        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect).astype(np.float32)
        edges = [(box[i], box[(i+1)%4]) for i in range(4)]

        hits=[]
        for a,b in edges:
            res = self._line_intersection(p0, v, a, b)
            if res is not None:
                pt,t,_ = res
                hits.append((pt,t))

        if len(hits) < 2:
            (cx,cy),(rw,rh),ang = rect
            L = max(rw,rh)*self.extend_ratio
            theta = np.deg2rad(ang if rw>=rh else ang+90.0)
            d = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            p1 = np.array([cx,cy], dtype=np.float32) - 0.5*L*d
            p2 = np.array([cx,cy], dtype=np.float32) + 0.5*L*d
        else:
            hits.sort(key=lambda x:x[1])
            p1, p2 = hits[0][0], hits[-1][0]
            if self.extend_ratio != 1.0:
                c=(p1+p2)*0.5; half=(p2-p1)*0.5*self.extend_ratio
                p1, p2 = c-half, c+half

        h,w = image.shape[:2]
        p1 = (int(np.clip(p1[0],0,w-1)), int(np.clip(p1[1],0,h-1)))
        p2 = (int(np.clip(p2[0],0,w-1)), int(np.clip(p2[1],0,h-1)))
        cv2.line(draw, p1, p2, self.color, self.thickness)
        return draw

    @staticmethod
    def _bbox_from_poly(p):
        x,y = p[:,0], p[:,1]
        return np.array([x.min(),y.min(),x.max(),y.max()], dtype=np.float32)

    @staticmethod
    def _iou_xyxy(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
        inter = iw*ih
        ua = max(0.0, (ax2-ax1)*(ay2-ay1))
        ub = max(0.0, (bx2-bx1)*(by2-by1))
        return inter / (ua + ub - inter + 1e-6)

    @staticmethod
    def _fit_angle_deg(poly):
        pts = poly.astype(np.float32)
        vx,vy,_,_ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        ang = np.degrees(np.arctan2(vy, vx))
        if ang < 0: ang += 180.0
        return ang

    @staticmethod
    def _angle_diff(a, b):
        d = abs(a - b)
        return min(d, 180.0 - d)

    @staticmethod
    def _median_depth_mm(depth, poly):
        if depth is None: return None
        H,W = depth.shape[:2]
        m = np.zeros((H,W), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        vals = depth[m==255]; vals = vals[vals>0]
        return float(np.median(vals)) if len(vals)>0 else None

    def _merge_polys_to_binary(self, polys, shape):
        H,W = shape[:2]
        binm = np.zeros((H,W), np.uint8)
        for p in polys:
            cv2.fillPoly(binm, [p.astype(np.int32)], 255)
        if self.morph_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.morph_dilate,self.morph_dilate))
            binm = cv2.dilate(binm, k, 1)
        if self.morph_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.morph_close,self.morph_close))
            binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k, 1)
        return binm

    @staticmethod
    def _largest_contour(binm):
        cnts,_ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        return max(cnts, key=cv2.contourArea).reshape(-1,2)

    # ---------- 공개 메서드 ----------

    def draw_single_line_per_object(self,
                                    image,
                                    masks,
                                    classes,
                                    depth_mm=None,
                                    include_class_ids: Optional[Iterable[int]] = None):
        """
        같은 물체로 보이는 분리 마스크들을 그룹핑→병합하고, 그룹당 선 1개만 그려 반환.
        """
        H,W = image.shape[:2]
        draw = image.copy()
        include_set = set(int(x) for x in include_class_ids) if include_class_ids else None

        # 1) 필터링 + 특성계산
        items=[]
        for poly, cid in zip(masks.xy, classes):
            cid=int(cid)
            if include_set is not None and cid not in include_set: 
                continue
            if cid in self.exclude_class_ids:
                continue
            p=np.asarray(poly, dtype=np.float32)
            if p.ndim!=2 or p.shape[0]<3: 
                continue
            bb=self._bbox_from_poly(p)
            cx,cy = 0.5*(bb[0]+bb[2]), 0.5*(bb[1]+bb[3])
            ang = self._fit_angle_deg(p)
            dep = self._median_depth_mm(depth_mm, p) if depth_mm is not None else None
            items.append({"cid":cid,"poly":p,"bbox":bb,"center":(cx,cy),"angle":ang,"depth":dep})

        # 2) 같은 클래스 내 greedy 클러스터링
        groups_by_class={}
        for it in items:
            cid=it["cid"]
            groups_by_class.setdefault(cid, [])
            placed=False
            for g in groups_by_class[cid]:
                rep=g[0]
                iou = self._iou_xyxy(it["bbox"], rep["bbox"])
                dist= np.hypot(it["center"][0]-rep["center"][0], it["center"][1]-rep["center"][1])
                adiff= self._angle_diff(it["angle"], rep["angle"])
                ddiff= abs(it["depth"]-rep["depth"]) if (it["depth"] is not None and rep["depth"] is not None) else 0.0
                if (iou>=self.iou_thresh or dist<=self.dist_thresh) and adiff<=self.angle_thresh_deg and ddiff<=self.depth_thresh_mm:
                    g.append(it); placed=True; break
            if not placed:
                groups_by_class[cid].append([it])

        # 3) 그룹 병합 → 외곽선 → 선 1개
        for cid, groups in groups_by_class.items():
            for g in groups:
                polys=[x["poly"] for x in g]
                binm = self._merge_polys_to_binary(polys, (H,W))
                merged = self._largest_contour(binm)
                if merged is None: 
                    continue
                draw = self._draw_fitline_over_minarearect(draw, merged)

        return draw


class HandObjectDepthAssessor:

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

    def colorize_hands_on_head_grad(
        self,
        head_image_bgr: np.ndarray,
        head_masks_xy: List[np.ndarray],
        head_classes: List[int],
        left_info: Dict[str, Any],
        right_info: Dict[str, Any],
        min_mm: float = 50.0,
        max_mm: float = 300.0,
        alpha: float = 0.55
    ) -> np.ndarray:
        LEFT_ID, RIGHT_ID = self.HAND_CLASSES
        left_masks  = [m for m, c in zip(head_masks_xy or [], head_classes or []) if c == LEFT_ID]
        right_masks = [m for m, c in zip(head_masks_xy or [], head_classes or []) if c == RIGHT_ID]

        # 1) wrist에서 넘어온 A/B(mm)로 near_norm 계산
        def safe_get(info: Optional[Dict[str, Any]], k: str):
            return None if info is None else info.get(k)

        l_near_raw = self._near_norm_from_pair(safe_get(left_info, "A_mm"),  safe_get(left_info, "B_mm"),  min_mm, max_mm)
        r_near_raw = self._near_norm_from_pair(safe_get(right_info, "A_mm"), safe_get(right_info, "B_mm"), min_mm, max_mm)

        # 2) no_object/unknown/None 이면 멀다(0.0)로 강제
        def _norm_or_fallback(info: Optional[Dict[str, Any]], near_val: Optional[float]) -> float:
            verdict = (info or {}).get("verdict")
            if verdict in ("no_object", "no_data", "unknown") or near_val is None:
                return 0.0  # 원하면 0.2~0.3으로 완충 가능
            return float(near_val)

        l_near = _norm_or_fallback(left_info,  l_near_raw)
        r_near = _norm_or_fallback(right_info, r_near_raw)

        # 3) EMA로 부드럽게
        l_sm = self._ema_norm("left",  l_near)
        r_sm = self._ema_norm("right", r_near)

        # 4) 색상 결정
        l_color = self._grad_color_far_to_near(l_sm)
        r_color = self._grad_color_far_to_near(r_sm)

        # 5) 오버레이
        out = head_image_bgr.copy()
        out = self.overlay_polys(out, left_masks,  l_color, alpha=alpha)
        out = self.overlay_polys(out, right_masks, r_color, alpha=alpha)

        # (옵션) dtype 안정화
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out
    # ---------- 손별(왼/오) 최근접 물체 찾기 ----------
    def _nearest_object_to_box(self,
                           depth_mm: np.ndarray,
                           boxes_xyxy: np.ndarray,
                           classes: np.ndarray,
                           target_idx: int,
                           max_xy_dist_px: float = 250.0,   # 손-물체 화면거리 허용 한계
                           max_depth_gap_mm: float = 200.0  # 손-물체 깊이차 허용 한계
                           ) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """
        손(target_idx) 기준으로 '화면 2D 중심 거리(픽셀)'가 가장 가까운 물체를 찾는다.
        단, 너무 멀리 떨어진 후보는 제외하기 위해 깊이 차이 한계도 적용.
        반환: (nearest_idx, hand_mm, obj_mm)
        """
        # ---- 방어: target_idx 범위 체크 ----
        if target_idx < 0 or target_idx >= len(boxes_xyxy):
            return None, None, None

        # ---- 손 중심 좌표 구하기 ----
        h_center = self.box_center_xyxy(boxes_xyxy[target_idx])
        if not h_center or len(h_center) != 2:
            return None, None, None

        # ---- 손 깊이 ----
        h_mm = self.robust_depth_at(depth_mm, h_center)

        nearest_idx, nearest_xy = None, None
        nearest_mm = None

        for i, cls in enumerate(classes):
            if i == target_idx:
                continue
            if cls in self.HAND_CLASSES:  # 다른 손은 제외
                continue

            # ---- 물체 중심 좌표 구하기 ----
            o_center = self.box_center_xyxy(boxes_xyxy[i])
            if not o_center or len(o_center) != 2:
                continue

            # ---- 2D(화면) 중심 거리 ----
            xy_dist = float(np.hypot(o_center[0] - h_center[0], o_center[1] - h_center[1]))
            if xy_dist > max_xy_dist_px:
                continue

            # ---- 물체 깊이 ----
            o_mm = self.robust_depth_at(depth_mm, o_center)
            if o_mm is None:
                continue

            # ---- 깊이 차이가 너무 크면 제외 ----
            if h_mm is not None and abs(o_mm - h_mm) > max_depth_gap_mm:
                continue

            # ---- 가장 가까운 물체 갱신 ----
            if (nearest_idx is None) or (xy_dist < nearest_xy):
                nearest_idx, nearest_xy, nearest_mm = i, xy_dist, o_mm

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

    
    