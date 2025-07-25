import cv2
import numpy as np

def overlayhand(l_path, r_path, originimg, alpha=0.7):
    DISPLAYSIZE = (100, 100)
    l_img = cv2.imread(l_path) 
    r_img = cv2.imread(r_path)

    l_img = cv2.resize(l_img, DISPLAYSIZE)
    r_img = cv2.resize(r_img, DISPLAYSIZE)

    # 3. 위치 지정 (좌상단과 우상단)
    h, w = originimg.shape[:2]
    l_x, l_y = 50, h - DISPLAYSIZE[1] - 50
    r_x, r_y = w - DISPLAYSIZE[0] - 50, h - DISPLAYSIZE[1] - 50

    def blend_overlay(background, overlay, x, y):
        bh, bw = background.shape[:2]
        oh, ow = overlay.shape[:2]
        if x+ow > bw or y+oh > bh:
            print("Overlay 위치가 잘못되었습니다.")
            return background

        # 분리: BGR + Alpha
        overlay_bgr = overlay[:, :, :3]
        overlay_mask = overlay[:, :, 3:] / 255.0  # 0~1

        # 배경 영역 잘라오기
        roi = background[y:y+oh, x:x+ow]

        # 합성
        blended = (1 - overlay_mask) * roi + overlay_mask * overlay_bgr
        background[y:y+oh, x:x+ow] = blended.astype(np.uint8)
        return background

    # 4. 합성
    originimg = blend_overlay(originimg, l_img, l_x, l_y)
    originimg = blend_overlay(originimg, r_img, r_x, r_y)

    return originimg
