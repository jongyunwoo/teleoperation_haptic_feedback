# #!/usr/bin/env python3
# # ------------------------------------------------------------
# # URDF + IMU + qpos  →  손끝 3‑D 위치  →  영상 픽셀 좌표
# # ------------------------------------------------------------
# import math, json, cv2, numpy as np
# from urchin import URDF
# def deg2rad(deg):
#     """도(degree) → 라디안(radian) 변환."""
#     return deg * math.pi / 180.0

# # deg= deg2rad(1.37)
# # # ====================== 사용자 설정 ==========================
# URDF_PATH  = "/home/scilab/teleoperation/avp_teleoperate/assets/g1_inspire_hand_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
# DATA_PATH  = "/home/scilab/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0001/data.json"
# IMAGE_PATH = "/home/scilab/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0001/colors/000052_color_0.jpg"

# FRAME        = 0                 # data.json 인덱스
# LEFTRIGHT    = "left"           # "left" or "right"
# WHICH_FINGER = "thumb"           # thumb / index / middle / ring / pinky

# # IMU 자세 (deg) : 월드→바디
# IMU_ROLL, IMU_PITCH, IMU_YAW =  1.86, -14.88, 0.00

# # 카메라 내부 파라미터
# fx, fy = 605.9278, 606.1917
# # fx = 426.7954
# # fy = 426.7954
# width, height = 848, 480
# cx, cy = width / 2, height / 2

# # 바디→카메라(camera_link) 고정 변환 (rad, m)
# CAMERA_RPY_B = (0.0, 0.8307767239493009, 0.0)     # roll, pitch, yaw
# CAMERA_T_B   = np.array([0.0576235, 0.01753, 0.42987])
# # ============================================================


# # ---------- 유틸리티 ----------
# def rpy_to_mat(roll, pitch, yaw, *, deg=False):
#     """Extrinsic Z‑Y‑X 회전행렬 (yaw‑pitch‑roll)."""
#     if deg:
#         roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
#     cr, sr = math.cos(roll),  math.sin(roll)
#     cp, sp = math.cos(pitch), math.sin(pitch)
#     cy, sy = math.cos(yaw),   math.sin(yaw)

#     Rx = np.array([[1, 0,  0],
#                    [0, cr, -sr],
#                    [0, sr,  cr]])
#     Ry = np.array([[ cp, 0, sp],
#                    [  0, 1,  0],
#                    [-sp, 0, cp]])
#     Rz = np.array([[cy, -sy, 0],
#                    [sy,  cy, 0],
#                    [ 0,   0, 1]])
#     return Rz @ Ry @ Rx


# # ---------- 1. URDF · 데이터 로드 ----------
# finger_map = {"thumb":5, "index":3, "middle":2, "ring":1, "pinky":0}

# robot = URDF.load(URDF_PATH, lazy_load_meshes=True)
# with open(DATA_PATH) as f:
#     data = json.load(f)

# arm_joint_src   = data["info"]["joint_names"][f"{LEFTRIGHT}_arm"]
# hand_joint_src  = data["info"]["joint_names"][f"{LEFTRIGHT}_hand"]

# # arm 이름 치환 (kLShoulderPitch → l_shoulder_pitch_joint 형태)
# parts = [LEFTRIGHT, "shoulder", "elbow", "wrist", "pitch", "roll", "yaw"]
# arm_joint_names = []
# for j in arm_joint_src:
#     j_mod, tmp = j[1:], ""
#     for p in parts:
#         idx = j_mod.lower().find(p)
#         if idx != -1:
#             tmp += j_mod[: idx + len(p)] + "_"
#             j_mod = j_mod[idx + len(p):]
#     arm_joint_names.append((tmp[:-1] + "_joint").lower())

# hand_map = {
#     f"k{LEFTRIGHT.capitalize()}HandThumbRotation": f"{LEFTRIGHT}_thumb_1_joint",
#     f"k{LEFTRIGHT.capitalize()}HandThumbBend":     f"{LEFTRIGHT}_thumb_3_joint",
#     f"k{LEFTRIGHT.capitalize()}HandIndex":         f"{LEFTRIGHT}_index_1_joint",
#     f"k{LEFTRIGHT.capitalize()}HandMiddle":        f"{LEFTRIGHT}_middle_1_joint",
#     f"k{LEFTRIGHT.capitalize()}HandRing":          f"{LEFTRIGHT}_ring_1_joint",
#     f"k{LEFTRIGHT.capitalize()}HandPinky":         f"{LEFTRIGHT}_little_1_joint",
# }
# hand_joint_names = [hand_map[h] for h in hand_joint_src]

# arm_qpos  = data["data"][FRAME]["states"][f"{LEFTRIGHT}_arm"]["qpos"]
# hand_qpos = data["data"][FRAME]["states"][f"{LEFTRIGHT}_hand"]["qpos"]

# # ---------- 2. FK ----------
# q_dict = {n: q for n, q in zip(arm_joint_names, arm_qpos)}
# q_dict[hand_joint_names[finger_map[WHICH_FINGER]]] = hand_qpos[finger_map[WHICH_FINGER]]

# ee_link_name = hand_joint_names[finger_map[WHICH_FINGER]].replace("_joint", "")
# T_WE = robot.link_fk(q_dict)[robot.link_map[ee_link_name]]            # ^W T_E

# # ---------- 3. 월드→카메라(camera_link) ----------
# R_W_B = rpy_to_mat(IMU_ROLL, IMU_PITCH, IMU_YAW, deg=True)            # 월드→바디
# R_B_C = rpy_to_mat(*CAMERA_RPY_B, deg=False)                          # 바디→cam_link
# t_B_C = CAMERA_T_B

# R_W_C = R_W_B @ R_B_C
# t_W_C = R_W_B @ t_B_C
# T_W_C = np.eye(4);  T_W_C[:3,:3] = R_W_C;  T_W_C[:3,3] = t_W_C

# # ---------- 4. camera_link → camera_color_optical_frame ----------
# T_C_W = np.linalg.inv(T_W_C)
# p_CL  = (T_C_W @ T_WE)[:3, 3]                                         # camera_link 좌표

# # 올바른 link→optical 회전 (x_fwd, y_left, z_up → x_right, y_down, z_fwd)
# R_CL_CO = np.array([[ 0, -1,  0],   # X_CO = -Y_CL
#                     [ 0,  0, -1],   # Y_CO = -Z_CL
#                     [ 1,  0,  0]])  # Z_CO =  X_CL
# p_CO = R_CL_CO @ p_CL                                                 # optical 좌표

# X, Y, Z = p_CO                                                        # X right, Y down, Z forward
# if Z <= 0:
#     raise ValueError("Z(depth) ≤ 0 : 카메라 뒤쪽입니다. 변환행렬을 확인하세요.")

# # ---------- 5. 투영 ----------
# u = fx * X / Z + cx
# v = fy * Y / Z + cy
# pixel = (int(round(u)), int(round(v)))
# print(f"[INFO] fingertip pixel = {pixel}")

# # ---------- 6. 시각 확인 ----------
# img = cv2.imread(IMAGE_PATH)
# if img is None:
#     raise FileNotFoundError(IMAGE_PATH)
# cv2.circle(img, pixel, 8, (0,0,255), -1)
# cv2.putText(img, str(pixel), (pixel[0]+8, pixel[1]-8),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# out_path = f"projected_{FRAME:04d}.png"
# cv2.imwrite(out_path, img)
# print(f"[INFO] saved → {out_path}")


import cv2
import mediapipe as mp

# 1) MediaPipe Hands 초기화
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2) 영상 열기 (파일 or 카메라)
# ▶ 파일을 재생하려면:
video_path = "/home/scilab/teleoperation/avp_teleoperate/teleop/utils/datanalysis/output/episode_0001.mp4"
cap = cv2.VideoCapture(video_path)

# ▶ 실시간 카메라를 쓰려면:
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(f"비디오 열기 실패: {video_path}")

while True:
    ret, frame = cap.read()     # ← 인자는 절대 NO
    if not ret:
        break

    # BGR → RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # 인덱스 손가락 끝 (landmark #8)
            lm8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            u, v = int(lm8.x * w), int(lm8.y * h)
            cv2.circle(frame, (u, v), 8, (0,0,255), -1)
            cv2.putText(frame, f"IdxTip:({u},{v})", (u+10, v-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # 전체 랜드마크도 함께 그리기
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
        break

cap.release()
cv2.destroyAllWindows()
