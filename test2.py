#!/usr/bin/env python3
# ------------------------------------------------------------
# 1) data.json   → 각 프레임별 joint qpos 추출
# 2) pixels.csv  → 사람이 찍은 손끝 (u,v)
# 3) least_squares 로 BODY→CAMERA extrinsic (rx,ry,rz, tx,ty,tz) 보정
# ------------------------------------------------------------
import json, math, csv
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from urchin import URDF

# =========== 파일 경로 =============
ROOT       = Path("/home/scilab/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0001")
URDF_PATH  = 
JSON_PATH  = ROOT / "data.json"
PIXELS_CSV = ROOT / "pixels.csv"   # ← ❶ 직접 작성
# ===================================
URDF_PATH  = "/home/scilab/teleoperation/avp_teleoperate/assets/g1_inspire_hand_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
# DATA_PATH  = "/home/scilab/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0001/data.json"
# IMAGE_PATH = "/home/scilab/teleoperation/avp_teleoperate/teleop/utils/datanalysis/episode_0001/colors/000052_color_0.jpg"

# ---------- 카메라 내부 파라미터 ----------
fx, fy = 605.9278, 606.1917
width, height = 848, 480
cx, cy = width/2, height/2

# ---------- 고정값: IMU (월드→바디) ----------
IMU_DEG = (1.72, -15.86, -0.08)
def rpy(roll,pitch,yaw,deg=False):
    if deg: roll,pitch,yaw = map(math.radians,(roll,pitch,yaw))
    cr,sr = math.cos(roll), math.sin(roll)
    cp,sp = math.cos(pitch),math.sin(pitch)
    cy,sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz@Ry@Rx
R_W_B = rpy(*IMU_DEG,deg=True)

# ---------- link→optical 회전 ----------
R_CL_CO = np.array([[ 0,-1, 0],
                    [ 0, 0,-1],
                    [ 1, 0, 0]])

# ---------- URDF 로드 ----------
robot = URDF.load(str(URDF_PATH), lazy_load_meshes=True)

# ---------- finger→link 매핑 ----------
def map_hand_joint(side,fname):
    m = {"thumb":"thumb_1","index":"index_1","middle":"middle_1",
         "ring":"ring_1","pinky":"little_1"}
    return f"{side}_{m[fname]}_link"

# ---------- 데이터 로드 ----------
with open(JSON_PATH) as f: data = json.load(f)

# joint name 테이블 만들기 한 번
def canon_arm_names(side):
    parts=[side,"shoulder","elbow","wrist","pitch","roll","yaw"]
    out=[]
    for j in data["info"]["joint_names"][f"{side}_arm"]:
        j_mod,tmp=j[1:],""          # kLeftShoulderPitch …
        for p in parts:
            ix=j_mod.lower().find(p)
            if ix!=-1:
                tmp+=j_mod[:ix+len(p)]+"_"
                j_mod=j_mod[ix+len(p):]
        out.append((tmp[:-1]+"_joint").lower())
    return out
ARM_NAMES = {s: canon_arm_names(s) for s in ("left","right")}

# 모든 프레임 qpos 캐시
QPOS = {}
for frame in data["data"]:
    idx = frame["idx"]
    QPOS[idx]={}
    for side in ("left","right"):
        QPOS[idx][f"{side}_arm"]  = frame["states"][f"{side}_arm"]["qpos"]
        QPOS[idx][f"{side}_hand"] = frame["states"][f"{side}_hand"]["qpos"]

# ---------- 픽셀 CSV 읽기 ----------
samples=[]
with open(PIXELS_CSV) as f:
    rdr=csv.DictReader(f)
    for row in rdr:
        samples.append({"idx":int(row["idx"]),
                        "side":row["side"],
                        "finger":row["finger"],
                        "uv":(float(row["u"]),float(row["v"]))})

# ---------- FK: 월드 손끝 좌표 ----------
finger_ord={"thumb":5,"index":3,"middle":2,"ring":1,"pinky":0}
def fk_fingertip(sample):
    idx,side,finger = sample["idx"],sample["side"],sample["finger"]
    q_arm  = QPOS[idx][f"{side}_arm"]
    q_hand = QPOS[idx][f"{side}_hand"]
    # build dict
    qd={n:q for n,q in zip(ARM_NAMES[side], q_arm)}
    # hand joint name
    hand_src = data["info"]["joint_names"][f"{side}_hand"]
    hand_map = {
        f"k{side.capitalize()}HandThumbRotation": f"{side}_thumb_1_joint",
        f"k{side.capitalize()}HandThumbBend":     f"{side}_thumb_3_joint",
        f"k{side.capitalize()}HandIndex":         f"{side}_index_1_joint",
        f"k{side.capitalize()}HandMiddle":        f"{side}_middle_1_joint",
        f"k{side.capitalize()}HandRing":          f"{side}_ring_1_joint",
        f"k{side.capitalize()}HandPinky":         f"{side}_little_1_joint"}
    hand_names=[hand_map[h] for h in hand_src]
    qd[hand_names[finger_ord[finger]]] = q_hand[finger_ord[finger]]
    # FK
    link_name=map_hand_joint(side,finger)
    T=robot.link_fk(qd)[robot.link_map[link_name]]
    return T[:3,3]   # (3,)

# ---------- 최적화 ----------
def reproj_err(param):
    rx,ry,rz, tx,ty,tz = param
    R_B_C = rpy(rx,ry,rz)
    t_B_C = np.array([tx,ty,tz])
    errs=[]
    for s in samples:
        p_W = fk_fingertip(s)
        p_C = R_W_B @ p_W + R_W_B@t_B_C + R_B_C@np.zeros(3)  # body→cam (R,t)
        p_CL = R_B_C @ p_W + t_B_C        # 더 단순히:  p_CL = R_B_C @ (R_W_B @ p_W) + t_B_C
        p_CO = R_CL_CO @ p_CL
        X,Y,Z = p_CO
        u_hat = fx*X/Z + cx
        v_hat = fy*Y/Z + cy
        errs.extend([u_hat - s["uv"][0], v_hat - s["uv"][1]])
    return errs

print(f"Loaded {len(samples)} pixel annotations.")
p0 = np.array([0.0, 0.8307767239493009, 0.0,   # rx,ry,rz (rad)
               0.0576235, 0.01753, 0.42987])   # tx,ty,tz (m)
res=least_squares(reproj_err, p0, verbose=2, method="lm", max_nfev=2000)
rx,ry,rz, tx,ty,tz = res.x
print("\n===== 최적화 결과 =====")
print(f"rx={rx:.5f}  ry={ry:.5f}  rz={rz:.5f} rad   (yaw deg = {math.degrees(rz):.3f}°)")
print(f"tx={tx:.4f}  ty={ty:.4f}  tz={tz:.4f}  [m]")
print(f"평균 픽셀 오차 = {res.cost/len(samples):.2f}")
