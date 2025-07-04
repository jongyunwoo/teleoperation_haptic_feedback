import numpy as np
import os

# ① npy 파일들이 있는 디렉토리로 수정하세요
tactile_dir = "/home/scilab/Documents/g1_tele/avp_teleoperate/teleop/utils/data/episode_0001/tactiles"

for fname in os.listdir(tactile_dir):
    if fname.endswith("_tactile.npy"):
        npy_path = os.path.join(tactile_dir, fname)
        csv_path = os.path.join(tactile_dir, fname.replace(".npy", ".csv"))

        # ② numpy 배열 로드
        arr = np.load(npy_path)

        # 1D 배열이면 한 열, 2D 배열이면 각 열을 CSV 컬럼으로 저장
        # delimiter="," 로 구분
        np.savetxt(csv_path, arr, delimiter=",", fmt="%g")

        print(f"Converted {npy_path} → {csv_path}")
