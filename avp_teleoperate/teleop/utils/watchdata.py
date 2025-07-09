import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ① 열고 싶은 .npy 파일 경로
npy_path = '/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/data/episode_0001/tactiles/000280_left_tactile.npy'
# csv_path = '/home/scilab/Documents/g1_tele/episode_0033/tactiles/000055_left_tactile.csv'
if not os.path.exists(npy_path):
    raise FileNotFoundError(f"{npy_path} 경로가 맞는지 확인하세요!")

data = np.load(npy_path)  
data_norm = data 
# ⑤ CSV 저장 (delimiter 지정 가능)
# np.savetxt(csv_path, data, delimiter=',', fmt='%.6f')
# print(f"Saved CSV → {csv_path}")
# # ② 파일 로드

print("==== 기본 정보 ====")
print("shape :", data.shape)
print("dtype :", data.dtype)
print("min   :", data.min())
print("max   :", data.max())
print("first 10 values :", data[:10])
print("len :", len(data))

# print("==== Norm 기본 정보 ====")
# print("shape :", data_norm.shape)
# print("dtype :", data_norm.dtype)
# print("min   :", data_norm.min())
# print("max   :", data_norm.max())
# print("first 10 values :", data_norm[:])
# print("len :", len(data_norm))

stats = {
    'shape':    [data_norm.shape],
    'dtype':    [str(data_norm.dtype)],
    'min':      [float(np.min(data_norm))],
    'max':      [float(np.max(data_norm))],
    'mean':     [float(np.mean(data_norm))],
    'std':      [float(np.std(data_norm))],
    'len':      [len(data_norm)],
}
df_norm_info = pd.DataFrame(stats)
print(df_norm_info)
df_norm_info.to_csv('tactile_basic_info.csv', index=False)
import numpy as np
import matplotlib.pyplot as plt


sensor_count = len(data_norm)

# 2. Plot
plt.figure(figsize=(15, 4))
plt.bar(range(sensor_count), data_norm, color='blue')
plt.title(f"Tactile Sensor Reading (Right Hand) - Total: {sensor_count}")
plt.xlabel("Sensor Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()




