import numpy as np
import matplotlib.pyplot as plt
class KalmanFilter1D:
    def __init__(self, q=1e-5, r=1e-2, p0=1.0, x0=0.0):
        # q: 프로세스 잡음 분산(Q), r: 측정 잡음 분산(R)
        self.A = 1.0     # 상태 전이 계수
        self.H = 1.0     # 측정 모델 계수
        self.Q = q       # 프로세스 잡음 공분산
        self.R = r       # 측정 잡음 공분산
        self.P = p0      # 추정 오차 공분산
        self.x = x0      # 상태 추정치

    def predict(self):
        # 예측 단계
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q

    def update(self, z):
        # 보정 단계
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1.0 - K * self.H) * self.P
        return self.x


if __name__ == '__main__':
    
    data = np.load('/home/scilab/Documents/teleoperation/avp_teleoperate/teleop/utils/episode_0001/tactiles/000306_right_tactile.npy')
    filt_kal = np.empty_like(data)
    
    kalfil = KalmanFilter1D(q=1e-2, r=1e-1)
    for i in range(len(data)):
        kalfil.predict()
        filt_kal[i]= kalfil.update(data[i])
    
    sensor_count = len(data)    
    plt.figure(figsize=(15, 4))
    plt.bar(range(sensor_count), data, color='blue')
    plt.bar(range(sensor_count), filt_kal, color='red')
    plt.title(f"Tactile Sensor Reading (Right Hand) - Total: {sensor_count}")
    plt.xlabel("Sensor Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    