from playsound import playsound
import time

def warn_beep(distance_m):
    """
    distance_m : 센서·카메라로 구한 거리 (미터)
    0.2m 이하부터 경고음, 가까워질수록 간격을 줄임
    """
    if distance_m > 0.2:
        return                 # 20 cm 초과면 무음

    # 5 cm → 0.1 s, 20 cm → 0.5 s 로 매핑
    interval = max(0.1, 0.5 * (distance_m / 0.2))
    playsound("/home/scilab/Documents/teleoperation/avp_teleoperate/hapticfeedback/beep-125033.mp3", block=False)
    time.sleep(interval)
    
    
def grap_sound(left_tactile_sensor,right_tactile_sensor, left_min, left_max, right_min, right_max):
    if left_tactile_sensor >= left_min:
        playsound('/home/scilab/Documents/teleoperation/avp_teleoperate/hapticfeedback/bell-notification-337658.mp3', block=False)
    elif left_tactile_sensor < left_min and right_tactile_sensor > right_max:
        playsound('/home/scilab/Documents/teleoperation/avp_teleoperate/hapticfeedback/error-83494.mp3', block=False)
    if right_tactile_sensor >= right_min:
        playsound('/home/scilab/Documents/teleoperation/avp_teleoperate/hapticfeedback/bell-notification-337658.mp3', block=False)
    elif right_tactile_sensor < right_min and right_tactile_sensor > right_max:
        playsound('/home/scilab/Documents/teleoperation/avp_teleoperate/hapticfeedback/error-83494.mp3', block=False)
         