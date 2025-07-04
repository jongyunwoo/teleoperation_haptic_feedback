import os, sys

# 1) 이 파일이 있는 디렉터리
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  
# 2) 한 단계 위 → avp_teleoperate
ONE_UP   = os.path.dirname(THIS_DIR)                 
# 3) 또 한 단계 위 → g1_tele
TWO_UP   = os.path.dirname(ONE_UP)                    
# 4) 거기에 있는 tact-python
TACT_PYTHON_DIR = os.path.join(TWO_UP, "tact-python")

# 5) 검색 경로에 추가
if not os.path.isdir(TACT_PYTHON_DIR):
    raise RuntimeError(f"tact-python 폴더를 찾을 수 없습니다: {TACT_PYTHON_DIR}")
sys.path.insert(0, TACT_PYTHON_DIR)

from bhaptics import better_haptic_player as player
player.initialize()

print("remote:", player.ws.remote_address) 