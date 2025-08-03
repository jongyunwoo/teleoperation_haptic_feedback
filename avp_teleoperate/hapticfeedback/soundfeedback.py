from pydub import AudioSegment
from pydub.playback import play
import threading
from bhaptics import better_haptic_player as player

haptics_paused = False

class StereoSoundFeedbackManager:
    def __init__(self, grip_sound_path):
        self.prev_grip_state = {"left": False, "right": False}
        self.grip_sound = AudioSegment.from_file(grip_sound_path)

    def play_sound(self, hand="right"):
        if hand == "right":
            sound = sound.pan(1.0)   # 오른쪽 출력
        elif hand == "left":
            sound = sound.pan(-1.0)  # 왼쪽 출력
        threading.Thread(target=play, args=(sound,), daemon=True).start()

    def update(self, grip_detected, hand="right"):
        global haptics_paused
        if grip_detected and not self.prev_grip_state[hand]:
            self.play_sound(hand)
            player.stop_all()        # 모든 haptic 중단
            haptics_paused = True    # Haptic 루프에서 중단 플래그 확인
        
        # Grip 해제 → Haptic 재개
        elif not grip_detected and self.prev_grip_state[hand]:
            haptics_paused = False   # 다시 Haptic 재개 가능

        # 상태 갱신
        self.prev_grip_state[hand] = grip_detected