import numpy as np
from collections import deque
import threading
class RobustGripDetector:
    def __init__(self, touch_dict, contact_threshold=50, grip_force_threshold=60,
                 min_fingers=2, grip_hold_time=0.0, fps=30):
        self.contact_threshold = contact_threshold
        self.grip_force_threshold = grip_force_threshold
        self.min_fingers = min_fingers
        self.history = deque(maxlen=int(grip_hold_time * fps)) if grip_hold_time > 0 else None
        self.finger_indices = self._create_finger_indices(touch_dict)

    def _create_finger_indices(self, touch_dict):
        indices = {}
        start = 0
        for name, count in touch_dict.items():
            indices[name] = range(start, start + count)
            start += count
        return indices

    def update(self, tactile_values):
        active_sensors = tactile_values > self.contact_threshold
        active_fingers = []

        for name, idx in self.finger_indices.items():
            if "palm" in name:
                continue
            ratio = np.sum(active_sensors[idx]) / len(idx)
            if ratio > 0.05:
                finger_name = name.split("_")[0]
                if finger_name not in active_fingers:
                    active_fingers.append(finger_name)

        contact_detected = len(active_fingers) > 0
        avg_force_active = np.mean(tactile_values[active_sensors]) if contact_detected else 0

        if self.history is not None:
            self.history.append(avg_force_active)
            grip_detected = (
                contact_detected
                and len(active_fingers) >= self.min_fingers
                and len(self.history) == self.history.maxlen
                and np.min(self.history) > self.grip_force_threshold
            )
        else:
            grip_detected = (
                contact_detected
                and len(active_fingers) >= self.min_fingers
                and avg_force_active > self.grip_force_threshold
            )

        return contact_detected, grip_detected, active_fingers, avg_force_active