from dataclasses import dataclass
from typing import NewType

import numpy as np

Mm = NewType('Mm', float)
Meters = NewType('Meters', float)

def meters_to_mm(m: Meters) -> Mm : return Mm(m * 1000)
def mm_to_meters(mm: Mm) -> Meters: return Meters(mm / 1000)

class SafeZed:
    def __init__(self, zed):
        self._zed = zed

    def __getattr__(self, name):
        if self._zed is None:
            raise RuntimeError("ZED frame expired — use inside 'with grabbed_frame()'")
        return getattr(self._zed, name)

    def _invalidate(self):
        self._zed = None


@dataclass
class Zedpack:
    left_image: np.ndarray
    left_depth: np.ndarray
    right_image: np.ndarray
    left_K: np.ndarray
    right_K: np.ndarray

@dataclass
class Capture:
    offset: np.ndarray
    robot2base: np.ndarray
    euler_target_to_robot: np.ndarray
    left_image: np.ndarray
    left_depth: np.ndarray
    right_image: np.ndarray
    left_calib: np.ndarray
