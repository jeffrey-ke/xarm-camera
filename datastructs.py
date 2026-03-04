from dataclasses import dataclass
from typing import NewType

import numpy as np

Mm = NewType('Mm', float)
Meters = NewType('Meters', float)

def meters_to_mm(m: Meters) -> Mm : return Mm(m * 1000)
def mm_to_meters(mm: Mm) -> Meters: return M(m / 1000)

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
    euler_target_to_robot: np.ndarray
    left_image: np.ndarray
    left_depth: np.ndarray
    right_image: np.ndarray
    left_calib: np.ndarray
