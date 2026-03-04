from dataclasses import dataclass

import numpy as np


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
