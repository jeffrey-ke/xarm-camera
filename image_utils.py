import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def pick_point(image_buf: np.ndarray) -> np.ndarray:
    is_bgr = len(image_buf.shape) == 3 and image_buf.shape[2] == 3
    display = cv2.cvtColor(image_buf, cv2.COLOR_BGR2RGB) if is_bgr else image_buf

    fig, ax = plt.subplots()
    ax.imshow(display)
    ax.set_title("Click to place a point")

    point = plt.ginput(1, timeout=0)
    plt.close(fig)

    return np.array(point[0], dtype=np.float64)


def annotate(image_buf: np.ndarray) -> np.ndarray:
    points = [pick_point(image_buf) for _ in range(4)]
    return np.stack(points)
