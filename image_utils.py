import cv2
import numpy as np


def pick_point(image_buf: np.ndarray) -> np.ndarray:
    point = []
    display = image_buf.copy()
    window_name = "Click to place a point"

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if point:
            return
        point.append((x, y))
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_click)
    cv2.imshow(window_name, display)

    while not point:
        cv2.waitKey(50)

    cv2.destroyWindow(window_name)
    return np.array(point[0], dtype=np.float64)


def annotate(image_buf: np.ndarray) -> np.ndarray:
    points = [pick_point(image_buf) for _ in range(4)]
    return np.stack(points)
