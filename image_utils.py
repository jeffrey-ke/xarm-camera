from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from eval import draw_coordinate_in_image
from pose_utils import make_se3
from .goto_capture import grabbed_frame, retrieve_stereo_images, get_intrinsics
from .xarm_datastructs import Capture


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

def detect(image: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray]:
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(image, None)
    assert desc is not None, f"No features detected in image of shape {image.shape}"
    return kp, desc

def ratio_test(desc_a: np.ndarray, desc_b: np.ndarray,
               ratio: float = 0.75) -> list[cv2.DMatch]:
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    return [m for m, n in raw if m.distance < ratio * n.distance]

def homography_inliers(pts_src: np.ndarray, pts_dst: np.ndarray,
                       reproj_thresh: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only correspondences consistent with a single planar homography.

    Returns (pts_src_inliers, pts_dst_inliers, H) where H maps src -> dst.
    """
    assert len(pts_src) >= 4, f"Need >= 4 points for homography, got {len(pts_src)}"
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, reproj_thresh)
    inlier = mask.ravel().astype(bool)
    return pts_src[inlier], pts_dst[inlier], H

def match(left_image: np.ndarray, right_image: np.ndarray,
          ratio: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    kp_left, desc_left = detect(left_image)
    kp_right, desc_right = detect(right_image)
    good = ratio_test(desc_left, desc_right, ratio)
    pts_left = np.array([kp_left[m.queryIdx].pt for m in good], dtype=np.float64)
    pts_right = np.array([kp_right[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts_left, pts_right

def bilinear_sample_3d(pixel_coords: np.ndarray, image_shape: tuple[int, int],
                       corners_3d: np.ndarray) -> np.ndarray:
    """Map pixel coordinates to 3D via bilinear interpolation over a planar quad.

    corners_3d: (4, 3) — top-left, bottom-left, bottom-right, top-right.
    """
    h, w = image_shape[:2]
    u = pixel_coords[:, 0] / (w - 1)
    v = pixel_coords[:, 1] / (h - 1)

    tl, bl, br, tr = corners_3d[0], corners_3d[1], corners_3d[2], corners_3d[3]
    return ((1 - u) * (1 - v))[:, None] * tl \
         + ((1 - u) * v)[:, None] * bl \
         + (u * v)[:, None] * br \
         + (u * (1 - v))[:, None] * tr

def stitch_correspondences(left_img: np.ndarray, right_img: np.ndarray,
                           left_pts: np.ndarray, right_pts: np.ndarray) -> np.ndarray:
    h1, h2 = left_img.shape[0], right_img.shape[0]
    h = max(h1, h2)
    if h1 != h:
        scale = h / h1
        left_img = cv2.resize(left_img, (int(left_img.shape[1] * scale), h))
        left_pts = left_pts * scale
    if h2 != h:
        scale = h / h2
        right_img = cv2.resize(right_img, (int(right_img.shape[1] * scale), h))
        right_pts = right_pts * scale
    def to_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    left_img, right_img = to_bgr(left_img), to_bgr(right_img)
    w1 = left_img.shape[1]
    canvas = np.hstack([left_img, right_img])
    for (x1, y1), (x2, y2) in zip(left_pts.astype(int), right_pts.astype(int)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(canvas, (x1, y1), 4, color, -1)
        cv2.circle(canvas, (x2 + w1, y2), 4, color, -1)
        cv2.line(canvas, (x1, y1), (x2 + w1, y2), color, 1)
    return canvas

def visualize_correspondences(left_pts: np.ndarray, right_pts: np.ndarray,
                              left_img: np.ndarray, right_img: np.ndarray):
    import matplotlib.pyplot as plt
    canvas = stitch_correspondences(left_img, right_img, left_pts, right_pts)
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def detect_aruco_se3(image, K, marker_length: float,
                     aruco_dict_type: int = cv2.aruco.DICT_4X4_50,
                     marker_id: int | None = None) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    all_corners, ids, _ = detector.detectMarkers(gray)

    assert ids is not None and len(ids) > 0, "No ArUco markers detected"

    if marker_id is not None:
        idx = np.where(ids.ravel() == marker_id)[0]
        assert len(idx) > 0, f"Marker {marker_id} not found (detected: {ids.ravel()})"
        corners_2d = all_corners[idx[0]][0]
    else:
        corners_2d = all_corners[0][0]

    half = marker_length / 2
    obj_pts = np.array([
        [0, -half, half],
        [0, half, half],
        [0, half, -half],
        [0, -half, -half],
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, corners_2d, K, np.zeros(4), flags=cv2.SOLVEPNP_IPPE)
    assert success, "solvePnP failed on ArUco corners"

    marker2camera = make_se3(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    return np.linalg.inv(marker2camera)


def aruco_pose(camera, baseline2left: np.ndarray) -> np.ndarray:
    with grabbed_frame(camera) as frame:
        left, _ = retrieve_stereo_images(frame, rectified=True)

    left_K, _ = get_intrinsics(camera)
    left2target = detect_aruco_se3(left, left_K, marker_length=0.1, marker_id=0)
    return left2target @ baseline2left

def pnp_box(image, corners_3d_top_left_CCW, K, D):
    points_2d_top_left_CCW = annotate(image)
    success, rvec, tvec = cv2.solvePnP(corners_3d_top_left_CCW, points_2d_top_left_CCW, K, D, flags=cv2.SOLVEPNP_SQPNP)
    assert success, "solvePnP failed"
    target2camera = make_se3(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    return np.linalg.inv(target2camera)

def sanity_check_gt(calculated_baseline2target: np.ndarray, camera, baseline2left: np.ndarray):
    target2left = baseline2left @ np.linalg.inv(calculated_baseline2target)

    with grabbed_frame(camera) as frame:
        left, _ = retrieve_stereo_images(frame, rectified=True)
    left_K, _ = get_intrinsics(camera)

    target_origin = np.zeros(3)
    annotated = draw_coordinate_in_image(left, target_origin, target2left, left_K, color=(0, 255, 0))
    return annotated

@dataclass
class UndistortParams:
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray
    new_k_left: np.ndarray
    new_k_right: np.ndarray

def load_calibration_params(path: Path) -> UndistortParams:
    fs = cv2.FileStorage(str(path), cv2.FileStorage_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
    size    = fs.getNode("Size").mat()
    k_left  = fs.getNode("K_LEFT").mat()
    k_right = fs.getNode("K_RIGHT").mat()
    d_left  = fs.getNode("D_LEFT").mat()
    d_right = fs.getNode("D_RIGHT").mat()
    fs.release()
    imsize = tuple(int(v) for v in np.asarray(size).reshape(-1)[:2])
    return _build_undistort_params(imsize, k_left, d_left, k_right, d_right)

def _build_undistort_params(imsize, k_left, d_left, k_right, d_right) -> UndistortParams:
    new_k_left,  _ = cv2.getOptimalNewCameraMatrix(k_left,  d_left,  imsize, 0.0, imsize)
    new_k_right, _ = cv2.getOptimalNewCameraMatrix(k_right, d_right, imsize, 0.0, imsize)
    map1_left,  map2_left  = cv2.initUndistortRectifyMap(k_left,  d_left,  None, new_k_left,  imsize, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(k_right, d_right, None, new_k_right, imsize, cv2.CV_32FC1)
    return UndistortParams(map1_left, map2_left, map1_right, map2_right, new_k_left, new_k_right)

_remap_image = lambda arr, m1, m2: cv2.remap(arr, m1, m2, cv2.INTER_LINEAR)
_remap_depth = lambda arr, m1, m2: cv2.remap(arr, m1, m2, cv2.INTER_NEAREST)

def undistort_capture(capture: Capture, params: UndistortParams) -> Capture:
    return Capture(
        offset=capture.offset,
        robot2base=capture.robot2base,
        euler_target_to_robot=capture.euler_target_to_robot,
        left_image=_remap_image(capture.left_image,   params.map1_left,  params.map2_left),
        right_image=_remap_image(capture.right_image, params.map1_right, params.map2_right),
        left_depth=_remap_depth(capture.left_depth,   params.map1_left,  params.map2_left),
        left_calib=params.new_k_left,
    )

