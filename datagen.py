from dataclasses import dataclass, fields
import pdb
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

from convert_capture import convert_to_dataset
from goto_capture import connect_arm, capture_zed_images, goto_pose, init_zed_camera, fallback, enable_arm, grabbed_frame, retrieve_stereo_images, get_intrinsics, get_distortion, get_baseline, hardware_init
from image_utils import annotate
from pose_utils import generate_random_offsets, xarmpose_to_se3, visualize_poses, add_rotation, offset_to_4x4, make_se3
from xarm_datastructs import Capture, Mm, Meters, meters_to_mm, mm_to_meters
from eval import draw_coordinate_in_image

@dataclass
class Config:
    corners_3d_top_left_CCW: tuple[tuple[float, float, float], ...] = (
        (0, -0.225/2, 0.122),
        (0, -0.225/2, 0),
        (0, 0.225/2, 0),
        (0, 0.225/2, 0.122),
    )
    canonical_image_path: str = 'canonical_image.png'

    xrange: tuple[Meters, Meters] = (.300, .500)
    yrange: tuple[Meters, Meters] = (-0.12, 0.12)
    zrange: tuple[Meters, Meters] = (0.02, 0.08)
    target_to_ee_ypr_desired: tuple[float, float, float] = (90, 0, -90)
    init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters] = (0.3, 0, 0.06)

    no_kfs: int = 1
    target_in_base_offset: tuple[Meters, Meters, Meters] = (0.745, 0, -0.116 + 0.093)
    base_to_target_ypr: tuple[float, float, float] = (180, 0, 0)
    ip: str = '192.168.1.241'
    tcp_origin: tuple[Meters, Meters, Meters] = (0, 0, 0.055)
    tcp_flange_to_tool_euler: tuple[float, float, float] = (90, 0, 90)

    params_in_meters: tuple[str, ...] = ('xrange', 'yrange', 'zrange', 'target_in_base_offset', 'tcp_origin')

    dataset_dir: str = '/tmp/test'
    idx: int = 0

    def __post_init__(self):
        assert Path(self.dataset_dir).parent.exists(), f"{self.dataset_dir} doesn't exist!"

def plan_poses(target_to_ee_ypr_desired, xrange, yrange, zrange, no_kfs, target2base: np.ndarray):
    yaw, pitch, roll = target_to_ee_ypr_desired
    target_frame_poses = np.array([
        add_rotation(offset_to_4x4(offset), z=yaw, y=pitch, x=roll)
        for offset in generate_random_offsets(xrange, yrange, zrange, no_kfs)
    ])
    base_frame_poses = target2base @ target_frame_poses
    return base_frame_poses

def make_capture(robot2base, target2base, zed_pack):
    robot2target = np.linalg.inv(target2base) @ robot2base
    return Capture(
        offset=robot2target[:3, -1],
        robot2base=robot2base,
        euler_target_to_robot=R.from_matrix(robot2target[:3, :3]).as_euler('ZYX', degrees=True),
        left_image=zed_pack.left_image,
        left_depth=zed_pack.left_depth,
        right_image=zed_pack.right_image,
        left_calib=zed_pack.left_K,
    )

def move_to_se3(xarm, pose: np.ndarray, speed=30):
    x, y, z = [meters_to_mm(m) for m in pose[:3, 3]]
    yaw, pitch, roll = R.from_matrix(pose[:3, :3]).as_euler('ZYX', degrees=True)
    goto_pose(xarm, x, y, z, roll, pitch, yaw, speed)

def ee2base(xarm):
    code, xarmpose = xarm.get_position()
    assert code == 0, f"get_position failed with code {code}"
    pose_meters = [mm_to_meters(mm) for mm in xarmpose[:3]]
    return xarmpose_to_se3([*pose_meters, *xarmpose[3:]])

def make_index(target_to_ee_ypr_desired: tuple[float, float, float],
               xrange, yrange, zrange,
               no_kfs: int,
               xarm,
               camera,
               target2base: np.ndarray,
               dataset_dir, idx):

    base_frame_poses = plan_poses(target_to_ee_ypr_desired, xrange, yrange, zrange, no_kfs, target2base
    )
    visualize the poses interactively, prompt for valid or invalid
    on invalid, kill. on valid, continue

    captures = [
        make_capture(robot2base, target2base, capture_zed_images(camera))
        for robot2base in map(xarmpose_to_se3, move_to(xarm, base_frame_poses))
    ]
    convert_to_dataset(captures, dataset_dir, idx)

def move_to(xarm, poses, speed=30):
    for pose in poses:
        yaw, pitch, roll = R.from_matrix(pose[:3, :3]).as_euler('ZYX', degrees=True) # need the base to target roll
        pose_mm = list(map(meters_to_mm, pose[:3, -1]))
        goto_pose(xarm, *pose_mm, roll, pitch, yaw, speed)
        code, xarmpose = xarm.get_position()
        fallback(xarm) if code != 0 else None
        pose_meters = [mm_to_meters(mm) for mm in xarmpose[:3]]
        yield [*pose_meters, *xarmpose[3:]]


def dry_run_datagen_arm_init(
        target_to_ee_ypr_desired, init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters],
        camera,
        corners_3d_top_left_CCW,
        xarm,
        baseline2left: np.ndarray
    ):
    starting_ee2target_desired = make_se3(init_ee_in_target_offset_desired, R.from_euler('ZYX', target_to_ee_ypr_desired, degrees=True).as_matrix())
    ee2target_current = run_registration(camera, corners_3d_top_left_CCW, baseline2left)
    print(R.from_matrix(ee2target_current[:3, :3]).as_euler('ZYX', degrees=True))
    print(f"Translation: {ee2target_current[:3, -1]}")
    target2base = ee2base(xarm) @ np.linalg.inv(ee2target_current)
    print(R.from_matrix(target2base[:3, :3]).as_euler('ZYX', degrees=True))
    init_pose = target2base @ starting_ee2target_desired
    return init_pose, target2base

def datagen_arm_init(
        target_to_ee_ypr_desired, init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters],
        camera,
        corners_3d_top_left_CCW,
        xarm,
        baseline2left: np.ndarray
    ):
    starting_ee2target_desired = make_se3(init_ee_in_target_offset_desired, R.from_euler('ZYX', target_to_ee_ypr_desired, degrees=True).as_matrix())
    ee2target_current = run_registration(camera, corners_3d_top_left_CCW, baseline2left)
    annotated = sanity_check_gt(ee2target_current, camera, baseline2left)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGRA2RGB))
    plt.axis('off')
    plt.show()
    target2base = ee2base(xarm) @ np.linalg.inv(ee2target_current)
    init_pose = target2base @ starting_ee2target_desired
    move_to_se3(xarm, init_pose, speed=10)
    return init_pose, target2base

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


def run_registration(camera, four_corners_3d, baseline2left: np.ndarray):
    with grabbed_frame(camera) as frame:
        left, _ = retrieve_stereo_images(frame)
    left_K, _ = get_intrinsics(camera)
    four_corners_3d = np.array(four_corners_3d, dtype=np.float64)

    points_2d = annotate(left)
    success, rvec, tvec = cv2.solvePnP(four_corners_3d, points_2d, left_K, np.zeros(4), flags=cv2.SOLVEPNP_SQPNP)
    assert success, "solvePnP failed"
    target2left = make_se3(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    left2target = np.linalg.inv(target2left)

    baseline2target = left2target @ baseline2left

    return baseline2target

def pnp_box(image, corners_3d_top_left_CCW, K, D):
    points_2d_top_left_CCW = annotate(image)
    success, rvec, tvec = cv2.solvePnP(corners_3d_top_left_CCW, points_2d_top_left_CCW, K, D, flags=cv2.SOLVEPNP_SQPNP)
    assert success, "solvePnP failed"
    target2camera = make_se3(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    return np.linalg.inv(target2camera)

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
        left, _ = retrieve_stereo_images(frame)

    left_K, _ = get_intrinsics(camera)
    left2target = detect_aruco_se3(left, left_K, marker_length=0.1, marker_id=0)
    return left2target @ baseline2left

def sanity_check_gt(calculated_baseline2target: np.ndarray, camera, baseline2left: np.ndarray):
    target2left = baseline2left @ np.linalg.inv(calculated_baseline2target)

    with grabbed_frame(camera) as frame:
        left, _ = retrieve_stereo_images(frame)
    left_K, _ = get_intrinsics(camera)

    target_origin = np.zeros(3)
    annotated = draw_coordinate_in_image(left, target_origin, target2left, left_K, color=(0, 255, 0))
    return annotated

if __name__ == '__main__':
    config = tyro.cli(Config)
    xarm, camera = hardware_init(config.ip, config.tcp_origin, config.tcp_flange_to_tool_euler)
    baseline2left = np.eye(4)
    baseline2left[0, 3] = get_baseline(camera) / 2

    init_pose, target2base = datagen_arm_init(
        config.target_to_ee_ypr_desired, config.init_ee_in_target_offset_desired,
        camera, 
        config.corners_3d_top_left_CCW,
        xarm,
        baseline2left)

    make_index(
        config.target_to_ee_ypr_desired,
        config.xrange, config.yrange, config.zrange,
        config.no_kfs,
        xarm,
        camera,
        target2base,
        config.dataset_dir, config.idx,
    )

