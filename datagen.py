from dataclasses import dataclass, fields
import pdb
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

from convert_capture import convert_to_dataset
from goto_capture import connect_arm, capture_zed_images, goto_pose, init_zed_camera, fallback, enable_arm, grabbed_frame, retrieve_stereo_images, get_intrinsics, get_distortion
from image_utils import annotate
from pose_utils import generate_offsets, xarmpose_to_se3, visualize_poses, add_rotation, offset_to_4x4, make_se3
from xarm_datastructs import Capture, Mm, Meters, meters_to_mm, mm_to_meters

@dataclass
class Config:
    init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters]
    corners_3d_top_left_CCW: tuple[tuple[float, float, float], ...]

    xrange: tuple[Meters, Meters] = (.300, .500)
    yrange: tuple[Meters, Meters] = (-0.12, 0.12)
    zrange: tuple[Meters, Meters] = (0.02, 0.08)
    target_to_ee_ypr_desired: tuple[float, float, float] = (90, 0, -90)

    no_kfs: int = 1
    target_in_base_offset: tuple[Meters, Meters, Meters] = (0.745, 0, -0.116 + 0.093)
    base_to_target_ypr: tuple[float, float, float] = (180, 0, 0)
    ip: str = '192.168.1.241'
    tcp_origin: tuple[Meters, Meters, Meters] = (0, 0, 0.055)
    tcp_flange_to_tool_euler: tuple[float, float, float] = (90, 0, 90)
    baseline: Meters = 0.063

    params_in_meters: tuple[str, ...] = ('xrange', 'yrange', 'zrange', 'target_in_base_offset', 'tcp_origin')

    dataset_dir: str = '/tmp/test'
    idx: int = 0

    def __post_init__(self):
        assert Path(self.dataset_dir).parent.exists(), f"{self.dataset_dir} doesn't exist!"

def plan_poses(target_to_ee_ypr_desired, xrange, yrange, zrange, no_kfs, target2base: np.ndarray):
    yaw, pitch, roll = target_to_ee_ypr_desired
    target_frame_poses = np.array([
        add_rotation(offset_to_4x4(offset), z=yaw, y=pitch, x=roll)
        for offset in generate_offsets(xrange, yrange, zrange, no_kfs)
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

def move_to_se3(xarm, pose: np.ndarray):
    x, y, z = [meters_to_mm(m) for m in pose[:3, 3]]
    yaw, pitch, roll = R.from_matrix(pose[:3, :3]).as_euler('ZYX', degrees=True)
    goto_pose(xarm, x, y, z, roll, pitch, yaw)

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

    captures = [
        make_capture(robot2base, target2base, capture_zed_images(camera))
        for robot2base in map(xarmpose_to_se3, move_to(xarm, base_frame_poses))
    ]
    convert_to_dataset(captures, dataset_dir, idx)

def move_to(xarm, poses):
    for pose in poses:
        yaw, pitch, roll = R.from_matrix(pose[:3, :3]).as_euler('ZYX', degrees=True) # need the base to target roll
        pose_mm = list(map(meters_to_mm, pose[:3, -1]))
        goto_pose(xarm, *pose_mm, roll, pitch, yaw)
        code, xarmpose = xarm.get_position()
        fallback(xarm) if code != 0 else None
        pose_meters = [mm_to_meters(mm) for mm in xarmpose[:3]]
        yield [*pose_meters, *xarmpose[3:]]


def datagen_arm_init(
        target_to_ee_ypr_desired, init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters],
        camera,
        corners_3d_top_left_CCW,
        xarm,
        baseline2left: np.ndarray
    ):
    starting_ee2target_desired = make_se3(init_ee_in_target_offset_desired, R.from_euler('ZYX', target_to_ee_ypr_desired, degrees=True).as_matrix())
    ee2target_current = run_registration(camera, corners_3d_top_left_CCW, baseline2left)
    target2base = ee2base(xarm) @ np.linalg.inv(ee2target_current)
    init_pose = target2base @ starting_ee2target_desired
    move_to_se3(xarm, init_pose)
    return target2base

if __name__ == '__main__':
    config = tyro.cli(Config)
    xarm = connect_arm(config.ip)
    xarm.set_tcp_offset([*map(meters_to_mm, config.tcp_origin), *config.tcp_flange_to_tool_euler], is_radian=False)
    enable_arm(xarm)
    camera = init_zed_camera()
    baseline2left = np.eye(4)
    baseline2left[0, 3] = config.baseline / 2

    target2base = datagen_arm_init(
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

def run_registration(camera, four_corners_3d, baseline2left: np.ndarray):
    with grabbed_frame(camera) as frame:
        left, _ = retrieve_stereo_images(frame)
    left_K, _ = get_intrinsics(camera)
    left_D, _ = get_distortion(camera)
    left2target = pnp_box(left, four_corners_3d, left_K, left_D)
    baseline2target = left2target @ baseline2left

    return baseline2target

def pnp_box(image, corners_3d_top_left_CCW, K, D):
    points_2d_top_left_CCW = annotate(image)
    success, rvec, tvec = cv2.solvePnP(corners_3d_top_left_CCW, points_2d_top_left_CCW, K, D, flags=cv2.SOLVEPNP_IPPE)
    assert success, "solvePnP failed"
    target2camera = make_se3(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    return np.linalg.inv(target2camera)

