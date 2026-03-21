from dataclasses import dataclass, fields
import pdb
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro

from convert_capture import convert_to_dataset
from goto_capture import connect_arm, capture_zed_images, goto_pose, init_zed_camera, fallback, enable_arm
from pose_utils import generate_offsets, xarmpose_to_se3, visualize_poses, add_rotation, offset_to_4x4
from datastructs import Capture, Mm, Meters, meters_to_mm, mm_to_meters

@dataclass
class Config:
    xrange: tuple[Meters, Meters] = (.300, .500)
    yrange: tuple[Meters, Meters] = (-0.12, 0.12)
    zrange: tuple[Meters, Meters] = (0.02, 0.08)
    target_to_ee_ypr: tuple[float, float, float] = (90, 0, -90)

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

def plan_poses(target_to_ee_ypr, xrange, yrange, zrange, no_kfs, base_to_target_ypr, target_in_base_offset):
    yaw, pitch, roll = target_to_ee_ypr
    target_frame_poses = [
        add_rotation(offset_to_4x4(offset), z=yaw, y=pitch, x=roll)
        for offset in generate_offsets(xrange, yrange, zrange, no_kfs)
    ]
    target2base = np.eye(4)
    target2base[:3, :3] = R.from_euler('ZYX', base_to_target_ypr, degrees=True).as_matrix()
    target2base[:3, -1] = target_in_base_offset
    base_frame_poses = target2base @ target_frame_poses
    return base_frame_poses, target2base

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

def make_index(ip, tcp_origin, tcp_flange_to_tool_euler,
               target_to_ee_ypr, xrange, yrange, zrange, no_kfs, base_to_target_ypr, target_in_base_offset,
               dataset_dir, idx):
    xarm = connect_arm(ip)
    xarm.set_tcp_offset([*map(meters_to_mm, tcp_origin), *tcp_flange_to_tool_euler], is_radian=False)
    enable_arm(xarm)
    camera = init_zed_camera()

    base_frame_poses, target2base = plan_poses(
        target_to_ee_ypr, xrange, yrange, zrange, no_kfs, base_to_target_ypr, target_in_base_offset,
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


if __name__ == '__main__':
    config = tyro.cli(Config)
    make_index(
        config.ip, config.tcp_origin, config.tcp_flange_to_tool_euler,
        config.target_to_ee_ypr, config.xrange, config.yrange, config.zrange,
        config.no_kfs, config.base_to_target_ypr, config.target_in_base_offset,
        config.dataset_dir, config.idx,
    )
