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

def make_index(config): 
    xarm = connect_arm(config.ip)
    xarm.set_tcp_offset([*map(meters_to_mm, config.tcp_origin), *config.tcp_flange_to_tool_euler], is_radian=False)
    enable_arm(xarm)
    camera = init_zed_camera()

    yaw, pitch, roll = config.target_to_ee_ypr
    target_frame_poses = [
        add_rotation(
            offset_to_4x4(offset), z=yaw, y=pitch, x=roll
        )
        for offset in generate_offsets(config.xrange, config.yrange, config.zrange, config.no_kfs)
    ]

    target2base = np.eye(4)
    target2base[:3, :3] = R.from_euler('ZYX', config.base_to_target_ypr, degrees=True).as_matrix()
    target2base[:3, -1] = config.target_in_base_offset
    base_frame_poses = target2base @ target_frame_poses
    # visualize_poses(base_frame_poses, extra_frames={'target': target2base})

    captures = []
    for robot2base in map(xarmpose_to_se3, move_to(xarm, base_frame_poses)):
        zed_pack = capture_zed_images(camera)
        robot2target = np.linalg.inv(target2base) @ robot2base
        captures.append(
            Capture(
                offset=robot2target[:3, -1],
                euler_target_to_robot=R.from_matrix(robot2target[:3, :3]).as_euler('ZYX', degrees=True),
                left_image=zed_pack.left_image,
                left_depth=zed_pack.left_depth,
                right_image=zed_pack.right_image,
                left_calib=zed_pack.left_K
            )
        )
    convert_to_dataset(captures, config.dataset_dir, config.idx)

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
    make_index(config)
