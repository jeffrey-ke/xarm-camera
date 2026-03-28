from dataclasses import asdict, dataclass, fields
import json
import pdb
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import trimesh

import trimesh_wrapper as tw
from scene_labels import add_label_billboards
from pose_utils import generate_random_offsets, generate_grid_offsets, xarmpose_to_se3, visualize_poses, add_rotation, offset_to_4x4, make_se3

from .convert_capture import convert_to_dataset
from .goto_capture import connect_arm, capture_zed_images, goto_pose, init_zed_camera, fallback, enable_arm, grabbed_frame, retrieve_stereo_images, get_intrinsics, get_distortion, get_baseline, hardware_init, ee2base, move_to, move_to_se3
from .image_utils import annotate, sanity_check_gt
from .xarm_datastructs import Capture, Mm, Meters


@dataclass
class Config:
    grid_dims: tuple[int, int, int] | None
    dataset_dir: str
    idx: int
    rectified: bool

    corners_3d_top_left_CCW: tuple[tuple[float, float, float], ...] = (
        (0, -0.225/2, 0.122),
        (0, -0.225/2, 0),
        (0, 0.225/2, 0),
        (0, 0.225/2, 0.122),
    )

    xrange: tuple[Meters, Meters] = (.450, .700)
    yrange: tuple[Meters, Meters] = (-0.12, 0.12)
    zrange: tuple[Meters, Meters] = (0.02, 0.10)
    target_to_ee_ypr_desired: tuple[float, float, float] = (90, 0, -90)
    init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters] = (0.3, 0, 0.06)

    target_in_base_offset: tuple[Meters, Meters, Meters] = (0.745, 0, -0.116 + 0.093)
    base_to_target_ypr: tuple[float, float, float] = (180, 0, 0)
    ip: str = '192.168.1.241'
    tcp_origin: tuple[Meters, Meters, Meters] = (0, 0, 0.055)
    tcp_flange_to_tool_euler: tuple[float, float, float] = (90, 0, 90)

    params_in_meters: tuple[str, ...] = ('xrange', 'yrange', 'zrange', 'target_in_base_offset', 'tcp_origin')

    no_kfs: int = 1

    def __post_init__(self):
        assert Path(self.dataset_dir).parent.exists(), f"{self.dataset_dir} doesn't exist!"
        render_dir = Path(self.dataset_dir) / "render" / f"render{self.idx:03d}"
        if render_dir.exists() and any(render_dir.iterdir()):
            raise FileExistsError(
                f"Scene {self.idx} already exists at {render_dir}. "
                f"Choose a different idx or delete the existing directory."
            )
        self.render_dir = render_dir
        self.render_dir.mkdir(parents=True, exist_ok=True)

def plan_poses(target_to_ee_ypr_desired, xrange, yrange, zrange,
               sampling: int | tuple[int, int, int],
               target2base: np.ndarray):
    yaw, pitch, roll = target_to_ee_ypr_desired
    if isinstance(sampling, tuple):
        offsets = generate_grid_offsets(xrange, yrange, zrange, *sampling)
    else:
        offsets = generate_random_offsets(xrange, yrange, zrange, sampling)
    target_frame_poses = np.array([
        add_rotation(offset_to_4x4(offset), z=yaw, y=pitch, x=roll)
        for offset in offsets
    ])
    base_frame_poses = target2base @ target_frame_poses
    return base_frame_poses

def poses_to_scene(base_frame_poses: np.ndarray, target2base: np.ndarray) -> trimesh.Scene:
    scene = trimesh.Scene()
    axis = tw.Geometry(trimesh.creation.axis(origin_size=0.004, axis_length=0.04), 'axis')
    base = tw.Node(geometry=axis, name='base_frame')
    target = tw.Node(geometry=axis, name='target_frame')
    tw.add_node(scene, base)
    tw.add_node(scene, target, base, transform=target2base)
    for i, pose in enumerate(base_frame_poses):
        tw.add_node(scene, tw.Node(geometry=axis, name=f'pose_{i}'), parent=base, transform=pose)

    first = tw.Node(geometry=axis, name='first')
    last = tw.Node(geometry=axis, name='last')
    tw.add_node(scene, first, parent=base, transform=base_frame_poses[0])
    tw.add_node(scene, last, parent=base, transform=base_frame_poses[-1])

    tw.ground_plane(scene, z=target2base[2, 3], parent=base)
    add_label_billboards(scene, [base, target, first, last], height=0.012)

    return scene

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

def make_index(target_to_ee_ypr_desired: tuple[float, float, float],
               xrange, yrange, zrange,
               sampling: int | tuple[int, int, int],
               xarm,
               camera,
               target2base: np.ndarray,
               *,
               rectified: bool = True,
               ):

    base_frame_poses = plan_poses(target_to_ee_ypr_desired, xrange, yrange, zrange, sampling, target2base)
    distance_from_target = lambda pose: np.linalg.norm((np.linalg.inv(target2base) @ pose)[:3, -1])
    distances = [distance_from_target(pose) for pose in base_frame_poses]
    closest_pose = min(base_frame_poses, key=distance_from_target)

    try:
        scene = poses_to_scene(base_frame_poses, target2base)
        big_axis = tw.Geometry(trimesh.creation.axis(origin_size=0.008, axis_length=0.08), 'big_axis')
        closest_node = tw.Node(geometry=big_axis, name='closest')
        tw.add_node(scene, closest_node, transform=closest_pose)
        add_label_billboards(scene, [closest_node], height=0.012)
        scene.show()
    except KeyboardInterrupt:
        pass
    if input("Accept poses? [y/n]: ").strip().lower() != 'y':
        raise SystemExit("Poses rejected")

    captures = [
        make_capture(robot2base, target2base, capture_zed_images(camera, rectified=rectified))
        for robot2base in map(xarmpose_to_se3, move_to(xarm, base_frame_poses, speed=60))
    ]
    return closest_pose, captures

def datagen_arm_init(
        target_to_ee_ypr_desired, init_ee_in_target_offset_desired: tuple[Meters, Meters, Meters],
        camera,
        corners_3d_top_left_CCW,
        xarm,
        baseline2left: np.ndarray,
    ):
    ee2target_current = run_registration(camera, corners_3d_top_left_CCW, baseline2left)
    annotated = sanity_check_gt(ee2target_current, camera, baseline2left)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGRA2RGB))
    plt.axis('off')
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    if input("GT looks correct? [y/n]: ").strip().lower() != 'y':
        raise SystemExit("GT rejected")
    starting_ee2target_desired = make_se3(ee2target_current[:3, -1], R.from_euler('ZYX', target_to_ee_ypr_desired, degrees=True).as_matrix())
    target2base = ee2base(xarm) @ np.linalg.inv(ee2target_current)
    init_pose = target2base @ starting_ee2target_desired
    move_to_se3(xarm, init_pose, speed=10)
    return init_pose, target2base

def run_registration(camera, four_corners_3d, baseline2left: np.ndarray):
    with grabbed_frame(camera) as frame:
        left, _ = retrieve_stereo_images(frame, rectified=True)
    left_K, _ = get_intrinsics(camera)

    four_corners_3d = np.array(four_corners_3d, dtype=np.float64)

    points_2d = annotate(left)
    success, rvec, tvec = cv2.solvePnP(four_corners_3d, points_2d, left_K, np.zeros(4), flags=cv2.SOLVEPNP_SQPNP)
    assert success, "solvePnP failed"
    target2left = make_se3(tvec.ravel(), cv2.Rodrigues(rvec)[0])
    left2target = np.linalg.inv(target2left)

    baseline2target = left2target @ baseline2left

    return baseline2target

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
        baseline2left,
    )

    np.save(Path(config.render_dir) / 'target2base.npy', target2base)
    with open(Path(config.render_dir) / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)

    sampling = config.grid_dims if config.grid_dims is not None else config.no_kfs
    closest_pose, captures = make_index(
        config.target_to_ee_ypr_desired,
        config.xrange, config.yrange, config.zrange,
        sampling,
        xarm,
        camera,
        target2base,
        rectified=config.rectified,
    )

    convert_to_dataset(captures, config.dataset_dir, config.idx)

    print('Arm moving to closest_pose to target')
    move_to_se3(xarm, closest_pose, speed=50)
    xarm.set_mode(2)
    xarm.set_state(0)
    print("Arm is now in teach mode.")
