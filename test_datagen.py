import os
import pdb
import sys
sys.path.insert(0, '..')

from eval import orbit_capture
from dataset import StereoImageDataset

sys.path.insert(0, '.')

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import imageio

from datagen import plan_poses, pnp_box, dry_run_datagen_arm_init, Config
from goto_capture import connect_arm, enable_arm, init_zed_camera
from xarm_datastructs import meters_to_mm
from pose_utils import make_se3
import trimesh_wrapper as tw
import tyro


def build_target2base(offset, ypr_degrees):
    rot = R.from_euler('ZYX', ypr_degrees, degrees=True).as_matrix()
    return make_se3(offset, rot)


def test_plan_poses():
    target2base = build_target2base(
        offset=(0.745, 0, -0.116 + 0.093),
        ypr_degrees=(180, 0, 0),
    )

    poses = plan_poses(
        target_to_ee_ypr_desired=(90, 0, -90),
        xrange=(0.300, 0.500),
        yrange=(-0.12, 0.12),
        zrange=(0.02, 0.08),
        no_kfs=20,
        target2base=target2base,
    )

    scene = trimesh.Scene()
    axis = tw.Geometry(trimesh.creation.axis(origin_size=0.004, axis_length=0.04), 'axis')

    base = tw.Node(geometry=axis, name='base_frame')
    target = tw.Node(geometry=axis, name='target_frame')

    tw.add_node(scene, base)
    tw.add_node(scene, target, base, transform=target2base)

    for i, pose in enumerate(poses):
        node = tw.Node(geometry=axis, name=f'pose_{i}')
        tw.add_node(scene, node, parent=base, transform=pose)

    frames = list(orbit_capture(scene, N=60, point_size=10.0))
    imageio.mimsave('test_plan_poses.gif', frames, duration=100, loop=0)
    print(f'Saved {len(frames)}-frame orbit gif of {len(poses)} poses to test_plan_poses.gif')


def calib_path_from_image_path(left_image_path):
    render_dir = os.path.dirname(os.path.dirname(left_image_path))
    idx_str = os.path.basename(left_image_path).replace('rgb_', '').replace('.png', '')
    return os.path.join(render_dir, 'gripper_left_camera_calib_npy', f'calib_{idx_str}.npy')


def sample_left_image_as_hwc(sample):
    return cv2.imread(sample.original_image_path[0], cv2.IMREAD_COLOR)


def test_box_pnp():
    dataset = StereoImageDataset('../datasets/real')
    target_path = 'render/render001/gripper_left_rgb/rgb_0042.png'
    sample = next(s for s in dataset if s.original_image_path[0].endswith(target_path))

    image = sample_left_image_as_hwc(sample)

    corners_3d = np.array([
        [0, -0.225/2, 0.122],   # top-left
        [0, -0.225/2, 0],   # next corner CCW
        [0, 0.225/2, 0],   # opposite corner
        [0, 0.225/2, 0.122],   # last corner CCW
    ], dtype=np.float64)

    K = np.load(calib_path_from_image_path(sample.original_image_path[0]))
    D = np.zeros(4)

    baseline2left = np.eye(4)
    baseline2left[:3, -1] = [0.063/2, 0, 0]
    ground_truth = sample.offset.numpy()
    R_target2baseline = R.from_euler('ZYX', (90, 0, -90), degrees=True).as_matrix().T
    ground_truth_baseline_frame = np.dot(R_target2baseline, -ground_truth)
    ground_truth = np.dot(baseline2left, np.concatenate((ground_truth_baseline_frame, [1])))[:-1]
    ground_truth = np.dot(R_target2baseline.T, -ground_truth)
    camera2target = pnp_box(image, corners_3d, K, D)
    estimated_translation = camera2target[:3, 3]


    print(f"Estimated translation:    {estimated_translation}")
    print(f"Ground truth offset:      {ground_truth}")
    print(f"Difference:               {estimated_translation - ground_truth}")

def test_datagen_dry_run():
    config = tyro.cli(Config)
    xarm = connect_arm(config.ip)
    xarm.set_tcp_offset([*map(meters_to_mm, config.tcp_origin), *config.tcp_flange_to_tool_euler], is_radian=False)
    enable_arm(xarm)
    camera = init_zed_camera()
    baseline2left = np.eye(4)
    baseline2left[0, 3] = config.baseline / 2

    init_pose, target2base = dry_run_datagen_arm_init(
        config.target_to_ee_ypr_desired, config.init_ee_in_target_offset_desired,
        camera,
        np.array(config.corners_3d_top_left_CCW),
        xarm,
        baseline2left,
    )

    scene = trimesh.Scene()
    axis = tw.Geometry(trimesh.creation.axis(origin_size=0.004, axis_length=0.04), 'axis')

    base = tw.Node(geometry=axis, name='base_frame')
    target = tw.Node(geometry=axis, name='target_frame')
    init_node = tw.Node(geometry=axis, name='init_pose')

    tw.add_node(scene, base)
    tw.add_node(scene, target, base, transform=target2base)
    # tw.add_node(scene, init_node, parent=base, transform=init_pose)
    pdb.set_trace()

    frames = list(orbit_capture(scene, N=60, point_size=10.0))
    imageio.mimsave('test_dry_run.gif', frames, duration=100, loop=0)
    print(f'Saved {len(frames)}-frame orbit gif to test_dry_run.gif')


if __name__ == '__main__':
    test_datagen_dry_run()
