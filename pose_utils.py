import pdb

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def cv2opengl(pose: np.ndarray) -> np.ndarray:
    return pose @ np.diag([1.0, -1.0, -1.0, 1.0])

def add_rotation(se3_matrix, z, y, x):
    rotation = R.from_euler('ZYX', [z, y, x], degrees=True).as_matrix()
    mat = se3_matrix.copy()
    mat[:3, :3] = rotation
    return mat

def circular_coordinates(origin: np.ndarray, radius: float, N: int):
    rad_ticks = np.linspace(0, 2 * np.pi, N, endpoint=False)
    coords_3d = np.stack((np.cos(rad_ticks), np.sin(rad_ticks), np.zeros_like(rad_ticks)), axis=-1) * radius
    return coords_3d + origin

def look_at(at_coord: np.ndarray, from_coord: np.ndarray):
    z_axis = (at_coord - from_coord) / np.linalg.norm(at_coord - from_coord)
    x_axis = np.cross(z_axis, [0, 0, 1])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack((x_axis, y_axis, z_axis), axis=-1)
    se3 = np.eye(4)
    se3[:3, :3] = rotation
    se3[:3, -1] = from_coord
    return se3

def generate_n_poses_with_rotation(xrange, yrange, zrange, Nx, Ny, Nz, anglez, angley, anglex) -> np.ndarray:
    """
    angle* are the active zyx intrinsic rotation parameters.
    """
    box_frame_poses = generate_offsets(xrange, yrange, zrange, Nx * Ny * Nz)
    se3_box_frame_poses = np.array(
        [
            add_rotation(offset_to_4x4(offset), anglez, angley, anglex)
            for offset in box_frame_poses
        ]
    )
    return se3_box_frame_poses

def generate_offsets(xrange, yrange, zrange, num):
    return np.random.uniform(*zip(xrange, yrange, zrange), size=(num, 3))

def offset_to_4x4(offset):
    assert offset.shape == (3,)
    se3 = np.eye(4)
    se3[:3, -1] = offset
    return se3

def xarmpose_to_se3(xarm_pose):
    ee_in_base = xarm_pose[:3]
    base_to_ee_euler = xarm_pose[3:]
    Ree2base = R.from_euler('ZYX', base_to_ee_euler, degrees=True).as_matrix()
    ee2base = np.eye(4)
    ee2base[:3, :3] = Ree2base
    ee2base[:3, -1] = ee_in_base
    return ee2base

def plot_coordinate_frame(ax, T, scale=1.0, alpha=0.8, label=None):
    """T is child2parent: columns are child's basis vectors in parent coords."""
    origin = T[:3, 3]
    for i, color in enumerate(['red', 'green', 'blue']):
        axis = T[:3, i] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                  axis[0], axis[1], axis[2],
                  color=color, alpha=alpha, arrow_length_ratio=0.1)
    if label:
        ax.text(origin[0], origin[1], origin[2], f'  {label}', fontsize=9)


def _arrow_scale_from_origins(origins):
    positions = np.array(origins)
    ranges = positions.max(axis=0) - positions.min(axis=0)
    max_range = ranges.max() if ranges.max() > 0 else 1.0
    return max_range / 12

def visualize_poses(poses, extra_frames=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    all_origins = [np.zeros(3)]
    if extra_frames:
        all_origins += [T[:3, 3] for T in extra_frames.values()]
    all_origins += [T[:3, 3] for T in poses]

    scale = _arrow_scale_from_origins(all_origins)

    plot_coordinate_frame(ax, np.eye(4), scale=scale * 1.4, alpha=1.0, label='base')
    if extra_frames:
        for name, T in extra_frames.items():
            plot_coordinate_frame(ax, T, scale=scale * 1.2, alpha=1.0, label=name)
    for T in poses:
        plot_coordinate_frame(ax, T, scale=scale)

    max_scale = scale * 1.4
    extremes = []
    for T, s in [(np.eye(4), scale * 1.4)] + \
                [(T, scale * 1.2) for T in (extra_frames or {}).values()] + \
                [(T, scale) for T in poses]:
        origin = T[:3, 3]
        extremes.append(origin)
        for i in range(3):
            extremes.append(origin + T[:3, i] * s)

    all_positions = np.array(extremes)
    margin = max_scale
    mins = all_positions.min(axis=0) - margin
    maxs = all_positions.max(axis=0) + margin
    max_extent = (maxs - mins).max()
    centers = (mins + maxs) / 2
    for setter, i in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        setter(centers[i] - max_extent / 2, centers[i] + max_extent / 2)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{len(poses)} SE3 poses')

    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='X-axis'),
        Line2D([0], [0], color='green', lw=2, label='Y-axis'),
        Line2D([0], [0], color='blue', lw=2, label='Z-axis'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

