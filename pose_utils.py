import pdb

import numpy as np
from scipy.spatial.transform import Rotation as R

def add_rotation(se3_matrix, z, y, x):
    rotation = R.from_euler('ZYX', [z, y, x], degrees=True).as_matrix()
    mat = se3_matrix.copy()
    mat[:3, :3] = rotation
    return mat


def generate_n_poses_with_rotation(xrange, yrange, zrange, Nx, Ny, Nz, anglez, angley, anglex) -> np.ndarray:
    """
    angle* are the active zyx intrinsic rotation parameters.
    """
    box_frame_poses = generate_offsets(xrange, yrange, zrange, Nx, Ny, Nz)
    se3_box_frame_poses = np.array(
        [
            add_rotation(offset_to_4x4(offset), anglez, angley, anglex)
            for offset in box_frame_poses
        ]
    )
    return se3_box_frame_poses

def generate_offsets(xrange, yrange, zrange, Nx, Ny, Nz):
    xx, yy, zz = np.meshgrid(
        np.linspace(*xrange, num=Nx), 
        np.linspace(*yrange, num=Ny), 
        np.linspace(*zrange, num=Nz), 
        indexing='xy'
    )
    return np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

def offset_to_4x4(offset):
    assert offset.shape == (3,)
    se3 = np.eye(4)
    se3[:3, -1] = offset
    return se3

