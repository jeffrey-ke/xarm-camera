import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R
from goto_capture import goto_pose, connect_arm, enable_arm
from pose_utils import generate_n_poses_with_rotation


Nx, Ny, Nz = 4, 4, 4
xrange = (200, 400)
yrange = (-150, 150)
zrange = (20, 50)
#TODO: Need to encode this angle config to knowledge
anglez, angley, anglex = 90, 0, -90
poses = generate_n_poses_with_rotation(xrange, yrange, zrange, Nx, Ny, Nz, anglez, angley, anglex)
target2base = np.eye(4)
target2base[:3, :3] = R.from_euler('z', [180], degrees=True).as_matrix()
target2base[:3, -1] = [205+530, 0, -80]
poses_base = np.einsum('ij, ...jk -> ...ik', target2base, poses)
arm = connect_arm('192.168.1.241')
arm.set_tcp_offset([0, 0, 65, 90, 0, 90], is_radian=False)
enable_arm(arm)

np.set_printoptions(suppress=True)
try:
    for pose in poses_base:
        x, y, z = pose[:3, -1]
        yaw, pitch, roll = R.from_matrix(pose[:3, :3]).as_euler('ZYX', degrees=True)
        rotn = R.from_matrix(pose[:3, :3])
        goto_pose(arm, x, y, z, roll, pitch, yaw)
        capture = capture_zed_images()
finally:
    arm.disconnect()

