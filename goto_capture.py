#!/usr/bin/env python3
"""
xarm_zed_capture.py
-------------------
Move xArm to poses and capture ZED camera images for visual servoing.
"""

import atexit
import pdb
import time
import os
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from collections.abc import Iterator

from xarm.wrapper import XArmAPI
import pyzed.sl as sl
import numpy as np
import cv2

from convert_capture import convert_to_dataset
from xarm_datastructs import Zedpack, Mm, Meters, SafeZed


def connect_arm(ip: str):
    """Connect to the xArm robot."""
    try:
        arm = XArmAPI(ip)
        print(f"✓ Connected to xArm at {ip}")
        atexit.register(arm.disconnect)
        return arm
    except Exception as e:
        print(f"✗ Failed to connect to arm: {e}")
        return None


def enable_arm(arm: XArmAPI):
    """Enable the arm for motion."""
    print("Enabling arm...")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(0)  # Position control mode
    arm.set_state(state=0)
    time.sleep(0.5)

    code, state = arm.get_state()
    if state in (4, 5):
        print(f"⚠ Arm in {'STOP' if state == 4 else 'ERROR'} state. Recovering...")
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)
        time.sleep(0.5)
        code, state = arm.get_state()

    if state not in (1, 2, 3):
        print(f"✗ Arm not ready after enable (state={state})")
        return False

    print("✓ Arm enabled")
    return True

def test_capture(zed):
    """Grab one frame from a ZED camera and save the left image to a user-specified path."""
    path = Path(input("Absolute path to save image: ").strip())
    assert path.parent.is_dir(), f"Parent directory does not exist: {path.parent}"

    capture = capture_zed_images(zed)
    pdb.set_trace()

    cv2.imwrite(str(path), capture.left_image)
    print(f"✓ Saved {path}  ({capture.left_image.shape})")


def init_zed_camera():
    """Initialize ZED camera."""
    print("\nInitializing ZED camera...")
    
    # Create a Camera object
    zed = sl.Camera()
    
    # Create configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"✗ Failed to open ZED camera: {err}")
        return None
    
    print("✓ ZED camera initialized")
    atexit.register(zed.close)
    return zed


def get_intrinsics(zed):
    """Get camera intrinsic parameters."""

    info = zed.get_camera_information()
    left_calib = info.camera_configuration.calibration_parameters.left_cam
    right_calib = info.camera_configuration.calibration_parameters.right_cam

    left_K = np.diag([left_calib.fx, left_calib.fy, 1])
    left_K[:2, -1] = [left_calib.cx, left_calib.cy]

    right_K = np.diag([right_calib.fx, right_calib.fy, 1])
    right_K[:2, -1] = [right_calib.cx, right_calib.cy]

    return left_K, right_K


def get_distortion(zed):
    """Get camera distortion coefficients [k1, k2, p1, p2, k3]."""

    info = zed.get_camera_information()
    left_calib = info.camera_configuration.calibration_parameters.left_cam
    right_calib = info.camera_configuration.calibration_parameters.right_cam

    left_D = np.array(left_calib.disto)
    right_D = np.array(right_calib.disto)

    return left_D, right_D

@contextmanager
def grabbed_frame(zed: sl.Camera) -> Iterator[SafeZed]:
    if not isinstance(zed, sl.Camera):
        raise TypeError(f"expected sl.Camera, got {type(zed).__name__}")
    if zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED grab failed")
    frame = SafeZed(zed)
    try:
        yield frame
    finally:
        frame._invalidate()


def retrieve_stereo_images(frame: SafeZed) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(frame, SafeZed), f"expected SafeZed, got {type(frame).__name__}"
    left, right = sl.Mat(), sl.Mat()
    frame.retrieve_image(left, sl.VIEW.LEFT)
    frame.retrieve_image(right, sl.VIEW.RIGHT)
    return np.array(left.get_data(), copy=True), np.array(right.get_data(), copy=True)


def retrieve_depth(frame: SafeZed) -> np.ndarray:
    assert isinstance(frame, SafeZed), f"expected SafeZed, got {type(frame).__name__}"
    depth = sl.Mat()
    frame.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return np.array(depth.get_data(), copy=True)


def capture_zed_images(zed) -> Zedpack:
    with grabbed_frame(zed) as frame:
        left_image, right_image = retrieve_stereo_images(frame)
        left_K, right_K = get_intrinsics(zed)
        return Zedpack(
            left_image=left_image,
            left_depth=retrieve_depth(frame),
            right_image=right_image,
            left_K=left_K,
            right_K=right_K,
        )


def save_raw_captures(results, save_dir):
    """Save raw capture results to disk for archival."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        pose_name = result['pose_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(save_dir / f"{pose_name}_{timestamp}_left.png"), result['left_image'])
        cv2.imwrite(str(save_dir / f"{pose_name}_{timestamp}_right.png"), result['right_image'])
        np.save(str(save_dir / f"{pose_name}_{timestamp}_depth.npy"), result['left_depth'])


def goto_pose(arm: XArmAPI, x: Mm, y: Mm, z: Mm, roll, pitch, yaw, wait=True):
    """
    Move arm to specified Cartesian pose.
    
    Args:
        arm: XArmAPI instance
        x, y, z: Target position in mm
        roll, pitch, yaw: Target orientation in degrees
        speed: Movement speed in mm/s
        wait: Wait for motion to complete
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get current position
    code, current = arm.get_position()
    if code != 0:
        print(f"✗ Failed to get current position")
        return False
    
    # Calculate movement distance
    dx = x - current[0]
    dy = y - current[1]
    dz = z - current[2]
    distance = (dx**2 + dy**2 + dz**2)**0.5
    
    print(f"\nMoving to pose:")
    print(f"  Current: [{', '.join(f'{p:.2f}' for p in current[:3])}] mm")
    print(f"  Target:  [{x:.2f}, {y:.2f}, {z:.2f}] mm")
    print(f"  Distance: {distance:.2f} mm")
    
    # Execute movement
    code = arm.set_position(
        x=x, y=y, z=z,
        roll=roll, pitch=pitch, yaw=yaw,
        wait=wait,
        radius=-1.0
    )
    
    if code == 0:
        print("  ✓ Movement completed")
        # Verify position
        code, final = arm.get_position()
        if code == 0:
            print(f"  Final: [{', '.join(f'{p:.2f}' for p in final[:3])}] mm")
        return True
    else:
        print(f"  ✗ Movement failed with code: {code}")
        code, state = arm.get_state()
        code, err = arm.get_err_warn_code()
        print(f"  State: {state}, Error: {err[0]}, Warning: {err[1]}")
        return False

def fallback(xarm):
    xarm.disconnect()
    assert False, "Crash!"

def capture_at_pose(arm: XArmAPI, zed, x, y, z, roll, pitch, yaw,
                    pose_name, speed=30):
    """
    Move to pose and capture images.

    Returns:
        dict with pose_name, target_pose, actual_pose,
        left_image, right_image, left_depth — or None on failure.
    """
    print(f"\n{'='*60}")
    print(f"Pose: {pose_name}")
    print(f"{'='*60}")

    success = goto_pose(arm, x, y, z, roll, pitch, yaw)
    if not success:
        return None

    time.sleep(1.0)

    capture = capture_zed_images(zed)

    code, final_pose = arm.get_position()

    return {
        'pose_name': pose_name,
        'target_pose': [x, y, z, roll, pitch, yaw],
        'actual_pose': final_pose if code == 0 else None,
        'left_image': capture.left_image,
        'right_image': capture.right_image,
        'left_depth': capture.left_depth,
    }


# def main():
#     # Configuration
#     XARM_IP = "192.168.1.241"
#     SAVE_DIR = Path("./zed_captures")
#     SAVE_DIR.mkdir(exist_ok=True)
#
#     DATASET_DIR = Path("./datasets/real_robot")
#     SCENE_NUM = 1
#     TRANSFORM_MATRIX_PATH = Path("./calibration/T_target_from_robot.npy")
#
#     print("=" * 60)
#     print("xArm + ZED Camera Data Collection")
#     print("=" * 60)
#
#     # Connect to arm
#     arm = connect_arm(XARM_IP)
#     if arm is None:
#         return
#
#     # Initialize ZED camera
#     zed = init_zed_camera()
#     test_capture(zed)
#
#     if zed is None:
#         arm.disconnect()
#         return
#
#     try:
#         # Enable arm
#         enable_arm(arm)
#
#         # Get current pose as home
#         code, home_pose = arm.get_position()
#         if code != 0:
#             print("✗ Failed to get home pose")
#             return
#
#         print(f"\nHome pose: [{', '.join(f'{p:.2f}' for p in home_pose)}]")
#
#         # Generate pose variations
#         variation_range = 30.0  # mm
#                                         variation_range=variation_range)
#
#         print(f"\n✓ Generated {len(poses)} poses (1 home + {len(poses)-1} variations)")
#         print(f"  Variation range: ±{variation_range} mm")
#
#         # Capture data at each pose
#         results = []
#
#
#
#         # Convert to dataset format
#         index_path = convert_to_dataset(results, DATASET_DIR, SCENE_NUM, T)
#
#         print(f"\n{'='*60}")
#         print(f"✓ Data collection complete!")
#         print(f"  Captured: {len(results)}/{len(poses)} poses")
#         print(f"  Raw saves: {SAVE_DIR}")
#         print(f"  Dataset index: {index_path}")
#         print(f"{'='*60}")
#
#     except KeyboardInterrupt:
#         print("\n\n⚠ Interrupted by user")
#
#     finally:
#         # Cleanup
#         if zed is not None:
#             zed.close()
#             print("✓ ZED camera closed")
#
#         arm.disconnect()
#         print("✓ Arm disconnected")



# if __name__ == "__main__":
#     main()
