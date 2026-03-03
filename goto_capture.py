#!/usr/bin/env python3
"""
xarm_zed_capture.py
-------------------
Move xArm to poses and capture ZED camera images for visual servoing.
"""

from xarm.wrapper import XArmAPI
import pyzed.sl as sl
import numpy as np
import cv2
import time
import os
from pathlib import Path
from datetime import datetime

from convert_capture import convert_to_dataset


def connect_arm(ip: str):
    """Connect to the xArm robot."""
    try:
        arm = XArmAPI(ip)
        print(f"✓ Connected to xArm at {ip}")
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
    assert capture is not None, "ZED grab failed"

    cv2.imwrite(str(path), capture["left_data"])
    print(f"✓ Saved {path}  ({capture['left_data'].shape})")


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
    return zed


def capture_zed_images(zed):
    """
    Capture images from ZED camera.

    Returns:
        dict with left_data, right_data, depth_data as numpy arrays,
        or None on failure.
    """
    runtime_params = sl.RuntimeParameters()

    image_left = sl.Mat()
    image_right = sl.Mat()
    depth_map = sl.Mat()

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return None

    zed.retrieve_image(image_left, sl.VIEW.LEFT)
    zed.retrieve_image(image_right, sl.VIEW.RIGHT)
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    return {
        'left_data': image_left.get_data(),
        'right_data': image_right.get_data(),
        'depth_data': depth_map.get_data(),
    }


def save_raw_captures(results, save_dir):
    """Save raw capture results to disk for archival."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        pose_name = result['pose_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(save_dir / f"{pose_name}_{timestamp}_left.png"), result['left_data'])
        cv2.imwrite(str(save_dir / f"{pose_name}_{timestamp}_right.png"), result['right_data'])
        np.save(str(save_dir / f"{pose_name}_{timestamp}_depth.npy"), result['depth_data'])


def goto_pose(arm: XArmAPI, x, y, z, roll, pitch, yaw, wait=True):
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


def capture_at_pose(arm: XArmAPI, zed, x, y, z, roll, pitch, yaw,
                    pose_name, speed=30):
    """
    Move to pose and capture images.

    Returns:
        dict with pose_name, target_pose, actual_pose,
        left_data, right_data, depth_data — or None on failure.
    """
    print(f"\n{'='*60}")
    print(f"Pose: {pose_name}")
    print(f"{'='*60}")

    success = goto_pose(arm, x, y, z, roll, pitch, yaw, speed=speed)
    if not success:
        return None

    time.sleep(1.0)

    capture = capture_zed_images(zed)
    if capture is None:
        print("  ✗ Failed to capture images")
        return None

    code, final_pose = arm.get_position()

    return {
        'pose_name': pose_name,
        'target_pose': [x, y, z, roll, pitch, yaw],
        'actual_pose': final_pose if code == 0 else None,
        'left_data': capture['left_data'],
        'right_data': capture['right_data'],
        'depth_data': capture['depth_data'],
    }

def generate_poses_from_home():
    """
    Generate pose variations around a home pose.
    
    Args:
        home_pose: [x, y, z, roll, pitch, yaw] home position
        num_variations: Number of pose variations to generate
        variation_range: Maximum variation in mm for position
        
    Returns:
        list: List of pose variations
    """
    poses = []
    
    # Add home pose
    poses.append({
        'name': 'home',
        'pose': home_pose
    })
    
    # Generate variations
    variations = [
        ('pos_x', [variation_range, 0, 0, 0, 0, 0]),
        ('neg_x', [-variation_range, 0, 0, 0, 0, 0]),
        ('pos_y', [0, variation_range, 0, 0, 0, 0]),
        ('neg_y', [0, -variation_range, 0, 0, 0, 0]),
        ('pos_z', [0, 0, variation_range, 0, 0, 0]),
        ('neg_z', [0, 0, -variation_range, 0, 0, 0]),
        ('diag_1', [variation_range/2, variation_range/2, 0, 0, 0, 0]),
        ('diag_2', [variation_range/2, -variation_range/2, 0, 0, 0, 0]),
        ('up_forward', [variation_range/2, 0, variation_range/2, 0, 0, 0]),
    ]
    
    for i, (name, delta) in enumerate(variations[:num_variations]):
        varied_pose = [
            home_pose[j] + delta[j] for j in range(6)
        ]
        poses.append({
            'name': f'var_{i+1}_{name}',
            'pose': varied_pose
        })
    
    return poses


def main():
    # Configuration
    XARM_IP = "192.168.1.241"
    SAVE_DIR = Path("./zed_captures")
    SAVE_DIR.mkdir(exist_ok=True)

    DATASET_DIR = Path("./datasets/real_robot")
    SCENE_NUM = 1
    TRANSFORM_MATRIX_PATH = Path("./calibration/T_target_from_robot.npy")
    
    print("=" * 60)
    print("xArm + ZED Camera Data Collection")
    print("=" * 60)
    
    # Connect to arm
    arm = connect_arm(XARM_IP)
    if arm is None:
        return

    # Initialize ZED camera
    zed = init_zed_camera()
    test_capture(zed)

    if zed is None:
        arm.disconnect()
        return
    
    try:
        # Enable arm
        enable_arm(arm)
        
        # Get current pose as home
        code, home_pose = arm.get_position()
        if code != 0:
            print("✗ Failed to get home pose")
            return
        
        print(f"\nHome pose: [{', '.join(f'{p:.2f}' for p in home_pose)}]")
        
        # Generate pose variations
        variation_range = 30.0  # mm
        poses = generate_poses_from_home(home_pose, num_variations=8, 
                                        variation_range=variation_range)
        
        print(f"\n✓ Generated {len(poses)} poses (1 home + {len(poses)-1} variations)")
        print(f"  Variation range: ±{variation_range} mm")
        
        # Capture data at each pose
        results = []

        for i, pose_info in enumerate(poses):
            pose_name = pose_info['name']
            pose = pose_info['pose']

            result = capture_at_pose(
                arm, zed,
                x=pose[0], y=pose[1], z=pose[2],
                roll=pose[3], pitch=pose[4], yaw=pose[5],
                pose_name=pose_name,
                speed=30
            )

            if result:
                results.append(result)
            else:
                print(f"  ⚠ Failed to capture at pose {pose_name}")

            time.sleep(0.5)

        # Save raw captures for archival
        save_raw_captures(results, SAVE_DIR)

        # Convert to dataset format
        T = np.load(str(TRANSFORM_MATRIX_PATH))
        index_path = convert_to_dataset(results, DATASET_DIR, SCENE_NUM, T)

        print(f"\n{'='*60}")
        print(f"✓ Data collection complete!")
        print(f"  Captured: {len(results)}/{len(poses)} poses")
        print(f"  Raw saves: {SAVE_DIR}")
        print(f"  Dataset index: {index_path}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        
    finally:
        # Cleanup
        if zed is not None:
            zed.close()
            print("✓ ZED camera closed")
        
        arm.disconnect()
        print("✓ Arm disconnected")



if __name__ == "__main__":
    main()
