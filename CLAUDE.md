# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Data collection pipeline for visual servoing: drives an xArm robot to random poses around a target, captures stereo images from a ZED camera at each pose, and writes a dataset consumable by `StereoImageDataset`.

## Running

```bash
# Main entry point — all config via tyro CLI flags
python datagen.py --dataset-dir /path/to/output --idx 0 --no-kfs 10

# Earlier prototype (hardcoded config, no target frame, units in mm throughout)
python main.py
```

Requires physical hardware: xArm (via `xarm-python-sdk`) and ZED camera (via `pyzed`). No tests or linting configured.

## How `datagen.py` Composes the Mechanisms

`datagen.py` is the orchestrator. It imports pure mechanisms from three modules and sequences them into a data-collection run:

1. **Pose generation** (`pose_utils`): `generate_random_offsets` produces random xyz positions in the target frame's bounding box. `offset_to_4x4` + `add_rotation` turn each offset into a full SE3 pose (position + fixed end-effector orientation from `Config.target_to_ee_ypr`).

2. **Frame transform**: A `target2base` SE3 matrix is built from `Config.target_in_base_offset` and `Config.base_to_target_ypr`. Batch-multiplying `target2base @ target_frame_poses` brings all poses into the robot's base frame.

3. **Motion execution** (`goto_capture`): `move_to` iterates the base-frame poses, converts meters→mm, calls `goto_pose` (which drives `xarm.set_position`), reads back the actual pose from the robot, converts mm→meters, and yields it. On error, `fallback` disconnects and crashes.

4. **Image capture** (`goto_capture`): After each move, `capture_zed_images` grabs stereo RGB + depth + intrinsics from the ZED, packed into a `Zedpack`.

5. **Record assembly**: For each (actual_pose, zed_pack) pair, `datagen` computes `robot2target = inv(target2base) @ robot2base` and builds a `Capture` dataclass with the offset, full base pose, euler angles, images, depth, and calibration.

6. **Serialization** (`convert_capture`): `convert_to_dataset` writes an index JSON (offsets per keyframe) and a render directory tree (left/right RGB PNGs, depth .npy, calibration .npy, base-pose .npy) in the layout `StereoImageDataset` expects.

### Key design point

All poses in `Config` and throughout the pipeline are in **meters**. The mm↔meters boundary lives only in `move_to` (outbound to xArm API) and when reading back `xarm.get_position()`. The `Mm`/`Meters` newtypes in `datastructs` make this explicit.

## Module Roles

| Module | Role |
|---|---|
| `datagen.py` | **Policy**: Config, orchestration, the `make_index` pipeline |
| `pose_utils.py` | **Mechanism**: SE3 math — offset generation, rotation composition, pose visualization |
| `goto_capture.py` | **Mechanism**: Hardware I/O — xArm connection/motion, ZED camera capture |
| `convert_capture.py` | **Mechanism**: Dataset serialization to `StereoImageDataset` layout |
| `datastructs.py` | **Mechanism**: Shared data types (`Capture`, `Zedpack`, `Mm`/`Meters` newtypes + converters) |
| `image_utils.py` | **Mechanism**: OpenCV point-picking UI for manual annotation |
| `main.py` | Legacy prototype of the same pipeline (hardcoded, mm units, no target frame) |

## Dataset Layout (produced by `convert_capture.py`)

```
dataset_dir/
  index/index{scene:03d}.json          # {"key_frames": [{"index": i, "offset": [x,y,z]}, ...]}
  render/render{scene:03d}/
    gripper_left_rgb/rgb_{i:04d}.png
    gripper_right_rgb/rgb_{i:04d}.png
    gripper_left_depth/distance_to_camera_{i:04d}.npy
    gripper_left_camera_calib_npy/calib_{i:04d}.npy
    gripper_base_pose/se3_pose_{i:04d}.npy
```

## Coordinate Conventions

- **Euler angles**: ZYX intrinsic (yaw, pitch, roll) in degrees everywhere via `scipy.spatial.transform.Rotation`.
- **xArm API**: Positions in mm, euler angles in degrees (`roll, pitch, yaw` argument order to `set_position`).
- **Target frame**: Defined by `target_in_base_offset` (translation from base) and `base_to_target_ypr` (rotation). The target frame is where the object of interest sits.
- **TCP offset**: `tcp_origin` (meters, converted to mm) + `tcp_flange_to_tool_euler` (degrees) — set once on the arm at startup.

## trimesh & Scene Graph Reference

API docs for `trimesh_wrapper.py` and trimesh library quirks: [`trimesh_wrapper.md`](../trimesh_wrapper.md) in the outer directory.

## ZED SDK Reference

API docs for the ZED Python SDK (pyzed/sl) — rectification, calibration, and ZED Mini specifics: [`zed.md`](zed.md) in this directory.

## xArm SDK Reference

Full API reference: [`xarm.md`](xarm.md) in this directory. SDK source: `/home/jeffk/repo/visual_servoing/xArm-Python-SDK/`.

### xArm API Functions Used in This Project

| Function | Where used | Purpose |
|----------|-----------|---------|
| `XArmAPI(ip)` | `goto_capture.connect_arm` | Connect to robot |
| `arm.disconnect()` | `goto_capture.connect_arm` (atexit), `fallback` | Clean shutdown |
| `arm.clean_error()` | `goto_capture.enable_arm` | Clear error state before enabling |
| `arm.clean_warn()` | `goto_capture.enable_arm` | Clear warning state |
| `arm.motion_enable(enable=True)` | `goto_capture.enable_arm` | Enable servo motors |
| `arm.set_mode(0)` | `goto_capture.enable_arm` | Position control mode |
| `arm.set_state(state=0)` | `goto_capture.enable_arm` | Ready for motion |
| `arm.get_state()` | `goto_capture.enable_arm` | Check if arm is ready |
| `arm.set_tcp_offset([x,y,z,r,p,y])` | `datagen.__main__` | Set tool center point offset (mm, degrees) |
| `arm.set_position(x,y,z,roll,pitch,yaw)` | `goto_capture.goto_pose` | Move to Cartesian pose (mm, degrees) |
| `arm.get_position()` | `goto_capture.goto_pose`, `datagen.ee2base`, `datagen.move_to` | Read current TCP pose |
| `arm.get_err_warn_code()` | `goto_capture.goto_pose` | Diagnose motion failures |

### Critical Preconditions

1. **Enable sequence is mandatory** before any motion:
   ```python
   arm.clean_error()
   arm.clean_warn()
   arm.motion_enable(enable=True)
   arm.set_mode(0)
   arm.set_state(state=0)
   time.sleep(0.5)
   ```

2. **`set_tcp_offset` uses mm and degrees** — this project stores TCP origin in meters, so convert before calling:
   ```python
   arm.set_tcp_offset([*map(meters_to_mm, tcp_origin), *euler_degrees], is_radian=False)
   ```

3. **`set_position` uses mm and degrees** — the mm/meters boundary lives in `move_to` and `ee2base`:
   ```python
   # Outbound (meters -> mm for xArm)
   x, y, z = [meters_to_mm(m) for m in pose[:3, 3]]
   arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)

   # Inbound (mm -> meters from xArm)
   code, xarmpose = arm.get_position()
   pose_meters = [mm_to_meters(mm) for mm in xarmpose[:3]]
   ```

4. **Always check return codes** — 0 means success, negative means error:
   ```python
   code = arm.set_position(...)
   if code != 0:
       code, (err, warn) = arm.get_err_warn_code()
       # Recovery: clean_error -> motion_enable -> set_state(0)
   ```

5. **`set_tcp_offset` is not persisted** across reboot unless `arm.save_conf()` is called.
