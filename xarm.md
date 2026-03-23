# xArm Python SDK Reference

Comprehensive reference for the `xarm-python-sdk` (`from xarm.wrapper import XArmAPI`).
SDK source: `/home/jeffk/repo/visual_servoing/xArm-Python-SDK/`.

## Architecture

Three layers:
1. **Transport** (`xarm/core/`) — TCP sockets or serial, binary protocol encoding
2. **Functional mixins** (`xarm/x3/`) — `Base`, `Gripper`, `Servo`, `GPIO`, `FtSensor`, `LinearMotor`, `Events`, etc. composed via multiple inheritance into `XArm`
3. **User API** (`xarm/wrapper/xarm_api.py`) — `XArmAPI` wraps `XArm`, handles unit conversion (radians/degrees via `is_radian`)

Entry point: `from xarm.wrapper import XArmAPI`

## Unit Conventions

| Quantity | xArm API unit | Notes |
|----------|---------------|-------|
| Position (x, y, z) | **mm** | Always mm, unaffected by `is_radian` |
| Angles (roll, pitch, yaw) | **degrees** (default) | Radians if `is_radian=True` at init or per-call |
| TCP speed | mm/s | Default: 100 |
| TCP acceleration | mm/s^2 | Default: 2000 |
| Joint speed | deg/s (default) | Default: 20 deg/s |
| Joint acceleration | deg/s^2 (default) | Default: 500 deg/s^2 |
| TCP jerk | mm/s^3 | |
| Linear motor position | mm | |
| Linear motor speed | mm/s | Range 1-1000 |

## Connection & Lifecycle

### `XArmAPI(port, is_radian=False, do_not_open=False, **kwargs)`
- `port`: IP address string (e.g. `'192.168.1.241'`) or serial device path
- `is_radian`: default unit for all angle parameters
- `do_not_open`: if `True`, must call `connect()` manually

### `connect(port=None)`
Establishes TCP connection, starts report thread, checks firmware version.

### `disconnect()`
Closes streams, joins threads.

### `connected` (property) -> bool

## Enabling the Arm (Required Sequence)

**You must run this sequence before any motion command will work:**

```python
arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)
arm.set_mode(0)       # 0 = position control
arm.set_state(state=0) # 0 = ready for motion
time.sleep(0.5)       # let state settle
```

### `motion_enable(enable=True, servo_id=None)`
- `servo_id`: joint 1-7, or `None`/8 for all
- Must be called before any motion

### `set_mode(mode=0, detection_param=0)`
| Mode | Description |
|------|-------------|
| 0 | Position control (default, for `set_position`/`set_servo_angle`) |
| 1 | Servo motion (for `set_servo_angle_j`, `set_servo_cartesian`) |
| 2 | Joint teaching mode |
| 4 | Joint velocity control |
| 5 | Cartesian velocity control |
| 6 | Joint online trajectory planning |
| 7 | Cartesian online trajectory planning |

### `set_state(state=0)`
| State | Meaning |
|-------|---------|
| 0 | Ready / sport state |
| 3 | Pause |
| 4 | Stop (emergency) |
| 6 | Deceleration stop |

## Cartesian Motion

### `set_position(x=None, y=None, z=None, roll=None, pitch=None, yaw=None, radius=None, speed=None, mvacc=None, mvtime=None, relative=False, is_radian=None, wait=False, timeout=None, **kwargs)`
- **x, y, z**: mm. `None` reuses last commanded position.
- **roll, pitch, yaw**: degrees (default). `None` reuses last.
- **radius**: `None` or <0 = linear interpolation. >=0 = arc (blended) motion.
- **speed**: mm/s. `None` = last used (default 100).
- **mvacc**: mm/s^2. `None` = last used (default 2000).
- **relative**: if `True`, values are relative to current position.
- **wait**: block until motion completes.
- **timeout**: max wait seconds (`None` = infinite).
- **kwargs**:
  - `motion_type`: 0 (linear planning), 1 (prioritize linear), 2 (joint planning). fw >= 1.11.100.
- **Returns**: `code` (int). 0 = success.
- **Preconditions**: Arm must be enabled, mode 0, state 0.
- **xArm5 constraint**: roll must be +/-180 deg, pitch must be 0.

### `get_position(is_radian=None)` -> `(code, [x, y, z, roll, pitch, yaw])`
- Returns current TCP position in mm and degrees (default).

### `set_tool_position(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, speed=None, mvacc=None, wait=False, timeout=None, radius=None)`
- Relative motion in the **tool coordinate** frame.

### `move_gohome(wait=True, timeout=None)`
- Returns to home (zero) joint position.

### `wait_move(timeout=None)` -> `code`
- Block until arm stops moving.

## Joint Motion

### `set_servo_angle(servo_id=None, angle=None, speed=None, mvacc=None, relative=False, is_radian=None, wait=False, timeout=None, radius=None)`
- `servo_id`: 1-7 for individual joint, `None`/8 for all joints.
- `angle`: single value (if servo_id 1-7) or list of 7 (if servo_id None/8).
- `speed`: deg/s (default 20).
- `mvacc`: deg/s^2 (default 500).

### `set_servo_angle_j(angles, speed=None, mvacc=None, is_radian=None)`
- For **servo motion mode** (mode 1). Real-time joint control.

### `set_servo_cartesian(mvpose, speed=None, mvacc=None, is_radian=None, is_tool_coord=False)`
- For **servo motion mode** (mode 1). Real-time Cartesian control.
- `mvpose`: [x, y, z, roll, pitch, yaw]

### `move_circle(pose1, pose2, percent, speed=None, mvacc=None, wait=False, timeout=None)`
- Circular motion through two waypoints.

## TCP Configuration

### `set_tcp_offset(offset, is_radian=None, wait=True)`
- `offset`: [x, y, z, roll, pitch, yaw] in mm and degrees.
- **Not persisted across reboot** unless `save_conf()` called.
- Reset with `set_tcp_offset([0, 0, 0, 0, 0, 0])`.

### `set_tcp_load(weight, center_of_gravity, wait=False)`
- `weight`: kg
- `center_of_gravity`: [x, y, z] in mm

### `set_tcp_jerk(jerk)` — mm/s^3
### `set_tcp_maxacc(acc)` — mm/s^2

## Safety & Limits

### `set_collision_sensitivity(value, wait=True)`
- `value`: 0-5. 0 = disabled, 5 = most sensitive.

### `set_collision_rebound(on_off)`
- Enable/disable automatic rebound after collision detection.

### `set_self_collision_detection(on_off)`
- Enable/disable self-collision checking.

### `set_collision_tool_model(tool_type, *args)`
- `tool_type`: `XCONF.CollisionToolType.BOX` or `.CYLINDER`
- BOX: `x, y, z, x_offset, y_offset, z_offset` (mm)
- CYLINDER: `radius, height, x_offset, y_offset, z_offset` (mm)

### `set_reduced_mode(on_off)`
- Restricted motion envelope for safety.

### `set_reduced_max_tcp_speed(speed)` — mm/s
### `set_reduced_max_joint_speed(speed, is_radian=None)`
### `set_reduced_tcp_boundary(boundary)` — [x_min, x_max, y_min, y_max, z_min, z_max] mm
### `set_reduced_joint_range(joint_range, is_radian=None)` — [j1_min, j1_max, ..., j7_min, j7_max]

### `set_fence_mode(on)`
- Digital fence boundary protection. fw >= 1.2.11.

### `set_gravity_direction(direction, wait=True)`
- `direction`: [x, y, z] unit vector (e.g. [0, 0, -1] for floor-mounted).

### `set_mount_direction(base_tilt_deg, rotation_deg, is_radian=None)`
- Sets gravity direction from mounting angles.

## Error Handling

### `get_err_warn_code(show=False, lang='en')` -> `(code, [error_code, warn_code])`

### `clean_error()` -> `code`
### `clean_warn()` -> `code`

### `emergency_stop()`
- Calls `set_state(4)` then re-enables. Does **not** auto-clear errors.

### Error Recovery Sequence
```python
arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
```

## Kinematics

### `get_inverse_kinematics(pose, input_is_radian=None, return_is_radian=None, limited=True, ref_angles=None)`
- `pose`: [x, y, z, roll, pitch, yaw]
- Returns: `(code, [j1, ..., j7])`

### `get_forward_kinematics(angles, input_is_radian=None, return_is_radian=None)`
- Returns: `(code, [x, y, z, roll, pitch, yaw])`

## Configuration Persistence

### `save_conf()`
- Saves current config (TCP offset, loads, sensitivities) to persistent storage. Survives reboot.

### `clean_conf()`
- Restores all settings to factory defaults.

## Properties (Read-Only)

| Property | Type | Description |
|----------|------|-------------|
| `connected` | bool | Connection status |
| `position` | list | [x, y, z, roll, pitch, yaw] mm/deg |
| `angles` | list | Joint angles [j1-j7] |
| `last_used_position` | list | Defaults for next `set_position` |
| `last_used_angles` | list | Defaults for next `set_servo_angle` |
| `tcp_offset` | list | Current TCP offset |
| `state` | int | Current state |
| `mode` | int | Current mode |
| `error_code` | int | Current error (0 = none) |
| `warn_code` | int | Current warning (0 = none) |
| `has_error` | bool | Error exists |
| `has_warn` | bool | Warning exists |
| `is_moving` | bool | Currently in motion |
| `cmd_num` | int | Commands queued on controller |
| `version` | str | Firmware version |
| `axis` | int | Number of joints (5, 6, or 7) |
| `sn` | str | Serial number |
| `temperatures` | list | Joint temperatures (C) |
| `voltages` | list | Joint voltages |
| `currents` | list | Joint currents |
| `joints_torque` | list | Joint torques |
| `tcp_speed_limit` | list | [min, max] mm/s |
| `tcp_acc_limit` | list | [min, max] mm/s^2 |
| `collision_sensitivity` | int | 0-5 |
| `gravity_direction` | list | [x, y, z] |
| `motor_brake_states` | list | Per-motor brake status |
| `motor_enable_states` | list | Per-motor enable status |

## Callbacks

### Registration
```python
arm.register_error_warn_changed_callback(fn)     # fn(dict) with error_code, warn_code
arm.register_state_changed_callback(fn)           # fn(dict) with state
arm.register_mode_changed_callback(fn)            # fn(dict)
arm.register_connect_changed_callback(fn)         # fn(dict)
arm.register_report_callback(fn)                  # fn(dict) full state
arm.register_report_location_callback(fn)         # fn(dict) position/angles only
arm.register_temperature_changed_callback(fn)     # fn(dict)
arm.register_count_changed_callback(fn)           # fn(dict)
```

### Release
Each has a corresponding `release_*_callback()` method.

## Gripper Control

### Standard Gripper
```python
arm.set_gripper_enable(True)
arm.set_gripper_mode(0)        # 0=position, 1=speed
arm.set_gripper_speed(speed)   # 0-2000 rpm
arm.set_gripper_position(pos, wait=False, timeout=5)  # 0-850 (0.1mm units)
code, pos = arm.get_gripper_position()
arm.clean_gripper_error()
```

### BIO Gripper
```python
arm.set_bio_gripper_enable(True)
arm.set_bio_gripper_speed(speed)
arm.open_bio_gripper(speed=0, wait=True, timeout=5)
arm.close_bio_gripper(speed=0, wait=True, timeout=5)
code, status = arm.get_bio_gripper_status()
arm.clean_bio_gripper_error()
```

### RobotIQ Gripper
```python
arm.robotiq_reset()
arm.robotiq_set_activate(True)
arm.robotiq_set_position(pos)  # 0-255
arm.robotiq_open() / arm.robotiq_close()
code, status = arm.robotiq_get_status()
```

### Lite6 Gripper (G2)
```python
arm.close_lite6_gripper(sync=True)
arm.open_lite6_gripper(sync=True)
```

## GPIO

### Tool GPIO (TGPIO)
```python
code, val = arm.get_tgpio_digital(ionum)    # ionum 0-1
arm.set_tgpio_digital(ionum, value)          # 0/1
code, val = arm.get_tgpio_analog(ionum)      # ionum 0-1
```

### Controller GPIO (CGPIO)
```python
code, val = arm.get_cgpio_digital(ionum)     # ionum 0-15
arm.set_cgpio_digital(ionum, value)
code, val = arm.get_cgpio_analog(ionum)      # ionum 0-1
arm.set_cgpio_analog(ionum, value)
code, states = arm.get_cgpio_state()         # full state dump
```

Position-triggered GPIO: `set_cgpio_digital_with_xyz()`, `set_cgpio_analog_with_xyz()`, `set_tgpio_digital_with_xyz()`.

## Linear Motor / Rail

Requires firmware >= 1.8.0. Communicates via RS485 modbus.

### Initialization (Required on First Power-On)

```python
# Home the motor first — required before any position commands
code = arm.set_linear_motor_back_origin(wait=True, auto_enable=True, timeout=10)

# Verify at zero
code, on_zero = arm.get_linear_motor_on_zero()
assert on_zero == 1, "Motor not at zero after homing"
```

### Basic Motion

```python
# Enable
arm.set_linear_motor_enable(True)

# Set speed (1-1000 mm/s)
arm.set_linear_motor_speed(500)

# Move to absolute position (mm), blocking
code = arm.set_linear_motor_pos(300, wait=True, timeout=100)

# Move to position with speed override
code = arm.set_linear_motor_pos(700, speed=200, wait=True)

# Stop immediately
arm.set_linear_motor_stop()
```

### Position Ranges by Model

| Model | Range |
|-------|-------|
| AL1300 | 0-700 mm |
| AL1301 | 0-1000 mm |
| AL1302 | 0-1500 mm |

### Status & Diagnostics

```python
code, pos = arm.get_linear_motor_pos()          # current position (mm)
code, status = arm.get_linear_motor_status()     # status flags
code, error = arm.get_linear_motor_error()       # error code
code, enabled = arm.get_linear_motor_is_enabled() # 0 or 1
code, on_zero = arm.get_linear_motor_on_zero()   # 0 or 1
code, version = arm.get_linear_motor_version()   # firmware version string
code, sn = arm.get_linear_motor_sn()             # 14-char serial number
```

**Status flags:**
- `status & 0x00`: motion finished
- `status & 0x01`: in motion
- `status & 0x02`: has stopped

### Full Status Register Query

```python
code, info = arm.get_linear_motor_registers()
# info = {'pos': float, 'status': int, 'error': int, 'is_enabled': int,
#         'on_zero': int, 'sci': int, 'sco': [int, int]}
```

### Error Handling

```python
arm.clean_linear_motor_error()
```

### Linear Motor Error Codes

| Code | Meaning |
|------|---------|
| 10 | Current detection error |
| 11 | Current overlimit |
| 12 | Speed overlimit |
| 13 | Large position deviation |
| 14 | Position command overlimit |
| 20 | Driver IC hardware error |
| 21 | Driver IC initialization error |
| 25 | Command over software limit |
| 26 | Feedback position software limit |
| 33 | Drive overloaded |
| 34 | Motor overload |
| 35 | Motor type error |
| 36 | Driver type error |
| 39 | Over voltage |
| 40 | Undervoltage |
| 49 | EEPROM read/write error |

### API Return Codes (Linear Motor)

| Code | Meaning |
|------|---------|
| 0 | Success |
| 80 | Linear motor has error |
| 81 | Linear motor SCI is low |
| 82 | Linear motor not initialized (not at zero) |
| 100 | Wait finish timeout |

### Advanced

```python
arm.set_linear_motor_default_parmas()  # Reset to factory parameters
arm.set_linear_motor_sn(sn)           # Set serial number (14 chars exactly)
```

### Key Preconditions for Linear Motor

1. **Homing first**: `set_linear_motor_back_origin()` must succeed before any `set_linear_motor_pos()` call. Check `get_linear_motor_on_zero()` returns 1.
2. **Enable**: Motor must be enabled. Most commands auto-enable if `auto_enable=True` (default).
3. **Error clear**: If motor is in error state, call `clean_linear_motor_error()` before retrying.
4. **Speed range**: 1-1000 mm/s.

## Trajectory Recording & Playback

```python
arm.start_record_trajectory()
# ... perform motions ...
arm.stop_record_trajectory()
arm.save_record_trajectory('my_traj', wait=True)

arm.playback_trajectory('my_traj', times=1, wait=True)
arm.delete_trajectory('my_traj')
code, traj_list = arm.get_trajectory_list()
```

## Force/Torque Sensor

```python
arm.set_ft_sensor_enable(True)
arm.set_ft_sensor_mode(0)              # 0=raw, 1=filtered
arm.set_ft_sensor_zero()               # calibrate zero
code, data = arm.get_ft_sensor_data()  # [fx, fy, fz, tx, ty, tz]

# Auto-identify payload
arm.iden_ft_sensor_load_offset()

# Set known payload
arm.set_ft_sensor_load_offset(weight, center, [x_off, y_off, z_off])
```

## Return Code Reference

### Success
- **0**: Success

### API Errors (negative)
| Code | Meaning |
|------|---------|
| -1 | Disconnected |
| -2 | Not ready (motion disabled or wrong state) |
| -4 | Command does not exist |
| -6 | Cartesian position out of limit |
| -7 | Joint angle out of limit |
| -8 | Out of range |
| -9 | Emergency stop |

### Communication Errors (positive)
| Code | Meaning |
|------|---------|
| 1 | Pending errors not cleared |
| 2 | Pending warnings not cleared |
| 3 | Response timeout |
| 9 | State not ready for motion |
| 12 | Parameter error |

### Controller Error Codes
| Code | Meaning |
|------|---------|
| 1-3 | Emergency stop / IO |
| 10-17 | Servo errors (10=general, 11-17=per-joint) |
| 18 | F/T sensor communication |
| 21 | Kinematic error |
| 22 | Self-collision |
| 23 | Joint limit |
| 24 | Speed limit |
| 25 | Planning error |
| 31 | Collision (abnormal current) |
| 40 | No IK solution available |
| 50-53 | F/T sensor errors |

## Common Patterns

### Safe Motion with Error Check
```python
code = arm.set_position(x=300, y=0, z=200, roll=-180, pitch=0, yaw=0, wait=True)
if code != 0:
    code, (err, warn) = arm.get_err_warn_code()
    print(f"Error: {err}, Warning: {warn}")
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_state(0)
```

### Blocking vs Non-Blocking
```python
# Non-blocking (default) — returns immediately, motion queued
arm.set_position(x=300, y=0, z=200, roll=-180, pitch=0, yaw=0)
# ... do other work ...
arm.wait_move()  # block until done

# Blocking — returns when motion completes
arm.set_position(x=300, y=0, z=200, roll=-180, pitch=0, yaw=0, wait=True, timeout=10)
```

### Relative Motion
```python
# Move 50mm in +X from current position
arm.set_position(x=50, relative=True, wait=True)

# Move in tool frame
arm.set_tool_position(z=50, wait=True)  # 50mm along tool Z axis
```
