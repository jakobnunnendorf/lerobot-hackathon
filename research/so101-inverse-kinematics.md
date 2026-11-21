# SO-101 kinematics and Cartesian motion for the ball hand-over

Researched 2026-11-21. LeRobot facts are read from the source at tag `v0.7.0` (raw files from GitHub); "docs" means the `docs/source/*.mdx` files at that tag, the built website was not opened. The arm model is `so101_new_calib.urdf` from TheRobotStudio/SO-ARM100 at commit `141aea7`. Everything under "Checked on Tian's laptop" was run (Windows 11, Python 3.12, numpy, ikpy); nothing ran with LeRobot installed or on a robot. **[unverified]** marks what I could not confirm. Hand tracking and camera calibration are covered in `research/hand-tracking-and-camera-calibration.md`, not here.

LeRobot links have the form `https://github.com/huggingface/lerobot/blob/v0.7.0/<path>#L<line>`.

## Summary

1. Use our closed-form numpy kinematics on every machine. For the team's loop that is `data/kinematics/handover/kinematics.py`. Checked here: it matches ikpy to 3e-16 m, forward(inverse(p)) returns p to 2 µm, one call takes 0.2 ms. LeRobot's own `RobotKinematics` needs `placo`, which has no Windows wheel and is not installed on the robot laptop.
2. The URDF's zero is the middle of each joint's range, and so is LeRobot's 0 with `use_degrees=True` (the 0.7.0 default). LeRobot's examples feed its degrees into the URDF unchanged. That holds only if every joint was swept to both end stops at calibration. Check it on the arm (checklist at the end).
3. **The gripper cannot point straight down at hover height.** It can below about 8 cm. Above that, wrist_flex's ±95° limit and the reach stop it. The steepest reachable tilt is about -86° at 10 cm, -68° at 15 cm, -52° at 20 cm and -36° at 25 cm. Let the solver pick the steepest reachable tilt.
4. Drive the arm with joint targets from our own 30 Hz loop: low-pass filter plus 1.5 cm dead band on the hand position, 0.15 m/s on the Cartesian goal, the IK, then a per-joint speed limit (45°/s in the team's loop). No `max_relative_target`. The gripper gets a constant command, the value it was closed with, never its measured position.
5. Drop when two hands are seen and the tip has been within 3 cm of their midpoint for 1 s. Our `kinematics.py` and `following.py` for the team's loop (`data/kinematics/handover/`) ran inside its dry run: follow, hold, home, drop and done came out as intended.
6. Only the arm can settle the joint offsets and signs, where the ball sits in the gripper, the gripper hold value, HOME, and whether the ball falls cleanly from a tilted gripper.

## Recommended approach

- **Robot laptop, as reported on the team board** (not verified by us): the robot program runs in WSL2 with LeRobot 0.7.0, `use_degrees=True` and calibration id `mi_follower`. The gripper reads about 11-16 (0-100 scale) while it holds the ball. `placo` is not installed there.
  - So its calibration file is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/mi_follower.json` inside WSL (path rule in section 2).
- **Kinematics:** our numpy code runs on every machine, the WSL robot laptop included, with no extra dependency.
  - `data/kinematics/handover/kinematics.py` is the version built for the team's loop: `forward_kinematics(joints_deg) -> xyz` and `inverse_kinematics(target_xyz, seed_deg) -> joints | None`, both in LeRobot degrees and the robot frame below.
  - `data/kinematics/so101_kinematics.py` is the same mathematics with a lower-level interface.
  - LeRobot's placo-based `RobotKinematics` is useful only as a cross-check on a Linux machine where placo installs.
- **Motion:** the team's loop `scripts/handover/handover_loop.py` (branch `jakobnunnendorf/handover-loop`) imports `kinematics.py` and `following.py`; ours are in `data/kinematics/handover/`. The work is split as follows:
  - The loop clamps the workspace (radius 12-33 cm, height ≥ 8 cm), calls the IK with the commanded joints as seed (tolerance 2 cm), limits each joint to 1.5° per tick (45°/s), holds the gripper value until the drop, and sends.
  - `Follower.step(hands, tip_xyz, now)` returns a mode and a Cartesian target that is already filtered, glitch-free, dead-banded, speed-limited to 0.15 m/s and reachable.
  - The hover height is a fixed 18 cm above the table, because z from one camera is too rough to follow (their decision, kept).
  - Out of reach for 3 s, or no hand for 3 s, means HOME.
  - Two hands, with the tip within 3 cm of their midpoint for 1 s, means DROP, then DONE after 1 s.
  - `data/kinematics/hand_follow.py` is my earlier stand-alone draft of the same logic. The `handover/` files supersede it.
- **Order of work** (estimates):
  1. 10 min, nothing moves: read the calibration file and compare its ranges with the URDF limits (checklist item 1).
  2. 30 min, torque off, someone holds the arm: touch 8 to 12 marks with the gripper tip and fit joint offsets (`data/kinematics/fit_offsets.py`).
  3. 1 to 2 h at a desk: feed the tracker's hand positions into the loop with `send_action` replaced by a print, and check the printed targets while a hand moves in front of the camera.
  4. 1 h at the table: first real motion with no ball, half speed (0.08 m/s, 45°/s) and the hand kept away from the arm; then normal limits.
  5. 30 min: with the ball, test holding during following, then the drop into cupped hands.

  About half a day for one person, mostly integration and tuning. The kinematics itself is done.

---

## 1. LeRobot 0.7.0's own kinematics

### `RobotKinematics` in `src/lerobot/model/kinematics.py`

- **Constructor:** `RobotKinematics(urdf_path: str, target_frame_name: str = "gripper_frame_link", joint_names: list[str] | None = None)`. It loads `placo.RobotWrapper(urdf_path)` and a `placo.KinematicsSolver`, fixes the base with `mask_fbase(True)` and adds a frame task on the target frame. Without `joint_names` it uses every joint in the URDF. ([kinematics.py#L44-L74](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/model/kinematics.py#L44-L74))
- **`forward_kinematics(joint_pos_deg) -> np.ndarray`:**
  - Takes degrees, converts them with `np.deg2rad` and uses the first `len(joint_names)` values.
  - Returns `robot.get_T_world_frame(target_frame_name)`, a 4x4 homogeneous transform in metres. ([L76-L98](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/model/kinematics.py#L76-L98))
  - The world frame is `base_link`, because the base is fixed at the origin. This is my reading of placo's behaviour **[unverified]**.
- **`inverse_kinematics(current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01, max_iters=8) -> np.ndarray`:**
  - Takes the current joints in degrees as the initial guess and the desired pose as a 4x4 matrix.
  - Sets the frame task to `"soft"` with the two weights and runs `solver.solve(True)` 8 times.
  - Returns degrees. Values beyond `joint_names` (the gripper) are copied through unchanged. ([L100-L156](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/model/kinematics.py#L100-L156))
  - So it is a numerical solver with a fixed number of iterations per call, not an exact one.
- **Orientation with 5 joints:** the default weight is 0.01 for orientation against 1.0 for position, so position dominates. The IK processor step says so in its docstring: "Set to `0.0` for position-only IK on under-actuated arms; a small nonzero weight gives soft-orientation IK on the 5-DOF SO-101, where the wrist tracks orientation only partially". ([robot_kinematic_processor.py#L290-L294](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L290-L294))
- **`joint_names` in the examples:** they pass `joint_names=list(robot.bus.motors.keys())`, which is all six motor names including `gripper`. The URDF has a joint with the same name (the moving jaw), so placo moves it too, without effect on `gripper_frame_link`. ([examples/phone_to_so100/teleoperate.py#L125-L129](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L125-L129))

### Ready-made steps in `src/lerobot/robots/so_follower/robot_kinematic_processor.py` (path confirmed at the tag)

| Step | What it does | Source |
| --- | --- | --- |
| `EEReferenceAndDelta(kinematics, end_effector_step_sizes, motor_names, use_latched_reference=True, use_ik_solution=False)` | Turns a relative command (`enabled`, `target_x..target_wz`, `gripper_vel`) plus forward kinematics of the measured joints into an absolute `ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz` (rotation vector). While disabled it keeps sending the last pose | [L40-L188](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L40-L188) |
| `EEBoundsAndSafety(end_effector_bounds={"min": [...], "max": [...]}, max_ee_step_m=0.05, raise_on_jump=True)` | Clips the position to a box (orientation is not bounded). Limits the step per frame: with `raise_on_jump=True` a larger step raises `ValueError` and ends the loop, with `False` it is rate-limited and logged | [L191-L272](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L191-L272) |
| `InverseKinematicsEEToJoints(kinematics, motor_names, initial_guess_current_joints=True, orientation_weight=0.01)` | `ee.*` plus `ee.gripper_pos` → `<motor>.pos`. The gripper value is copied from `ee.gripper_pos` | [L275-L371](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L275-L371) |
| `GripperVelocityToJoint(speed_factor=20.0, clip_min=0.0, clip_max=100.0, discrete_gripper=False)` | Adds a velocity times `speed_factor` to the **measured** gripper position | [L374-L437](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L374-L437) |
| `ForwardKinematicsJointsToEE(kinematics, motor_names)` and its observation-only and action-only variants | Joints → `ee.x..ee.wz`, `ee.gripper_pos` | [L440-L556](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L440-L556) |
| `InverseKinematicsRLStep` | The IK step for the RL pipeline; stores the solution as `IK_solution` | [L559-L651](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L559-L651) |

For us:

- Our hand positions are absolute, so `EEReferenceAndDelta` does not fit. Only `EEBoundsAndSafety` and `InverseKinematicsEEToJoints` would.
- `GripperVelocityToJoint` integrates from the measured gripper position. When the jaws are stalled on the ball, that position is the ball's width, so the step would quietly loosen the grip (inferred from the code).

### How the phone example wires them

- **Robot and solver:** `SO100FollowerConfig(port, id, use_degrees=True, disable_torque_on_disconnect=False)` ([teleoperate.py#L110-L115](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L110-L115)) and `RobotKinematics(urdf_path="./SO101/so101_new_calib.urdf", target_frame_name="gripper_frame_link", joint_names=list(robot.bus.motors.keys()))` ([L125-L129](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L125-L129)).
- **Pipeline:**
  - `MapPhoneActionToRobotAction`
  - → `EEReferenceAndDelta(end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5}, use_latched_reference=True)`
  - → `EEBoundsAndSafety(end_effector_bounds ±1 m, max_ee_step_m=0.10)`
  - → `GripperVelocityToJoint(speed_factor=20.0)`
  - → `InverseKinematicsEEToJoints(initial_guess_current_joints=True)`

  ([L132-L158](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L132-L158))
- **Loop:** at 30 fps, `robot.get_observation()` → `pipeline((phone_obs, robot_obs))` → `robot.send_action(joint_action)` ([L181-L200](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L181-L200)).
- **Homing:** at start the example moves every body joint to raw encoder value 2047 (the pose held during calibration) and leaves the gripper alone. Its docstring warns that this "is not the same as commanding 0 in the robot's normalized units, which is the middle of each joint's *recorded range of motion*" ([L53-L71](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L53-L71)).
- **Initial guess:** the docs use the measured joints in closed loop and the previous solution when replaying ([phone_teleop.mdx#L181-L201](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/phone_teleop.mdx#L181-L201)).
- **Isaac example:** it bounds z at `base_link` z ≥ 0, steps at most 0.1 m per frame and uses orientation weight 0.01 ([examples/isaac_teleop_to_so101/common.py#L158-L165](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/isaac_teleop_to_so101/common.py#L158-L165), [L321-L333](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/isaac_teleop_to_so101/common.py#L321-L333)).

### Extra to install, and platforms

- **The extra:** `kinematics = ["lerobot[placo-dep]"]`, and `placo-dep = ["placo>=0.9.6,<0.9.16", "cmeel-urdfdom>=4,<5", "cmeel-tinyxml2<11"]`. The cap is there because placo 0.9.16 links a urdfdom that Ubuntu 24.04 does not ship. ([pyproject.toml#L158-L164](https://github.com/huggingface/lerobot/blob/v0.7.0/pyproject.toml#L158-L164), [#L222](https://github.com/huggingface/lerobot/blob/v0.7.0/pyproject.toml#L222))
  - The docs' install line is `pip install 'lerobot[phone,kinematics,feetech,core_scripts]'` ([phone_teleop.mdx#L14](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/phone_teleop.mdx#L14)).
- **Our repo:** we depend on `lerobot[all,...]`, which includes `kinematics`. `uv.lock` already pins placo 0.9.15 and pin 3.4.0 (Pinocchio), with wheels for macOS (x86_64, arm64) and Linux (x86_64, aarch64) only (read in our `uv.lock`).
- **PyPI today** (read 2026-11-21 from the JSON API):
  - placo's latest is 0.12.0, uploaded 2026-11-18, with wheels for CPython 3.10-3.14 on macOS and manylinux_2_28 x86_64/aarch64 and **no Windows wheel**. 0.9.15 has the same platform set. ([PyPI placo](https://pypi.org/project/placo/#files))
  - pin 4.1.0 has the same platforms. ([PyPI pin](https://pypi.org/project/pin/#files))
  - This agrees with `kb/kinematics-on-windows.md`.
- **conda-forge:** has win-64 builds of placo 0.9.17-0.10.0 and of pinocchio 4.1.0 ([placo files](https://anaconda.org/conda-forge/placo/files), [pinocchio files](https://anaconda.org/conda-forge/pinocchio/files)). LeRobot caps placo below 0.9.16, so on Windows LeRobot's code would run against a placo it was not pinned to **[unverified]**. Not worth it.
- **The URDF is not shipped:** the docs say to copy `Simulation/SO101` from SO-ARM100 next to the example ([phone_teleop.mdx#L76-L84](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/phone_teleop.mdx#L76-L84)). The Isaac example downloads it from the Hugging Face bucket `lerobot/robot-urdfs` ([common.py#L168-L183](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/isaac_teleop_to_so101/common.py#L168-L183)).

---

## 2. The SO-101 URDF and LeRobot's joint values

### Which file

- The file is `Simulation/SO101/so101_new_calib.urdf` ([permalink at 141aea7](https://github.com/TheRobotStudio/SO-ARM100/blob/141aea7bcac2744d92998e30268c413202a56f96/Simulation/SO101/so101_new_calib.urdf)). Every LeRobot example names it ([teleoperate.py#L124](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L124)).
- It was generated with onshape-to-robot. A copy is in `data/kinematics/urdf/`, with the base and gripper meshes in `urdf/assets/`.
- **History of the file** ([history](https://github.com/TheRobotStudio/SO-ARM100/commits/main/Simulation/SO101/so101_new_calib.urdf)):
  - 2025-07-02, last change: "Fixing origin frame offsets".
  - 2025-06-25: "updated the new so101 calibration urdf with fixed direction and frame names", "Renamming joints to match LeRobot middleware naming convention", and "shift joint(3) elbow flex by five degrees to adhere to the offset on the real robot".
- **The calibration convention, from the SO-ARM100 README** ([README](https://github.com/TheRobotStudio/SO-ARM100/blob/141aea7bcac2744d92998e30268c413202a56f96/Simulation/SO101/README.md)):
  - "New Calibration (Default): Each joint's virtual zero is set to the **middle** of its joint range".
  - `so101_old_calib` puts zero where the arm is "fully extended horizontally". It does not match LeRobot 0.7.0.
  - The README also says that in LeRobot the gripper is 0 = fully closed and 100 = fully open, and that this "is not yet reflected" in the URDF.

### Joints, tool frame, limits ([URDF lines 293-451](https://github.com/TheRobotStudio/SO-ARM100/blob/141aea7bcac2744d92998e30268c413202a56f96/Simulation/SO101/so101_new_calib.urdf#L293-L451))

| Joint (same name as the LeRobot motor) | Parent → child | Limits (rad) | Limits (deg) |
| --- | --- | --- | --- |
| `shoulder_pan` | `base_link` → `shoulder_link` | ±1.91986 | ±110.0 |
| `shoulder_lift` | `shoulder_link` → `upper_arm_link` | ±1.74533 | ±100.0 |
| `elbow_flex` | `upper_arm_link` → `lower_arm_link` | ±1.69 | ±96.8 (comment: "5-degree calibration offset applied to joint limits") |
| `wrist_flex` | `lower_arm_link` → `wrist_link` | ±1.65806 | ±95.0 |
| `wrist_roll` | `wrist_link` → `gripper_link` | −2.74385 … 2.84121 | −157.2 … 162.8 |
| `gripper` | `gripper_link` → `moving_jaw_so101_v1_link` | −0.174533 … 1.74533 | −10 … 100 |
| `gripper_frame_joint` (fixed) | `gripper_link` → `gripper_frame_link` | none | none |

- **Axes:** every revolute joint turns about its local z axis.
- **Tool frame:** `gripper_frame_link`, a massless link on the fixed jaw. At zero pose its z axis points along the gripper.
- **Where the tool point sits** (from the fixed-jaw mesh bounds, `data/kinematics/explore_urdf.py`): its origin is 6 mm short of the end of the fixed jaw, and 7.9 mm off the wrist_roll axis.

### Dimensions computed from the joint origins

These come from multiplying out the URDF chain at zero pose with wrist_roll 0 (`check_kinematics.py`, section "geometry"). They are in the robot frame defined here: **origin on the table straight below the shoulder_pan axis, x forward (where the arm points at shoulder_pan = 0), y to the robot's left, z up.** That frame is `base_link` shifted by (−3.88 cm, 0, +0.24 cm). It is the frame proposed for the hand tracker, with the origin pinned to the pan axis, because the IK needs exactly that point.

| Quantity | Value |
| --- | --- |
| shoulder_lift pivot | 3.04 cm in front of the pan axis, 11.90 cm above the table |
| Upper arm (lift pivot → elbow pivot) | 11.60 cm, at 76.0° above horizontal at zero |
| Forearm (elbow pivot → wrist_flex pivot) | 13.50 cm, at 2.2° at zero |
| Hand (wrist_flex pivot → `gripper_frame_link`) | 15.94 cm: 15.92 cm along the gripper axis, 0.79 cm below it |
| Gripper axis at zero | horizontal, through the wrist_flex pivot; it is also the wrist_roll axis |
| Tip at zero pose | x 35.25 cm, y 0.00, z 22.89 cm |
| Lateral offset of the tip from the plane through the pan axis | 9 µm at wrist_roll 0. At ±90° roll the tip moves 8 mm sideways |
| `base_link` origin | 3.88 cm behind the pan axis, 2.4 mm above the table (the base mesh's underside is at z = −2.4 mm) |
| Base footprint (mesh `base_so101_v2.stl`) | x −2.2 … +6.5 cm, y ±5.5 cm in `base_link`: the pan axis is 6.1 cm from the base's back edge, 2.6 cm from its front edge, centred left to right |

- **Zero pose:** the line between the upper arm's two pivots leans 14° forward of vertical, the forearm and the gripper are horizontal and point forward. That is roughly the pose in the calibration diagram ([so_follower.py#L32-L51](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L32-L51)).
- **Signs in the URDF** (computed):
  - shoulder_pan: the axis points down, so +30° swings the tip from y = 0 to y = −17.6 cm (the robot's right).
  - shoulder_lift, elbow_flex and wrist_flex: the three axes are parallel and horizontal (to within 1e-5; the URDF writes π/2 as 1.5708). Positive values tip the next link down and forward: +30° on shoulder_lift moves the zero-pose tip from z = 22.9 to 5.3 cm.
  - wrist_roll: +90° moves the tip from below the roll axis to the robot's right side.
- **Jaws:** at wrist_roll 0 they open in the arm's vertical plane. The moving jaw's hinge is 2.4 cm above the roll axis, so the moving jaw is the upper one when the gripper points forward (computed from the URDF).

### LeRobot's joint values

- **`use_degrees: bool = True`** is the 0.7.0 default ([config_so_follower.py#L41-L42](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/config_so_follower.py#L41-L42)).
  - With it the five body joints use `MotorNormMode.DEGREES`, without it `RANGE_M100_100`. The gripper is always `RANGE_0_100`. ([so_follower.py#L68-L79](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L68-L79))
- **The three modes** (`motors_bus.py`):
  - **DEGREES:** `deg = (raw − (range_min + range_max) / 2) × 360 / 4095`. 0 is the middle of the range recorded during calibration. Not clamped, and the drive mode is not applied. ([motors_bus.py#L1030-L1033](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1030-L1033), inverse at [#L1060-L1063](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1060-L1063))
  - **RANGE_M100_100 (without `use_degrees`):** −100 … 100, linear over the recorded range ([#L1024-L1026](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1024-L1026)).
  - **RANGE_0_100, the gripper:** 0 … 100 over the recorded range ([#L1027-L1029](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1027-L1029)).
- **Calibration:**
  - The operator holds every joint "in the middle of their ranges". A homing offset makes that pose read raw 2047, half a turn. Then the operator "move[s] each joint through its full range of motion". ([so101.mdx#L399](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/so101.mdx#L399), [motors_bus.py#L931-L1008](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L931-L1008))
  - wrist_roll is a full-turn motor with range 0 … 4095 ([so_follower.py#L140](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L140), [motors_bus.py#L995-L997](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L995-L997)).
- **The recorded range is a hard limit:** it is written to the servos as `Min_Position_Limit` and `Max_Position_Limit` ([motors_bus.py#L786-L796](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L786-L796)), so goals outside it are not reached (the servo-side clamp is **[unverified]**). In LeRobot degrees those limits are ±half the recorded range, symmetric by construction. `so101_kinematics.joint_limits_from_calibration(path)` reads them.
- **Calibration file:** `$HF_LEROBOT_CALIBRATION/robots/so_follower/<robot.id>.json`.
  - By default that is `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json` ([robot.py#L62-L65](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L62-L65), [constants.py#L92-L101](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/utils/constants.py#L92-L101)).
  - The folder is `so_follower` for both SO-100 and SO-101 in 0.7.0 ([so_follower.py#L61](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L61)).
  - Each motor entry holds `id, drive_mode, homing_offset, range_min, range_max` ([motors_bus.py#L139-L144](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L139-L144)).
  - WSL and Windows keep separate copies (`kb/machines.md`).

### Can LeRobot degrees go into the URDF unchanged?

**By design, yes.** The README puts the URDF's zero at the middle of each joint's range, and LeRobot puts its 0 at the middle of the recorded range. LeRobot's own examples rely on this: `RobotKinematics.forward_kinematics` only converts degrees to radians ([kinematics.py#L88](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/model/kinematics.py#L88)).

It goes wrong in two ways:

1. **Shifted zero:** a joint not pushed to both end stops during calibration, or pushed further on one side, has its zero shifted by half the difference. Simulated here, offsets of up to 6° put the tip up to 7 cm off in the hover region (`fit_offsets.py`).
2. **Sign:** a motor whose positive direction is opposite to the URDF axis. The 2025-06-25 commit says the directions were fixed to match LeRobot. I could not check this without an arm.

**Not the same thing:** the calibration pose (raw 2047) is not LeRobot's 0 (phone example, [teleoperate.py#L56-L60](https://github.com/huggingface/lerobot/blob/v0.7.0/examples/phone_to_so100/teleoperate.py#L56-L60)). The calibration file gives the gap per joint as `(2047 − (range_min + range_max) / 2) × 360 / 4095`.

**What the file can tell without moving anything:** each joint's half range, to compare with the URDF's 110 / 100 / 96.8 / 95°. A joint whose half range is clearly smaller was probably not swept to both stops, so its zero is suspect.

**What only the arm can tell:** the actual offsets and signs. The procedure is in the checklist. Fitting offsets from 8 to 12 touched marks brought the simulated tip error from 56-70 mm down to 5-7 mm (`fit_output.txt`).

---

## 3. Inverse kinematics without placo

### Options

| Option | On Windows | Measured or read | Verdict |
| --- | --- | --- | --- |
| **Closed form, ours** (`so101_kinematics.py`, `handover/kinematics.py`) | numpy only | 25-40 µs per solve at one tilt; 0.2 ms for the whole vectorised tilt search; exact to 2 µm; the gripper's tilt is an explicit input | Use it |
| **ikpy 4.5.0** | pure-Python wheel; needs numpy, scipy, sympy ([PyPI](https://pypi.org/project/ikpy/)); installed and ran here | Forward kinematics identical to ours (3e-16 m). Position-only IK took 27-37 ms per solve, and from its default start 2 of 30 reachable targets were missed by more than 1 mm (worst 8.6 cm) | Good as a cross-check. Too slow and unreliable inside a 30 Hz loop without warm starts (warm starts not measured) |
| **Pinocchio** (conda-forge) | win-64 builds of 4.1.0 exist ([files](https://anaconda.org/conda-forge/pinocchio/files)), no Windows wheel on PyPI | not tried | Needs conda and a hand-written IK loop. No gain for a 5-joint arm |
| **placo** (LeRobot's) | no Windows wheel on PyPI | not tried here | Fine in WSL or on macOS. It is numerical (8 steps per call) and treats orientation as soft |

### Closed-form derivation

The SO-101 is a base yaw joint (shoulder_pan), three parallel pitch joints (shoulder_lift, elbow_flex, wrist_flex) and a roll joint (wrist_roll) along the gripper axis. Numbers are from the table above, angles in the arm's vertical plane measured from horizontal, positive up.

1. **Roll is an input.** The tool point is 7.9 mm off the roll axis, so wrist_roll changes where the tip is. Fix roll first (0 during following).
   - At fixed roll, the tip's distance `d` from the vertical plane through the pan axis is a constant, because the pitch joints rotate about axes perpendicular to that plane. At roll 0, `d` is 9 µm.
2. **Yaw.**
   - Target (x, y, z) in the robot frame, ρ = √(x² + y²).
   - The in-plane forward distance is u = √(ρ² − d²).
   - shoulder_pan = atan2(d, u) − atan2(y, x). The sign follows from the pan axis pointing down in the URDF.
3. **Wrist pivot.**
   - Choose the gripper's tilt φ: the angle of the gripper axis from horizontal, so −90° is straight down.
   - The hand vector is 15.94 cm long at angle γ = −2.84° relative to the gripper axis.
   - Wrist pivot W = (u, z) − 15.94 cm · (cos(φ + γ), sin(φ + γ)).
4. **Two links from the shoulder S** (3.04 cm, 11.90 cm) to W, with L1 = 11.60 cm and L2 = 13.50 cm:
   - D = |W − S| and cos β = (D² − L1² − L2²) / (2 L1 L2).
   - No solution if |cos β| > 1: W is out of reach, D > 25.1 cm.
   - β = −acos(…) (elbow up).
   - α1 = atan2(W − S) − atan2(L2 sin β, L1 + L2 cos β) is the upper arm's absolute angle.
5. **Joint angles.** A positive pitch joint lowers the in-plane angle of everything after it.
   - shoulder_lift = 76.03° − α1
   - elbow_flex = (2.21° − 76.03°) − β = −73.82° − β, where −73.82° is the fully stretched elbow
   - wrist_flex = −φ − shoulder_lift − elbow_flex (plus 0.001° for the gripper axis at zero)
6. **Limits.** Reject the solution if any joint is outside the limits (URDF or calibration) minus a margin.
   - Only the elbow-up branch is used: elbow-down needs elbow_flex < −96.8° unless the arm is almost straight.

How the code applies it:

- The code derives every constant in steps 3-5 from the URDF chain at import, so the derivation and the forward kinematics cannot drift apart.
- `inverse_kinematics(target, pitch_deg=-90, wrist_roll_deg=0, tool_point=(0, 0, 0), margin_deg=0, limits=JOINT_LIMITS_DEG)` returns the five URDF angles in degrees, or raises `Unreachable`.
  - `tool_point` moves the controlled point within `gripper_frame_link`; for example `(0, 0, -0.03)` is 3 cm back toward the wrist, about where a held ball's centre might be (to be measured).
- `inverse_kinematics_any_pitch` tries tilts from −90° upward and returns the steepest that is reachable.

### Reachable workspace (tip = `gripper_frame_link`, wrist_roll 0, radius from the pan axis)

Pointing straight down (−90°) and tilted 45° (−45°):

| Height of the tip above the table | −90°, URDF limits | −45°, URDF limits | −45°, limits minus 5° |
| --- | --- | --- | --- |
| 10 cm | none | 12.2-36.8 cm | 13.2-36.8 cm |
| 15 cm | none | 11.2-33.8 cm | 12.5-33.8 cm |
| 20 cm | none | 15.2-29.0 cm | 17.2-28.7 cm |
| 25 cm | none | none | none |

- **Why straight down fails:** with the gripper pointing down the wrist pivot sits 16 cm above the tip. At 10 cm that needs wrist_flex 98-104° against a ±95° limit; from 15 cm up the wrist pivot is also out of reach (D > 25.1 cm).
- **Where it does work:** straight down works at 5 cm for radii 7.5-22.5 cm, and at 8 cm (checked at 15 cm radius, wrist_flex 89.7°).
- **Yaw** does not change lift, elbow or wrist anywhere inside the pan range (checked).

The steepest reachable tilt, 1° steps, URDF limits:

| Height \ radius (cm) | 10 | 12.5 | 15 | 17.5 | 20 | 22.5 | 25 | 27.5 | 30 | 35 | 40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 cm | −90 | −90 | −90 | −90 | −90 | −90 | −89 | −82 | −74 | −57 | −37 |
| 10 cm | – | −82 | −86 | −86 | −84 | −82 | −78 | −73 | −66 | −51 | −32 |
| 15 cm | – | −53 | −63 | −67 | −68 | −67 | −65 | −61 | −55 | −41 | −24 |
| 20 cm | −25 | −37 | −44 | −49 | −51 | −52 | −50 | −47 | −42 | −30 | −13 |
| 25 cm | −17 | −24 | −30 | −34 | −36 | −36 | −35 | −32 | −28 | −16 | +1 |

Variants (full tables in `data/kinematics/check_output.txt`):

- **Controlled point 3 cm behind the tip (the ball):** straight down works at 10 cm for radii 10-22.5 cm, and the steepest tilt at 15 cm is −78°.
- **5° margin on all limits:** costs about 4-10° of tilt at 10-15 cm.
- **If the real wrist_flex reached ±110°:** straight down would work at 10 cm up to 22.5 cm radius.

**For the demo:** hover the ball 10-15 cm above the palms at 15-25 cm from the pan axis. That is the best region, with −65° to −86° of tilt for the tip.

---

## 4. Driving the arm from our own loop

### What `send_action` expects

- **Keys:** `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` ([so_follower.py#L83-L104](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L83-L104)). Body joints are in degrees (with `use_degrees`), the gripper 0-100.
- **Only the keys present are written.** `send_action` builds `Goal_Position` from the keys ending in `.pos` and sync-writes exactly those motors, then returns the goals it sent ([so_follower.py#L187-L213](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L187-L213), [motors_bus.py#L1254-L1295](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L1254-L1295)). An action without `gripper.pos` leaves the gripper's goal as it was.
- **`max_relative_target`** defaults to `None` ([config_so_follower.py#L33-L36](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/config_so_follower.py#L33-L36)). When set, every `send_action` first reads all positions, then clamps each goal to the measured position ± the cap ([so_follower.py#L204-L209](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L204-L209), [robots/utils.py#L136-L170](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/utils.py#L136-L170)). It must be a `float` or a dict whose keys equal the keys sent: a 5-key action against a 6-key dict raises `ValueError`.
  - The clamp is against the measured position, so it would also hold the gripper's goal near the ball's width and loosen the grip (inferred). Leave it off and limit speed in our own code.
- **Servo speed:**
  - The follower uses six STS3215 with 1/345 gearing ([so101.mdx#L35](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/so101.mdx#L35)).
  - Their no-load speed is 0.222 s per 60° at 12 V and 0.192 s per 60° at 7.4 V, about 270-310°/s ([Waveshare ST3215 wiki](https://www.waveshare.com/wiki/ST3215_Servo), [servodatabase](https://servodatabase.com/servo/feetech/sts3215)).
  - At connect LeRobot sets `Acceleration` and `Maximum_Acceleration` to 254 ([motors_bus.py#L572-L592](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py#L572-L592)), with position gains P = 16, I = 0, D = 32 ([config_so_follower.py#L44-L47](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/config_so_follower.py#L44-L47)). So any step in the goal is driven at close to full speed, and our code must ramp the goals itself.
  - Encoder resolution: 4096 steps per turn (LeRobot divides by 4095), 0.088° per step.
- **Gripper protection:** the gripper motor is limited at connect to `Max_Torque_Limit` 500, `Protection_Current` 250 and `Overload_Torque` 25 ([so_follower.py#L154-L157](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py#L154-L157)). Whether a long squeeze on the ball trips the overload protection is **[unverified]**, and is a test at the table.

### Design for following a hand with a person nearby

The numbers are starting values to tune. Items 1-6, 10-12 are in `data/kinematics/handover/following.py` (`FollowSettings`); items 7-8 sit in the team's loop and in `kinematics.inverse_kinematics`. The stand-alone draft `data/kinematics/hand_follow.py` has all of them in one class.

1. **Hover target.** The handover version uses a fixed height: the tip 18 cm above the table, over the mean of the palms. That is the team's choice, because one camera's z is too rough. The draft used palm z + 12 cm, clamped to 8-25 cm.
2. **Glitch rejection.** A detection more than 15 cm from the filtered target is ignored unless it persists for 5 frames. A tracker that jumps to the wrong hand for a frame or two does not move the arm.
3. **Low-pass filter.** Exponential smoothing with a 0.25 s time constant (α = 0.125 at 30 Hz). Gaussian noise drops to about a quarter, at the cost of about 0.25 s of lag (standard formulas; the dry run below measures the effect).
4. **Dead band with hysteresis.** The Cartesian goal starts moving only when the filtered target is more than 1.5 cm away, and then follows until it is within 0.5 cm. A trembling hand leaves the arm still, and a real move is followed to the end.
5. **Workspace clamp and out of reach.** If the target cannot be solved, the goal goes to the farthest solvable point on the line from the current goal toward the target (8 bisection steps), so the arm reaches toward the hand as far as it can.
6. **Cartesian speed limit.** The goal moves at most 0.15 m/s, 5 mm per tick. Moves are straight lines in space, so the ball does not dip between targets. In the handover version an approach from below the hover height rises first, then moves sideways. In the draft the first approach from HOME is a joint-space move instead (see 8).
7. **IK.** Use the steepest reachable tilt and keep wrist_roll fixed. Handover version: −90° to 0° in 1° steps, all at once (vectorised), 3° margin on the joint limits, wrist_roll from the seed. Draft: −90° to −30° in 2° steps with a 5° margin. Use the calibration file's limits once they have been read.
8. **Joint speed limit.** Each joint's command moves a limited step per tick toward the IK solution: 1.5° per tick (45°/s) in the team's loop, 3° per tick (90°/s) in the draft. This also turns HOME moves into smooth joint-space moves.
9. **Minimum heights.** The tip never goes below 8 cm above the table or 12 cm above the higher palm. In the dry run the lowest commanded joint frame was the fixed shoulder_lift pivot (11.9 cm). With elbow-up solutions the elbow and wrist stay high.
10. **Hand lost.** Hold, meaning keep the last command: the servos keep their goal with torque on. After 3 s without a hand, go HOME and keep the ball. When a reachable hand returns, follow again.
11. **Out of reach.** Hold at the edge. After 3 s, go HOME, and come back only when the hand is reachable again.
12. **Drop.**
    - Two hands must be seen, and the ball point (forward kinematics of the *measured* joints, `tool_point` = the ball centre) must have stayed within 3 cm of the hover point for 1.0 s.
    - Then command the gripper open, keep the arm still for 1.0 s, and go HOME with the gripper open.
    - With a tilted gripper the ball should roll off the lower, fixed jaw once the upper jaw opens (geometry, not tested).
13. **Gripper value during the motion.** Every action carries the same `gripper.pos`: the value commanded when the gripper closed on the ball, for example the last value the PICK skill or teleoperation sent, or a fixed value measured at the table. Never send the measured `gripper.pos` back: the jaws are stalled at the ball's width, so commanding that value removes the position error, which is what makes the servo squeeze (position control, P = 16). Omitting `gripper.pos` altogether also keeps the old goal, but the explicit value is safer after a restart.
14. **No reset, reconnect or disconnect while holding.** `research/lerobot-multi-skill-runner.md` section 2 explains why. Use `disable_torque_on_disconnect=False` as the phone example does, and disconnect only after the drop with the arm at rest.

### Wiring it into the LeRobot loop

The team's `handover_loop.py` already does this; see "Checked on Tian's laptop" for its dry run with our two files. Changes it needs:

1. **The `PLACEHOLDER` checks.** Our files have no `PLACEHOLDER` flag, so `handover_loop.py` line 177 (`kinematics.PLACEHOLDER or following.PLACEHOLDER`) and `read_pose.py` line 22 raise `AttributeError` without `--dry-run`. Delete both checks.
2. **`DRY_RUN_START_DEG = [0, 20, 60, 90, 0]`.** With the real URDF this puts the tip on top of the base (3.4 cm behind the pan axis, 6.3 cm high), and the joint-space path out of it dips to 1.7 cm above the table. Use `[0.0, -55.3, 5.6, 91.8, 0.0]` (tip at 15 cm forward, 20 cm high). It only affects the dry run.
3. **Recommended, one line after `connect`:** `kinematics.JOINT_LIMITS_DEG.update(kinematics.joint_limits_from_calibration(robot.calibration_fpath))`.
   - The solver picks the steepest tilt, so wrist_flex sits at its limit minus 3° (about 92°) while hovering.
   - If the calibrated range is smaller, the servo stops short and the tip lands off target.
4. **No change needed:** a hand beyond the loop's `max_radius` of 33 cm leaves the arm hovering at the 33 cm edge in FOLLOW. The IK itself reaches about 40 cm at 18 cm height, so the follower's out-of-reach → HOME only triggers beyond that.

The stand-alone sketch below wires the earlier draft (`hand_follow.py`) directly. It is an **untested sketch**: the controller ran only against a simulated arm.

```python
# UNTESTED with LeRobot or hardware; the controller itself passed data/kinematics/dryrun_hand_follow.py.
import time

import so101_kinematics as kin
from hand_follow import FollowConfig, HandFollower
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
robot = SO101Follower(SO101FollowerConfig(
    port="/dev/ttyACM0", id="<same id as calibration>", use_degrees=True,
    disable_torque_on_disconnect=False,  # cameras={}: the hand tracker owns the cameras in its own process
))
robot.connect()
kin.LEROBOT_OFFSETS_DEG.update(FITTED_OFFSETS)  # from fit_offsets.py at the table, or leave at 0
config = FollowConfig(
    home_deg=HOME_DEG, gripper_hold=GRIPPER_HOLD, gripper_open=GRIPPER_OPEN, tool_point=BALL_POINT,
    limits=kin.joint_limits_from_calibration(robot.calibration_fpath),
)
follower = HandFollower(config, kin.urdf_from_lerobot(robot.get_observation()))
last = time.perf_counter()
while not stop_requested():
    t0 = time.perf_counter()
    obs = robot.get_observation()
    hands = tracker.latest_palms(max_age_s=0.2)  # list of [x, y, z] in the robot frame, metres
    command, gripper = follower.step(t0, t0 - last, hands, kin.urdf_from_lerobot(obs))
    last = t0
    robot.send_action({**kin.lerobot_action_from_urdf(command), "gripper.pos": gripper})
    precise_sleep(max(1 / FPS - (time.perf_counter() - t0), 0.0))
# stop: send nothing more (the arm holds); disconnect only once the ball is gone and the arm rests
```

- `robot.calibration_fpath` is the calibration file's path ([robot.py#L65](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/robot.py#L65)).
- The controller works in URDF degrees. `urdf_from_lerobot` and `lerobot_action_from_urdf` apply the per-joint `LEROBOT_SIGNS` and `LEROBOT_OFFSETS_DEG`, which start as identity.
- Time per controller step on this laptop: about 0.5 ms normally, up to 16 ms while a hand is out of reach (the bisection and the reachability test). If the loop misses 30 Hz there, use coarser tilt steps.

---

## 5. Prior art (ten minutes)

- **Vision-based hand shadowing, SO-ARM101** ([blog](https://chichonnade.github.io/posts/vision-based-hand-shadowing), [arXiv 2603.11383](https://arxiv.org/abs/2603.11383), [code](https://github.com/chichonnade/Vision-Based-Hand-Shadowing-for-Robotic-Manipulation-via-Inverse-Kinematics)):
  - Setup: MediaPipe Hands plus a RealSense D400 depth camera, with landmarks deprojected to 3D. IK is PyBullet `calculateInverseKinematics` (damped least squares, up to 100 iterations).
  - Smoothing: exponential smoothing with α = 0.8 on landmarks and α = 0.5 on the IK solutions, plus a binary gripper to stop jaw jitter. Targets below z = 0.05 m are rejected.
  - Speed: processing ran offline at about 5 fps (213 ms per frame), and trajectories were replayed.
  - Results: 45 of 50 lab pick-and-place runs succeeded (90%), but only 7 of 75 in the wild (9.3%). They note the SO-101 reaches about 30 cm against a human arm's 60 cm. No position accuracy is reported.
- **so101-ros-physical-ai** ([forum post](https://discourse.openrobotics.org/t/interactive-so-101-ik-in-ros-2-with-viser-robokin/53850), [so101_kinematics](https://github.com/legalaspro/so101-ros-physical-ai/tree/main/so101_kinematics)):
  - ROS 2 with numerical placo-based IK through `robokin`. The author calls the SO-101 "really a 5-DOF arm for arm kinematics" and accepts approximate solutions.
  - MoveIt's PickIK "worked for reachable poses, but not as well for approximate poses". No accuracy numbers.
- **box2ai-robotics/lerobot-kinematics** ([GitHub](https://github.com/box2ai-robotics/lerobot-kinematics)): FK and IK examples for the SO-100 and SO-101 with MuJoCo and keyboard or Joy-Con control, plus a "new calibration" variant for the SO-101. The README states no method or accuracy.
- **Hand-sign control of an SO-101** ([Hackster](https://www.hackster.io/AlbertaBeef/hand-controlled-robotic-arm-29a93b)): ROS 2 Jazzy. Recognised hand signs drive 4 of the 6 axes; it does not follow the hand's position.
- **Other arms:** an AgileX Piper mirrors the hand with MediaPipe and Pinocchio IK ([Hackster](https://www.hackster.io/agilexrobotics/control-piper-arm-with-hand-gestures-7c2779)). A ThinkRobotics guide runs MediaPipe hand tracking for an SO-101 on a Raspberry Pi 5; I did not check its IK ([guide](https://thinkrobotics.com/blogs/learn/lerobot-so101-hand-tracking)).
- **Takeaway:** nobody published a closed-form SO-101 IK with measured accuracy. Everyone filters the tracker output. The hard part reported is the 3D hand position (depth noise, occlusion), not the IK.

---

## Checked on Tian's laptop

Environment: `data/kinematics/.venv`, made with the system Python 3.12.10, numpy 2.5.5, ikpy 4.5.0 (scipy 1.18.1, sympy 1.14.0). The URDF and five meshes were downloaded to `data/kinematics/urdf/`. Commands, run from `data/kinematics/`:

```
.venv/Scripts/python.exe -I check_kinematics.py   > check_output.txt
.venv/Scripts/python.exe -I dryrun_hand_follow.py > dryrun_output.txt
.venv/Scripts/python.exe -I fit_offsets.py        > fit_output.txt
.venv/Scripts/python.exe -I explore_urdf.py urdf/so101_new_calib.urdf
cd handover && ../.venv/Scripts/python.exe -I check_handover.py > check_handover_output.txt
```

`check_handover.py` copies the team's `handover_loop.py`, `hand_socket.py` and `fake_tracker.py` (from `git show origin/jakobnunnendorf/handover-loop:scripts/handover/...`, commit `90bb1d3`) and our two files into `data/kinematics/handover_dryrun/`, then runs them there. The git checkout is not touched.

| Check | Result |
| --- | --- |
| Constants in `so101_kinematics.py` against the URDF file | identical (max difference 0) |
| Forward kinematics against ikpy (`Chain.from_urdf_file`), 500 random joint vectors inside the limits | max position difference 3.3e-16 m, rotation 2.4e-6° |
| Forward kinematics against a second, generic URDF walk (Rodrigues rotations, parsed from the file) | 1.5e-16 m, 2.4e-6° |
| forward(inverse(p)): grid x −15…45 cm, y ±40 cm, z 0…38 cm in 2 cm steps; tilts −90, −60, −45, 0°; roll 0, 45, −90°; tool point at the tip and 3 cm back | 135,452 of 610,080 targets solvable inside the URDF limits. Max position error 2.0e-6 m, max tilt error 4.6e-4° |
| inverse(forward(q)), 20,000 random joint vectors | 14,670 have an elbow-up solution inside the limits, position error ≤ 2.9e-6 m. The 10,756 that were elbow-up to begin with came back within 0.09° per joint (99% within 0.008°; the largest near the stretched elbow) |
| Time per solve | closed form 25-40 µs; ikpy position-only 27-37 ms, 2 of 30 reachable targets missed by more than 1 mm from its default start |
| Workspace tables | see section 3 |
| `joint_limits_from_calibration` on a synthetic calibration file | half ranges as expected (for example 1000 steps → ±87.9°, wrist_roll ±180°) |
| Offset fit (`fit_offsets.py`), 20 trials, random true offsets up to 6°, 3 mm touch error | tip error in the hover region without correction up to 56-70 mm. After fitting 8 table touches: 7 mm; 8 touches at two heights (0 and 10 cm): 6 mm; 12 touches at two heights: 5 mm. Single offsets stay uncertain by 3-4° because lift, elbow and wrist trade off, but the tip position does not |
| Controller dry run against a simulated arm (first-order lag 0.1 s, 250°/s max): one hand, a move, a dropout, a false detection, a second hand | Follows from 1.0 s; ball within 3 cm of the hover point 0.33 s later. 5 mm tracker noise with the hand still: 0.0 mm of arm motion. 3-frame false detection: 0.0 mm. Hand moving at 16 cm/s: ball trails by up to 4.9 cm. Drop at 9.0 s (second hand at 8.0 s plus 1.0 s), home at 10.0 s. Joint speeds capped at 90°/s. Lowest commanded ball point 15 cm above the table |
| Dry run, hand moves out of reach at 2 s | reaches as far as it can, goes home at 5.2 s, stays home |
| Dry run, hand lost from 2 to 6 s | goes home at 5.0 s, follows again at 6.0 s |

### The two files for the team's loop (`data/kinematics/handover/check_handover.py` → `check_handover_output.txt`)

| Check | Result |
| --- | --- |
| `forward_kinematics` against ikpy, 500 random joint vectors | 3.2e-16 m |
| forward(inverse(p)) on a 7,800-point grid, random wrist_roll seeds | 5,366 solved, error ≤ 1.9e-6 m, wrist_roll kept from the seed exactly. Same with offsets (4, −3, 2.5, −6)° and wrist_flex's sign flipped |
| Unreachable targets (60 cm out, on the pan axis, 60 cm high, behind the base) | `None` |
| Vectorised tilt search against one `solve` per tilt, 3,000 random targets | 0 mismatches |
| Time per `inverse_kinematics` call | 0.2 ms, reachable or not |
| Tilt chosen at 18 cm hover height | −54° at 25 cm, −48° at 30 cm radius; −66° at 15 cm height and 20 cm radius |
| Team loop's own functions (`arm_goal`, `step_towards`, `Limits`) driven with simulated time from a pose 20 cm up | 5 mm noise: 0.0 mm of tip motion. 3-frame glitch: 0.0 mm. Second hand at 8 s → drop at 9.0 s, done at 10.0 s. Lowest commanded tip 16.7 cm. Hand beyond reach at 2 s → HOME at 5.1 s. Hand lost 2-6 s → HOLD at 2.0, HOME at 5.0, FOLLOW at 6.0 s |
| **Their real dry run:** `handover_loop.py --dry-run --duration 22` with `fake_tracker.py`, both unchanged, over the socket, our two files swapped in | Modes: hold 0, follow 1.43, hold 6.43, home 9.47, follow 10.43, drop 16.17, done 17.2 s. Largest joint step 1.50° per tick. Lowest tip 1.7 cm, because their start pose `[0, 20, 60, 90, 0]` folds the arm onto its base (see section 4) |
| Same dry run, scratch copy with only `DRY_RUN_START_DEG` changed to `[0, −55.3, 5.6, 91.8, 0]` | Same mode sequence (drop at 16.17 s, 1 s after the tip reached the two hands' midpoint (0.200, 0.057)). Lowest tip 16.8 cm. Back at HOME (0.15, 0, 0.20) at the end |

Not run: anything with LeRobot, placo or a robot. The dry-run arms are crude models (no load, no gravity, no servo deadband).

### Cross-check against LeRobot's placo kinematics on Linux (to compare like with like)

- **URDF:** `Simulation/SO101/so101_new_calib.urdf` from TheRobotStudio/SO-ARM100 at commit `141aea7bcac2744d92998e30268c413202a56f96`. The file was last changed on 2025-07-02, so `main` gives the same file today. Meshes are not needed for kinematics.
- **Solver:** `RobotKinematics(urdf_path, target_frame_name="gripper_frame_link", joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])`.
- **What to compare:** placo's `forward_kinematics(q_deg)[:3, 3]` is in `base_link`. Ours is in the robot frame, so expect `ours = placo + (−0.0388353, 8.97657e-09, +0.0024)` m.
  - This holds with `kinematics.TOOL_POINT = (0, 0, 0)` (the default, the `gripper_frame_link` origin) and identity `LEROBOT_SIGNS` / `LEROBOT_OFFSETS_DEG`.
  - Rotations should be identical: `kinematics.tool_pose(q)[:3, :3]`.
  - Expect agreement to 1e-6 m or better, as with ikpy.

## Only verifiable on the arm (checklist for the person at the table)

Nothing below needs a command that moves the arm, except the last items, which are run by the person at the table.

1. **Calibration file, nothing moves.**
   - Open `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json` on the robot laptop (inside WSL if the arm runs there). For each body joint compute `(range_max − range_min) / 2 × 360 / 4095`.
   - Compare with the URDF: pan 110, lift 100, elbow 96.8, wrist_flex 95°. Several degrees less on a joint means its zero is suspect. Note the numbers on the board.
   - `kin.joint_limits_from_calibration(path)` gives the same as limits for the IK.
2. **Signs, torque off, someone holds the arm.**
   - Connect, call `robot.bus.disable_torque()` (the arm goes limp; hold it), and print `get_observation()` while moving one joint at a time.
   - Expected with identity signs:
     - Turning the arm to its **right** makes `shoulder_pan` **increase**.
     - Tipping the upper arm forward and down makes `shoulder_lift` increase.
     - Folding the forearm down makes `elbow_flex` increase.
     - Tipping the gripper down makes `wrist_flex` increase.
     - Positive `wrist_roll` moves the jaw tip from below the gripper's axis to the robot's right.
   - A joint that goes the other way gets `LEROBOT_SIGNS[name] = -1`.
3. **Zero, same session.**
   - Hold the arm so that every body joint reads about 0.
   - Expected: forearm and gripper horizontal and pointing forward, and the upper arm leaning a little forward. The line between its two pivot axes is 14° from vertical.
   - Gross differences (more than about 10°) mean a calibration problem; recalibrate rather than patch.
4. **Touch points, same session.**
   - Mark 8-12 points on the table at known positions in the robot frame. Use 15-30 cm radius and both sides, and put some of them on a 10 cm box.
   - Origin: the point on the table below the pan axis, 6.1 cm in front of the base's back edge and centred left to right (mesh numbers; measure).
   - Touch each mark with the tip at different tilts and record the joint readings.
   - Compare `kin.forward_kinematics(kin.urdf_from_lerobot(obs))` with the mark. Within 1.5 cm everywhere is good. Otherwise run `fit_offsets.fit_offsets(samples)` and put the result into `LEROBOT_OFFSETS_DEG`.
   - This check also settles the table plane and the robot-frame origin the hand tracker must use.
5. **Ball point.**
   - With the ball held, measure where its centre is relative to the tip, along the gripper and across it. That becomes `tool_point`, in metres, in `gripper_frame_link`, whose z axis points out of the gripper: (0, 0, −0.03) is 3 cm back toward the wrist.
   - Also note which wrist_roll value keeps the jaws opening up and down.
6. **Gripper values.** The commanded value that holds the ball firmly (`GRIPPER_HOLD`) and one that releases it (`GRIPPER_OPEN`).
   - The board reports a *measured* 11-16 while holding. The command must be the closed goal: `handover_loop.py` reads `Goal_Position` at start, which is right.
   - Hold the ball for 60 s while the arm moves and check that it does not slip (overload protection).
7. **HOME.** Pick a pose above the bowl with the ball, read its joints, and put them into `home_deg`.
8. **The person at the table runs it:**
   - The follower with `send_action` disabled (print only) while someone moves a hand. Check that the printed targets make sense.
   - Then with motion at half speed (`max_speed_m_s=0.08`, `max_joint_speed_deg_s=45`) and no ball.
   - Then with the ball, and the drop into cupped hands at 10-15 cm hover: does the ball leave the tilted gripper cleanly, and where does it land relative to the hover point?

## Unverified

- That placo's world frame equals `base_link` with a fixed base. Read from usage, not run.
- That LeRobot's positive directions match the URDF axes for every joint. The SO-ARM100 commit message says so; not checked on an arm.
- That the servos clamp goals to the recorded range (inferred from the position-limit registers LeRobot writes).
- That `max_relative_target` would weaken the grip, and that sending the measured gripper position would loosen it. Both are inferred from position control, not tested.
- The gripper's overload behaviour during a long hold.
- How the ball leaves a tilted gripper.
- That the base sits flat on the table with its mesh underside at z = −2.4 mm (it is clamped, which could raise or tilt it).
- The STS3215 speed figures are vendor no-load numbers.
- The controller dry run used a simulated arm. Real servo lag, gravity sag of a loaded arm (the joint error under load shifts the tip) and the tracker's real noise and latency will change the tuning.
