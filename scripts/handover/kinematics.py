import json
from functools import lru_cache
from pathlib import Path

import numpy as np

# Joint origins and limits copied from so101_new_calib.urdf (TheRobotStudio/SO-ARM100, Simulation/SO101, commit 141aea7),
# the file LeRobot 0.7.0's examples load. Its zero is the middle of each joint's range, which is also what LeRobot
# reports as 0 with use_degrees=True, so LeRobot degrees map 1:1 unless the arm shows an offset or a flipped sign
# (research/so101-inverse-kinematics.md). The rounded 3.14159 and 1.5708 are kept to match placo and ikpy.
URDF_CHAIN = (
    ("shoulder_pan", (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159), (-1.91986, 1.91986)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0), (-1.74533, 1.74533)),
    ("elbow_flex", (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708), (-1.69, 1.69)),
    ("wrist_flex", (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708), (-1.65806, 1.65806)),
    ("wrist_roll", (5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159), (-2.74385, 2.84121)),
)
GRIPPER_FRAME_ORIGIN = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

ARM_JOINTS = tuple(name for name, *_ in URDF_CHAIN)
JOINT_LIMITS_DEG = {name: tuple(np.degrees(limits)) for name, _, _, limits in URDF_CHAIN}
# The servos stop at the range recorded at calibration, not at the URDF limits; keep clear of both.
LIMIT_MARGIN_DEG = 3.0
# A fully stretched arm is singular: millimetres of tip travel need tens of degrees at shoulder and elbow.
# Measured on 8 mm waypoints at hover height: joint steps up to 16 deg without this bend, up to 9 deg with it.
MIN_ELBOW_BEND_DEG = 20.0

# Robot frame: origin on the table straight below the shoulder_pan axis, x forward (where the arm points at
# shoulder_pan = 0), y to the robot's left, z up. The underside of the base mesh is 2.4 mm below base_link.
BASE_LINK_IN_ROBOT = np.array([-0.0388353, 8.97657e-09, 0.0024])

# The point both functions work with, in gripper_frame_link coordinates (z points out of the gripper). Its origin
# until the held ball's centre has been measured at the table.
TOOL_POINT = (0.0, 0.0, 0.0)

# Gripper axis angle from horizontal, steepest first: as close to straight down as the joint limits allow.
PITCHES_DEG = tuple(range(-90, 1))

LEROBOT_SIGNS = dict.fromkeys(ARM_JOINTS, 1.0)
LEROBOT_OFFSETS_DEG = dict.fromkeys(ARM_JOINTS, 0.0)


class Unreachable(ValueError):
    pass


def forward_kinematics(joints_deg: np.ndarray) -> np.ndarray:
    return tool_pose(joints_deg)[:3, 3]


def inverse_kinematics(target_xyz: np.ndarray, seed_deg: np.ndarray, limits=JOINT_LIMITS_DEG) -> np.ndarray | None:
    wrist_roll = urdf_degrees(seed_deg)[4]
    joints, valid = _solve_pitches(target_xyz, PITCHES_DEG, wrist_roll, None, LIMIT_MARGIN_DEG, limits)
    return lerobot_degrees(joints[np.argmax(valid)]) if valid.any() else None


def tool_pose(joints_deg, tool_point=None):
    return urdf_frames(urdf_degrees(joints_deg), tool_point)[-1]


def urdf_degrees(lerobot_deg):
    values = np.asarray(lerobot_deg, dtype=float)[: len(ARM_JOINTS)]
    return np.array([LEROBOT_SIGNS[n] * (v - LEROBOT_OFFSETS_DEG[n]) for n, v in zip(ARM_JOINTS, values, strict=True)])


def lerobot_degrees(urdf_deg):
    return np.array([LEROBOT_SIGNS[n] * v + LEROBOT_OFFSETS_DEG[n] for n, v in zip(ARM_JOINTS, urdf_deg, strict=True)])


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _transform(rotation=np.eye(3), translation=(0.0, 0.0, 0.0)):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_JOINT_ORIGINS = tuple(_transform(_rpy_matrix(*rpy), xyz) for _, xyz, rpy, _ in URDF_CHAIN)
_GRIPPER_FRAME = _transform(_rpy_matrix(*GRIPPER_FRAME_ORIGIN[1]), GRIPPER_FRAME_ORIGIN[0])
_ROBOT_FROM_BASE_LINK = _transform(translation=BASE_LINK_IN_ROBOT)


def urdf_frames(urdf_deg, tool_point=None):
    frames = []
    t = _ROBOT_FROM_BASE_LINK
    for origin, angle in zip(_JOINT_ORIGINS, np.radians(urdf_deg), strict=True):
        t = t @ origin @ _transform(_rot_z(angle))
        frames.append(t)
    frames.append(t @ _GRIPPER_FRAME @ _transform(translation=TOOL_POINT if tool_point is None else tool_point))
    return frames


def tool_pitch_deg(pose):
    direction = pose[:3, 2]
    return float(np.degrees(np.arctan2(direction[2], np.hypot(direction[0], direction[1]))))


@lru_cache(maxsize=64)
def _planar_geometry(wrist_roll_deg, tool_point):
    frames = urdf_frames([0.0, 0.0, 0.0, 0.0, wrist_roll_deg], tool_point)
    pan_axis = np.array([frames[0][0, 3], frames[0][1, 3], 0.0])
    shoulder, elbow, wrist, tool = (frames[i][:3, 3] - pan_axis for i in (1, 2, 3, 5))

    def link(start, end):
        delta = end - start
        return float(np.hypot(delta[0], delta[2])), float(np.arctan2(delta[2], delta[0]))

    direction = frames[5][:3, 2]
    return {
        "pan_axis": (float(pan_axis[0]), float(pan_axis[1])),
        "shoulder": (float(shoulder[0]), float(shoulder[2])),
        "upper_arm": link(shoulder, elbow),
        "forearm": link(elbow, wrist),
        "hand": link(wrist, tool),
        "pointing": float(np.arctan2(direction[2], direction[0])),
        "lateral": float(tool[1]),
    }


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def solve(target_xyz, pitch_deg, wrist_roll_deg, tool_point=None, margin_deg=0.0, limits=JOINT_LIMITS_DEG):
    joints, valid = _solve_pitches(target_xyz, (pitch_deg,), wrist_roll_deg, tool_point, margin_deg, limits)
    if not valid[0]:
        raise Unreachable(f"{np.round(target_xyz, 3)} at pitch {pitch_deg} deg is out of reach or outside the limits")
    return joints[0]


# Closed form in URDF degrees for many gripper pitches at once, derived in research/so101-inverse-kinematics.md:
# yaw from atan2, then a planar two-link problem for the wrist_flex pivot. Only the elbow-up branch: elbow-down
# needs elbow_flex below -96.8 deg unless the arm is almost straight.
def _solve_pitches(target_xyz, pitches_deg, wrist_roll_deg, tool_point, margin_deg, limits):
    point = TOOL_POINT if tool_point is None else tool_point
    geometry = _planar_geometry(float(wrist_roll_deg), tuple(float(v) for v in point))
    pitch = np.radians(np.asarray(pitches_deg, dtype=float))
    x, y, z = np.asarray(target_xyz, dtype=float) - (*geometry["pan_axis"], 0.0)
    lateral = geometry["lateral"]
    radius_sq = x * x + y * y
    if radius_sq <= lateral * lateral:
        return np.zeros((len(pitch), 5)), np.zeros(len(pitch), dtype=bool)
    forward = np.sqrt(radius_sq - lateral * lateral)
    pan = _wrap(np.arctan2(lateral, forward) - np.arctan2(y, x))

    hand_length, hand_angle = geometry["hand"]
    hand_direction = hand_angle - geometry["pointing"] + pitch
    du = forward - hand_length * np.cos(hand_direction) - geometry["shoulder"][0]
    dz = z - hand_length * np.sin(hand_direction) - geometry["shoulder"][1]

    upper_length, upper_angle = geometry["upper_arm"]
    fore_length, fore_angle = geometry["forearm"]
    cos_bend = (du * du + dz * dz - upper_length**2 - fore_length**2) / (2 * upper_length * fore_length)
    in_reach = (cos_bend >= -1.0) & (cos_bend <= np.cos(np.radians(MIN_ELBOW_BEND_DEG)))
    bend = -np.arccos(np.clip(cos_bend, -1.0, 1.0))
    upper_abs = np.arctan2(dz, du) - np.arctan2(fore_length * np.sin(bend), upper_length + fore_length * np.cos(bend))

    lift = _wrap(upper_angle - upper_abs)
    elbow = _wrap(fore_angle - upper_angle - bend)
    wrist = _wrap(geometry["pointing"] - pitch - lift - elbow)
    joints = np.degrees(np.column_stack([np.full_like(pitch, pan), lift, elbow, wrist, np.zeros_like(pitch)]))
    joints[:, 4] = wrist_roll_deg
    low = np.array([limits[name][0] for name in ARM_JOINTS]) + margin_deg
    high = np.array([limits[name][1] for name in ARM_JOINTS]) - margin_deg
    return joints, in_reach & np.all((joints >= low) & (joints <= high), axis=1)


# In LeRobot degrees the servo limits are +-half the range recorded at calibration (0 is its middle);
# STS3215 has 4096 steps and LeRobot divides by 4095. Returned in URDF degrees, ready for the limits argument.
def joint_limits_from_calibration(path):
    calibration = json.loads(Path(path).read_text())
    limits = {}
    for name in ARM_JOINTS:
        half_range = (calibration[name]["range_max"] - calibration[name]["range_min"]) / 2 * 360 / 4095
        ends = [LEROBOT_SIGNS[name] * (v - LEROBOT_OFFSETS_DEG[name]) for v in (-half_range, half_range)]
        limits[name] = (min(ends), max(ends))
    return limits
