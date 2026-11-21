import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handover"))
import kinematics  # noqa: E402

ARM_JOINTS = kinematics.ARM_JOINTS
JOINTS_FILE = Path(__file__).resolve().parents[2] / "data" / "calib" / "joints.json"
MAX_AGE_S = 1.0
NUMBER = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"


def parse_joints(text):
    named = dict(re.findall(rf"(\w+)=\s*({NUMBER})", text))
    if all(name in named for name in ARM_JOINTS):
        return np.array([float(named[name]) for name in ARM_JOINTS])
    numbers = re.findall(NUMBER, text)
    if len(numbers) == len(ARM_JOINTS):
        return np.array(numbers, float)
    raise ValueError(f"Expected a line from read_pose.py or the five values {', '.join(ARM_JOINTS)} in degrees.")


def ask_joints():
    while True:
        try:
            return parse_joints(input(f"Paste the joints ({', '.join(ARM_JOINTS)}, LeRobot degrees): "))
        except ValueError as error:
            print(error)


def write_joints_file(path, positions):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps({"t": time.time(), "joints": {name: float(positions[name]) for name in ARM_JOINTS}}))
    os.replace(temporary, path)


def read_joints_file(path, max_age_s=MAX_AGE_S):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if time.time() - data["t"] > max_age_s:
        return None
    return np.array([data["joints"][name] for name in ARM_JOINTS])


def tool_point_in_base(joints_deg, tool_point_m):
    return kinematics.tool_pose(joints_deg, tuple(tool_point_m))[:3, 3]
