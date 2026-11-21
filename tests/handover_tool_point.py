import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handover"))
import kinematics  # noqa: E402
from read_pose import FIXED_JAW_END, jaw_point  # noqa: E402


def steepest_pose_over(xy):
    return kinematics.inverse_kinematics(np.array([*xy, 0.10]), np.zeros(len(kinematics.ARM_JOINTS)))


def main() -> None:
    joints = steepest_pose_over((0.20, 0.0))
    pose = kinematics.tool_pose(joints)
    frame, end = pose[:3, 3], jaw_point(joints)
    pitch = kinematics.tool_pitch_deg(pose)
    drop = frame[2] - end[2]
    expected = 0.0063 * np.sin(np.radians(-pitch))
    print(f"gripper pitch {pitch:.1f} deg; jaw end is {drop * 1000:.2f} mm below the frame point (expected {expected * 1000:.2f})")
    assert np.isclose(np.linalg.norm(end - frame), np.linalg.norm(FIXED_JAW_END))
    assert pitch < -30 and drop > 0 and np.isclose(drop, expected, atol=1e-6)
    print("PASS")


if __name__ == "__main__":
    main()
