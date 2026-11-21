import argparse
import time

import numpy as np

from kinematics import ARM_JOINTS, forward_kinematics, tool_pose

# The URDF's gripper_frame_link origin lies on the inner (gripping) face of the fixed jaw, centred across its width,
# 6.3 mm back from that jaw's end (SO-ARM100 meshes at commit 141aea7; derived here and by Kamel's agent, board
# comment 7061703346). The jaw's end is what a person can put on a mark, so that point is printed too.
FIXED_JAW_END = (0.0, 0.0, 0.0063)


def jaw_point(joints_deg: np.ndarray, tool_point=FIXED_JAW_END) -> np.ndarray:
    return tool_pose(joints_deg, tool_point)[:3, 3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="mi_follower")
    p.add_argument("--tool-point", type=float, nargs=3, default=FIXED_JAW_END,
                   help="point in gripper_frame_link coordinates (m) printed as 'jaw tip'; default: end of the fixed jaw")
    args = p.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    # calibrate=False: a calibration mismatch must not start the interactive calibration at the table.
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print("Torque is OFF. Pose the arm by hand; Ctrl-C to stop.")
    try:
        while True:
            obs = robot.get_observation()
            joints = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS])
            frame, jaw = forward_kinematics(joints), jaw_point(joints, args.tool_point)
            print("  ".join(f"{j}={v:7.2f}" for j, v in zip(ARM_JOINTS, joints)),
                  f"gripper={obs['gripper.pos']:6.2f}  frame (m) x={frame[0]:+.3f} y={frame[1]:+.3f} z={frame[2]:+.3f}"
                  f"  jaw tip (m) x={jaw[0]:+.3f} y={jaw[1]:+.3f} z={jaw[2]:+.3f}", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
