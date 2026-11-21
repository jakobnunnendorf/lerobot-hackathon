import argparse
import time

import numpy as np

from kinematics import ARM_JOINTS, forward_kinematics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="mi_follower")
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
            tip = forward_kinematics(joints)
            print("  ".join(f"{j}={v:7.2f}" for j, v in zip(ARM_JOINTS, joints)),
                  f"gripper={obs['gripper.pos']:6.2f}  FK tip (m) x={tip[0]:+.3f} y={tip[1]:+.3f} z={tip[2]:+.3f}", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
