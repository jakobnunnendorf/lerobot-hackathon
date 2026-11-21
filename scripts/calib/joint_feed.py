import argparse
import sys
import time
from pathlib import Path

from joints import ARM_JOINTS, JOINTS_FILE, write_joints_file


def main():
    parser = argparse.ArgumentParser(description="Runs in the LeRobot environment. Reads the follower's joints and keeps "
                                                 "the newest in a file for calibrate_extrinsics_touch.py and validate.py.")
    parser.add_argument("--port", default="/dev/so101_follower")
    parser.add_argument("--robot-id", default="mi_follower")
    parser.add_argument("--file", type=Path, default=JOINTS_FILE)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    if not robot.calibration:
        sys.exit(f"No LeRobot calibration file for id {args.robot_id!r} ({robot.calibration_fpath}).")
    # Only the bus, not robot.connect(): that writes the motor settings, switches the torque on, and on a calibration
    # mismatch starts the interactive calibration. Reading positions changes nothing on the arm, torque stays as it is.
    robot.bus.connect()
    args.file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading joints into {args.file}; nothing is sent to the arm. Ctrl+C stops.")
    try:
        while True:
            positions = robot.bus.sync_read("Present_Position", num_retry=2)
            write_joints_file(args.file, positions)
            print("  ".join(f"{name}={positions[name]:7.2f}" for name in ARM_JOINTS), end="\r", flush=True)
            time.sleep(1 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
