import argparse
import re

import numpy as np

from kinematics import ARM_JOINTS
from read_pose import FIXED_JAW_END, jaw_point

JOINT_VALUE = {name: re.compile(rf"{name}=\s*(-?\d+(?:\.\d+)?)") for name in ARM_JOINTS}


BOARD_SIDES_MM = {("1", "2"): 172.0, ("1", "4"): 116.0, ("2", "3"): 116.0, ("4", "3"): 172.0}


def parse_line(line: str) -> tuple[str, np.ndarray, np.ndarray]:
    measured, printed = line.split("|", 1)
    label, ruler = measured.split(":")
    ruler_x_mm, ruler_y_mm = (float(v) for v in ruler.split(","))
    joints = np.array([float(JOINT_VALUE[name].search(printed).group(1)) for name in ARM_JOINTS])
    return label.strip(), np.array([ruler_x_mm / 1000, ruler_y_mm / 1000, 0.0]), joints


def print_sides(label: str, points: dict[str, np.ndarray]) -> None:
    sides = [f"{a}-{b} {np.linalg.norm(points[a][:2] - points[b][:2]) * 1000:.0f} (board {mm:.0f})"
             for (a, b), mm in BOARD_SIDES_MM.items() if a in points and b in points]
    print(f"{label} side lengths in mm: " + ", ".join(sides))


def main() -> None:
    p = argparse.ArgumentParser(description="corner file: one line per corner, '<1-4>: <ruler x mm>, <ruler y mm> | <read_pose.py line>'")
    p.add_argument("corner_file")
    p.add_argument("--tool-point", type=float, nargs=3, default=FIXED_JAW_END, help="same as read_pose.py --tool-point")
    args = p.parse_args()
    corners = [parse_line(line) for line in open(args.corner_file) if line.strip() and not line.startswith("#")]

    errors, rulers, jaws = [], {}, {}
    for label, ruler, joints in corners:
        jaw = jaw_point(joints, args.tool_point)
        rulers[label], jaws[label] = ruler, jaw
        errors.append(jaw - ruler)
        print(f"{label:>10}  ruler ({ruler[0]:+.3f},{ruler[1]:+.3f},{ruler[2]:+.3f})  kinematics ({jaw[0]:+.3f},{jaw[1]:+.3f},{jaw[2]:+.3f})"
              f"  error x {errors[-1][0] * 100:+5.1f} y {errors[-1][1] * 100:+5.1f} z {errors[-1][2] * 100:+5.1f} cm")
    # Side lengths do not depend on where the ruler origin was put, so they separate a joint error from an origin error.
    print_sides("ruler     ", rulers)
    print_sides("kinematics", jaws)
    errors = np.array(errors)
    mean, spread = errors.mean(axis=0), np.linalg.norm(errors - errors.mean(axis=0), axis=1).max()
    print(f"mean error x {mean[0] * 100:+.1f} y {mean[1] * 100:+.1f} z {mean[2] * 100:+.1f} cm; "
          f"largest deviation from the mean {spread * 100:.1f} cm")
    if np.linalg.norm(mean) < 0.01 and spread < 0.01:
        print("verdict: kinematics matches the ruler within 1 cm")
    elif spread < 0.01:
        print("verdict: the same error at every corner, so a joint zero offset (or the frame origin is measured from the wrong point)")
    else:
        print("verdict: errors point in different directions, so a joint sign or a wrong joint reading; post the file on the board")


if __name__ == "__main__":
    main()
