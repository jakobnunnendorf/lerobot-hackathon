import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handover"))
import kinematics  # noqa: E402

WAYPOINT_M = 0.008
HOVER_HEIGHTS_M = (0.18, 0.22, 0.25)
RADIUS_M = (0.14, 0.36)
MAX_ROUND_TRIP_M = 1e-5
MAX_JOINT_STEP_DEG = 10.0


def solve(point):
    return kinematics.inverse_kinematics(point, np.zeros(len(kinematics.ARM_JOINTS)))


def round_trip_error(rng, samples=5000):
    points = np.column_stack([rng.uniform(0.08, 0.42, samples), rng.uniform(-0.3, 0.3, samples), rng.uniform(0.05, 0.32, samples)])
    solved = [(p, solve(p)) for p in points]
    errors = [np.linalg.norm(kinematics.forward_kinematics(joints) - p) for p, joints in solved if joints is not None]
    return len(errors), max(errors)


def polar(radius, angle, height):
    return np.array([radius * np.cos(angle), radius * np.sin(angle), height])


def largest_joint_step(rng, height, paths=200):
    largest = 0.0
    for _ in range(paths):
        start = polar(rng.uniform(*RADIUS_M), rng.uniform(-1.2, 1.2), height)
        end = polar(rng.uniform(*RADIUS_M), rng.uniform(-1.2, 1.2), height)
        count = max(int(np.linalg.norm(end - start) / WAYPOINT_M), 1)
        previous = None
        for step in range(count + 1):
            joints = solve(start + (end - start) * step / count)
            if joints is not None and previous is not None:
                largest = max(largest, float(np.abs(joints - previous).max()))
            previous = joints
    return largest


def main():
    rng = np.random.default_rng(0)
    solved, error = round_trip_error(rng)
    print(f"round trip: {solved} reachable points, largest |fk(ik(p)) - p| {error:.1e} m")
    steps = {height: largest_joint_step(rng, height) for height in HOVER_HEIGHTS_M}
    for height, step in steps.items():
        print(f"hover {height:.2f} m: largest joint change between {1000 * WAYPOINT_M:.0f} mm waypoints {step:.1f} deg")
    ok = error < MAX_ROUND_TRIP_M and max(steps.values()) < MAX_JOINT_STEP_DEG
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
