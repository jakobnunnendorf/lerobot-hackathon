import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import kinematics
from following import Decision, Follower, FollowSettings, Mode
from hand_socket import HandReceiver
from kinematics import ARM_JOINTS, forward_kinematics, inverse_kinematics

ARM_KEYS = [f"{joint}.pos" for joint in ARM_JOINTS]
DRY_RUN_START_DEG = np.array([0.0, -55.3, 5.6, 91.8, 0.0])
FRAME_EVERY_TICKS = 2
DRY_RUN_GRIPPER = 13.0


@dataclass
class Limits:
    max_step_deg: float = 1.5
    gripper_step: float = 4.0
    max_step_m: float = 0.008
    min_height: float = 0.08
    max_height: float = 0.30
    min_radius: float = 0.12
    max_radius: float = 0.38
    ik_tolerance: float = 0.02


def clamp_workspace(xyz: np.ndarray, lim: Limits) -> np.ndarray:
    radius = np.hypot(xyz[0], xyz[1])
    direction = xyz[:2] / radius if radius > 1e-6 else np.array([1.0, 0.0])
    radius = np.clip(radius, lim.min_radius, lim.max_radius)
    return np.array([*(direction * radius), np.clip(xyz[2], lim.min_height, lim.max_height)])


def table_and_base_violation(xyz: np.ndarray, lim: Limits) -> float:
    return max(lim.min_height - xyz[2], 0.0) + max(lim.min_radius - np.hypot(xyz[0], xyz[1]), 0.0)


# The IK can ask for up to ~9 deg in one waypoint (board comment 7061460387); while the speed limit spreads
# that over several ticks the tip leaves the waypoint line. A shorter step is taken whenever the full one would
# bring the tip closer to the table or the base than both where it is and where it is going.
def guarded_step(commanded: np.ndarray, goal: np.ndarray, lim: Limits) -> np.ndarray:
    allowed = max(table_and_base_violation(forward_kinematics(commanded), lim),
                  table_and_base_violation(forward_kinematics(goal), lim)) + 1e-6
    for fraction in (1.0, 0.5, 0.25, 0.125):
        candidate = step_towards(commanded, commanded + fraction * (goal - commanded), lim.max_step_deg)
        if table_and_base_violation(forward_kinematics(candidate), lim) <= allowed:
            return candidate
    return commanded


def step_towards(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    delta = goal - current
    largest = np.abs(delta).max(initial=0.0)
    return current + (delta if largest <= max_step else delta * max_step / largest)


def solve_reachable(target: np.ndarray, seed: np.ndarray, lim: Limits) -> np.ndarray | None:
    joints = inverse_kinematics(target, seed)
    if joints is None or np.linalg.norm(forward_kinematics(joints) - target) > lim.ik_tolerance:
        return None
    return joints


class SimArm:
    def __init__(self, joints_deg: np.ndarray, gripper: float):
        self.joints, self.gripper = joints_deg.astype(float), gripper

    def read(self) -> tuple[np.ndarray, float]:
        return self.joints.copy(), self.gripper

    def send(self, joints_deg: np.ndarray, gripper: float) -> None:
        self.joints, self.gripper = joints_deg.copy(), gripper


class RobotArm:
    def __init__(self, robot):
        self.robot = robot
        self.obs = None

    def read(self) -> tuple[np.ndarray, float]:
        self.obs = self.robot.get_observation()
        return np.array([self.obs[k] for k in ARM_KEYS]), self.obs["gripper.pos"]

    def top_frame(self) -> np.ndarray:
        return self.obs["top"]

    def send(self, joints_deg: np.ndarray, gripper: float) -> None:
        self.robot.send_action({**dict(zip(ARM_KEYS, joints_deg.tolist())), "gripper.pos": gripper})


def unreachable_hover_points(hover_height: float, lim: Limits) -> list[tuple[float, float]]:
    seed = np.zeros(len(ARM_JOINTS))
    return [(r, deg) for r in (0.15, 0.20, 0.25, 0.30, 0.35) for deg in (-45, 0, 45)
            if solve_reachable(np.array([r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg)), hover_height]), seed, lim) is None]


# The arm moves along a straight tip line in short Cartesian waypoints, each clamped to the workspace:
# stepping the joints straight to the goal let the tip dip 4 cm below the hover height between targets (dry run).
class Mover:
    def __init__(self, home: np.ndarray, lim: Limits):
        self.home, self.lim = home, lim
        self.homing_in_joints = False

    def goal(self, decision: Decision, commanded: np.ndarray) -> tuple[np.ndarray, bool]:
        homing = decision.mode in (Mode.HOME, Mode.DONE)
        self.homing_in_joints &= homing
        if decision.mode in (Mode.HOLD, Mode.DROP) or self.homing_in_joints:
            return (self.home if self.homing_in_joints else commanded), True
        tip = forward_kinematics(commanded)
        goal_xyz = forward_kinematics(self.home) if homing else clamp_workspace(decision.target_xyz, self.lim)
        # Latched: the last centimetre to the exact HOME joints flipped back and forth between two IK branches.
        if homing and np.linalg.norm(goal_xyz - tip) < 0.01:
            self.homing_in_joints = True
            return self.home, True
        waypoint = clamp_workspace(tip + step_towards(np.zeros(3), goal_xyz - tip, self.lim.max_step_m), self.lim)
        solved = solve_reachable(waypoint, commanded, self.lim)
        if solved is not None:
            return solved, True
        # HOME must always be reachable, even if the straight line is not; a stuck follow target just holds.
        return (self.home, True) if homing else (commanded, False)


def run(arm, receiver: HandReceiver, follower: Follower, home: np.ndarray, gripper_hold: float,
        gripper_open: float, lim: Limits, fps: int, duration_s: float | None, on_tick, frame_writer=None) -> None:
    commanded, _ = arm.read()
    mover = Mover(home, lim)
    gripper_cmd = gripper_hold
    start = time.monotonic()
    tick_start, last_warning, tick = start, 0.0, 0
    while duration_s is None or tick_start - start < duration_s:
        measured, gripper_measured = arm.read()
        tip = forward_kinematics(measured)
        decision = follower.step(receiver.newest_hands(), tip, tick_start)

        goal, reachable = mover.goal(decision, commanded)
        if not reachable and tick_start - last_warning > 1.0:
            print(f"target out of reach, holding: {np.round(decision.target_xyz, 3).tolist()}", flush=True)
            last_warning = tick_start
        commanded = guarded_step(commanded, goal, lim)
        # Until the drop, the holding value is sent every tick, so a lost hand or a solver failure never loosens the grip.
        released = decision.mode in (Mode.DROP, Mode.DONE)
        gripper_cmd = step_towards(np.array(gripper_cmd), np.array(gripper_open), lim.gripper_step).item() if released else gripper_hold
        arm.send(commanded, gripper_cmd)
        on_tick(tick_start - start, decision, commanded, gripper_cmd, forward_kinematics(commanded), forward_kinematics(goal))
        if frame_writer is not None and tick % FRAME_EVERY_TICKS == 0:
            frame_writer.write(arm.top_frame(), time.perf_counter_ns())
        tick += 1

        tick_start = max(tick_start + 1 / fps, time.monotonic())
        time.sleep(max(tick_start - time.monotonic(), 0.0))


class Reporter:
    def __init__(self, print_every: int, csv_path: str | None):
        self.print_every, self.count, self.rows = print_every, 0, []
        self.csv_path = csv_path

    def __call__(self, t, decision, commanded, gripper, tip, goal_tip) -> None:
        self.rows.append([t, decision.mode.value, *commanded, gripper, *tip, *goal_tip])
        if self.count % self.print_every == 0:
            joints = " ".join(f"{v:7.1f}" for v in commanded)
            print(f"t={t:6.2f}s {decision.mode.value:6s} joints[{joints}] grip={gripper:5.1f} "
                  f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})", flush=True)
        self.count += 1

    def save(self, plot_path: str | None) -> None:
        header = ["t", "mode", *ARM_KEYS, "gripper.pos", "tip_x", "tip_y", "tip_z", "goal_x", "goal_y", "goal_z"]
        if self.csv_path:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerows([header, *self.rows])
        if plot_path:
            plot(self.rows, plot_path)


def plot(rows: list, path: str) -> None:
    import matplotlib.pyplot as plt

    data = np.array([[r[0], *r[2:]] for r in rows], dtype=float)
    fig, (ax_j, ax_t) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for i, name in enumerate([*ARM_KEYS, "gripper.pos"]):
        ax_j.plot(data[:, 0], data[:, 1 + i], label=name)
    for i, name in enumerate(["tip_x", "tip_y", "tip_z"]):
        ax_t.plot(data[:, 0], data[:, 7 + i], label=name)
    ax_j.set_ylabel("commanded (deg, gripper 0..100)")
    ax_t.set_ylabel("tip (m)")
    ax_t.set_xlabel("t (s)")
    ax_j.legend(fontsize=7)
    ax_t.legend(fontsize=7)
    fig.savefig(path, dpi=100)


def connect_robot(port: str, robot_id: str, top_camera: str):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    camera = int(top_camera) if top_camera.isdigit() else top_camera
    # Only one process can open the top camera, so the loop owns it and shares frames with the tracker.
    # MJPG as in the recordings and the calibration: another pixel format can change the field of view.
    cameras = {"top": OpenCVCameraConfig(index_or_path=camera, width=640, height=480, fps=30, fourcc="MJPG")}
    # Torque stays on at exit: the default disconnect switches it off and would drop a held ball.
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras=cameras, disable_torque_on_disconnect=False))
    robot.connect()
    kinematics.JOINT_LIMITS_DEG.update(kinematics.joint_limits_from_calibration(robot.calibration_fpath))
    return robot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--robot", action="store_true", help="connect to and move the real arm")
    p.add_argument("--port", help="robot serial port; required with --robot")
    p.add_argument("--top-camera", help="top camera index or device path; required with --robot")
    p.add_argument("--frames-shm", default="top_camera_frames", help="shared-memory name the hand tracker reads")
    p.add_argument("--robot-id", default="mi_follower")
    p.add_argument("--listen", default="127.0.0.1:8766")
    p.add_argument("--any-hand", action="store_true", help="also follow hands the tracker marks closed (default: open palms only)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, help="seconds; default runs until Ctrl-C")
    p.add_argument("--home", type=float, nargs=5, help="HOME joints in degrees; default: the pose at start")
    p.add_argument("--gripper-hold", type=float, help="default: the gripper goal at start (robot) or 13 (dry run)")
    p.add_argument("--gripper-open", type=float, default=60.0)
    p.add_argument("--hover-height", type=float, default=FollowSettings.hover_height)
    p.add_argument("--lost-to-home", type=float, default=FollowSettings.lost_to_home_s)
    p.add_argument("--drop-radius", type=float, default=FollowSettings.drop_radius)
    p.add_argument("--drop-dwell", type=float, default=FollowSettings.drop_dwell_s)
    p.add_argument("--max-step-deg", type=float, default=Limits.max_step_deg)
    p.add_argument("--max-step-m", type=float, default=Limits.max_step_m)
    p.add_argument("--min-height", type=float, default=Limits.min_height)
    p.add_argument("--max-height", type=float, default=Limits.max_height)
    p.add_argument("--min-radius", type=float, default=Limits.min_radius)
    p.add_argument("--max-radius", type=float, default=Limits.max_radius)
    p.add_argument("--print-every", type=int, default=15)
    p.add_argument("--csv")
    p.add_argument("--plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.robot and not (args.port and args.top_camera):
        sys.exit("--robot needs --port and --top-camera")

    host, port = args.listen.rsplit(":", 1)
    receiver = HandReceiver(host, int(port), open_only=not args.any_hand)
    lim = Limits(args.max_step_deg, Limits.gripper_step, args.max_step_m, args.min_height, args.max_height,
                 args.min_radius, args.max_radius)
    if missing := unreachable_hover_points(args.hover_height, lim):
        print(f"WARNING: hover height {args.hover_height} m not reachable at (radius m, pan deg) {missing}", flush=True)
    follower = Follower(FollowSettings(args.hover_height, args.lost_to_home, args.drop_radius, args.drop_dwell))
    reporter = Reporter(args.print_every, args.csv)

    robot, frame_writer = None, None
    if args.dry_run:
        arm = SimArm(DRY_RUN_START_DEG, DRY_RUN_GRIPPER)
        gripper_hold = args.gripper_hold if args.gripper_hold is not None else DRY_RUN_GRIPPER
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handtrack"))
        from frame_link import FrameWriter

        robot = connect_robot(args.port, args.robot_id, args.top_camera)
        arm = RobotArm(robot)
        frame_writer = FrameWriter(args.frames_shm)
        gripper_hold = args.gripper_hold if args.gripper_hold is not None else robot.bus.sync_read("Goal_Position")["gripper"]
    home = np.array(args.home) if args.home else arm.read()[0]
    print(f"listening on {args.listen}; home={home.round(1).tolist()} gripper_hold={gripper_hold:.1f}", flush=True)

    try:
        run(arm, receiver, follower, home, gripper_hold, args.gripper_open, lim, args.fps, args.duration, reporter,
            frame_writer)
    except KeyboardInterrupt:
        pass
    finally:
        reporter.save(args.plot)
        if frame_writer is not None:
            frame_writer.close()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
