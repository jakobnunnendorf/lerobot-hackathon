import argparse
import csv
import sys
import time
from dataclasses import dataclass

import numpy as np

import following
import kinematics
from following import Decision, Follower, FollowSettings, Mode
from hand_socket import HandReceiver
from kinematics import ARM_JOINTS, forward_kinematics, inverse_kinematics

ARM_KEYS = [f"{joint}.pos" for joint in ARM_JOINTS]
DRY_RUN_START_DEG = np.array([0.0, 20.0, 60.0, 90.0, 0.0])
DRY_RUN_GRIPPER = 13.0


@dataclass
class Limits:
    max_step_deg: float = 1.5
    gripper_step: float = 4.0
    min_height: float = 0.08
    min_radius: float = 0.12
    max_radius: float = 0.33
    ik_tolerance: float = 0.02


def clamp_workspace(xyz: np.ndarray, lim: Limits) -> np.ndarray:
    radius = np.hypot(xyz[0], xyz[1])
    direction = xyz[:2] / radius if radius > 1e-6 else np.array([1.0, 0.0])
    radius = np.clip(radius, lim.min_radius, lim.max_radius)
    return np.array([*(direction * radius), max(xyz[2], lim.min_height)])


def step_towards(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    return current + np.clip(goal - current, -max_step, max_step)


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

    def read(self) -> tuple[np.ndarray, float]:
        obs = self.robot.get_observation()
        return np.array([obs[k] for k in ARM_KEYS]), obs["gripper.pos"]

    def send(self, joints_deg: np.ndarray, gripper: float) -> None:
        self.robot.send_action({**dict(zip(ARM_KEYS, joints_deg.tolist())), "gripper.pos": gripper})


def arm_goal(decision: Decision, commanded: np.ndarray, home: np.ndarray, lim: Limits) -> np.ndarray:
    if decision.mode is Mode.FOLLOW:
        solved = solve_reachable(clamp_workspace(decision.target_xyz, lim), commanded, lim)
        return commanded if solved is None else solved
    if decision.mode in (Mode.HOME, Mode.DONE):
        return home
    return commanded


def run(arm, receiver: HandReceiver, follower: Follower, home: np.ndarray, gripper_hold: float,
        gripper_open: float, lim: Limits, fps: int, duration_s: float | None, on_tick) -> None:
    commanded, _ = arm.read()
    gripper_cmd = gripper_hold
    start = time.monotonic()
    tick_start = start
    while duration_s is None or tick_start - start < duration_s:
        measured, gripper_measured = arm.read()
        tip = forward_kinematics(measured)
        decision = follower.step(receiver.newest_hands(), tip, tick_start)

        commanded = step_towards(commanded, arm_goal(decision, commanded, home, lim), lim.max_step_deg)
        # Until the drop, the holding value is sent every tick, so a lost hand or a solver failure never loosens the grip.
        released = decision.mode in (Mode.DROP, Mode.DONE)
        gripper_cmd = step_towards(np.array(gripper_cmd), np.array(gripper_open), lim.gripper_step).item() if released else gripper_hold
        arm.send(commanded, gripper_cmd)
        on_tick(tick_start - start, decision, commanded, gripper_cmd, forward_kinematics(commanded))

        tick_start = max(tick_start + 1 / fps, time.monotonic())
        time.sleep(max(tick_start - time.monotonic(), 0.0))


class Reporter:
    def __init__(self, print_every: int, csv_path: str | None):
        self.print_every, self.count, self.rows = print_every, 0, []
        self.csv_path = csv_path

    def __call__(self, t, decision, commanded, gripper, tip) -> None:
        self.rows.append([t, decision.mode.value, *commanded, gripper, *tip])
        if self.count % self.print_every == 0:
            joints = " ".join(f"{v:7.1f}" for v in commanded)
            print(f"t={t:6.2f}s {decision.mode.value:6s} joints[{joints}] grip={gripper:5.1f} "
                  f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f})", flush=True)
        self.count += 1

    def save(self, plot_path: str | None) -> None:
        header = ["t", "mode", *ARM_KEYS, "gripper.pos", "tip_x", "tip_y", "tip_z"]
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


def connect_robot(port: str, robot_id: str):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    # Torque stays on at exit: the default disconnect switches it off and would drop a held ball.
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, disable_torque_on_disconnect=False))
    robot.connect()
    return robot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--port", help="robot serial port; required without --dry-run")
    p.add_argument("--robot-id", default="mi_follower")
    p.add_argument("--listen", default="127.0.0.1:8766")
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
    p.add_argument("--min-height", type=float, default=Limits.min_height)
    p.add_argument("--min-radius", type=float, default=Limits.min_radius)
    p.add_argument("--max-radius", type=float, default=Limits.max_radius)
    p.add_argument("--print-every", type=int, default=15)
    p.add_argument("--csv")
    p.add_argument("--plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and (kinematics.PLACEHOLDER or following.PLACEHOLDER):
        sys.exit("Placeholder kinematics/following still in place; only --dry-run is allowed.")
    if not args.dry_run and not args.port:
        sys.exit("--port is required without --dry-run")

    host, port = args.listen.rsplit(":", 1)
    receiver = HandReceiver(host, int(port))
    lim = Limits(args.max_step_deg, Limits.gripper_step, args.min_height, args.min_radius, args.max_radius)
    follower = Follower(FollowSettings(args.hover_height, args.lost_to_home, args.drop_radius, args.drop_dwell))
    reporter = Reporter(args.print_every, args.csv)

    robot = None
    if args.dry_run:
        arm = SimArm(DRY_RUN_START_DEG, DRY_RUN_GRIPPER)
        gripper_hold = args.gripper_hold if args.gripper_hold is not None else DRY_RUN_GRIPPER
    else:
        robot = connect_robot(args.port, args.robot_id)
        arm = RobotArm(robot)
        gripper_hold = args.gripper_hold if args.gripper_hold is not None else robot.bus.sync_read("Goal_Position")["gripper"]
    home = np.array(args.home) if args.home else arm.read()[0]
    print(f"listening on {args.listen}; home={home.round(1).tolist()} gripper_hold={gripper_hold:.1f}", flush=True)

    try:
        run(arm, receiver, follower, home, gripper_hold, args.gripper_open, lim, args.fps, args.duration, reporter)
    except KeyboardInterrupt:
        pass
    finally:
        reporter.save(args.plot)
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
