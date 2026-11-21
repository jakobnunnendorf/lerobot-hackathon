import argparse
import sys
from pathlib import Path

import numpy as np
from commands import Commands
from machine import Aborted, Sensors, Settings, Skills, Supervisor, TransitionLog
from skills import (
    PRESENT_POSE,
    SCRIPTS,
    LoadedPolicy,
    PolicySkill,
    PresentAndDrop,
    Replay,
    Settled,
    TopCameraVisibility,
    capped,
    dataset_features,
    give_sequence,
    holding_move,
    import_from,
    load_episode_actions,
    pose_of,
    put_down_at,
)

DEFAULT_PICK = "molnrb/act_pick_ball_combined"


class RobotArm:
    def __init__(self, robot, max_step_deg: float | None = None):
        self.robot, self.max_step_deg = robot, max_step_deg
        goal = robot.bus.sync_read("Goal_Position")
        self.last_sent = {f"{motor}.pos": float(value) for motor, value in goal.items()}
        self.last_obs = None

    def observe(self) -> dict:
        self.last_obs = self.robot.get_observation()
        return self.last_obs

    def send(self, action: dict) -> None:
        action = capped(action, self.last_sent, self.max_step_deg)
        self.robot.send_action(action)
        self.last_sent = dict(action)


def camera_source(value: str):
    return int(value) if value.isdigit() else value


def make_robot(args):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    # Camera names and size as in molnrb/act_pick_ball_combined's config (observation.images.top / .wrist, 640x480).
    cameras = {name: OpenCVCameraConfig(index_or_path=camera_source(source), width=640, height=480, fps=args.fps)
               for name, source in (("top", args.top_camera), ("wrist", args.wrist_camera))}
    # Torque stays on at exit: the default disconnect switches it off and would drop a held ball.
    return SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id, cameras=cameras,
                                             disable_torque_on_disconnect=False))


def settings_from(args) -> Settings:
    return Settings(fps=args.fps, look=args.look, clear_gray=args.clear_gray_policy is not None,
                    max_clear_attempts=args.max_clear_attempts, max_pick_attempts=args.max_pick_attempts,
                    pick_timeout_s=args.pick_timeout, clear_timeout_s=args.clear_timeout,
                    give_timeout_s=args.give_timeout, put_down_timeout_s=args.put_down_timeout,
                    hold_range=tuple(args.hold_range), gripper_hold=args.gripper_hold, hand_wait_s=args.hand_wait,
                    min_visible_area_px=args.min_visible_area)


def put_down_skill(args, arm):
    if args.put_down_episode:
        repo_id, episode = args.put_down_episode.rsplit(":", 1)
        actions = load_episode_actions(repo_id, int(episode))
        return lambda: Replay("put_down", actions)
    held_pose = np.array(args.put_down_pose[:5])
    return lambda: put_down_at(np.concatenate([held_pose, [arm.last_sent["gripper.pos"]]]), args.gripper_open)


def run_robot(args) -> None:
    if not (args.port and args.top_camera and args.wrist_camera):
        sys.exit("--robot needs --port, --top-camera and --wrist-camera")
    if not (args.put_down_episode or args.put_down_pose):
        sys.exit("--robot needs --put-down-episode REPO:EPISODE or --put-down-pose (5 joints)")
    handover_dir = Path(args.handover_dir)
    has_loop = (handover_dir / "handover_loop.py").exists()
    follow = args.give_mode == "follow" or (args.give_mode == "auto" and has_loop)
    if follow and not has_loop:
        sys.exit(f"--give-mode follow needs #13's handover_loop.py in {handover_dir}")
    has_receiver = (handover_dir / "hand_socket.py").exists()
    commands = Commands()
    commands.listen_keyboard()
    if args.voice:
        commands.listen_voice(args.voice_host)

    robot = make_robot(args)
    features = dataset_features(robot)
    # Policies load before connecting: loading takes seconds and nothing should wait on it with torque on.
    rename = dict(pair.split("=", 1) for pair in args.camera_rename)
    pick_policy = LoadedPolicy(args.pick_policy, args.device, args.n_action_steps, rename)
    clear_policy = LoadedPolicy(args.clear_gray_policy, args.device, None, rename) if args.clear_gray_policy else None
    receiver = import_from(handover_dir, "hand_socket").HandReceiver(*split_address(args.hand_listen)) if has_receiver else None
    frame_link_dir = SCRIPTS / "handtrack"
    frame_writer = (import_from(frame_link_dir, "frame_link").FrameWriter(args.frames_shm)
                    if (frame_link_dir / "frame_link.py").exists() else None)

    robot.connect()
    log = TransitionLog(args.log)
    try:
        arm = RobotArm(robot, args.max_step_deg)
        first = arm.observe()
        rest = np.array(args.rest) if args.rest else pose_of(first)
        settings = settings_from(args)
        if settings.hold_range[0] <= first["gripper.pos"] <= settings.hold_range[1]:
            sys.exit(f"gripper reads {first['gripper.pos']:.1f} (holding range) at start: empty the gripper first")
        if follow:
            kinematics = import_from(handover_dir, "kinematics")
            kinematics.JOINT_LIMITS_DEG.update(kinematics.joint_limits_from_calibration(robot.calibration_fpath))
        palms_seen = (lambda: len(receiver.newest_hands()) >= 2) if receiver else (lambda: False)
        pick_finished = lambda: Settled(rest, settings.hold_range, args.settle_tol_deg, 1.0, 3.0)
        if follow:
            def give_skill():
                return give_sequence(arm, args.present_pose, handover_dir=handover_dir, receiver=receiver,
                                     abort=commands.abort, timeout_s=settings.give_timeout_s,
                                     gripper_open=args.gripper_open, frame_writer=frame_writer, fps=args.fps)
        elif receiver:
            def give_skill():
                return PresentAndDrop(palms_seen, args.gripper_open)
        else:
            give_skill = None
        skills = Skills(
            pick=lambda: PolicySkill("pick", pick_policy, features, robot.robot_type, args.pick_task, pick_finished()),
            clear_gray=(lambda: PolicySkill("clear_gray", clear_policy, features, robot.robot_type)) if clear_policy else None,
            give=give_skill,
            present=lambda: holding_move("to_present", arm, args.present_pose),
            put_down=put_down_skill(args, arm),
        )
        sensors = Sensors(palms_seen=palms_seen, visible_area=TopCameraVisibility() if args.look or clear_policy else None,
                          destination=commands.destination)
        supervisor = Supervisor(arm, skills, sensors, settings, rest, log, commands.abort, frame_writer=frame_writer,
                                on_tick=commands.poll_voice)
        log.write(event="start", rest=rest.round(1).tolist(), give="follow" if follow else "present" if receiver else None, look=args.look,
                  clear_gray=clear_policy is not None, pick_policy=args.pick_policy)
        print("Enter starts a round. h/c + Enter: destination hand/container, x: no preference, a: ABORT.", flush=True)
        for number in range(1, args.rounds + 1):
            if not args.auto:
                commands.go.clear()
                try:
                    supervisor.idle_until(commands.go)
                except Aborted:
                    break
            if supervisor.run_round(number).outcome == "aborted":
                print("ABORT: arm holds its pose with torque on. Hold the arm, then switch the power off.", flush=True)
                break
    finally:
        robot.disconnect()


def split_address(address: str) -> tuple[str, int]:
    host, port = address.rsplit(":", 1)
    return host, int(port)


def run_fake(args) -> None:
    from fakes import SCENARIOS, build

    chosen = SCENARIOS if args.fake == "all" else [s for s in SCENARIOS if s.name == args.fake]
    if not chosen:
        sys.exit(f"unknown scenario; choose from: all, {', '.join(s.name for s in SCENARIOS)}")
    log = TransitionLog(args.log, echo=args.verbose)
    for number, scenario in enumerate(chosen, 1):
        arm, skills, sensors, settings, abort, clock, _world = build(scenario)
        rest = pose_of(arm.last_sent)
        result = Supervisor(arm, skills, sensors, settings, rest, log, abort, clock).run_round(number)
        print(f"{scenario.name:32s} {result.outcome:9s} {' -> '.join(s.value for s in result.states)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", metavar="SCENARIO", help="scripted fake skills and arm; 'all' runs every scenario")
    mode.add_argument("--robot", action="store_true", help="connect to and move the real arm")
    p.add_argument("--log", help="JSONL file for transitions and round results (appended)")
    p.add_argument("--verbose", action="store_true", help="fake mode: print every transition")
    p.add_argument("--port")
    p.add_argument("--top-camera")
    p.add_argument("--wrist-camera")
    p.add_argument("--robot-id", default="mi_follower")
    p.add_argument("--device", help="policy device; default: the checkpoint's, falling back as LeRobot does")
    p.add_argument("--pick-policy", default=DEFAULT_PICK)
    p.add_argument("--n-action-steps", type=int, help="override the pick checkpoint's n_action_steps (ACT: 100)")
    p.add_argument("--camera-rename", nargs="*", default=[], metavar="OURS=THEIRS",
                   help="camera names the checkpoint expects, e.g. top=camera1 wrist=camera2 for SmolVLA")
    p.add_argument("--pick-task", default="", help="task sentence; ACT ignores it, SmolVLA needs it")
    p.add_argument("--clear-gray-policy", help="enables LOOK + CLEAR_GRAY; off by default")
    p.add_argument("--put-down-episode", help="REPO_ID:EPISODE replayed as PUT_DOWN")
    p.add_argument("--put-down-pose", type=float, nargs=5, help="joints (deg) to carry the ball to, then open")
    p.add_argument("--rest", type=float, nargs=6, help="rest pose incl. gripper; default: the pose at start")
    p.add_argument("--handover-dir", default=str(SCRIPTS / "handover"), help="where #13's handover_loop.py lives")
    p.add_argument("--present-pose", type=float, nargs=5, default=PRESENT_POSE,
                   help="joints (deg) the ball is carried to before GIVE; the hand decision is made there")
    p.add_argument("--give-mode", choices=("auto", "follow", "present"), default="auto",
                   help="follow: #13 hand-following loop; present: hold the present pose, drop on two palms "
                        "(no camera calibration needed); auto: follow if #13 is there")
    p.add_argument("--hand-listen", default="127.0.0.1:8766")
    p.add_argument("--frames-shm", default="top_camera_frames")
    p.add_argument("--voice", action="store_true", help="receive voice commands (scripts/voice/link.py) on port 8765")
    p.add_argument("--voice-host", default="127.0.0.1", help="0.0.0.0 when this runs in WSL and voice on Windows")
    p.add_argument("--look", action="store_true", help="log the top-camera visibility check before each pick")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--auto", action="store_true", help="start rounds without waiting for Enter")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-pick-attempts", type=int, default=Settings.max_pick_attempts)
    p.add_argument("--max-clear-attempts", type=int, default=Settings.max_clear_attempts)
    p.add_argument("--pick-timeout", type=float, default=Settings.pick_timeout_s)
    p.add_argument("--clear-timeout", type=float, default=Settings.clear_timeout_s)
    p.add_argument("--give-timeout", type=float, default=Settings.give_timeout_s)
    p.add_argument("--put-down-timeout", type=float, default=Settings.put_down_timeout_s)
    p.add_argument("--hold-range", type=float, nargs=2, default=Settings.hold_range)
    p.add_argument("--hand-wait", type=float, default=Settings.hand_wait_s)
    p.add_argument("--min-visible-area", type=int, default=Settings.min_visible_area_px)
    p.add_argument("--settle-tol-deg", type=float, default=8.0, help="PICK ends early within this of the rest pose")
    p.add_argument("--gripper-open", type=float, default=60.0)
    p.add_argument("--gripper-hold", type=float,
                   help="gripper goal while holding; default: keep the pick policy's last goal")
    p.add_argument("--max-step-deg", type=float, help="speed cap: largest arm joint change per tick, all skills")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.robot:
        run_robot(args)
    else:
        run_fake(args)


if __name__ == "__main__":
    main()
