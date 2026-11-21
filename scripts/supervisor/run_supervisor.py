import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from commands import Commands
from grasp_check import GraspCheckSettings
from machine import Aborted, Sensors, Settings, Skills, Supervisor, TransitionLog
from skills import (
    KEYS,
    PRESENT_POSE,
    SCRIPTS,
    GiveSettings,
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
from status import STATUS

DEFAULT_PICK = "molnrb/act_pick_ball_combined"


class RobotArm:
    def __init__(self, robot, max_step_deg: float | None = None, wrist_writer=None):
        self.robot, self.max_step_deg, self.wrist_writer = robot, max_step_deg, wrist_writer
        goal = robot.bus.sync_read("Goal_Position")
        self.last_sent = {f"{motor}.pos": float(value) for motor, value in goal.items()}
        self.last_obs = None
        self.observes = 0

    # Every skill observes through here, GIVE's loop included, so the wrist frames and live values reach the
    # dashboard in every state.
    def observe(self) -> dict:
        self.last_obs = self.robot.get_observation()
        self.observes += 1
        if self.wrist_writer is not None and self.observes % 2 == 0:
            self.wrist_writer.write(self.last_obs["wrist"], time.perf_counter_ns())
        STATUS.update(joints=[round(float(self.last_obs[key]), 1) for key in KEYS[:5]],
                      gripper=round(float(self.last_obs["gripper.pos"]), 1), observes=self.observes)
        return self.last_obs

    # Grasp check: the gripper servo's Present_Load (STS3215 register 60, sign-magnitude, 0.1% of max torque in the
    # Feetech docs; LeRobot 0.7.0's bus reads any register through rustypot, unnormalised). Not read on the arm yet.
    def gripper_load(self) -> float:
        return float(self.robot.bus.sync_read("Present_Load", "gripper")["gripper"])

    def send(self, action: dict) -> None:
        action = capped(action, self.last_sent, self.max_step_deg)
        self.robot.send_action(action)
        self.last_sent = dict(action)


def camera_source(value: str):
    return int(value) if value.isdigit() else value


def camera_configs(args) -> dict:
    from lerobot.cameras.opencv import OpenCVCameraConfig

    # Camera names and size as in molnrb/act_pick_ball_combined's config (observation.images.top / .wrist, 640x480).
    # MJPG as in PR #6's scripts/so101/common.sh: two uncompressed 640x480/30 fps streams usually do not both fit
    # on one USB bus (about 147 vs 9 Mbps per camera, research/lerobot-training-feasibility.md on that branch).
    fourcc = None if args.camera_fourcc.lower() == "none" else args.camera_fourcc
    return {name: OpenCVCameraConfig(index_or_path=camera_source(source), width=640, height=480, fps=args.fps, fourcc=fourcc)
            for name, source in (("top", args.top_camera), ("wrist", args.wrist_camera))}


def make_robot(args):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cameras = camera_configs(args)

    # Torque stays on at exit: the default disconnect switches it off and would drop a held ball.
    return SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id, cameras=cameras,
                                             disable_torque_on_disconnect=False))


def settings_from(args) -> Settings:
    return Settings(fps=args.fps, look=args.look, clear_gray=args.clear_gray_policy is not None,
                    max_clear_attempts=args.max_clear_attempts, max_pick_attempts=args.max_pick_attempts,
                    pick_timeout_s=args.pick_timeout, clear_timeout_s=args.clear_timeout,
                    give_timeout_s=args.give_timeout, put_down_timeout_s=args.put_down_timeout,
                    hold_range=tuple(args.hold_range), gripper_hold=args.gripper_hold, hand_wait_s=args.hand_wait,
                    min_visible_area_px=args.min_visible_area, min_held_target_area_px=args.min_held_target_area,
                    held_check_colours=tuple(args.held_check_colours), held_unseen_floor_px=args.held_unseen_floor,
                    min_held_area_px={**Settings().min_held_area_px, **dict(args.min_held_area)},
                    grasp_check=None if args.no_grasp_check else GraspCheckSettings(
                        trigger=args.grasp_trigger, armed_above=args.grasp_armed_above, close_goal=args.grasp_close_goal,
                        settle_s=args.grasp_settle, min_error=args.grasp_min_error, hold_range=tuple(args.hold_range),
                        min_load=args.grasp_min_load, backoff_s=args.grasp_backoff,
                        max_tip_height=args.grasp_max_tip_height))


def colour_area(value: str) -> tuple[str, int]:
    colour, _, area = value.partition("=")
    if colour not in ("striped", "red", "blue") or not area.isdigit():
        raise argparse.ArgumentTypeError(f"expected striped|red|blue=PX, got {value!r}")
    return colour, int(area)


# Ellipse (centre x, centre y, radius x, radius y) as frame fractions: where the jaws hold the ball in the wrist
# frame at rest, the lower middle (the camera faces the ceiling). From PR #30 (kb/held-ball-wrist-areas.md,
# board 7063126045): the striped ball's centre at x 0.49-0.62, y 0.64-0.77 over 183 offline detections of
# molnrb/so101_pick_rb_mixed, checked on overlay frames. The first guess 0.5 0.5 0.25 0.3 saw mostly ceiling.
WRIST_HELD_ROI = (0.52, 0.72, 0.2, 0.25)


def held_ball_area(vision_dir: Path, roi: tuple[float, float, float, float] = WRIST_HELD_ROI):
    """#24's detector (branch jakobnunnendorf/colour-pick) as Sensors.held_ball_area; None if its files are not here.

    Without a merge: --vision-dir <colour-pick worktree>/scripts/vision. The top camera's bowl ROI is replaced by
    `roi`, the region of the wrist frame where the jaws hold the ball, and the ball size limit is dropped (the
    held ball is close). Without the region blue read up to ~218,000 px on whole wrist frames (board 7063041111).
    """
    if not (vision_dir / "supervisor_adapter.py").exists():
        return None
    detector_module = import_from(vision_dir, "colour_detector")
    adapter = import_from(vision_dir, "supervisor_adapter")
    base = detector_module.load_detector()
    specs = tuple(replace(spec, use_roi=True, max_area=sys.maxsize) for spec in base.specs)
    visibility = adapter.ColourVisibility(detector_module.ColourDetector(specs, roi=tuple(roi)))

    def area(wrist_frame: np.ndarray, wanted: str) -> int:
        if not visibility.set_target(wanted):
            raise ValueError(f"the detector has no ball {wanted!r}")
        return visibility(wrist_frame)
    return area


def held_check_hook(args):
    if args.no_held_check:
        print("WARNING: --no-held-check: only the gripper band decides, a red/blue ball reads like the target", flush=True)
        return None
    hook = held_ball_area(Path(args.vision_dir), tuple(args.held_check_roi))
    if hook is None:
        print(f"WARNING: no #24 colour detector in {args.vision_dir}: held-ball check off, only the gripper band decides "
              "(a red/blue ball reads like the target)", flush=True)
    if hook is not None and "blue" in args.held_check_colours:
        print("WARNING: blue is not usable on the wrist camera (up to 27,466 px with the target held, "
              "kb/held-ball-wrist-areas.md): without --min-held-area blue=PX a blue check only logs", flush=True)
    if hook is not None and args.min_held_target_area is None and "striped" not in dict(args.min_held_area):
        print("WARNING: --min-held-target-area not measured: the held-ball area is logged at every VERIFY_HOLD but "
              "decides nothing. Suggested start from offline data: 120; confirm at the table (wrist frame at rest, "
              "target vs red/blue ball)", flush=True)
    return hook


def wrist_frame_saver(log_path: str | None):
    """Sensors.save_wrist_frame: one JPEG per VERIFY_HOLD next to the --log file, to see what the wrist camera saw.

    None without --log. Best effort: the supervisor catches and logs any error, the round carries on.
    """
    if not log_path:
        return None
    log = Path(log_path)

    def save(frame: np.ndarray, name: str) -> str:
        from PIL import Image

        path = log.parent / f"{log.stem}_{name}_{datetime.now().strftime('%H%M%S_%f')[:-3]}.jpg"
        # Robot observations are RGB uint8 (OpenCVCamera's default colour mode), as PIL expects.
        Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(path, quality=90)
        return str(path)
    return save


def container_skills(args, arm) -> dict:
    # One taught pose per sorting container. Not taught yet (placeholder): the colour falls back to --put-down-pose.
    # Teach both at the table with read_pose.py, like the present pose.
    poses = {"red": args.container_pose_red, "blue": args.container_pose_blue}
    for colour, pose in poses.items():
        if pose is None:
            print(f"WARNING: --container-pose-{colour} not taught: {colour} balls go to the generic put-down pose", flush=True)
    return {colour: (lambda pose=np.array(pose): put_down_at(np.concatenate([pose, [arm.last_sent["gripper.pos"]]]),
                                                              args.gripper_open))
            for colour, pose in poses.items() if pose is not None}


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
    frame_link_dir = Path(args.handtrack_dir)
    frame_writer, wrist_writer = None, None
    if (frame_link_dir / "frame_link.py").exists():
        frame_link = import_from(frame_link_dir, "frame_link")
        unlink_stale_block(args.frames_shm)
        frame_writer = frame_link.FrameWriter(args.frames_shm)
        unlink_stale_block(args.wrist_shm)
        wrist_writer = frame_link.FrameWriter(args.wrist_shm)
    else:
        print(f"WARNING: no frame_link.py in {frame_link_dir}: the hand tracker gets no top-camera frames", flush=True)
    try:
        run_connected(args, robot, receiver, frame_writer, handover_dir, follow, commands, features, pick_policy,
                      clear_policy, wrist_writer)
    finally:
        # A crash or Ctrl-C must not leave the shared-memory blocks: the next start would fail with FileExistsError.
        for writer in (frame_writer, wrist_writer):
            if writer is not None:
                try:
                    writer.close()
                except Exception as error:  # noqa: BLE001 - closing must never hide the original error
                    print(f"frame writer close failed: {error}", file=sys.stderr, flush=True)


def unlink_stale_block(name: str) -> None:
    """Remove a shared-memory block left behind by a crashed run, so creating it again cannot fail."""
    try:
        stale = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    stale.close()
    stale.unlink()
    print(f"removed stale shared-memory block {name!r}", flush=True)


def sync_goal_to_present(bus) -> None:
    """Goal_Position = Present_Position (raw steps) on a bus of its own, torque untouched, before robot.connect().

    LeRobot 0.7.0's connect() switches the torque on without writing a goal, so after the arm was moved by hand each
    servo jumps to its stale stored goal. Tian's fix (TianK003/hand-tracking-test), as in #13's connect_robot().
    """
    bus.connect()
    try:
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position", normalize=False), normalize=False)
    finally:
        bus.disconnect(disable_torque=False)


def give_settings_from(args) -> GiveSettings:
    return GiveSettings(hover_height=args.give_hover_height, min_height=args.give_min_height,
                        max_step_deg=args.give_max_step_deg, max_step_m=args.give_max_step_m, celebrate=args.celebrate)


def run_connected(args, robot, receiver, frame_writer, handover_dir, follow, commands, features, pick_policy,
                  clear_policy, wrist_writer=None) -> None:
    sync_goal_to_present(robot.bus)
    robot.connect()
    log = TransitionLog(args.log)
    try:
        arm = RobotArm(robot, args.max_step_deg, wrist_writer)
        first = arm.observe()
        rest = np.array(args.rest) if args.rest else pose_of(first)
        settings = settings_from(args)
        if settings.hold_range[0] <= first["gripper.pos"] <= settings.hold_range[1]:
            sys.exit(f"gripper reads {first['gripper.pos']:.1f} (holding range) at start: empty the gripper first")
        if follow:
            kinematics = import_from(handover_dir, "kinematics")
            kinematics.JOINT_LIMITS_DEG.update(kinematics.joint_limits_from_calibration(robot.calibration_fpath))
        palms_seen = (lambda: palm_count(receiver) >= 2) if receiver else (lambda: False)
        pick_finished = lambda: Settled(rest, settings.hold_range, args.settle_tol_deg, 1.0, 3.0)
        if follow:
            def give_skill():
                return give_sequence(arm, args.present_pose, handover_dir=handover_dir, receiver=receiver,
                                     abort=commands.abort, timeout_s=settings.give_timeout_s,
                                     gripper_open=args.gripper_open, frame_writer=frame_writer, fps=args.fps,
                                     settings=give_settings_from(args))
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
            put_down_container=container_skills(args, arm),
            # pick_colour: HOOK for the colour pick (branch jakobnunnendorf/colour-pick). Until it is plugged in, a
            # round that asks for "red" or "blue" is refused before the arm moves.
        )
        tip_height = None
        if (handover_dir / "kinematics.py").exists():
            kinematics = import_from(handover_dir, "kinematics")
            tip_height = lambda joints: float(kinematics.forward_kinematics(np.asarray(joints))[2])  # noqa: E731
        elif args.grasp_max_tip_height is not None:
            sys.exit(f"--grasp-max-tip-height needs #13's kinematics.py in {handover_dir}")
        sensors = Sensors(palms_seen=palms_seen, gripper_load=arm.gripper_load, tip_height=tip_height, visible_area=TopCameraVisibility() if args.look or clear_policy else None,
                          destination=commands.destination,
                          target=commands.target, finish_round=commands.clear, held_ball_area=held_check_hook(args),
                          save_wrist_frame=wrist_frame_saver(args.log))
        supervisor = Supervisor(arm, skills, sensors, settings, rest, log, commands.abort, frame_writer=frame_writer,
                                on_tick=commands.poll_voice)
        log.write(event="start", rest=rest.round(1).tolist(), give="follow" if follow else "present" if receiver else None, look=args.look,
                  clear_gray=clear_policy is not None, pick_policy=args.pick_policy,
                  held_check=sensors.held_ball_area is not None, min_held_target_area_px=args.min_held_target_area,
                  held_check_roi=list(args.held_check_roi), held_check_colours=list(args.held_check_colours),
                  min_held_area_px=settings.min_held_area_px, camera_fourcc=args.camera_fourcc,
                  grasp_check=None if settings.grasp_check is None else vars(settings.grasp_check),
                  n_action_steps=pick_policy.policy.config.n_action_steps, budgets_s=budgets(settings),
                  put_down_pose=args.put_down_pose)
        publish_status(args, skills, receiver, settings)
        print("Enter starts a round. h/c + Enter: destination hand/container, r/b/s + Enter: target red/blue/striped, x: clear, a: ABORT.", flush=True)
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


def budgets(settings: Settings) -> dict:
    return {"PICK": settings.pick_timeout_s, "GIVE": settings.give_timeout_s, "PUT_DOWN": settings.put_down_timeout_s,
            "REST": settings.move_timeout_s, "CLEAR_GRAY": settings.clear_timeout_s}


def publish_status(args, skills: Skills, receiver, settings: Settings) -> None:
    targets = ["striped", *(["red", "blue"] if skills.pick_colour else [])]
    destinations = [*(["hand"] if skills.give else []), "container"]
    STATUS.update(available={"targets": targets, "destinations": destinations}, pick_policy=args.pick_policy,
                  log_path=str(Path(args.log).resolve()) if args.log else None, budgets_s=budgets(settings),
                  hold_range=list(settings.hold_range), min_held_target_area_px=args.min_held_target_area)
    if receiver is not None:
        STATUS.add_reader(lambda: {"palms": palm_count(receiver),
                                   "hands": [np.round(hand, 3).tolist() for hand in receiver.newest_hands()]})
    STATUS.start(port=args.status_port)


def palm_count(receiver) -> int:
    # #13's receiver counts open palms, from pixels too when the tracker has no calibration; older ones only the 3D hands.
    counter = getattr(receiver, "newest_palm_count", None)
    return counter() if counter else len(receiver.newest_hands())


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
    p.add_argument("--clear-gray-policy", help="OLD FOIL-BALL TABLE ONLY (gray foil balls, gone since board 7062034477): enables LOOK + CLEAR_GRAY; off by default")
    p.add_argument("--put-down-episode", help="REPO_ID:EPISODE replayed as PUT_DOWN")
    p.add_argument("--put-down-pose", type=float, nargs=5, help="joints (deg) to carry the ball to, then open")
    p.add_argument("--container-pose-red", type=float, nargs=5,
                   help="joints (deg) over the red sorting container, taught at the table; default: --put-down-pose")
    p.add_argument("--container-pose-blue", type=float, nargs=5,
                   help="joints (deg) over the blue sorting container, taught at the table; default: --put-down-pose")
    p.add_argument("--rest", type=float, nargs=6, help="rest pose incl. gripper; default: the pose at start")
    p.add_argument("--handover-dir", default=str(SCRIPTS / "handover"), help="where #13's handover_loop.py lives")
    p.add_argument("--handtrack-dir", default=str(SCRIPTS / "handtrack"),
                   help="where frame_link.py lives (shares top-camera frames with the hand tracker)")
    p.add_argument("--present-pose", type=float, nargs=5, default=PRESENT_POSE,
                   help="joints (deg) the ball is carried to before GIVE; the hand decision is made there")
    p.add_argument("--give-mode", choices=("auto", "follow", "present"), default="auto",
                   help="follow: #13 hand-following loop; present: hold the present pose, drop on two palms "
                        "(no camera calibration needed); auto: follow if #13 is there")
    p.add_argument("--give-hover-height", type=float, default=GiveSettings.hover_height,
                   help="follow GIVE: tip height over the hands (m); 0.28 ran on the arm (board 7064755928)")
    p.add_argument("--give-min-height", type=float, default=GiveSettings.min_height,
                   help="follow GIVE: no target below this (m), as --min-height of the run on the arm")
    p.add_argument("--give-max-step-deg", type=float, default=GiveSettings.max_step_deg,
                   help="follow GIVE: joint speed limit (deg/tick), half speed as on the arm")
    p.add_argument("--give-max-step-m", type=float, default=GiveSettings.max_step_m,
                   help="follow GIVE: tip step per tick (m), half speed as on the arm")
    p.add_argument("--celebrate", type=int, default=GiveSettings.celebrate,
                   help="follow GIVE: fast close-open gripper cycles after the drop (Tian's, 0 = off); then REST folds")
    p.add_argument("--hand-listen", default="127.0.0.1:8766")
    p.add_argument("--frames-shm", default="top_camera_frames")
    p.add_argument("--wrist-shm", default="wrist_camera_frames", help="shared-memory name the dashboard reads")
    p.add_argument("--status-port", type=int, default=8767, help="UDP port the dashboard reads live values from")
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
    p.add_argument("--vision-dir", default=str(SCRIPTS / "vision"),
                   help="where #24's colour_detector.py and supervisor_adapter.py live (held-ball check on the wrist frame)")
    p.add_argument("--min-held-target-area", type=int, default=Settings.min_held_target_area_px,
                   help="px of the wanted ball inside --held-check-roi of the wrist frame at rest for the held ball to count "
                        "as it (the striped ball's threshold). Without it the area is only logged. Suggested start: 120 "
                        "(offline wrist frames of molnrb/so101_pick_rb_mixed, kb/held-ball-wrist-areas.md); confirm at the table")
    p.add_argument("--min-held-area", type=colour_area, nargs="*", default=[], metavar="COLOUR=PX",
                   help="per-ball thresholds, e.g. red=2000 striped=120 (striped= overrides --min-held-target-area); "
                        "default red=2000, a candidate: confirm with one red ball held at rest. Blue has none (not usable)")
    p.add_argument("--held-check-roi", type=float, nargs=4, default=WRIST_HELD_ROI, metavar=("CX", "CY", "RX", "RY"),
                   help="ellipse (frame fractions) of the wrist frame where the jaws hold the ball at rest; "
                        "default from PR #30's offline measurement (kb/held-ball-wrist-areas.md)")
    p.add_argument("--held-check-colours", nargs="*", default=list(Settings.held_check_colours),
                   choices=("striped", "red", "blue"),
                   help="balls whose wrist area decides; others are logged only (held_check=untrusted). Add red once "
                        "confirmed at the table (--held-check-colours striped red); blue is not usable on the wrist camera")
    p.add_argument("--held-unseen-floor", type=int, default=Settings.held_unseen_floor_px,
                   help="px: when every ball reads below this in the wrist region (a deep grasp hides the ball above "
                        "the region), the held ball is unseen, not wrong: kept, logged as held_check=unseen")
    p.add_argument("--camera-fourcc", default="MJPG",
                   help="FOURCC of both cameras (as PR #6); 'none' leaves the format to OpenCV")
    p.add_argument("--no-held-check", action="store_true", help="turn the held-ball check off (gripper band only)")
    p.add_argument("--settle-tol-deg", type=float, default=8.0, help="PICK ends early within this of the rest pose")
    p.add_argument("--gripper-open", type=float, default=60.0)
    p.add_argument("--gripper-hold", type=float,
                   help="gripper goal while holding; default: keep the pick policy's last goal")
    g = GraspCheckSettings()
    p.add_argument("--no-grasp-check", action="store_true",
                   help="turn the grasp check off: PICK runs its whole chunk after a close, only VERIFY_HOLD judges")
    p.add_argument("--grasp-trigger", type=float, default=g.trigger,
                   help="grasp check: a commanded gripper falling below this pauses the pick (teleop closes: 0.3-9.2)")
    p.add_argument("--grasp-armed-above", type=float, default=g.armed_above,
                   help="grasp check: armed once the policy opened above this in the pick (78 of 80 rb_mixed episodes)")
    p.add_argument("--grasp-close-goal", type=float, default=g.close_goal,
                   help="grasp check: gripper goal while checking (teleop holds: leader median 1.1)")
    p.add_argument("--grasp-settle", type=float, default=g.settle_s, help="grasp check: seconds still before reading")
    p.add_argument("--grasp-min-error", type=float, default=g.min_error,
                   help="grasp check: reading minus goal for a ball stalling the jaw (held closes: p1 5.5, median 12.4)")
    p.add_argument("--grasp-min-load", type=float, default=g.min_load,
                   help="grasp check: |Present_Load| of the gripper servo for a hold; default: logged only (not measured)")
    p.add_argument("--grasp-backoff", type=float, default=g.backoff_s,
                   help="grasp check: on a miss, back the arm off along the commanded path of this many seconds")
    p.add_argument("--grasp-max-tip-height", type=float, default=g.max_tip_height,
                   help="grasp check: a close with the FK tip above this (m) is a miss at once; off by default "
                        "(model tip z at holding closes 0.02-0.16 m, empty closes overlap)")
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
