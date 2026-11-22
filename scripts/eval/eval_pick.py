# Written by Claude
# Repeated PICK evaluation on the SO-101: SPACE starts a run from HOME, the policy runs until the arm is back
# home holding a ball or the time cap hits, the result is written to a CSV. Several policies are compared on the
# same layouts, newest checkpoint of each by default. Runbook: docs/eval-pick-runbook.md.
# Own loop instead of lerobot-rollout: rollout disconnects after every run, which switches torque off and drops the ball.

import argparse
import csv
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors, make_robot_action
from lerobot.policies.factory import get_policy_class
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "so101"))
from gripper_limit import cap_gripper_torque  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
# Median first frame of the 80 episodes of molnrb/so101_pick_rb_mixed: the folded rest pose they all start and end in.
HOME = {
    "shoulder_pan.pos": -5.6,
    "shoulder_lift.pos": -102.8,
    "elbow_flex.pos": 96.6,
    "wrist_flex.pos": -100.1,
    "wrist_roll.pos": -5.2,
    "gripper.pos": 17.8,
}
# Only these three joints are tight at the end of every recorded episode (p5-p95 within 1 degree); pan and roll vary.
HOME_CHECK_JOINTS = ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")
HOME_TOLERANCE = 8.0
LEFT_HOME = 25.0
# Gripper readings while holding a ball lie between about 6 and 20 (board 7061759444); near 1 is an empty grasp.
HOLDING = (6.0, 20.0)
SETTLED_S = 1.0
GRIPPER_OPEN = 30.0
ESC = "\x1b"
# Task sentence of molnrb/so101_pick_rb_mixed (meta/tasks.parquet); only language-conditioned policies read it.
TASK = (
    "Pick up the striped ball from the bowl of red and blue balls, uncovering it first if it is buried, "
    "and return to the start position"
)
REPO_ROOT = Path(__file__).resolve().parents[2]

START_STATES = {"v": "visible", "p": "partly", "h": "hidden"}
RESULTS = {
    "s": "success",
    "w": "wrong_ball",
    "e": "empty",
    "d": "dropped",
    "t": "still_digging",
    "n": "never_started",
    "o": "other",
}


def say(text: str) -> None:
    if shutil.which("espeak-ng"):
        subprocess.Popen(["espeak-ng", "-s", "160", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def banner(text: str, colour: str = "30;43") -> None:
    print(f"\n\033[{colour}m\033[1m{('   ' + text).ljust(78)}\033[0m", flush=True)


@contextmanager
def keyboard():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # cbreak, not raw: Ctrl+C still raises KeyboardInterrupt
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key() -> str:
    # os.read, not sys.stdin.read: the buffered reader can swallow keys that select() then never reports.
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()


def key_pressed() -> str | None:
    return read_key() if select.select([sys.stdin], [], [], 0)[0] else None


def wait_key(allowed: str) -> str:
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    while not (k := read_key()) or k not in allowed:
        pass
    return k


def checkpoint_folder(paths, step: str = "") -> str | None:
    """Folder of the newest (or the given) step: `checkpoints/<step>` (LeRobot runs, Tomi's repo) or `<step>` (Kamel's repo)."""
    found = {}
    for p in paths:
        parts = p.split("/")
        if len(parts) > 2 and parts[0] == "checkpoints" and parts[1].isdigit():
            found[parts[1]] = f"checkpoints/{parts[1]}"
        elif len(parts) > 1 and parts[0].isdigit():
            found.setdefault(parts[0], parts[0])
    if step and step not in found:
        raise SystemExit(f"step {step} not found; available: {sorted(found)}")
    return found[step or max(found)] if found else None


def model_dir(path: Path) -> str:
    # LeRobot run dirs keep the weights in <step>/pretrained_model; the Hub repos keep them in <step>/ directly.
    return str(path / "pretrained_model") if (path / "pretrained_model").is_dir() else str(path)


def resolve_checkpoint(spec: str) -> tuple[str, str]:
    """`spec` is a Hub repo id or a local dir, optionally `@<step>`; without a step the newest checkpoint is used."""
    source, _, step = spec.partition("@")
    if Path(source).is_dir():
        root = Path(source)
        dirs = [*root.glob("checkpoints/*"), *root.glob("*")]
        folder = checkpoint_folder([d.relative_to(root).as_posix() + "/" for d in dirs if d.is_dir()], step)
        path = root / folder if folder else root
    else:
        folder = checkpoint_folder(HfApi().list_repo_files(source), step)
        local = Path(snapshot_download(source, allow_patterns=[f"{folder}/*"] if folder else None))
        path = local / folder if folder else local
    return model_dir(path), f"{source}@{folder.split('/')[-1]}" if folder else source


class Policy:
    def __init__(self, path: str, n_action_steps: int | None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        overrides = [f"--device={device}"]
        if n_action_steps:
            overrides.append(f"--n_action_steps={n_action_steps}")
        config = PreTrainedConfig.from_pretrained(path, cli_overrides=overrides)
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config)
        self.device = torch.device(device)
        self.type = config.type
        self.n_action_steps = config.n_action_steps
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.check_camera_names(config)

    def check_camera_names(self, config) -> None:
        # The robot's cameras are top/wrist; SmolVLA checkpoints trained with --rename_map expect camera1/camera2 and
        # carry that map in their saved preprocessor. A checkpoint without it would silently get no images.
        expected = {k.rsplit(".", 1)[-1] for k in config.image_features}
        if expected <= {"top", "wrist"}:
            return
        renamed = {v.rsplit(".", 1)[-1] for step in self.pre.steps for v in getattr(step, "rename_map", {}).values()}
        if not expected <= renamed | {"camera3"}:
            raise SystemExit(f"checkpoint expects cameras {sorted(expected)} but its preprocessor renames to {sorted(renamed)}")

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def act(self, frame: dict, features: dict, robot_type: str, task: str = "") -> dict:
        action = predict_action(
            observation=frame,
            policy=self.policy,
            device=self.device,
            preprocessor=self.pre,
            postprocessor=self.post,
            use_amp=self.policy.config.use_amp,
            task=task if self.type != "act" else "",
            robot_type=robot_type,
        )
        return make_robot_action(action, features)


class EpisodeBuffer:
    # Frames are JPEG-encoded on a worker thread (cv2 releases the GIL) instead of kept raw: 90 s of two 640x480
    # cameras is about 5 GB raw, 1 GB as JPEG. Nothing touches the dataset until the run is over.
    def __init__(self):
        self.jobs: queue.Queue = queue.Queue()
        self.frames: list = []
        self.count = 0
        self.worker = threading.Thread(target=self.encode_loop, daemon=True)
        self.worker.start()

    def encode_loop(self) -> None:
        while (job := self.jobs.get()) is not None:
            obs_frame, action_frame = job
            packed = {
                k: cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 95])[1] if v.ndim == 3 else v
                for k, v in obs_frame.items()
            }
            self.frames.append({**packed, **action_frame})

    def add(self, obs_frame: dict, action_frame: dict) -> None:
        self.count += 1
        self.jobs.put((obs_frame, action_frame))

    def drain(self):
        # A generator: decoding all frames up front would put the raw 5 GB back in memory.
        self.jobs.put(None)
        self.worker.join()
        while self.frames:
            f = self.frames.pop(0)
            yield {k: cv2.imdecode(v, cv2.IMREAD_COLOR) if v.ndim == 1 and v.dtype == np.uint8 else v for k, v in f.items()}


def open_dataset(root: Path, features: dict, robot_type: str, repo_id: str) -> LeRobotDataset:
    if (root / "meta" / "info.json").exists():
        return LeRobotDataset(repo_id, root=root, streaming_encoding=True)
    # streaming_encoding: videos are encoded while add_frame runs and no PNG staging is written, so save_episode takes
    # about 11 s for a 90 s run instead of 56 s, and each finished episode is already a complete mp4 on disk.
    return LeRobotDataset.create(
        repo_id, FPS, features=features, root=root, robot_type=robot_type, use_videos=True, streaming_encoding=True
    )


def save_episode(dataset: LeRobotDataset, buffer: EpisodeBuffer, task: str) -> int:
    for frame in buffer.drain():
        dataset.add_frame({**frame, "task": task})
    dataset.save_episode()
    return dataset.meta.total_episodes - 1


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def camera(path: str) -> OpenCVCameraConfig:
    # Same settings as every recording (scripts/so101/common.sh on molnrb/so101-robot-skill); backend 200 is V4L2.
    return OpenCVCameraConfig(index_or_path=Path(path), fps=FPS, width=640, height=480, fourcc="MJPG", backend=200)


def make_robot() -> SO101Follower:
    env = os.environ.get
    return SO101Follower(
        SO101FollowerConfig(
            port=env("SO101_FOLLOWER_PORT", "/dev/so101_follower"),
            id=env("SO101_FOLLOWER_ID", "mi_follower"),
            # Same cap as run_policy.sh: a bad prediction cannot slam the arm.
            max_relative_target=15.0,
            cameras={
                "top": camera(env("SO101_CAM_TOP", "/dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0")),
                "wrist": camera(
                    env(
                        "SO101_CAM_WRIST",
                        "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
                    )
                ),
            },
        )
    )


def sync_goal_to_present(bus) -> None:
    # connect() switches torque on without writing a goal, so each servo drives to its last stored goal: after `q`
    # (torque off, arm limp) or teleop that goal is stale and the arm jumps. Same fix as scripts/handover/handover_loop.py.
    bus.connect()
    try:
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position", normalize=False), normalize=False)
    finally:
        bus.disconnect(disable_torque=False)


def joints(robot: SO101Follower) -> dict:
    obs = robot.get_observation()
    return {k: obs[k] for k in HOME}


def move_to(robot: SO101Follower, target: dict, speed_deg_s: float = 30.0) -> None:
    start = joints(robot)
    goal = {**start, **target}
    steps = max(int(max(abs(goal[k] - start[k]) for k in goal) / speed_deg_s * FPS), FPS)
    for i in range(1, steps + 1):
        t = i / steps
        robot.send_action({k: start[k] + (goal[k] - start[k]) * t for k in goal})
        precise_sleep(1 / FPS)


def at_home_holding(obs: dict) -> bool:
    near = all(abs(obs[k] - HOME[k]) < HOME_TOLERANCE for k in HOME_CHECK_JOINTS)
    return near and HOLDING[0] <= obs["gripper.pos"] <= HOLDING[1]


def left_home(obs: dict) -> bool:
    return any(abs(obs[k] - HOME[k]) > LEFT_HOME for k in HOME_CHECK_JOINTS)


def run_once(
    robot: SO101Follower, policy: Policy, features: dict, cap_s: float, task: str = "", buffer: EpisodeBuffer | None = None
) -> dict:
    policy.reset()
    start = time.perf_counter()
    end, has_left, settled_since, late_ticks, ticks = "timeout", False, None, 0, 0
    while (elapsed := time.perf_counter() - start) < cap_s:
        tick = time.perf_counter()
        k = key_pressed()
        if k == " ":
            end = "operator"
            break
        if k in ("q", ESC):
            end = "aborted"
            break
        obs = robot.get_observation()
        has_left = has_left or left_home(obs)
        if has_left and at_home_holding(obs):
            settled_since = settled_since or tick
            if tick - settled_since >= SETTLED_S:
                end = "returned"
                break
        else:
            settled_since = None
        frame = build_dataset_frame(features, obs, prefix=OBS_STR)
        action = policy.act(frame, features, robot.robot_type, task)
        robot.send_action(action)
        if buffer:
            buffer.add(frame, build_dataset_frame(features, action, prefix=ACTION))
        ticks += 1
        spent = time.perf_counter() - tick
        late_ticks += spent > 1 / FPS
        precise_sleep(max(1 / FPS - spent, 0.0))
    # Nothing is sent from here on: the servos hold their last goal, so a held ball stays in the gripper.
    return {
        "seconds": round(time.perf_counter() - start, 1),
        "end": end,
        "gripper_end": round(joints(robot)["gripper.pos"], 1),
        "left_home": has_left,
        "late_ticks_pct": round(100 * late_ticks / max(ticks, 1), 1),
    }


def ask(prompt: str, options: dict) -> str:
    print(f"\033[1m  {prompt}  " + "  ".join(f"[{k}] {v}" for k, v in options.items()) + "\033[0m", flush=True)
    return options[wait_key("".join(options))]


def tally(rows: list) -> str:
    parts = []
    for state in START_STATES.values():
        done = [r for r in rows if r["start_state"] == state]
        if done:
            parts.append(f"{state} {sum(r['result'] == 'success' for r in done)}/{len(done)}")
    parts.append(f"all {sum(r['result'] == 'success' for r in rows)}/{len(rows)}")
    failures = Counter(r["result"] for r in rows if r["result"] != "success")
    if failures:
        parts.append("failures: " + ", ".join(f"{k} {v}" for k, v in failures.most_common()))
    return " | ".join(parts)


def summary(rows: list, names: list) -> str:
    return "\n".join(f"  {n}: {tally([r for r in rows if r['policy'] == n])}" for n in names if any(
        r["policy"] == n for r in rows))


def schedule(names: list, first_layout: int, layouts: int) -> list:
    # Every policy runs each layout once; the order flips every layout so no policy always goes first.
    return [(k, n) for k in range(first_layout, first_layout + layouts) for n in (names if k % 2 else names[::-1])]


def parse_policies(specs: list) -> dict:
    policies = {}
    for spec in specs:
        name, _, source = spec.rpartition("=")
        name = name or source.partition("@")[0].rstrip("/").split("/")[-1]
        policies[name] = source
    return policies


def main() -> None:
    p = argparse.ArgumentParser(description="Repeated PICK evaluation on the SO-101 (SPACE starts each run).")
    p.add_argument(
        "--policy",
        action="append",
        help="[name=]<Hub repo id or local run dir>[@step]; repeat to compare policies on the same layouts. "
        "Without @step the newest checkpoint is used. Default: Space0code/act_redblue_from_foil",
    )
    p.add_argument("--runs", type=int, default=10, help="layouts; every policy runs each layout once")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--n-action-steps", type=int, help="override the checkpoints' n_action_steps (default 100)")
    p.add_argument("--task", default=TASK, help="task sentence for language-conditioned policies (SmolVLA); ACT ignores it")
    p.add_argument("--record", type=Path, help="write every run as one episode of a LeRobot dataset in this directory")
    p.add_argument("--push-to-hub", metavar="REPO_ID", help="with --record: upload the dataset at the very end")
    p.add_argument("--log-dir", default=str(Path.home() / "so101" / "evals"))
    args = p.parse_args()
    if args.push_to_hub and not args.record:
        p.error("--push-to-hub needs --record")

    policies, labels = {}, {}
    for name, spec in parse_policies(args.policy or ["Space0code/act_redblue_from_foil"]).items():
        path, labels[name] = resolve_checkpoint(spec)
        print(f"Loading {name}: {labels[name]} from {path}", flush=True)
        policies[name] = Policy(path, args.n_action_steps)
    names = list(policies)
    robot = make_robot()
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
    }
    log = Path(args.log_dir) / f"eval_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "layout", "episode_index", "policy", "checkpoint", "started", "n_action_steps", "cap_s", "seconds",
              "end", "gripper_end", "left_home", "late_ticks_pct", "start_state", "result"]
    rows: list = []
    dataset = open_dataset(args.record, features, robot.robot_type, args.push_to_hub or "local/eval_pick") if args.record else None
    commit = git_commit()

    sync_goal_to_present(robot.bus)
    robot.connect()
    cap_gripper_torque(robot.bus)
    try:
        # The first inference initialises CUDA and takes seconds; do it before the first timed run.
        frame = build_dataset_frame(features, robot.get_observation(), prefix=OBS_STR)
        for policy in policies.values():
            policy.act(frame, features, robot.robot_type, args.task)
        with keyboard(), open(log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            queue, layouts, start_states = schedule(names, 1, args.runs), args.runs, {}
            run = 1
            while True:
                if not queue:
                    banner(f"{layouts} LAYOUTS DONE", "30;42")
                    print(summary(rows, names), flush=True)
                    banner("SPACE = 5 more layouts,  q = quit (arm goes HOME, then limp)")
                    if wait_key(" q") == "q":
                        break
                    queue, layouts = schedule(names, layouts + 1, 5), layouts + 5
                layout, name = queue.pop(0)
                policy = policies[name]
                setup = "the SAME layout as the last run" if layout in start_states else "a NEW layout"
                banner(f"RUN {run}  LAYOUT {layout}/{layouts}  POLICY {name}: set up {setup}. SPACE = start,  q = quit")
                say(f"Run {run}. Press space to start")
                if wait_key(" q") == "q":
                    break
                move_to(robot, HOME)
                banner(f"RUN {run} ({name}) RUNNING  -  max {args.seconds:.0f} s.  SPACE = end now,  ESC = abort (arm holds)", "97;41")
                say("Robot starting")
                started = datetime.now().isoformat(timespec="seconds")
                buffer = EpisodeBuffer() if dataset else None
                outcome = run_once(robot, policy, features, args.seconds, args.task, buffer)
                say("Run finished")
                print(
                    f"\n  ended: {outcome['end']} after {outcome['seconds']} s, gripper {outcome['gripper_end']}"
                    f" (holding is about {HOLDING[0]:.0f}-{HOLDING[1]:.0f}), slow ticks {outcome['late_ticks_pct']}%",
                    flush=True,
                )
                if layout not in start_states:
                    start_states[layout] = ask("Ball at start?", START_STATES)
                result = ask("Result?", RESULTS)
                episode_index = None
                if buffer and buffer.count:
                    print("  saving the episode (the arm holds its pose)...", flush=True)
                    episode_index = save_episode(dataset, buffer, args.task)
                row = {
                    "run": run,
                    "layout": layout,
                    "episode_index": episode_index,
                    "policy": name,
                    "checkpoint": labels[name],
                    "started": started,
                    "n_action_steps": policy.n_action_steps,
                    "cap_s": args.seconds,
                    **outcome,
                    "start_state": start_states[layout],
                    "result": result,
                }
                writer.writerow(row)
                f.flush()
                if episode_index is not None:
                    with open(args.record / "runs.jsonl", "a") as sidecar:
                        sidecar.write(json.dumps({**row, "policy_type": policy.type, "git_commit": commit}) + "\n")
                rows.append(row)
                print(summary(rows, names), flush=True)
                banner("Hold a hand under the gripper. SPACE = open the gripper")
                wait_key(" ")
                move_to(robot, {"gripper.pos": GRIPPER_OPEN})
                run += 1
            move_to(robot, HOME)
    finally:
        robot.disconnect()  # torque off: the arm goes limp, so it is sent HOME first on a normal quit
        print(f"\nLog: {log}\n{summary(rows, names) if rows else 'no runs'}", flush=True)
        if dataset:
            dataset.finalize()
            print(f"Dataset: {args.record}", flush=True)
            if args.push_to_hub:
                dataset.push_to_hub()


if __name__ == "__main__":
    main()
