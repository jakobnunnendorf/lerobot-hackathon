# Records a LeRobot dataset with the same lerobot-record run as PR #6's scripts/so101/record.sh, so the dashboard can
# drive and watch it: its buttons act as the recording keys, and the phase, counts and both cameras reach the page.
# Why lerobot-record in this process with hooks, not a loop of our own: decisions/dashboard.md.
import argparse
import json
import os
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from multiprocessing import shared_memory
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS / "supervisor"), str(SCRIPTS / "handtrack")]

from frame_link import FrameWriter  # noqa: E402
from status import STATUS  # noqa: E402

CONTROL_PORT = 8768
CONTROLS = {"next": "right", "discard": "left", "stop": "esc"}
REQUIRED_ENV = ("SO101_HF_USER", "SO101_FOLLOWER_PORT", "SO101_FOLLOWER_ID", "SO101_LEADER_PORT", "SO101_LEADER_ID",
                "SO101_CAM_TOP", "SO101_CAM_WRIST")
# lerobot-record speaks these lines (the last four only with molnrb's patch_record.py); they mark the phases.
PHASES = (("Press space to start", "waiting"), ("Recording episode", "recording"), ("Reset the environment", "reset"),
          ("Re-record episode", "discarded"), ("Stop recording", "finishing"), ("Exiting", "done"))


def stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def lerobot_home() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))


def episodes_in(root: Path) -> int:
    info = root / "meta" / "info.json"
    return int(json.loads(info.read_text())["total_episodes"]) if info.exists() else 0


# Same settings as scripts/so101/record.sh: every dataset so far was recorded with them, and the policies expect these
# camera names and sizes. reset_time_s 3600: the pause between episodes ends on "next", not on a timer. No rerun
# viewer (record.sh's --display_data): the dashboard shows the cameras.
def record_args(name: str, task: str, episodes: int, episode_s: int, env) -> list[str]:
    repo = f"{env['SO101_HF_USER']}/{name}"
    root = lerobot_home() / repo
    camera = "{type: opencv, index_or_path: %s, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200}"
    cameras = f"{{ top: {camera % env['SO101_CAM_TOP']}, wrist: {camera % env['SO101_CAM_WRIST']} }}"
    return ["--robot.type=so101_follower", f"--robot.port={env['SO101_FOLLOWER_PORT']}",
            f"--robot.id={env['SO101_FOLLOWER_ID']}", "--teleop.type=so101_leader",
            f"--teleop.port={env['SO101_LEADER_PORT']}", f"--teleop.id={env['SO101_LEADER_ID']}",
            f"--robot.cameras={cameras}", "--display_data=false", f"--dataset.repo_id={repo}",
            f"--dataset.root={root}", "--dataset.no_stamp=true", f"--dataset.single_task={task}",
            f"--dataset.num_episodes={episodes}", f"--dataset.episode_time_s={episode_s}",
            "--dataset.reset_time_s=3600", "--dataset.fps=30", "--dataset.streaming_encoding=true",
            "--dataset.encoder_threads=2", "--dataset.push_to_hub=false",
            f"--resume={'true' if (root / 'meta' / 'info.json').exists() else 'false'}"]


class Session:
    def __init__(self, log_path: Path, repo: str, task: str, target: int, before: int):
        self.log = open(log_path, "a", encoding="utf-8")
        self.lock = threading.Lock()
        self.saved, self.discarded, self.episode, self.observes = 0, 0, None, 0
        self.writers: dict[str, FrameWriter] = {}
        STATUS.update(mode="collect", dataset=repo, task=task, target=target, dataset_total=before, saved=0,
                      discarded=0, episode=None, log_path=str(log_path.resolve()))
        self.phase("starting")
        self.write(event="session_start", dataset=repo, task=task, target=target, dataset_before=before)

    def write(self, **event) -> None:
        with self.lock:
            self.log.write(json.dumps({"time": stamp(), **event}) + "\n")
            self.log.flush()

    def phase(self, name: str, **details) -> None:
        STATUS.update(phase=name, phase_started=time.time(), **details)

    def said(self, text: str) -> None:
        name = next((phase for prefix, phase in PHASES if text.startswith(prefix)), None)
        if name == "recording":
            self.episode = int(text.split()[-1])
            self.write(event="episode_start", episode=self.episode)
        elif name == "discarded":
            self.discarded += 1
            self.write(event="episode_discarded", episode=self.episode)
        if name:
            self.phase(name, episode=self.episode, discarded=self.discarded)

    def saving(self) -> None:
        self.phase("saving")

    def saved_episode(self, total: int) -> None:
        self.saved += 1
        STATUS.update(saved=self.saved, dataset_total=total)
        self.write(event="episode_saved", episode=self.episode, saved=self.saved, dataset_total=total)

    def control(self, name: str) -> None:
        self.write(event="control", control=name, phase=STATUS.snapshot().get("phase"))

    def observed(self, obs: dict) -> None:
        self.observes += 1
        STATUS.update(observes=self.observes)
        if self.observes % 2 == 0:
            for name, writer in self.writers.items():
                if name in obs:
                    writer.write(obs[name], time.perf_counter_ns())

    def close(self) -> None:
        self.phase("done")
        self.write(event="session_end", saved=self.saved, discarded=self.discarded)
        self.log.close()


def listen_controls(session: Session, events: dict, port: int) -> None:
    from lerobot.utils.keyboard_input import apply_recording_control

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        try:
            name = json.loads(sock.recv(1024)).get("control")
        except ValueError:
            continue
        if name in CONTROLS:
            session.control(name)
            apply_recording_control(CONTROLS[name], events)


def install_hooks(record_module, dataset_class, session: Session, control_port: int) -> None:
    open_keys, say, make_robot = (record_module.init_keyboard_listener, record_module.log_say,
                                  record_module.make_robot_from_config)
    save = dataset_class.save_episode

    # The original listener stays: terminal keys still work when there is a TTY.
    def init_keyboard_listener():
        listener, events = open_keys()
        threading.Thread(target=listen_controls, args=(session, events, control_port), daemon=True).start()
        return listener, events

    def log_say(text, *args, **kwargs):
        session.said(text)
        return say(text, *args, **kwargs)

    def make_robot_from_config(config):
        robot = make_robot(config)
        observe = robot.get_observation

        def get_observation(*args, **kwargs):
            obs = observe(*args, **kwargs)
            session.observed(obs)
            return obs

        robot.get_observation = get_observation
        return robot

    def save_episode(self, *args, **kwargs):
        session.saving()
        result = save(self, *args, **kwargs)
        session.saved_episode(self.num_episodes)
        return result

    record_module.init_keyboard_listener = init_keyboard_listener
    record_module.log_say = log_say
    record_module.make_robot_from_config = make_robot_from_config
    dataset_class.save_episode = save_episode


def frame_writers(names: dict[str, str]) -> dict[str, FrameWriter]:
    writers = {}
    for camera, block in names.items():
        try:
            stale = shared_memory.SharedMemory(name=block)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        writers[camera] = FrameWriter(block)
    return writers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lerobot-record as record.sh runs it, driven and watched by the dashboard. "
                                            "Needs the variables of ~/.so101.env exported.")
    p.add_argument("--name", required=True, help="dataset name under $SO101_HF_USER; an existing one gets more episodes")
    p.add_argument("--task", required=True, help="task sentence; keep the dataset's own when adding episodes")
    p.add_argument("--episodes", type=int, required=True, help="episodes to record in this session")
    p.add_argument("--episode-s", type=int, default=120, help="an episode ends by itself after this long")
    p.add_argument("--log-dir", default="~/so101/collect")
    p.add_argument("--control-port", type=int, default=CONTROL_PORT)
    p.add_argument("--status-port", type=int, default=8767)
    p.add_argument("--top-shm", default="top_camera_frames")
    p.add_argument("--wrist-shm", default="wrist_camera_frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        sys.exit(f"missing {', '.join(missing)}: export the variables of ~/.so101.env first (set -a; source ~/.so101.env)")
    os.environ.setdefault("SO101_WAIT_START", "1")
    os.environ.setdefault("SO101_BANNERS", "1")
    repo = f"{os.environ['SO101_HF_USER']}/{args.name}"
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    session = Session(log_dir / f"collect_{args.name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl", repo, args.task,
                      args.episodes, episodes_in(lerobot_home() / repo))
    STATUS.update(episode_limit_s=args.episode_s)
    STATUS.start(port=args.status_port)

    import lerobot.scripts.lerobot_record as record_module
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    install_hooks(record_module, LeRobotDataset, session, args.control_port)
    session.writers = frame_writers({"top": args.top_shm, "wrist": args.wrist_shm})
    sys.argv = ["lerobot-record", *record_args(args.name, args.task, args.episodes, args.episode_s, os.environ)]
    print("lerobot-record " + " ".join(sys.argv[1:]), flush=True)
    try:
        record_module.main()
    finally:
        session.close()
        for writer in session.writers.values():
            writer.close()
        time.sleep(0.5)  # the last status datagram (phase done) goes out before the process ends


if __name__ == "__main__":
    main()
