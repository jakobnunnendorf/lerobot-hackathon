# scripts/collect/record_session.py without the arm: the hooks exist in LeRobot 0.7.0's lerobot_record, the arguments
# match scripts/so101/record.sh, and a stand-in record() driven over the control port gives the right phases, counts,
# events and camera frames. Needs the lerobot environment.
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# tests/cuda.py would shadow the cuda package torch imports while loading lerobot_record.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path[:0] = [str(ROOT / "scripts" / "collect"), str(ROOT / "scripts" / "handtrack"), str(ROOT / "scripts" / "supervisor")]

import record_session  # noqa: E402
from frame_link import FrameReader  # noqa: E402
from status import STATUS  # noqa: E402

failures = []
ENV = {"SO101_HF_USER": "molnrb", "SO101_FOLLOWER_PORT": "/dev/so101_follower", "SO101_FOLLOWER_ID": "mi_follower",
       "SO101_LEADER_PORT": "/dev/so101_leader", "SO101_LEADER_ID": "mi_leader", "SO101_CAM_TOP": "/dev/top",
       "SO101_CAM_WRIST": "/dev/wrist"}


def check(name: str, ok: bool, detail="") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def hooks_exist() -> None:
    import lerobot.scripts.lerobot_record as record_module
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = ("init_keyboard_listener", "log_say", "make_robot_from_config", "main")
    missing = [n for n in names if not hasattr(record_module, n)] + [
        n for n in ("save_episode", "num_episodes") if not hasattr(LeRobotDataset, n)]
    check("LeRobot 0.7.0 still has every name record_session.py hooks", not missing, str(missing))


def arguments(home: Path) -> None:
    os.environ["HF_LEROBOT_HOME"] = str(home)
    fresh = record_session.record_args("so101_test", "Pick up the striped ball", 5, 120, ENV)
    (home / "molnrb" / "so101_test" / "meta").mkdir(parents=True)
    (home / "molnrb" / "so101_test" / "meta" / "info.json").write_text(json.dumps({"total_episodes": 7}))
    again = record_session.record_args("so101_test", "Pick up the striped ball", 5, 120, ENV)
    cameras = ("--robot.cameras={ top: {type: opencv, index_or_path: /dev/top, width: 640, height: 480, fps: 30, "
               "fourcc: MJPG, backend: 200}, wrist: {type: opencv, index_or_path: /dev/wrist, width: 640, height: 480, "
               "fps: 30, fourcc: MJPG, backend: 200} }")
    expected = {"--robot.type=so101_follower", "--robot.id=mi_follower", "--teleop.type=so101_leader",
                "--teleop.id=mi_leader", cameras, "--dataset.repo_id=molnrb/so101_test", "--dataset.no_stamp=true",
                "--dataset.single_task=Pick up the striped ball", "--dataset.num_episodes=5",
                "--dataset.episode_time_s=120", "--dataset.reset_time_s=3600", "--dataset.fps=30",
                "--dataset.streaming_encoding=true", "--dataset.encoder_threads=2", "--dataset.push_to_hub=false",
                "--robot.num_read_retries=10", "--teleop.num_read_retries=10"}
    check("arguments as record.sh: same cameras, ids and dataset settings; --resume once the dataset exists",
          expected <= set(fresh) and "--resume=false" in fresh and "--resume=true" in again
          and record_session.episodes_in(home / "molnrb" / "so101_test") == 7, str(sorted(expected - set(fresh))))


class FakeDataset:
    def __init__(self):
        self.num_episodes = 7

    def save_episode(self):
        self.num_episodes += 1


class FakeBus:
    def __init__(self):
        self.registers = {}

    def write(self, name, motor, value, normalize=True):
        self.registers[(name, motor)] = value

    def read(self, name, motor, normalize=True):
        return self.registers[(name, motor)]


class FakeRobot:
    def __init__(self):
        self.count = 0
        self.bus = FakeBus()

    def connect(self):
        pass

    def get_observation(self):
        self.count += 1
        return {"shoulder_pan.pos": 0.0, "top": np.full((480, 640, 3), self.count, np.uint8),
                "wrist": np.full((480, 640, 3), 100 + self.count, np.uint8)}


def send(port: int, control: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps({"control": control}).encode(), ("127.0.0.1", port))
    sock.close()


def wait_for(events: dict, key: str, seconds: float = 3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not events[key]:
        time.sleep(0.02)
    return events[key]


def stand_in_session(work: Path) -> None:
    port = 18768
    spoken, phases = [], []

    class Module:
        @staticmethod
        def init_keyboard_listener():
            return None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

        @staticmethod
        def log_say(text, play_sounds=True, blocking=False):
            spoken.append(text)

        @staticmethod
        def make_robot_from_config(config):
            return FakeRobot()

    session = record_session.Session(work / "collect_test.jsonl", "molnrb/so101_test", "Pick", 2, 7)
    session.writers = record_session.frame_writers({"top": "collect_test_top", "wrist": "collect_test_wrist"})
    record_session.install_hooks(Module, FakeDataset, session, port)

    def phase():
        phases.append(STATUS.snapshot()["phase"])

    listener, events = Module.init_keyboard_listener()
    robot, dataset = Module.make_robot_from_config(None), FakeDataset()
    robot.connect()
    check("connect caps the gripper: Torque_Limit 200 written after LeRobot's connect",
          robot.bus.registers == {("Torque_Limit", "gripper"): 200}, str(robot.bus.registers))
    time.sleep(0.2)
    Module.log_say("Press space to start")
    phase()
    send(port, "next")
    started = wait_for(events, "exit_early")
    events["exit_early"] = False
    Module.log_say(f"Recording episode {dataset.num_episodes}")
    phase()
    for _ in range(4):
        robot.get_observation()
    send(port, "next")
    finished = wait_for(events, "exit_early")
    events["exit_early"] = False
    dataset.save_episode()
    saved = STATUS.snapshot()
    Module.log_say("Reset the environment")
    phase()
    send(port, "discard")
    discarded = wait_for(events, "rerecord_episode")
    Module.log_say("Re-record episode")
    phase()
    send(port, "stop")
    stopped = wait_for(events, "stop_recording")
    Module.log_say("Stop recording")
    Module.log_say("Exiting")
    phase()
    top, _ = FrameReader("collect_test_top").latest()
    wrist, _ = FrameReader("collect_test_wrist").latest()
    session.end()
    session.close()
    for writer in session.writers.values():
        writer.close()
    log = [json.loads(line)["event"] for line in (work / "collect_test.jsonl").read_text().splitlines()]
    check("page controls reach lerobot-record's events: next ends the wait and the episode, discard re-records, "
          "stop ends", started and finished and discarded and stopped)
    check("phases follow lerobot-record's spoken lines, and the original log_say still runs (banners, voice)",
          phases == ["waiting", "recording", "reset", "discarded", "done"] and spoken[0] == "Press space to start",
          str(phases))
    check("a saved episode counts once, with the dataset's new total",
          saved["saved"] == 1 and saved["dataset_total"] == 8 and saved["phase"] == "saving", str(saved))
    check("both cameras reach the dashboard's blocks every second observation", int(top[0, 0, 0]) == 4
          and int(wrist[0, 0, 0]) == 104, f"top {int(top[0, 0, 0])}, wrist {int(wrist[0, 0, 0])}")
    check("session log: start, episode, controls, saved, discarded, end",
          log[0] == "session_start" and "episode_start" in log and "episode_saved" in log
          and "episode_discarded" in log and log[-1] == "session_end" and log.count("control") == 4, str(log))


def uploads(work: Path) -> None:
    import lerobot.datasets.lerobot_dataset as dataset_module

    pushes = []

    class HubDataset:
        fail = False

        def __init__(self, repo_id, root=None):
            self.repo_id, self.root = repo_id, root

        def push_to_hub(self, **kwargs):
            if HubDataset.fail:
                raise OSError("network unreachable")
            pushes.append((self.repo_id, self.root, kwargs))

    root = work / "home" / "molnrb" / "so101_test"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "file-000.mp4").write_bytes(b"\0" * 3_000_000)
    original, dataset_module.LeRobotDataset = dataset_module.LeRobotDataset, HubDataset
    try:
        session = record_session.Session(work / "upload_test.jsonl", "molnrb/so101_test", "Pick", 2, 7, upload=True)
        ok = record_session.upload(session, "molnrb/so101_test", root)
        after_ok = STATUS.snapshot()
        HubDataset.fail = True
        failed = record_session.upload(session, "molnrb/so101_test", root)
        after_fail = STATUS.snapshot()
        session.close()
    finally:
        dataset_module.LeRobotDataset = original
    log = [json.loads(line)["event"] for line in (work / "upload_test.jsonl").read_text().splitlines()]
    check("upload: push_to_hub of the local dataset with PR #6's tags; phase uploaded with the episode count",
          ok and pushes == [("molnrb/so101_test", root, {"tags": record_session.HUB_TAGS})]
          and after_ok["phase"] == "uploaded" and after_ok["dataset_total"] == 7 and after_ok["upload_mb"] == 3,
          str((pushes, after_ok.get("phase"), after_ok.get("upload_mb"))))
    check("upload: a failure is reported to the page and logged, nothing raised",
          not failed and after_fail["phase"] == "upload_failed" and "network unreachable" in after_fail["upload_error"]
          and log == ["session_start", "upload_start", "upload_done", "upload_start", "upload_failed"], str(log))


def main() -> None:
    hooks_exist()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        arguments(Path(work) / "home")
        stand_in_session(Path(work))
        uploads(Path(work))
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
