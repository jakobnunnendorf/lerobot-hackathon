# Stands in for scripts/collect/record_session.py so the dashboard's Collect mode can be tried without the arm: the
# same Session (status fields, session log, frame blocks), control port and lerobot-record's order of phases, where
# an episode is saved only when the pause after it ends. Never run it next to the real robot program (same ports).
import argparse
import json
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS / "collect"), str(SCRIPTS / "supervisor"), str(SCRIPTS / "handtrack"),
                str(SCRIPTS / "dashboard")]

from fake_robot import pattern  # noqa: E402
from record_session import CONTROL_PORT, Session, frame_writers  # noqa: E402
from status import STATUS  # noqa: E402

SAVE_S = 0.6
UPLOAD_S = 2.0
FAIL_MARKER = "fail_next_upload"  # a file of this name in --log-dir makes the next fake upload fail once


def fake_upload(session: Session, log_dir: Path, total: int) -> bool:
    session.uploading(400)
    time.sleep(UPLOAD_S)
    marker = log_dir / FAIL_MARKER
    if marker.exists():
        marker.unlink()
        session.upload_failed("HfHubHTTPError: fake network failure")
        return False
    session.uploaded(UPLOAD_S, total)
    return True


class FakeRecording:
    def __init__(self, session: Session, webcam: int | None):
        self.session, self.webcam = session, webcam
        self.events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        self.stopped = threading.Event()

    def control(self, name: str) -> None:
        self.session.control(name)
        self.events["exit_early"] = True
        if name == "discard":
            self.events["rerecord_episode"] = True
        if name == "stop":
            self.events["stop_recording"] = True

    def listen(self, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        while True:
            try:
                name = json.loads(sock.recv(1024)).get("control")
            except ValueError:
                continue
            if name in ("next", "discard", "stop"):
                self.control(name)

    def observe(self) -> None:
        capture = cv2.VideoCapture(self.webcam) if self.webcam is not None else None
        while not self.stopped.is_set():
            t = time.monotonic()
            ok, frame = capture.read() if capture is not None else (False, None)
            top = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB) if ok else pattern("top", t)
            wrist = np.ascontiguousarray(top[::-1]) if ok else pattern("wrist", t + 1.5)
            self.session.observed({"top": top, "wrist": wrist})
            time.sleep(1 / 30)

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.session.said(text)

    def loop(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.events["exit_early"]:
            time.sleep(0.02)
        self.events["exit_early"] = False

    def run(self, target: int, episode_s: float, total: int) -> int:
        time.sleep(1.0)
        self.say("Press space to start")
        self.loop(3600)
        self.events["rerecord_episode"] = False
        recorded = 0
        while recorded < target and not self.events["stop_recording"]:
            self.say(f"Recording episode {total}")
            self.loop(episode_s)
            if not self.events["stop_recording"] and (recorded < target - 1 or self.events["rerecord_episode"]):
                self.say("Reset the environment")
                self.loop(3600)
            if self.events["rerecord_episode"]:
                self.say("Re-record episode")
                self.events["rerecord_episode"] = self.events["exit_early"] = False
                continue
            self.session.saving()
            time.sleep(SAVE_S)
            total += 1
            recorded += 1
            self.session.saved_episode(total)
        self.say("Stop recording")
        time.sleep(SAVE_S)
        self.say("Exiting")
        return total


def main() -> None:
    p = argparse.ArgumentParser(description="Fake data collection session for trying the dashboard without the arm.")
    p.add_argument("--name", required=True)
    p.add_argument("--task", default="")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--episode-s", type=int, default=120)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--upload-only", action="store_true")
    p.add_argument("--log-dir", default="~/so101/collect")
    p.add_argument("--control-port", type=int, default=CONTROL_PORT)
    p.add_argument("--status-port", type=int, default=8767)
    p.add_argument("--top-shm", default="top_camera_frames")
    p.add_argument("--wrist-shm", default="wrist_camera_frames")
    p.add_argument("--webcam", type=int, help="camera index for real pictures instead of the test pattern")
    p.add_argument("--dataset-before", type=int, default=40, help="episodes the fake dataset already has")
    args = p.parse_args()
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    started = f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.upload_only:
        session = Session(log_dir / f"upload_{started}.jsonl", f"fake/{args.name}", "", 0, args.dataset_before,
                          upload=True, start_event="upload_retry")
        STATUS.update(fake=True)
        STATUS.start(port=args.status_port)
        ok = fake_upload(session, log_dir, args.dataset_before)
        session.close()
        time.sleep(0.5)
        sys.exit(0 if ok else 1)
    session = Session(log_dir / f"collect_{started}.jsonl", f"fake/{args.name}", args.task, args.episodes,
                      args.dataset_before, upload=args.upload)
    STATUS.update(episode_limit_s=args.episode_s, fake=True)
    STATUS.start(port=args.status_port)
    session.writers = frame_writers({"top": args.top_shm, "wrist": args.wrist_shm})
    recording = FakeRecording(session, args.webcam)
    threading.Thread(target=recording.listen, args=(args.control_port,), daemon=True).start()
    observing = threading.Thread(target=recording.observe, daemon=True)
    observing.start()
    total = args.dataset_before
    try:
        total = recording.run(args.episodes, args.episode_s, args.dataset_before)
    finally:
        recording.stopped.set()
        observing.join(timeout=2)
        session.end()
        for writer in session.writers.values():
            writer.close()
        time.sleep(0.5)
    if args.upload and session.saved:
        fake_upload(session, log_dir, total)
        time.sleep(0.5)
    session.close()


if __name__ == "__main__":
    main()
