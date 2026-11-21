# Stands in for run_supervisor.py so the dashboard can be tried without the arm: same command port, status port,
# frame blocks and log format, with scripted rounds. Never run it next to the real robot program (same ports).
import argparse
import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS / "voice"), str(SCRIPTS / "handtrack"), str(SCRIPTS / "supervisor")]

import link  # noqa: E402
from frame_link import FrameWriter  # noqa: E402
from status import Status  # noqa: E402

REST = [-5.6, -102.8, 96.6, -100.1, -5.2]
BUDGETS = {"PICK": 90.0, "GIVE": 45.0, "PUT_DOWN": 20.0, "REST": 15.0}


class Log:
    def __init__(self, path: Path):
        self.handle = open(path, "a", encoding="utf-8")

    def write(self, **event) -> None:
        self.handle.write(json.dumps({"time": datetime.now(UTC).isoformat(timespec="milliseconds"), **event}) + "\n")
        self.handle.flush()


class Aborted(Exception):
    pass


class FakeRobot:
    def __init__(self, log: Log, status: Status, speed: float):
        self.log, self.status, self.speed = log, status, speed
        self.abort = threading.Event()
        self.state, self.round = "REST", 0

    def wait(self, seconds: float, **live) -> None:
        end = time.monotonic() + seconds / self.speed
        while time.monotonic() < end:
            if self.abort.is_set():
                raise Aborted
            self.status.update(**live)
            time.sleep(0.05)

    def go(self, target_state: str, reason: str, **details) -> None:
        self.log.write(event="transition", round=self.round, source=self.state, target=target_state, reason=reason,
                       gripper=self.status.snapshot().get("gripper"), **details)
        self.state = target_state

    def run_round(self, target: str, destination: str) -> None:
        self.round += 1
        self.abort.clear()
        started, states = time.monotonic(), ["REST"]
        self.state = "REST"
        self.log.write(event="round_start", round=self.round, target=target, destination=destination)
        outcome = "aborted"
        try:
            self.wait(1.0, gripper=30.0)
            self.go("PICK", "at rest")
            states.append("PICK")
            self.wait(4.0, gripper=40.0, joints=[0, -60, 40, -60, 0])
            self.log.write(event="grasp_check", round=self.round, pick_attempt=1, phase="settle", reading=13.0, goal=1.0,
                           error=12.0, verdict="held")
            self.wait(3.0, gripper=13.0, joints=REST)
            self.go("VERIFY_HOLD", "pick attempt 1: ok", skill="settled")
            states.append("VERIFY_HOLD")
            self.wait(1.0)
            areas = {"striped": 274, "red": 0, "blue": 0}
            if destination == "container":
                self.go("PUT_DOWN", "holding; voice or key says container", held_check="target", held_areas_px=areas,
                        held_area_px=274, held_threshold_px=120)
                states.append("PUT_DOWN")
                self.wait(4.0, gripper=13.0)
                self.wait(1.0, gripper=60.0)
                self.go("REST", "ball put down", skill="ok")
                outcome = "put_down"
            else:
                self.go("GIVE", "holding; voice or key says hand", held_check="target", held_areas_px=areas,
                        held_area_px=274, held_threshold_px=120)
                states.append("GIVE")
                for mode, seconds, palms in (("follow", 3, 1), ("hold", 2, 0), ("follow", 2, 1), ("drop", 1, 1),
                                             ("done", 2, 1)):
                    self.wait(seconds, give_mode=mode, palms=palms, gripper=60.0 if mode in ("drop", "done") else 13.0,
                              tip=[0.25, 0.0, 0.28])
                self.status.update(give_mode=None, palms=0)
                self.go("REST", "ball given", skill="dropped")
                outcome = "given"
            states.append("REST")
            self.wait(2.0, gripper=30.0, joints=REST)
            self.go("DONE", f"round over: {outcome}")
            states.append("DONE")
        except Aborted:
            self.go("ABORT", "abort requested")
            states.append("ABORT")
            self.status.update(give_mode=None)
        self.log.write(event="round_end", round=self.round, outcome=outcome, pick_attempts=1, clear_attempts=0,
                       wrong_balls=0, duration_s=round(time.monotonic() - started, 2), states=states)
        self.state = "REST"


def pattern(label: str, t: float) -> np.ndarray:
    image = np.zeros((480, 640, 3), np.uint8)
    image[..., 0] = np.linspace(40, 120, 640, dtype=np.uint8)[None, :]
    image[..., 2] = np.linspace(40, 120, 480, dtype=np.uint8)[:, None]
    centre = (int(320 + 200 * np.cos(t)), int(240 + 120 * np.sin(1.3 * t)))
    cv2.circle(image, centre, 30, (230, 200, 40), -1)
    cv2.putText(image, f"FAKE {label} {time.strftime('%H:%M:%S')}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def feed_frames(writers: dict, webcam: int | None) -> None:
    capture = cv2.VideoCapture(webcam) if webcam is not None else None
    while True:
        t = time.monotonic()
        ok, frame = capture.read() if capture is not None else (False, None)
        top = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB) if ok else pattern("top", t)
        wrist = np.ascontiguousarray(top[::-1]) if ok else pattern("wrist", t + 1.5)
        writers["top"].write(top, time.perf_counter_ns())
        writers["wrist"].write(wrist, time.perf_counter_ns())
        time.sleep(1 / 15)


def main() -> None:
    p = argparse.ArgumentParser(description="Fake robot program for trying the dashboard without the arm.")
    p.add_argument("--log-dir", default="~/so101/demo")
    p.add_argument("--webcam", type=int, help="camera index for real pictures instead of the test pattern")
    p.add_argument("--speed", type=float, default=1.0, help="faster rounds, e.g. 3")
    p.add_argument("--command-port", type=int, default=link.PORT)
    p.add_argument("--status-port", type=int, default=8767)
    p.add_argument("--top-shm", default="top_camera_frames")
    p.add_argument("--wrist-shm", default="wrist_camera_frames")
    args = p.parse_args()
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"chain_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log = Log(log_path)
    status = Status()
    status.update(available={"targets": ["striped"], "destinations": ["hand", "container"]},
                  pick_policy="FAKE Space0code/act_redblue_from_foil@015000", log_path=str(log_path.resolve()),
                  budgets_s=BUDGETS, hold_range=[6.0, 20.0], min_held_target_area_px=120, gripper=30.0, joints=REST,
                  observes=0, fake=True)
    started = time.monotonic()
    status.add_reader(lambda: {"observes": int((time.monotonic() - started) * 30)})
    status.start(port=args.status_port)
    writers = {"top": FrameWriter(args.top_shm), "wrist": FrameWriter(args.wrist_shm)}
    threading.Thread(target=feed_frames, args=(writers, args.webcam), daemon=True).start()
    robot = FakeRobot(log, status, args.speed)
    log.write(event="start", pick_policy="FAKE", budgets_s=BUDGETS, fake=True)
    receiver = link.CommandReceiver("127.0.0.1", args.command_port)
    print(f"fake robot: commands on {args.command_port}, status on {args.status_port}, log {log_path}. Ctrl+C stops.",
          flush=True)
    pending: list[dict] = []

    def poll() -> None:
        while True:
            command = receiver.poll()
            if command is not None and command.get("abort"):
                robot.abort.set()
            elif command is not None:
                pending.append(command)
            time.sleep(0.05)

    threading.Thread(target=poll, daemon=True).start()
    try:
        while True:
            if pending:
                command = pending.pop(0)
                print(f"round: {command}", flush=True)
                robot.run_round(command.get("target"), command.get("destination"))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()
