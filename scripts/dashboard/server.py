# One page for testing and showing the demo chain: start chains (tiles, chat, push-to-talk), both camera feeds, the
# supervisor's state and live values, round results for the board. Runbook: docs/demo-chain-test.md. Why it works this
# way: decisions/dashboard.md. It never starts or moves the arm itself; it only sends commands to a running supervisor.
import argparse
import asyncio
import csv
import json
import socket
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

SCRIPTS = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
sys.path[:0] = [str(SCRIPTS / "voice"), str(SCRIPTS / "handtrack")]

import commands as voice  # noqa: E402
import link  # noqa: E402
import speech  # noqa: E402
from frame_link import FrameReader  # noqa: E402

STATUS_FRESH_S = 2.0
STATUS_GAP_S = 1.0
CAMERA_FPS = 15
CAMERA_STALE_S = 3.0
MAX_VOICE_S = 8.0
MIN_VOICE_S = 0.3
SILENCE_PEAK = 0.01
VOICE_PAD_S = 0.5
COMMAND_SETTLE_S = 3.0
VIDEO_FPS = 15
VIDEO_TAIL_S = 2.0
VIDEO_BAR_PX = 32  # under the frames, not over them: the wrist frame's bottom edge shows the held ball
RESULTS = ("success", "wrong_ball", "empty", "dropped", "still_digging", "never_started", "other")
DEFAULT_AVAILABLE = {"targets": ["striped"], "destinations": ["hand", "container"]}


class StatusListener:
    def __init__(self, port: int):
        self.latest: dict = {}
        self.received_at = 0.0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        threading.Thread(target=self._listen, args=(sock,), daemon=True).start()

    # The status went quiet for seconds during PICK in the VLA restart (take 1, board 7067318509), cause unknown. The
    # sender's own timestamps tell a stalled robot program apart from a stalled dashboard.
    def _listen(self, sock: socket.socket) -> None:
        while True:
            data = sock.recv(65536)
            try:
                latest = json.loads(data)
            except ValueError:
                continue
            now = time.time()
            if self.received_at and now - self.received_at > STATUS_GAP_S:
                sent_gap = float(latest.get("time", now)) - float(self.latest.get("time", now))
                print(f"{log_stamp(now)} status gap: {now - self.received_at:.1f} s since the last one; "
                      f"{sent_gap:.1f} s apart on the robot program's clock", flush=True)
            self.latest, self.received_at = latest, now

    def age(self) -> float | None:
        return None if not self.received_at else time.time() - self.received_at

    def fresh(self) -> bool:
        age = self.age()
        return age is not None and age < STATUS_FRESH_S


class LogTail:
    """Events of the supervisor's JSONL log: the file the newest status names, however old, else the newest
    chain_*.jsonl in log_dir. Requiring a fresh status lost the pick of take 1 when the status went quiet."""

    def __init__(self, log_dir: Path, status: StatusListener):
        self.log_dir, self.status = log_dir, status
        self.path: Path | None = None
        self.events: list[dict] = []
        self.generation = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._follow, daemon=True).start()

    def _current_file(self) -> Path | None:
        named = self.status.latest.get("log_path")
        if named and Path(named).exists():
            return Path(named)
        return newest_chain_log(self.log_dir)

    def _follow(self) -> None:
        handle, buffer = None, ""
        while True:
            path = self._current_file()
            if path != self.path:
                if handle:
                    handle.close()
                handle, buffer = (open(path, encoding="utf-8") if path else None), ""
                with self._lock:
                    self.path, self.events, self.generation = path, [], self.generation + 1
            if handle:
                buffer += handle.read()
                *lines, buffer = buffer.split("\n")
                parsed = [event for event in map(parse_event, lines) if event]
                if parsed:
                    with self._lock:
                        self.events.extend(parsed)
            time.sleep(0.25)

    def since(self, generation: int, index: int) -> tuple[int, int, list[dict]]:
        with self._lock:
            if generation != self.generation:
                return self.generation, len(self.events), list(self.events)
            return generation, len(self.events), self.events[index:]

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)


# --tick-log writes chain_<time>.ticks.jsonl next to the event log, and touches it 30 times a second.
def newest_chain_log(log_dir: Path) -> Path | None:
    logs = [p for p in log_dir.glob("chain_*.jsonl") if not p.name.endswith(".ticks.jsonl")]
    return max(logs, key=lambda p: p.stat().st_mtime, default=None)


def parse_event(line: str) -> dict | None:
    try:
        return json.loads(line) if line.strip() else None
    except ValueError:
        return None


class Camera:
    """Newest frame of one shared-memory block as JPEG. Reopens the block when the supervisor restarts."""

    def __init__(self, block: str, label: str):
        self.block, self.label = block, label
        self.jpeg, self.live, self.version = placeholder(label), False, 0
        self.frame, self.captured_at = None, None
        threading.Thread(target=self._run, daemon=True).start()

    def _set(self, jpeg: bytes, live: bool) -> None:
        self.jpeg, self.live, self.version = jpeg, live, self.version + 1
        if not live:
            self.frame, self.captured_at = None, None

    def _run(self) -> None:
        reader, last_new = None, 0.0
        while True:
            if reader is None:
                try:
                    reader, last_new = FrameReader(self.block), time.monotonic()
                except FileNotFoundError:
                    if self.live:
                        self._set(placeholder(self.label), False)
                    time.sleep(1.0)
                    continue
            got = reader.latest()
            if got is not None:
                last_new = time.monotonic()
                frame = cv2.cvtColor(got[0], cv2.COLOR_RGB2BGR)
                self.frame, self.captured_at = frame, wall_time(got[1])
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self._set(encoded.tobytes(), True)
            elif time.monotonic() - last_new > CAMERA_STALE_S:
                reader.close()
                reader = None
                self._set(placeholder(self.label), False)
                continue
            time.sleep(1 / CAMERA_FPS)


# The writers stamp frames with perf_counter_ns, which is the system-wide monotonic clock on Linux and Windows, so the
# capture time converts to the wall clock the supervisor's logs use.
def wall_time(captured_ns: int) -> float:
    return time.time() - (time.perf_counter_ns() - captured_ns) / 1e9


def log_stamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat(timespec="milliseconds")


def active_round(events: list[dict]) -> tuple[int | None, str]:
    number, state = None, "idle"
    for event in events:
        kind = event.get("event")
        if kind == "round_start":
            number, state = event["round"], "REST"
        elif kind == "transition" and event.get("round") == number:
            state = event["target"]
        elif kind == "round_end" and event.get("round") == number:
            number, state = None, "idle"
    return number, state


# A restarted robot program appends to the same log and counts rounds from 1 again: never overwrite an earlier video.
def free_stem(stem: Path) -> Path:
    candidate, n = stem, 1
    while candidate.with_name(candidate.name + ".mp4").exists():
        n += 1
        candidate = stem.with_name(f"{stem.name}_{n}")
    return candidate


class RoundVideo:
    def __init__(self, stem: Path, fps: int):
        stem = free_stem(stem)
        self.path = stem.with_name(stem.name + ".mp4")
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * 640, 480 + VIDEO_BAR_PX))
        self.times = open(stem.with_name(stem.name + ".frames.csv"), "w", newline="", encoding="utf-8")
        self.rows = csv.writer(self.times)
        self.rows.writerow(["frame", "time", "top_time", "wrist_time", "round", "state"])
        self.count = 0

    def write(self, cameras: dict, number: int, state: str) -> None:
        panels = [camera.frame if camera.frame is not None else blank(camera.label) for camera in cameras.values()]
        stamps = [camera.captured_at for camera in cameras.values()]
        shown = next((t for t in stamps if t is not None), time.time())
        image = np.vstack([np.hstack(panels), np.zeros((VIDEO_BAR_PX, 2 * 640, 3), np.uint8)])
        cv2.putText(image, f"{log_stamp(shown)[11:23]} UTC   round {number}   {state}", (10, 480 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.writer.write(image)
        self.rows.writerow([self.count, log_stamp(shown), *[log_stamp(t) if t else "" for t in stamps], number, state])
        self.count += 1

    def close(self) -> None:
        self.writer.release()
        self.times.close()


def blank(label: str) -> np.ndarray:
    image = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(image, f"no {label} frame", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
    return image


# One video per round next to the event log (<log stem>_roundNN.mp4): top | wrist, stamped with the capture time on
# the log's clock, plus <...>.frames.csv with every frame's times, to line the video up with the event and tick logs.
class RoundRecorder:
    def __init__(self, log: LogTail, cameras: dict, fps: int, live: Callable[[], bool]):
        self.log, self.cameras, self.fps, self.live = log, cameras, fps, live
        self.recording: str | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        video, recorded, tail_until, next_at = None, None, None, time.monotonic()
        while True:
            # An unfinished round in a crashed program's log is not running, so record only while camera frames arrive.
            # Not the status: it can go quiet for seconds in the middle of a pick.
            number, state = active_round(self.log.snapshot()) if self.live() else (None, "idle")
            if video is not None and number == recorded:
                tail_until = None
            elif video is not None:
                tail_until = tail_until or time.monotonic() + VIDEO_TAIL_S
                if number is not None or time.monotonic() >= tail_until:
                    video.close()
                    video, self.recording = None, None
            if video is None and number is not None and self.log.path is not None:
                video = RoundVideo(self.log.path.with_name(f"{self.log.path.stem}_round{number:02d}"), self.fps)
                recorded, tail_until, self.recording = number, None, video.path.name
            if video is not None:
                video.write(self.cameras, recorded, state if number == recorded else "after round")
            next_at = max(next_at + 1 / self.fps, time.monotonic())
            time.sleep(max(next_at - time.monotonic(), 0.0))


def placeholder(label: str) -> bytes:
    image = np.full((480, 640, 3), 32, np.uint8)
    cv2.putText(image, f"{label} camera idle", (150, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(image, "starts with the robot program", (130, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    return cv2.imencode(".jpg", image)[1].tobytes()


class Speech:
    """Parakeet, loaded in the background so the page is up at once (the first start downloads about 660 MB)."""

    def __init__(self, backend: str):
        self.transcribe, self.error = None, None
        threading.Thread(target=self._load, args=(backend,), daemon=True).start()

    def _load(self, backend: str) -> None:
        try:
            self.transcribe = speech.BACKENDS[backend]()
        except Exception as error:  # noqa: BLE001 - the page keeps working without voice
            self.error = repr(error)


class Results:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._lock = threading.Lock()

    def path_for(self, log: Path | None) -> Path:
        return self.log_dir / f"results_{log.stem if log else 'nolog'}.csv"

    def read(self, log: Path | None) -> list[dict]:
        path = self.path_for(log)
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        latest = {row["round"]: row for row in rows}
        return list(latest.values())

    def add(self, log: Path | None, row: dict) -> None:
        path = self.path_for(log)
        with self._lock:
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)


def round_ends(events: list[dict]) -> dict[str, dict]:
    starts = {str(e["round"]): e for e in events if e.get("event") == "round_start"}
    ends = {}
    for event in events:
        if event.get("event") == "round_end":
            number = str(event["round"])
            ends[number] = {**event, "target": starts.get(number, {}).get("target"),
                            "destination": starts.get(number, {}).get("destination")}
    return ends


def summary_markdown(events: list[dict], results: list[dict], log: Path | None, pick_policy: str | None) -> str:
    ends = round_ends(events)
    marked = {row["round"]: row for row in results}
    lines = [f"**Demo chain session** `{log.name if log else 'no log'}`, pick policy `{pick_policy or '?'}`", "",
             "| # | chain | outcome | s | picks | wrong balls | result | note |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for number, end in ends.items():
        row = marked.get(number, {})
        chain = f"{end.get('target') or 'striped'} -> {end.get('destination') or '(none)'}"
        lines.append(f"| {number} | {chain} | {end['outcome']} | {end.get('duration_s', '')} | {end.get('pick_attempts', '')} | "
                     f"{end.get('wrong_balls', '')} | {row.get('result', '')} | {row.get('note', '')} |")
    outcomes = Counter(end["outcome"] for end in ends.values())
    tally = Counter(row["result"] for row in results)
    lines += ["", f"Outcomes: {', '.join(f'{k} {v}' for k, v in outcomes.most_common()) or 'none'}",
              f"Marked by the tester: {', '.join(f'{k} {v}' for k, v in tally.most_common()) or 'none'} "
              f"({tally['success']}/{len(results)} success)"]
    checks = Counter(e.get("verdict") for e in events if e.get("event") == "grasp_check")
    if checks:
        lines.append(f"Grasp checks: {', '.join(f'{k} {v}' for k, v in checks.most_common())}")
    return "\n".join(lines)


def build_app(args) -> FastAPI:
    status = StatusListener(args.status_port)
    log = LogTail(Path(args.log_dir).expanduser(), status)
    cameras = {"top": Camera(args.top_shm, "top"), "wrist": Camera(args.wrist_shm, "wrist")}
    recorder = None if args.no_video else RoundRecorder(log, cameras, args.video_fps,
                                                        lambda: any(camera.live for camera in cameras.values()))
    voice_model = Speech(args.speech) if not args.no_speech else None
    results = Results(Path(args.log_dir).expanduser())
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def available() -> dict:
        return status.latest.get("available", DEFAULT_AVAILABLE) if status.fresh() else DEFAULT_AVAILABLE

    def unavailable_reason(target: str, destination: str) -> str | None:
        allowed = available()
        if target not in allowed["targets"]:
            return f"no {target} pick skill yet: the training data has only the striped ball"
        if destination not in allowed["destinations"]:
            return f"destination {destination!r} is not set up in this robot program"
        return None

    # The supervisor reads the destination again at VERIFY_HOLD, so a command during a round would change that round.
    # A command just sent counts as running too: its round_start reaches the log a moment later.
    last_sent = {"at": 0.0}

    def round_running() -> bool:
        events = log.snapshot()
        started = {e["round"] for e in events if e.get("event") == "round_start"}
        ended = {e["round"] for e in events if e.get("event") == "round_end"}
        return bool(started - ended) or time.monotonic() - last_sent["at"] < COMMAND_SETTLE_S

    def send(command: dict) -> dict:
        reason = "a round is running: wait for it to end, or ABORT" if round_running() else \
            unavailable_reason(command["target"], command["destination"])
        if reason:
            return {"ok": False, "reason": reason, "command": command}
        sent = link.send(command, port=args.command_port)
        if sent:
            last_sent["at"] = time.monotonic()
        return {"ok": sent, "command": command, "reason": None if sent else "robot program is not listening on 8765"}

    def handle_text(text: str, source: str) -> dict:
        command = voice.parse(text)
        if command is None:
            scores = voice.scores(text)
            return {"ok": False, "text": text, "reason": voice.rejection_reason(text),
                    "scores": {"destinations": scores.destinations, "colours": list(scores.colours)}}
        return {"text": text, **send({"target": command.target, "destination": command.destination,
                                      "transcript": text, "source": source})}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stream/{name}")
    async def stream(name: str):
        camera = cameras.get(name)
        if camera is None:
            raise HTTPException(404)

        async def frames():
            version, sent_at = -1, 0.0
            while True:
                if camera.version != version or time.monotonic() - sent_at > 1.0:
                    version, sent_at, jpeg = camera.version, time.monotonic(), camera.jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / CAMERA_FPS)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/events")
    async def events(request: Request):
        async def feed():
            generation, index = -1, 0
            while not await request.is_disconnected():
                new_generation, index, new = log.since(generation, index)
                message = {"status": status.latest if status.received_at else None, "status_age": status.age(),
                           "log": log.path.name if log.path else None, "reset": new_generation != generation,
                           "events": new, "cameras": {name: camera.live for name, camera in cameras.items()},
                           "recording": recorder.recording if recorder else None,
                           "speech": "ready" if voice_model and voice_model.transcribe else
                           (f"failed: {voice_model.error}" if voice_model and voice_model.error else
                            "off" if voice_model is None else "loading")}
                generation = new_generation
                yield f"data: {json.dumps(message, default=float)}\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(feed(), media_type="text/event-stream")

    @app.post("/api/command")
    async def command(request: Request):
        body = await request.json()
        return send({"target": body.get("target"), "destination": body.get("destination"), "transcript": "",
                     "source": "dashboard"})

    @app.post("/api/text")
    async def text(request: Request):
        body = await request.json()
        return handle_text(str(body.get("text", "")), "chat")

    @app.post("/api/voice")
    async def voice_command(request: Request):
        if voice_model is None or voice_model.transcribe is None:
            return {"ok": False, "reason": "speech model not loaded yet" if voice_model else "speech is off"}
        audio = np.frombuffer(await request.body(), np.float32)[: int(MAX_VOICE_S * speech.SAMPLE_RATE)]
        if len(audio) < MIN_VOICE_S * speech.SAMPLE_RATE or np.abs(audio).max() < SILENCE_PEAK:
            return {"ok": False, "reason": "only silence recorded: check the microphone", "text": ""}
        started = time.perf_counter()
        transcript = await asyncio.to_thread(voice_model.transcribe, np.pad(audio, round(VOICE_PAD_S * speech.SAMPLE_RATE)))
        return {**handle_text(transcript, "voice"), "seconds": round(time.perf_counter() - started, 2)}

    @app.post("/api/abort")
    def abort():
        return {"ok": link.send({"abort": True}, port=args.command_port)}

    @app.get("/api/results")
    def get_results():
        return results.read(log.path)

    @app.post("/api/result")
    async def add_result(request: Request):
        body = await request.json()
        if body.get("result") not in RESULTS:
            raise HTTPException(400, f"result must be one of {RESULTS}")
        number = str(body.get("round"))
        end = round_ends(log.snapshot()).get(number, {})
        results.add(log.path, {"time": datetime.now().isoformat(timespec="seconds"), "log": log.path.name if log.path else "",
                               "round": number, "target": end.get("target"), "destination": end.get("destination"),
                               "outcome": end.get("outcome"), "duration_s": end.get("duration_s"),
                               "pick_attempts": end.get("pick_attempts"), "wrong_balls": end.get("wrong_balls"),
                               "pick_policy": status.latest.get("pick_policy"), "result": body["result"],
                               "note": str(body.get("note", ""))[:300]})
        return {"ok": True}

    @app.get("/api/summary", response_class=PlainTextResponse)
    def summary():
        return summary_markdown(log.snapshot(), results.read(log.path), log.path, status.latest.get("pick_policy"))

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test and demo dashboard for the demo chain (open http://localhost:8080).")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so the Windows browser reaches it inside WSL")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--log-dir", default="~/so101/demo", help="where demo_chain.sh writes chain_*.jsonl")
    p.add_argument("--command-port", type=int, default=link.PORT)
    p.add_argument("--status-port", type=int, default=8767)
    p.add_argument("--top-shm", default="top_camera_frames")
    p.add_argument("--wrist-shm", default="wrist_camera_frames")
    p.add_argument("--speech", default="parakeet-en", choices=sorted(speech.BACKENDS))
    p.add_argument("--no-speech", action="store_true", help="skip loading the speech model (typed chat still works)")
    p.add_argument("--no-video", action="store_true", help="do not record a video of each round next to the log")
    p.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    return p.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    uvicorn.run(build_app(arguments), host=arguments.host, port=arguments.port, log_level="warning")
