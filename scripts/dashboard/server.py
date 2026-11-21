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
from datetime import datetime
from pathlib import Path

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
CAMERA_FPS = 15
CAMERA_STALE_S = 3.0
MAX_VOICE_S = 8.0
MIN_VOICE_S = 0.3
SILENCE_PEAK = 0.01
VOICE_PAD_S = 0.5
COMMAND_SETTLE_S = 3.0
RESULTS = ("success", "wrong_ball", "empty", "dropped", "still_digging", "never_started", "other")
DEFAULT_AVAILABLE = {"targets": ["striped"], "destinations": ["hand", "container"]}


class StatusListener:
    def __init__(self, port: int):
        self.latest: dict = {}
        self.received_at = 0.0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        threading.Thread(target=self._listen, args=(sock,), daemon=True).start()

    def _listen(self, sock: socket.socket) -> None:
        while True:
            data = sock.recv(65536)
            try:
                self.latest, self.received_at = json.loads(data), time.time()
            except ValueError:
                pass

    def age(self) -> float | None:
        return None if not self.received_at else time.time() - self.received_at

    def fresh(self) -> bool:
        age = self.age()
        return age is not None and age < STATUS_FRESH_S


class LogTail:
    """Events of the supervisor's JSONL log: the file the status names, else the newest chain_*.jsonl in log_dir."""

    def __init__(self, log_dir: Path, status: StatusListener):
        self.log_dir, self.status = log_dir, status
        self.path: Path | None = None
        self.events: list[dict] = []
        self.generation = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._follow, daemon=True).start()

    def _current_file(self) -> Path | None:
        named = self.status.latest.get("log_path") if self.status.fresh() else None
        if named and Path(named).exists():
            return Path(named)
        logs = sorted(self.log_dir.glob("chain_*.jsonl"), key=lambda p: p.stat().st_mtime)
        return logs[-1] if logs else None

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
        threading.Thread(target=self._run, daemon=True).start()

    def _set(self, jpeg: bytes, live: bool) -> None:
        self.jpeg, self.live, self.version = jpeg, live, self.version + 1

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
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self._set(encoded.tobytes(), True)
            elif time.monotonic() - last_new > CAMERA_STALE_S:
                reader.close()
                reader = None
                self._set(placeholder(self.label), False)
                continue
            time.sleep(1 / CAMERA_FPS)


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
    return p.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    uvicorn.run(build_app(arguments), host=arguments.host, port=arguments.port, log_level="warning")
