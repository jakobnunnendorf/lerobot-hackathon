# Dashboard server against scripts/dashboard/fake_robot.py on spare ports: streams, commands, refusals, abort,
# results and the board summary. Needs the dashboard environment (scripts/dashboard/requirements.txt).
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "http://127.0.0.1:18080"
PORTS = ["--command-port", "18865", "--status-port", "18867", "--top-shm", "smoke_top", "--wrist-shm", "smoke_wrist"]
failures = []


def check(name: str, ok: bool, detail="") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def call(path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        text = response.read().decode()
    return json.loads(text) if text.startswith(("{", "[")) else text


def first_event() -> dict:
    with urllib.request.urlopen(SERVER + "/events", timeout=10) as response:
        for line in response:
            if line.startswith(b"data: "):
                return json.loads(line[6:])
    return {}


def wait_for(condition, seconds: float):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = condition()
        if value:
            return value
        time.sleep(0.5)
    return None


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as logs:
        robot = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/dashboard/fake_robot.py"), "--log-dir", logs,
                                  "--speed", "6", "--quiet-pick", "3", *PORTS])
        server = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/dashboard/server.py"), "--host", "127.0.0.1",
                                   "--port", "18080", "--log-dir", logs, "--no-speech", "--fake-collect",
                                   "--collect-log-dir", f"{logs}/collect", "--collect-port", "18868", *PORTS])
        try:
            event = wait_for(lambda: _try(first_event), 30)
            check("events: live status from the robot program and its log", bool(event and event["status"]
                  and event["log"]), str(event and {k: event[k] for k in ("log", "cameras", "speech")}))
            with urllib.request.urlopen(SERVER + "/stream/top", timeout=10) as response:
                chunk = response.read(4000)
            check("top stream: MJPEG part with a JPEG", chunk.startswith(b"--frame") and b"\xff\xd8" in chunk)
            refused = call("/api/command", {"target": "red", "destination": "hand"})
            check("red is refused (no red pick skill)", not refused["ok"] and "red" in refused["reason"], refused["reason"])
            plan = call("/api/text", {"text": "take out 2 red balls, then give me the striped ball"})
            check("chat plan parsed and refused: this robot program has no red clear policy", not plan["ok"]
                  and plan["command"]["plan"][0] == {"target": "red", "count": 2} and "--clear-red-policy" in plan["reason"],
                  plan["reason"])
            started = call("/api/text", {"text": "give me the ball"})
            check("chat 'give me the ball' -> striped to hand, sent", started["ok"]
                  and started["command"]["destination"] == "hand")
            busy = call("/api/command", {"target": "striped", "destination": "container"})
            check("a second command right after the first is refused", not busy["ok"] and "running" in busy["reason"],
                  busy["reason"])
            given = wait_for(lambda: "| 1 | striped -> hand | given" in call("/api/summary"), 30)
            check("round 1 ends 'given' and shows in the summary", bool(given))
            round_video(Path(logs))
            call("/api/result", {"round": 1, "result": "success", "note": "smoke"})
            rows = call("/api/results")
            check("result row saved", rows and rows[0]["result"] == "success" and rows[0]["outcome"] == "given", str(rows))
            call("/api/command", {"target": "striped", "destination": "container"})
            time.sleep(1.0)
            call("/api/abort", {})
            aborted = wait_for(lambda: "| 2 | striped -> container | aborted" in call("/api/summary"), 15)
            check("ABORT ends round 2 'aborted'", bool(aborted), call("/api/summary").splitlines()[-3:])
            dance_round(Path(logs))
            sys.path.insert(0, str(ROOT / "scripts/dashboard"))
            no_overwrite(Path(logs))
            stale_status_keeps_log(Path(logs))
            refused = call("/api/collect/start", {"name": "smoke_set", "task": "Pick", "episodes": 2, "episode_s": 30})
            check("collect: refused while the robot program runs", not refused["ok"] and "robot program" in refused["reason"],
                  refused["reason"])
            robot.terminate()
            robot.wait(timeout=10)
            time.sleep(4.5)  # status stale (2 s), and the page lets go of the frame blocks (3 s): Windows keeps them while open
            collect_session(Path(logs) / "collect")
        finally:
            for process in (server, robot):
                process.terminate()
                process.wait(timeout=10)
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


def dance_round(logs: Path) -> None:
    mixed = call("/api/text", {"text": "dance and give me the ball"})
    check("chat: a dance and a ball command in one sentence is refused", not mixed["ok"] and "dance" in mixed["reason"],
          mixed["reason"])
    danced = wait_for(lambda: (lambda reply: reply if reply["ok"] else None)(call("/api/text", {"text": "let's dance"})), 10)
    check("chat 'let's dance' -> {\"dance\": true} sent", bool(danced) and danced["command"]["dance"] is True, str(danced))
    busy = call("/api/dance", {})
    check("Dance button right after: refused while the dance round runs", not busy["ok"] and "running" in busy["reason"],
          busy["reason"])
    ended = wait_for(lambda: "| 3 | dance FAKE/so101_gestures_dance:1 | danced" in call("/api/summary"), 20)
    video = wait_for(lambda: next(logs.glob("chain_*_round03.mp4"), None), 10)
    check("round 3 dances (REST -> DANCE -> REST), shows in the summary and is filmed", bool(ended) and bool(video),
          str(call("/api/summary").splitlines()[-4:]))
    again = wait_for(lambda: (lambda reply: reply if reply["ok"] else None)(call("/api/dance", {})), 10)
    check("Dance button once the round is over -> sent", bool(again) and again["command"]["source"] == "dashboard", str(again))
    wait_for(lambda: "| 4 | dance" in call("/api/summary"), 20)


def round_video(logs: Path) -> None:
    import csv
    from datetime import datetime

    import cv2

    log = next(p for p in logs.glob("chain_*.jsonl") if not p.name.endswith(".ticks.jsonl"))
    video = wait_for(lambda: next(logs.glob(f"{log.stem}_round01.mp4"), None), 10)
    times = log.with_name(f"{log.stem}_round01.frames.csv")
    closed = wait_for(lambda: times.exists() and cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT) > 0, 10)
    time.sleep(0.5)
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    with open(times, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    start = next(datetime.fromisoformat(e["time"]) for e in events if e.get("event") == "round_start" and e["round"] == 1)
    end = next(datetime.fromisoformat(e["time"]) for e in events if e.get("event") == "round_end" and e["round"] == 1)
    first, last = datetime.fromisoformat(rows[0]["time"]), datetime.fromisoformat(rows[-1]["time"])
    split = list(logs.glob(f"{log.stem}_round01_*.mp4"))
    check("round 1 video, through 3 s without a status in PICK: one file, top | wrist, one CSV row per frame, "
          "frame times on the log's clock",
          bool(closed) and ok and frame.shape == (512, 1280, 3) and frames == len(rows) > 10 and not split
          and abs((first - start).total_seconds()) < 1.0 and 0 <= (last - end).total_seconds() < 3.0,
          f"{video.name if video else None}: {frames} frames, {len(rows)} rows, first {(first - start).total_seconds():+.2f} s "
          f"from round_start, last {(last - end).total_seconds():+.2f} s from round_end")


def wait_phase(phase: str, seconds: float = 15):
    return wait_for(lambda: (lambda s: s if s.get("phase") == phase else None)((_try(first_event) or {}).get("status") or {}),
                    seconds)


def collect_session(collect_dir: Path) -> None:
    form = {"name": "smoke_set", "task": "Pick up the striped ball", "episodes": 2, "episode_s": 30, "upload": True}
    collect_dir.mkdir(parents=True, exist_ok=True)
    (collect_dir / "fail_next_upload").write_text("")  # fake_recorder.py fails this session's upload once
    started = call("/api/collect/start", form)
    again = call("/api/collect/start", form)
    waiting = wait_phase("waiting")
    check("collect: a session starts from the page; a second start is refused while it runs",
          started["ok"] and not again["ok"] and "running" in again["reason"] and bool(waiting), f"{started} {again}")
    busy = call("/api/command", {"target": "striped", "destination": "hand"})
    check("collect: chain commands are refused while a session owns the arm", not busy["ok"]
          and "collection" in busy["reason"], busy["reason"])
    steps = []
    for control, phase in (("next", "recording"), ("next", "reset"), ("discard", "recording"), ("next", "reset"),
                           ("next", "recording"), ("next", "upload_failed")):
        call("/api/collect/control", {"control": control})
        reached = wait_phase(phase)
        steps.append((control, phase, bool(reached)))
    final = reached or {}
    event = _try(first_event) or {}
    check("collect: next/discard walk lerobot-record's phases, then the upload; the discarded episode is recorded again",
          all(ok for _, _, ok in steps), str(steps))
    check("collect: counts at the end (2 saved, 1 thrown away, dataset 40 -> 42), the upload's error reaches the page",
          final.get("saved") == 2 and final.get("discarded") == 1 and final.get("dataset_total") == 42
          and "fake network failure" in (final.get("upload_error") or ""), str(final))
    ended = wait_for(lambda: not call("/api/collect/datasets") is None and not (_try(first_event) or {}).get(
        "collect", {}).get("running"), 10)
    check("collect: the session process ends by itself; its output reaches the page",
          bool(ended) and any("Recording episode" in line for line in event.get("collect", {}).get("output", [])),
          str(event.get("collect", {}).get("output", [])[-3:]))
    time.sleep(2.5)  # status stale: the ended session no longer counts as running
    retried = call("/api/collect/upload", {"name": "smoke_set"})
    uploading = wait_phase("uploading", 10)
    refused = call("/api/collect/start", form)
    uploaded = wait_phase("uploaded", 15)
    check("collect: Retry upload from the page; a new session is refused while it uploads",
          retried["ok"] and bool(uploading) and not refused["ok"] and "upload" in refused["reason"] and bool(uploaded),
          f"{retried} {refused} {uploaded}")


def stale_status_keeps_log(logs: Path) -> None:
    from types import SimpleNamespace

    from server import LogTail

    followed = logs / "exp" / "supervisor.jsonl"
    followed.parent.mkdir()
    followed.write_text("")
    (logs / "chain_29990101_000000.jsonl").write_text("")
    status = SimpleNamespace(latest={"log_path": str(followed)}, fresh=lambda: False)
    chosen = LogTail._current_file(SimpleNamespace(status=status, log_dir=logs))
    check("a status gone quiet still names the log to follow, not the newest chain_*.jsonl (VLA take 1)",
          chosen == followed, str(chosen))


def no_overwrite(logs: Path) -> None:
    from server import free_stem

    stem = logs / "supervisor_round01"
    first = free_stem(stem)
    first.with_name(first.name + ".mp4").write_bytes(b"x")
    second = free_stem(stem)
    second.with_name(second.name + ".mp4").write_bytes(b"x")
    check("a restarted robot program's round 1 gets a new video file, the earlier one stays",
          first.name == "supervisor_round01" and second.name == "supervisor_round01_2"
          and free_stem(stem).name == "supervisor_round01_3", f"{first.name}, {second.name}")


def _try(function):
    try:
        return function()
    except OSError:
        return None


if __name__ == "__main__":
    main()
