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
                                  "--speed", "6", *PORTS])
        server = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/dashboard/server.py"), "--host", "127.0.0.1",
                                   "--port", "18080", "--log-dir", logs, "--no-speech", *PORTS])
        try:
            event = wait_for(lambda: _try(first_event), 30)
            check("events: live status from the robot program and its log", bool(event and event["status"]
                  and event["log"]), str(event and {k: event[k] for k in ("log", "cameras", "speech")}))
            with urllib.request.urlopen(SERVER + "/stream/top", timeout=10) as response:
                chunk = response.read(4000)
            check("top stream: MJPEG part with a JPEG", chunk.startswith(b"--frame") and b"\xff\xd8" in chunk)
            refused = call("/api/command", {"target": "red", "destination": "hand"})
            check("red is refused (no red pick skill)", not refused["ok"] and "red" in refused["reason"], refused["reason"])
            started = call("/api/text", {"text": "give me the ball"})
            check("chat 'give me the ball' -> striped to hand, sent", started["ok"]
                  and started["command"]["destination"] == "hand")
            busy = call("/api/command", {"target": "striped", "destination": "container"})
            check("a second command right after the first is refused", not busy["ok"] and "running" in busy["reason"],
                  busy["reason"])
            given = wait_for(lambda: "| 1 | striped -> hand | given" in call("/api/summary"), 30)
            check("round 1 ends 'given' and shows in the summary", bool(given))
            call("/api/result", {"round": 1, "result": "success", "note": "smoke"})
            rows = call("/api/results")
            check("result row saved", rows and rows[0]["result"] == "success" and rows[0]["outcome"] == "given", str(rows))
            call("/api/command", {"target": "striped", "destination": "container"})
            time.sleep(1.0)
            call("/api/abort", {})
            aborted = wait_for(lambda: "| 2 | striped -> container | aborted" in call("/api/summary"), 15)
            check("ABORT ends round 2 'aborted'", bool(aborted), call("/api/summary").splitlines()[-3:])
        finally:
            for process in (server, robot):
                process.terminate()
                process.wait(timeout=10)
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


def _try(function):
    try:
        return function()
    except OSError:
        return None


if __name__ == "__main__":
    main()
