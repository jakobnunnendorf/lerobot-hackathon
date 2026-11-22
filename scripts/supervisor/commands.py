import re
import sys
import threading
import time

from skills import SCRIPTS, import_from

DESTINATION_KEYS = {"h": "hand", "c": "container"}
TARGETS = ("striped", "red", "blue")
TARGET_KEYS = {"r": "red", "b": "blue", "s": "striped"}
PLAN_STEP = re.compile(r"(\d*)([rbs])")

Plan = tuple[tuple[str, int], ...]


# Red/blue steps in any order, at most one striped step and only last: the striped ball ends the round (PICK chain).
def valid_plan(steps) -> Plan | None:
    try:
        plan = tuple((step["target"], step["count"]) for step in steps)
    except (TypeError, KeyError):
        return None
    if not plan or any(target not in TARGETS or type(count) is not int or count < 1 for target, count in plan):
        return None
    striped = [i for i, (target, _) in enumerate(plan) if target == "striped"]
    if striped and (striped != [len(plan) - 1] or plan[-1][1] != 1):
        return None
    return plan


# "p 3b 2r s h": 3 blue, 2 red, then the striped ball, to the hand. Returns (plan, destination) or None.
def plan_from_keys(line: str) -> tuple[Plan, str | None] | None:
    steps, destination = [], None
    for token in line.lower().split()[1:]:
        if token in DESTINATION_KEYS and destination is None:
            destination = DESTINATION_KEYS[token]
            continue
        match = PLAN_STEP.fullmatch(token)
        if match is None:
            return None
        steps.append({"target": TARGET_KEYS[match[2]], "count": int(match[1] or 1)})
    plan = valid_plan(steps)
    return None if plan is None else (plan, destination)


class Commands:
    def __init__(self):
        self.abort = threading.Event()
        self.go = threading.Event()
        self._lock = threading.Lock()
        self._destination: str | None = None
        self._target: str | None = None
        self._plan: Plan | None = None
        self._dance = False
        self._bite = False
        self.voice = None
        self.on_max_pick_attempts = None

    def set_destination(self, destination: str | None) -> None:
        if destination in (None, "hand", "container"):
            with self._lock:
                self._destination = destination

    def destination(self) -> str | None:
        with self._lock:
            return self._destination

    # A single target and a plan replace each other: the newest command wins.
    def set_target(self, target: str | None) -> None:
        if target in (None, *TARGETS):
            with self._lock:
                self._target, self._plan = target, None

    def set_plan(self, plan: Plan) -> None:
        with self._lock:
            self._plan, self._target = plan, None

    def plan(self) -> Plan | None:
        with self._lock:
            return self._plan

    def target(self) -> str | None:
        with self._lock:
            return self._target

    def request_dance(self) -> None:
        with self._lock:
            self._dance = True
        self.go.set()

    def take_dance(self) -> bool:
        with self._lock:
            wanted, self._dance = self._dance, False
            return wanted

    def request_bite(self) -> None:
        with self._lock:
            self._bite = True
        self.go.set()

    def take_bite(self) -> bool:
        with self._lock:
            wanted, self._bite = self._bite, False
            return wanted

    # A command is for one round: the supervisor calls this when the round ends, so the next round does not reuse a
    # stale "hand" or "container" (or a colour), and a dance or bite asked for during a round does not start after it.
    def clear(self) -> None:
        with self._lock:
            self._destination = self._target = self._plan = None
            self._dance = self._bite = False

    # Bite is "f" (finger): "b" already sets the blue target. Takes the whole line: a plan ("p 3b 2r s h") needs it.
    def handle_key(self, line: str) -> None:
        key = line.strip().lower()[:1]
        if key == "p":
            if not self._plan_line(line):
                return
        elif key in ("a", "q"):
            self.abort.set()
        elif key in DESTINATION_KEYS:
            self.set_destination(DESTINATION_KEYS[key])
        elif key in TARGET_KEYS:
            self.set_target(TARGET_KEYS[key])
        elif key == "x":
            self.clear()
        elif key == "d":
            self.request_dance()
        elif key == "f":
            self.request_bite()
        self.go.set()

    # Line-based (key + Enter) so it works the same in a Windows console, WSL and macOS.
    def listen_keyboard(self) -> None:
        def read():
            for line in sys.stdin:
                self.handle_key(line)
        threading.Thread(target=read, daemon=True).start()

    def _plan_line(self, line: str) -> bool:
        parsed = plan_from_keys(line)
        if parsed is None:
            print(f"plan not understood: {line.strip()!r}; e.g. p 3b 2r s h (3 blue, 2 red, then striped to the hand)",
                  flush=True)
            return False
        self.set_plan(parsed[0])
        self.set_destination(parsed[1])
        return True

    def listen_voice(self, host: str, port: int = 8765) -> None:
        self.voice = import_from(SCRIPTS / "voice", "link").CommandReceiver(host, port)

        # Also polled from a thread of its own: the supervisor's tick does not run inside GIVE's loop, and an abort
        # from the dashboard must land there too.
        def poll_forever() -> None:
            while True:
                self.poll_voice()
                time.sleep(0.05)

        threading.Thread(target=poll_forever, daemon=True).start()

    # Voice, the dashboard and link.py send {"target", "destination"} or {"plan", "destination"}; the dashboard's ABORT
    # sends {"abort": true}, its Dance {"dance": true} and its Bite {"bite": true}.
    def poll_voice(self) -> None:
        command = self.voice.poll() if self.voice is not None else None
        if command is not None and command.get("abort"):
            self.abort.set()
        elif command is not None and command.get("dance") is True:
            self.request_dance()
        elif command is not None and command.get("bite") is True:
            self.request_bite()
        elif command is not None and "plan" in command:
            plan = valid_plan(command["plan"])
            if plan is None:
                print(f"plan refused, not valid: {command['plan']}", flush=True)
                return
            self.set_plan(plan)
            self.set_destination(command.get("destination"))
            self.go.set()
        elif command is not None and "max_pick_attempts" in command:
            # No go.set(): the limit never starts a round, and during one it lands at once (a buried ball gets more picks).
            if self.on_max_pick_attempts is not None:
                self.on_max_pick_attempts(command["max_pick_attempts"])
        elif command is not None and command.get("target") in TARGETS:
            self.set_target(command["target"])
            self.set_destination(command.get("destination"))
            self.go.set()
