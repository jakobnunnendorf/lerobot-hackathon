import sys
import threading
import time

from skills import SCRIPTS, import_from

DESTINATION_KEYS = {"h": "hand", "c": "container"}
TARGETS = ("striped", "red", "blue")
TARGET_KEYS = {"r": "red", "b": "blue", "s": "striped"}


class Commands:
    def __init__(self):
        self.abort = threading.Event()
        self.go = threading.Event()
        self._lock = threading.Lock()
        self._destination: str | None = None
        self._target: str | None = None
        self._dance = False
        self.voice = None

    def set_destination(self, destination: str | None) -> None:
        if destination in (None, "hand", "container"):
            with self._lock:
                self._destination = destination

    def destination(self) -> str | None:
        with self._lock:
            return self._destination

    def set_target(self, target: str | None) -> None:
        if target in (None, *TARGETS):
            with self._lock:
                self._target = target

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

    # A command is for one round: the supervisor calls this when the round ends, so the next round does not reuse a
    # stale "hand" or "container" (or a colour), and a dance asked for during a round does not start after it.
    def clear(self) -> None:
        with self._lock:
            self._destination = self._target = None
            self._dance = False

    # Line-based (key + Enter) so it works the same in a Windows console, WSL and macOS.
    def listen_keyboard(self) -> None:
        def read():
            for line in sys.stdin:
                key = line.strip().lower()[:1]
                if key in ("a", "q"):
                    self.abort.set()
                elif key in DESTINATION_KEYS:
                    self.set_destination(DESTINATION_KEYS[key])
                elif key in TARGET_KEYS:
                    self.set_target(TARGET_KEYS[key])
                elif key == "x":
                    self.clear()
                elif key == "d":
                    self.request_dance()
                self.go.set()
        threading.Thread(target=read, daemon=True).start()

    def listen_voice(self, host: str, port: int = 8765) -> None:
        self.voice = import_from(SCRIPTS / "voice", "link").CommandReceiver(host, port)

        # Also polled from a thread of its own: the supervisor's tick does not run inside GIVE's loop, and an abort
        # from the dashboard must land there too.
        def poll_forever() -> None:
            while True:
                self.poll_voice()
                time.sleep(0.05)

        threading.Thread(target=poll_forever, daemon=True).start()

    # Voice, the dashboard and link.py send {"target", "destination"}; the dashboard's ABORT sends {"abort": true} and
    # its Dance {"dance": true}.
    def poll_voice(self) -> None:
        command = self.voice.poll() if self.voice is not None else None
        if command is not None and command.get("abort"):
            self.abort.set()
        elif command is not None and command.get("dance") is True:
            self.request_dance()
        elif command is not None and command.get("target") in TARGETS:
            self.set_target(command["target"])
            self.set_destination(command.get("destination"))
            self.go.set()
