import sys
import threading

from skills import SCRIPTS, import_from

DESTINATION_KEYS = {"h": "hand", "c": "container"}


class Commands:
    def __init__(self):
        self.abort = threading.Event()
        self.go = threading.Event()
        self._lock = threading.Lock()
        self._destination: str | None = None
        self.voice = None

    def set_destination(self, destination: str | None) -> None:
        if destination in (None, "hand", "container"):
            with self._lock:
                self._destination = destination

    def destination(self) -> str | None:
        with self._lock:
            return self._destination

    # Line-based (key + Enter) so it works the same in a Windows console, WSL and macOS.
    def listen_keyboard(self) -> None:
        def read():
            for line in sys.stdin:
                key = line.strip().lower()[:1]
                if key in ("a", "q"):
                    self.abort.set()
                elif key in DESTINATION_KEYS:
                    self.set_destination(DESTINATION_KEYS[key])
                elif key == "x":
                    self.set_destination(None)
                self.go.set()
        threading.Thread(target=read, daemon=True).start()

    def listen_voice(self, host: str, port: int = 8765) -> None:
        self.voice = import_from(SCRIPTS / "voice", "link").CommandReceiver(host, port)

    def poll_voice(self) -> None:
        command = self.voice.poll() if self.voice is not None else None
        if command is not None and command.get("target") == "striped":
            self.set_destination(command.get("destination"))
            self.go.set()
