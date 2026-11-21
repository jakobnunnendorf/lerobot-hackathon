import json
from datetime import UTC, datetime
from typing import Callable


def stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


# One line per control tick: what the arm measured, then what was decided and sent from that observation. A line is
# written when the next observation starts it, so each line pairs an observation with the action computed from it.
class TickLog:
    def __init__(self):
        self.file, self.pending = None, None
        self.context: Callable[[], dict] = dict

    @property
    def on(self) -> bool:
        return self.file is not None

    def open(self, path: str) -> None:
        self.file = open(path, "a", encoding="utf-8")

    def observed(self, measured: list[float], load: float | None) -> None:
        if self.file is None:
            return
        self._flush()
        self.pending = {"time": stamp(), **self.context(), "measured": measured, "load": load}

    def note(self, **fields) -> None:
        if self.pending is not None:
            self.pending.update(fields)

    def close(self) -> None:
        if self.file is not None:
            self._flush()
            self.file.close()
            self.file = None

    def _flush(self) -> None:
        if self.pending is not None:
            self.file.write(json.dumps(self.pending) + "\n")
            self.file.flush()
            self.pending = None


TICKS = TickLog()
