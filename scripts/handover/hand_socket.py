import json
import socket
import threading
import time

import numpy as np


# The tracker connects to us (board comment 7061160180); only the newest frame counts.
class HandReceiver:
    # open_only: count only hands the tracker marks "open" (a fist is not a palm). A tracker run with
    # --only-open already filters; without it, both fists and palms arrive. Messages without the key count as open.
    def __init__(self, host: str = "127.0.0.1", port: int = 8766, max_age_s: float = 0.5, open_only: bool = True):
        self.max_age_s = max_age_s
        self.open_only = open_only
        self._lock = threading.Lock()
        self._hands: list[np.ndarray] = []
        self._pixel_palms = 0
        self._received_at = 0.0
        self._server = socket.create_server((host, port))
        threading.Thread(target=self._accept_forever, daemon=True).start()

    def newest_hands(self) -> list[np.ndarray]:
        with self._lock:
            fresh = time.monotonic() - self._received_at <= self.max_age_s
            return list(self._hands) if fresh else []

    # Number of open palms, for "are two palms there" checks (present mode). Uses the 3D hands when the tracker is
    # calibrated, otherwise the pixel detections it sends instead ("pixels": no position, presence only).
    def newest_palm_count(self) -> int:
        with self._lock:
            fresh = time.monotonic() - self._received_at <= self.max_age_s
            return max(len(self._hands), self._pixel_palms) if fresh else 0

    def _accept_forever(self) -> None:
        while True:
            conn, _ = self._server.accept()
            try:
                with conn, conn.makefile("r", encoding="utf-8") as lines:
                    for line in lines:
                        self._store(line)
            except OSError:
                pass

    def _counts(self, hand: dict) -> bool:
        return not self.open_only or bool(hand.get("open", True))

    def _store(self, line: str) -> None:
        try:
            message = json.loads(line)
            hands = [np.array([h["x"], h["y"], h["z"]], dtype=float) for h in message["hands"] if self._counts(h)]
            pixel_palms = sum(1 for h in message.get("pixels") or [] if self._counts(h))
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        with self._lock:
            self._hands, self._pixel_palms, self._received_at = hands, pixel_palms, time.monotonic()
