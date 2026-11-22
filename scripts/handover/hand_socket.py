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
        self._pointing_tips: list[np.ndarray] = []
        self._messages = 0
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

    # Fingertips of pointing hands, for the bite round. Ignores open_only: a pointing hand is never an open palm.
    def newest_pointing_tips(self) -> list[np.ndarray]:
        return self.newest_pointing()[1]

    # With the number of the message they came from: the same message is handed out for up to max_age_s (15 ticks of
    # the 30 Hz loop), and counting it once per tick turned one false detection into a bite.
    def newest_pointing(self) -> tuple[int, list[np.ndarray]]:
        with self._lock:
            fresh = time.monotonic() - self._received_at <= self.max_age_s
            return self._messages, list(self._pointing_tips) if fresh else []

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

    # Parsed apart from the palms, so a bad tip field never drops them.
    @staticmethod
    def _pointing_tips_in(hands) -> list[np.ndarray]:
        tips = []
        for hand in hands:
            try:
                tip = np.array([hand["tip_x"], hand["tip_y"], hand["tip_z"]], dtype=float)
                if hand.get("pointing") is True and np.isfinite(tip).all():
                    tips.append(tip)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        return tips

    def _store(self, line: str) -> None:
        try:
            message = json.loads(line)
            hands = [np.array([h["x"], h["y"], h["z"]], dtype=float) for h in message["hands"] if self._counts(h)]
            pixel_palms = sum(1 for h in message.get("pixels") or [] if self._counts(h))
        except (ValueError, KeyError, TypeError, AttributeError):
            return
        tips = self._pointing_tips_in(message["hands"])
        with self._lock:
            self._hands, self._pixel_palms, self._pointing_tips = hands, pixel_palms, tips
            self._messages += 1
            self._received_at = time.monotonic()
