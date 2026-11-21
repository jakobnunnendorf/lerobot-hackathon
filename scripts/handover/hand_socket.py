import json
import socket
import threading
import time

import numpy as np


# The tracker connects to us (board comment 7061160180); only the newest frame counts.
class HandReceiver:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766, max_age_s: float = 0.5):
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._hands: list[np.ndarray] = []
        self._received_at = 0.0
        self._server = socket.create_server((host, port))
        threading.Thread(target=self._accept_forever, daemon=True).start()

    def newest_hands(self) -> list[np.ndarray]:
        with self._lock:
            fresh = time.monotonic() - self._received_at <= self.max_age_s
            return list(self._hands) if fresh else []

    def _accept_forever(self) -> None:
        while True:
            conn, _ = self._server.accept()
            try:
                with conn, conn.makefile("r", encoding="utf-8") as lines:
                    for line in lines:
                        self._store(line)
            except OSError:
                pass

    def _store(self, line: str) -> None:
        try:
            hands = [np.array([h["x"], h["y"], h["z"]], dtype=float) for h in json.loads(line)["hands"]]
        except (ValueError, KeyError, TypeError):
            return
        with self._lock:
            self._hands, self._received_at = hands, time.monotonic()
