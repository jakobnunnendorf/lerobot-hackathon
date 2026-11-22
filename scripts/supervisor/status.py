import json
import socket
import threading
import time
from collections.abc import Callable

PORT = 8767


# Live values for the dashboard. The control loop only stores into a dict; a thread sends the newest snapshot as one
# UDP datagram, so a slow or absent dashboard can never stall the 30 Hz loop. See decisions/dashboard.md.
class Status:
    def __init__(self):
        self._fields: dict = {}
        self._lock = threading.Lock()
        self._readers: list[Callable[[], dict]] = []

    def update(self, **fields) -> None:
        with self._lock:
            self._fields.update(fields)

    # Readers run in the publishing thread, never in the control loop.
    def add_reader(self, read: Callable[[], dict]) -> None:
        self._readers.append(read)

    def snapshot(self) -> dict:
        with self._lock:
            fields = dict(self._fields)
        for read in self._readers:
            try:
                fields.update(read())
            except Exception as error:  # noqa: BLE001 - a broken reader must not stop the status
                fields["reader_error"] = repr(error)
        return {"time": time.time(), **fields}

    def start(self, host: str = "127.0.0.1", port: int = PORT, hz: float = 5.0) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)

        def publish() -> None:
            while True:
                try:
                    sock.sendto(json.dumps(self.snapshot(), default=float).encode(), (host, port))
                except OSError:
                    pass
                time.sleep(1 / hz)

        threading.Thread(target=publish, daemon=True).start()


STATUS = Status()
