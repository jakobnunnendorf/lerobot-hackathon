# Standard library only. Contract agreed on the board (comment 7061160180): one JSON line per camera frame
# over local TCP, the tracker connects to the robot program on 127.0.0.1:8766,
# {"t": <epoch s>, "hands": [{"x", "y", "z", "palm_up", "score", ...}]} in metres in the robot base frame.
import json
import socket
import time

HOST = "127.0.0.1"
PORT = 8766


class HandLineSender:
    # Never blocks the tracker for long: if the robot program is not listening, lines are dropped and the
    # connection is retried at most once per second.
    def __init__(self, host=HOST, port=PORT):
        self.address = (host, port)
        self.connection = None
        self.next_attempt = 0.0

    def send(self, message):
        if self.connection is None and time.monotonic() >= self.next_attempt:
            try:
                self.connection = socket.create_connection(self.address, timeout=0.2)
                self.connection.settimeout(0.2)
            except OSError:
                self.next_attempt = time.monotonic() + 1.0
        if self.connection is None:
            return False
        try:
            self.connection.sendall((json.dumps(message) + "\n").encode())
            return True
        except OSError:
            self.connection.close()
            self.connection, self.next_attempt = None, time.monotonic() + 1.0
            return False
