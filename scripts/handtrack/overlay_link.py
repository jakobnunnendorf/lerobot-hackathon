# The dashboard's hand overlay gets its own UDP datagram, so the TCP lines to the robot program stay as agreed on the
# board. Fire and forget, like the status: a closed dashboard can never stall the tracker. See decisions/dashboard.md.
import json
import socket

from hand_features import is_offering

HOST = "127.0.0.1"
PORT = 8771


def overlay_hand(landmarks, hand, width, height):
    u, v = hand["palm_centre_px"]
    return {"handedness": hand["handedness"],
            "landmarks": [[round(float(lm.x), 3), round(float(lm.y), 3)] for lm in landmarks],
            "palm_centre": [round(float(u) / width, 3), round(float(v) / height, 3)],
            "offering": bool(is_offering(hand))}


def overlay_message(captured_epoch, width, height, landmark_sets, hands):
    return {"t": captured_epoch, "width": width, "height": height,
            "hands": [overlay_hand(lm, h, width, height) for lm, h in zip(landmark_sets, hands)]}


class OverlaySender:
    def __init__(self, port=PORT, host=HOST):
        self.address = (host, port)
        self.sock = None
        if port:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)

    def send(self, message):
        if self.sock is None:
            return False
        try:
            self.sock.sendto(json.dumps(message, separators=(",", ":")).encode(), self.address)
            return True
        except OSError:
            return False
