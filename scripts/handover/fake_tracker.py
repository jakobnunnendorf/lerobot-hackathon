import argparse
import json
import socket
import time

import numpy as np


def scenario(t: float) -> list[dict]:
    if t < 5:
        angle = 2 * np.pi * t / 5
        return [hand(0.22 + 0.04 * np.cos(angle), 0.06 * np.sin(angle))]
    if t < 9:
        return []
    if t < 13:
        return [hand(0.18, -0.08)]
    return [hand(0.20, 0.03), hand(0.20, 0.09)]


def hand(x: float, y: float) -> dict:
    return {"x": round(x, 4), "y": round(y, 4), "z": 0.12, "palm_up": True, "score": 0.9}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--connect", default="127.0.0.1:8766")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--duration", type=float, default=20)
    args = p.parse_args()
    host, port = args.connect.rsplit(":", 1)
    with socket.create_connection((host, int(port))) as sock:
        start = time.monotonic()
        while (t := time.monotonic() - start) < args.duration:
            sock.sendall((json.dumps({"t": time.time(), "hands": scenario(t)}) + "\n").encode())
            time.sleep(1 / args.fps)


if __name__ == "__main__":
    main()
