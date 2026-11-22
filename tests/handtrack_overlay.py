import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handtrack"))
from hand_features import PALM, describe  # noqa: E402
from overlay_link import OverlaySender, overlay_message  # noqa: E402

WIDTH, HEIGHT = 640, 480
# Right hand seen palm-up from above: index knuckle on the image's right of the pinky knuckle, fingers pointing up.
KNUCKLES = {5: 0.03, 9: 0.01, 13: -0.01, 17: -0.03, 1: 0.05}
NAMES = {5: "index", 9: "middle", 13: "ring", 17: "pinky", 1: "thumb"}


def chain(base_x, straight):
    turn = 0.0 if straight else np.radians(75)
    heading, point, points = np.radians(90), np.array([base_x, 0.0, 0.0]), []
    for _ in range(4):
        points.append(point.copy())
        point = point + 0.03 * np.array([np.cos(heading), np.sin(heading), 0.0])
        heading += turn
    return points


def fake_result(straight):
    world = np.zeros((21, 3))
    world[0] = [0.0, -0.05, 0.0]
    for start, base_x in KNUCKLES.items():
        world[start:start + 4] = chain(base_x, NAMES[start] in straight)
    landmarks = [SimpleNamespace(x=np.float32(0.5 + p[0] * 2), y=np.float32(0.6 - p[1] * 2), z=0.0) for p in world]
    world_landmarks = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in world]
    category = SimpleNamespace(category_name="Right", score=0.98)
    return SimpleNamespace(hand_landmarks=[landmarks], hand_world_landmarks=[world_landmarks], handedness=[[category]])


def built(straight):
    result = fake_result(straight)
    hands = [describe(result, 0, WIDTH, HEIGHT)]
    return result, overlay_message(1700000000.123, WIDTH, HEIGHT, result.hand_landmarks, hands)


def main():
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {got}" + ("" if ok else f" (want {want})"), flush=True)
        if not ok:
            failures.append(name)

    result, message = built({"index", "middle", "ring", "pinky", "thumb"})
    hand = message["hands"][0]
    points = hand["landmarks"]
    check("frame fields", (message["t"], message["width"], message["height"]), (1700000000.123, WIDTH, HEIGHT))
    check("21 landmarks of two coordinates", (len(points), {len(p) for p in points}), (21, {2}))
    check("landmarks within 0..1", all(0 <= c <= 1 for p in points for c in p), True)
    check("landmarks rounded to 3 decimals", all(c == round(c, 3) for p in points for c in p), True)
    check("landmarks are plain floats (JSON)", json.loads(json.dumps(message)) == message, True)
    palm = np.mean([[result.hand_landmarks[0][i].x, result.hand_landmarks[0][i].y] for i in PALM], axis=0)
    check("palm_centre is the palm's pixel centre, normalised",
          bool(np.allclose(hand["palm_centre"], palm, atol=0.001)), True)
    check("open palm up is offering", (hand["handedness"], hand["offering"]), ("Right", True))
    check("fist is not offering", built(set())[1]["hands"][0]["offering"], False)
    check("no hands, empty list", overlay_message(1.0, WIDTH, HEIGHT, [], [])["hands"], [])

    check("port 0 sends nothing", OverlaySender(0).send(message), False)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        sent = OverlaySender(receiver.getsockname()[1]).send(message)
        check("sender delivers one datagram", (sent, json.loads(receiver.recv(65536))), (True, message))
    check("a port nobody listens on does not raise", OverlaySender(9).send(message) in (True, False), True)

    print("PASS" if not failures else f"FAIL: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
