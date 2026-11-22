import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handtrack"))
from hand_features import describe, is_pointing  # noqa: E402

DOWNLOADS = Path("C:/Users/Tian/Desktop/lerobot-hackathon/data/handtrack/downloads")
WIDTH, HEIGHT = 640, 480
KNUCKLES = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
FINGER_STARTS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def finger_chain(base_x, straight):
    turn = 0.0 if straight else np.radians(75)
    heading, point, chain = np.radians(90), np.array([base_x, 0.0, 0.0]), []
    for _ in range(4):
        chain.append(point.copy())
        point = point + 0.03 * np.array([np.cos(heading), np.sin(heading), 0.0])
        heading += turn
    return chain


def synthetic_points(straight):
    points = np.zeros((21, 3))
    for name, base_x in KNUCKLES.items():
        chain = finger_chain(base_x, name in straight)
        start = FINGER_STARTS[name]
        points[start:start + 4] = chain
    thumb = finger_chain(-0.05, "thumb" in straight)
    points[1:5] = thumb
    return points


def fake_result(straight, handedness="Right"):
    world = synthetic_points(straight)
    landmarks = [SimpleNamespace(x=0.5 + p[0] * 2, y=0.8 - p[1] * 2, z=0.0) for p in world]
    world_landmarks = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in world]
    category = SimpleNamespace(category_name=handedness, score=0.99)
    return SimpleNamespace(hand_landmarks=[landmarks], hand_world_landmarks=[world_landmarks], handedness=[[category]])


def described(straight):
    return describe(fake_result(straight), 0, WIDTH, HEIGHT)


def mediapipe_pointing_photo():
    photo = DOWNLOADS / "images" / "pointing_up.jpg"
    model = DOWNLOADS / "models" / "hand_landmarker.task"
    if not photo.exists() or not model.exists():
        return None
    import mediapipe as mp

    vision = mp.tasks.vision
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)), num_hands=1))
    image = mp.Image.create_from_file(str(photo))
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return False
    return is_pointing(describe(result, 0, image.width, image.height))


def main():
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {got}" + ("" if ok else f" (want {want})"), flush=True)
        if not ok:
            failures.append(name)

    check("pointing: index straight, others curled", is_pointing(described({"index"})), True)
    check("pointing with the thumb straight", is_pointing(described({"index", "thumb"})), True)
    check("open palm is not pointing", is_pointing(described({"index", "middle", "ring", "pinky", "thumb"})), False)
    check("fist is not pointing", is_pointing(described(set())), False)
    check("victory is not pointing", is_pointing(described({"index", "middle"})), False)
    check("curled index is not pointing", is_pointing(described({"middle"})), False)

    hand = described({"index"})
    result = fake_result({"index"})
    tip = result.hand_landmarks[0][8]
    check("index_tip_px is landmark 8 in pixels",
          tuple(np.round(hand["index_tip_px"], 3)), (round(tip.x * WIDTH, 3), round(tip.y * HEIGHT, 3)))
    check("index_direction_px is a unit vector", round(float(np.linalg.norm(hand["index_direction_px"])), 6), 1.0)
    check("index_direction_px points from knuckle to tip", bool(hand["index_direction_px"][1] < 0), True)
    check("straightness has the four fingers", sorted(hand["straightness"]), ["index", "middle", "pinky", "ring"])
    check("index is straight and the middle finger is curled",
          (hand["straightness"]["index"] > 0.9, hand["straightness"]["middle"] < 0.75), (True, True))

    photo = mediapipe_pointing_photo()
    if photo is None:
        print("SKIP  MediaPipe on pointing_up.jpg (photo or model not downloaded)")
    else:
        check("MediaPipe sees pointing_up.jpg as pointing", photo, True)

    print("PASS" if not failures else f"FAIL: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
