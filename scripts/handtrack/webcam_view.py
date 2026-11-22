import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

from hand_features import describe, is_offering

vision = mp.tasks.vision
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
OFFERING, OTHER, JOINT, TEXT = (0, 200, 0), (0, 140, 255), (255, 255, 255), (255, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    parser = argparse.ArgumentParser(description="Show MediaPipe's hand skeleton on a webcam. No robot needed.")
    parser.add_argument("--model", type=Path, required=True, help="hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    landmarker = create_landmarker(args.model)
    last_ms = -1
    while True:
        ok, bgr = capture.read()
        if not ok:
            print(f"no picture from camera {args.camera}; try --camera 1")
            break
        last_ms = max(int(time.monotonic() * 1000), last_ms + 1)
        started = time.perf_counter()
        result = detect(landmarker, bgr, last_ms)
        annotate(bgr, result, 1000 * (time.perf_counter() - started))
        cv2.imshow("hand skeleton (q quits)", bgr)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    capture.release()
    cv2.destroyAllWindows()


def create_landmarker(model: Path):
    # same settings as hand_tracker.py, so the picture shows what the robot side would get
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))


def detect(landmarker, bgr, timestamp_ms):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)


def annotate(image, result, took_ms):
    height, width = image.shape[:2]
    for index, landmarks in enumerate(result.hand_landmarks):
        points = [(round(lm.x * width), round(lm.y * height)) for lm in landmarks]
        draw_hand(image, points, describe(result, index, width, height))
    status = f"{took_ms:.0f} ms per frame, {len(result.hand_landmarks)} hand(s); green = open palm up (would be sent)"
    cv2.putText(image, status, (10, 22), FONT, 0.5, TEXT, 1, cv2.LINE_AA)


def draw_hand(image, points, hand):
    colour = OFFERING if is_offering(hand) else OTHER
    for start, end in CONNECTIONS:
        cv2.line(image, points[start], points[end], colour, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 3, JOINT, -1, cv2.LINE_AA)
    centre = tuple(round(c) for c in hand["palm_centre_px"])
    cv2.drawMarker(image, centre, colour, cv2.MARKER_CROSS, 18, 2)
    side = "palm" if hand["palm_facing_camera"] else "back"
    label = f"{hand['handedness']} {hand['handedness_score']:.2f}, {side}, {len(hand['extended'])} fingers straight"
    cv2.putText(image, label, (centre[0] + 12, centre[1]), FONT, 0.5, colour, 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
