import argparse
import sys
import time
from pathlib import Path

import mediapipe as mp

from frame_link import SHAPE, FrameReader
from hand_features import describe, is_offering
from hand_link import HandLineSender

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calib"))
from pixel_to_base import TopCamera  # noqa: E402

vision = mp.tasks.vision


def load_geometry(intrinsics, extrinsics):
    if intrinsics is None or extrinsics is None:
        return None
    return TopCamera.load(intrinsics, extrinsics, image_size=(SHAPE[1], SHAPE[0]))


def pixel_message(hand):
    u, v = (round(float(c), 1) for c in hand["palm_centre_px"])
    return {"u": u, "v": v, "palm_up": hand["palm_facing_camera"], "score": hand["handedness_score"],
            "open": is_offering(hand), "handedness": hand["handedness"]}


def hand_message(hand, geometry, hand_z):
    xyz = geometry.pixel_to_base(*hand["palm_centre_px"], hand_z)
    if xyz is None:
        return None
    x, y, z = (round(float(c), 4) for c in xyz)
    return {"x": x, "y": y, "z": z, **pixel_message(hand)}


def frame_message(hands, geometry, hand_z, captured_epoch):
    # Without calibration the hands go under "pixels" and "hands" stays empty: the robot side turns
    # null coordinates into NaN (np.array([None], dtype=float)) instead of rejecting them.
    if geometry is None:
        return {"t": captured_epoch, "hands": [], "pixels": [pixel_message(h) for h in hands]}
    located = [hand_message(h, geometry, hand_z) for h in hands]
    return {"t": captured_epoch, "hands": [h for h in located if h is not None]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shm-name", default="top_camera_frames")
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--extrinsics", type=Path)
    parser.add_argument("--hand-z", type=float, default=0.15, help="assumed palm height above the table, metres")
    parser.add_argument("--only-open", action="store_true", help="send only hands that are open with the palm up")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seconds", type=float, default=0, help="stop after this long; 0 runs until Ctrl+C")
    args = parser.parse_args()

    geometry = load_geometry(args.intrinsics, args.extrinsics)
    sender = HandLineSender(port=args.port)
    reader = FrameReader(args.shm_name)
    # VIDEO mode, not LIVE_STREAM: the loop always takes the newest frame itself, so nothing queues up.
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    last_ms, end = -1, time.perf_counter() + (args.seconds or float("inf"))
    while time.perf_counter() < end:
        got = reader.latest()
        if got is None:
            time.sleep(0.002)
            continue
        rgb, captured_ns = got
        last_ms = max(captured_ns // 1_000_000, last_ms + 1)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms)
        hands = [describe(result, i, rgb.shape[1], rgb.shape[0]) for i in range(len(result.hand_landmarks))]
        if args.only_open:
            hands = [h for h in hands if is_offering(h)]
        captured_epoch = round(time.time() - (time.perf_counter_ns() - captured_ns) / 1e9, 3)
        sender.send(frame_message(hands, geometry, args.hand_z, captured_epoch))


if __name__ == "__main__":
    main()
