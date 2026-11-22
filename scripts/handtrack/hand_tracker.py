import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

from frame_link import SHAPE, FrameReader
from hand_features import describe, is_offering, is_pointing
from hand_link import HandLineSender

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calib"))
from common import camera_settings, load_config, open_camera  # noqa: E402
from pixel_to_base import TopCamera  # noqa: E402

vision = mp.tasks.vision
FRAME_TIMEOUT_S = 2.0


# Without the robot program the tracker opens the top camera itself, the same way the calibration did, so the
# hand positions can be checked while the arm stays still.
class CameraFrames:
    def __init__(self, camera):
        self.capture = open_camera(camera_settings(load_config(), camera or None))

    def latest(self):
        ok, bgr = self.capture.read()
        return (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.perf_counter_ns()) if ok else None

    def close(self):
        self.capture.release()


def load_geometry(intrinsics, extrinsics):
    if intrinsics is None or extrinsics is None:
        return None
    return TopCamera.load(intrinsics, extrinsics, image_size=(SHAPE[1], SHAPE[0]))


def pixel_message(hand):
    u, v = (round(float(c), 1) for c in hand["palm_centre_px"])
    tip_u, tip_v = (round(float(c), 1) for c in hand["index_tip_px"])
    return {"u": u, "v": v, "palm_up": hand["palm_facing_camera"], "score": hand["handedness_score"],
            "open": is_offering(hand), "handedness": hand["handedness"],
            "pointing": is_pointing(hand), "tip_u": tip_u, "tip_v": tip_v}


def fingertip_message(hand, geometry, finger_z):
    xyz = geometry.pixel_to_base(*hand["index_tip_px"], finger_z)
    if xyz is None:
        return {}
    return {f"tip_{axis}": round(float(c), 4) for axis, c in zip("xyz", xyz)}


def hand_message(hand, geometry, hand_z, finger_z):
    xyz = geometry.pixel_to_base(*hand["palm_centre_px"], hand_z)
    if xyz is None:
        return None
    x, y, z = (round(float(c), 4) for c in xyz)
    return {"x": x, "y": y, "z": z, **pixel_message(hand), **fingertip_message(hand, geometry, finger_z)}


def frame_message(hands, geometry, hand_z, finger_z, captured_epoch):
    # Without calibration the hands go under "pixels" and "hands" stays empty: the robot side turns
    # null coordinates into NaN (np.array([None], dtype=float)) instead of rejecting them.
    if geometry is None:
        return {"t": captured_epoch, "hands": [], "pixels": [pixel_message(h) for h in hands]}
    located = [hand_message(h, geometry, hand_z, finger_z) for h in hands]
    return {"t": captured_epoch, "hands": [h for h in located if h is not None]}


def summary(message):
    hands = message["hands"] or message.get("pixels") or []
    if not hands:
        return "no hand"
    return " | ".join(
        (f"x {h['x']:+.3f} y {h['y']:+.3f} m" if "x" in h else f"pixel ({h['u']:.0f}, {h['v']:.0f})")
        + (" open palm" if h["open"] else " pointing" if h["pointing"] else " not an open palm") for h in hands)


def open_frames(name):
    # The robot program creates the block, so the tracker may start first; wait instead of failing.
    announced = False
    while True:
        try:
            return FrameReader(name)
        except FileNotFoundError:
            if not announced:
                print(f"waiting for frames from the robot program ({name}) ...", flush=True)
                announced = True
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shm-name", default="top_camera_frames")
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--extrinsics", type=Path)
    parser.add_argument("--hand-z", type=float, default=0.15, help="assumed palm height above the table, metres")
    parser.add_argument("--finger-z", type=float, default=0.15, help="assumed fingertip height above the table, metres")
    parser.add_argument("--only-open", action="store_true", help="send only hands that are open with the palm up")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seconds", type=float, default=0, help="stop after this long; 0 runs until Ctrl+C")
    parser.add_argument("--camera", nargs="?", const="", help="open the top camera directly instead of the robot "
                        "program's frames (robot program must be stopped); no value = the path in calib/config.yaml")
    parser.add_argument("--print-every", type=float, default=0.5, help="seconds between printed hand positions; 0 = off")
    args = parser.parse_args()

    geometry = load_geometry(args.intrinsics, args.extrinsics)
    sender = HandLineSender(port=args.port)
    own_camera = args.camera is not None
    reader = CameraFrames(args.camera) if own_camera else open_frames(args.shm_name)
    last_frame = last_print = time.monotonic()
    # VIDEO mode, not LIVE_STREAM: the loop always takes the newest frame itself, so nothing queues up.
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    last_ms, end = -1, time.perf_counter() + (args.seconds or float("inf"))
    while time.perf_counter() < end:
        got = reader.latest()
        if got is None:
            if own_camera:
                sys.exit("The camera stopped delivering frames (in WSL: re-attach it with scripts/so101/attach_usb.sh).")
            # a restarted robot program writes into a new block; the one held here never gets another frame
            if time.monotonic() - last_frame > FRAME_TIMEOUT_S:
                reader.close()
                reader, last_frame = open_frames(args.shm_name), time.monotonic()
            time.sleep(0.002)
            continue
        last_frame = time.monotonic()
        rgb, captured_ns = got
        last_ms = max(captured_ns // 1_000_000, last_ms + 1)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms)
        hands = [describe(result, i, rgb.shape[1], rgb.shape[0]) for i in range(len(result.hand_landmarks))]
        if args.only_open:
            hands = [h for h in hands if is_offering(h)]
        captured_epoch = round(time.time() - (time.perf_counter_ns() - captured_ns) / 1e9, 3)
        message = frame_message(hands, geometry, args.hand_z, args.finger_z, captured_epoch)
        sender.send(message)
        if args.print_every and time.monotonic() - last_print >= args.print_every:
            print(summary(message), flush=True)
            last_print = time.monotonic()


if __name__ == "__main__":
    main()
