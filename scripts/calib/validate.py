import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import CONFIG_PATH, QUIT_KEYS, LatestFrame, Window, camera_settings, draw_text, load_config, open_camera, to_point
from joints import JOINTS_FILE, ARM_JOINTS, ask_joints, parse_joints, read_joints_file, tool_point_in_base
from pixel_to_base import EXTRINSICS_PATH, INTRINSICS_PATH, TopCamera

HAND_LANDMARKS = {"wrist": 0, "index tip": 8}
HAND_PRINT_INTERVAL_S = 0.5


def grid_points(settings):
    xs = np.arange(settings["grid_x_m"][0], settings["grid_x_m"][1] + 1e-9, settings["grid_step_m"])
    ys = np.arange(settings["grid_y_m"][0], settings["grid_y_m"][1] + 1e-9, settings["grid_step_m"])
    return np.array([(x, y, 0.0) for x in xs for y in ys])


def draw_grid(view, camera, points):
    for point, pixel in zip(points, camera.base_to_pixel(points)):
        if np.isfinite(pixel).all():
            on_ten_cm = np.allclose(np.round(point[:2] * 10) / 10, point[:2])
            cv2.circle(view, to_point(pixel), 3 if on_ten_cm else 2, (0, 255, 255) if on_ten_cm else (0, 160, 160), -1)
            if on_ten_cm:
                cv2.putText(view, f"{point[0]:.1f},{point[1]:.1f}", to_point(pixel + (4, -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


def draw_axes(view, camera):
    base = camera.base_to_pixel([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    for end, colour in zip(base[1:], ((0, 0, 255), (0, 200, 0), (255, 0, 0))):
        if np.isfinite([base[0], end]).all():
            cv2.arrowedLine(view, to_point(base[0]), to_point(end), colour, 2, tipLength=0.1)


def draw_tip(view, camera, tip):
    tip_pixel, below_pixel = camera.base_to_pixel([tip, (tip[0], tip[1], 0.0)])
    if np.isfinite([tip_pixel, below_pixel]).all():
        cv2.line(view, to_point(tip_pixel), to_point(below_pixel), (255, 0, 255), 1)
        cv2.circle(view, to_point(below_pixel), 3, (255, 0, 255), -1)
        cv2.circle(view, to_point(tip_pixel), 6, (255, 0, 255), 2)
        cv2.putText(view, f"tip z {tip[2]:.3f}", to_point(tip_pixel + (8, 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)


class HandProbe:
    def __init__(self, model_path, camera, hand_z):
        import mediapipe as mp

        self.mp, self.camera, self.hand_z, self.last_print = mp, camera, hand_z, 0.0
        vision = mp.tasks.vision
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.timestamp_ms = 0

    def run(self, frame, view):
        self.timestamp_ms += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp_ms)
        lines = []
        for hand in result.hand_landmarks:
            for name, index in HAND_LANDMARKS.items():
                u, v = hand[index].x * frame.shape[1], hand[index].y * frame.shape[0]
                xyz = self.camera.pixel_to_base(u, v, self.hand_z)
                cv2.circle(view, to_point((u, v)), 5, (0, 128, 255), -1)
                lines.append(f"{name} px ({u:.0f}, {v:.0f}) -> base {np.round(xyz, 3).tolist() if xyz is not None else None}")
        if lines and time.monotonic() - self.last_print > HAND_PRINT_INTERVAL_S:
            print(f"hands at z = {self.hand_z} m: " + "; ".join(lines))
            self.last_print = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description="Live check of the calibration: table grid, robot axes, the gripper's "
                                                 "tool point from the joints; click a pixel to print its robot-frame position.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--extrinsics", type=Path, default=EXTRINSICS_PATH)
    parser.add_argument("--camera", help="index or device path, instead of the config's")
    parser.add_argument("--joints-file", type=Path, nargs="?", const=JOINTS_FILE, help="live joints from joint_feed.py")
    parser.add_argument("--joints", help="fixed joints instead: a line from read_pose.py or five values in degrees")
    parser.add_argument("--plane-z", type=float, default=0.0, help="height of the plane a click is intersected with, m")
    parser.add_argument("--hand-model", type=Path, help="hand_landmarker.task: also print wrist and index tip in the robot frame")
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = camera_settings(config, args.camera)
    camera = TopCamera.load(args.intrinsics, args.extrinsics, image_size=(settings["width"], settings["height"]))
    hand_z = config["validate"]["hand_plane_z_m"]
    tool_point = config["touch"]["tool_point_m"]
    points = grid_points(config["validate"])
    fixed_joints = parse_joints(args.joints) if args.joints else None
    probe = HandProbe(args.hand_model, camera, hand_z) if args.hand_model else None

    def on_click(u, v):
        for z in (args.plane_z, hand_z):
            xyz = camera.pixel_to_base(u, v, z)
            print(f"pixel ({u:.1f}, {v:.1f}) at z = {z} m -> {np.round(xyz, 4).tolist() if xyz is not None else 'no intersection'}")

    stream = LatestFrame(open_camera(settings))
    window = Window("validate: q quits, j types joints", args.scale, on_click=on_click)
    number = 0
    try:
        while True:
            frame, number = stream.latest(after=number)
            view = frame.copy()
            draw_grid(view, camera, points)
            draw_axes(view, camera)
            joints = read_joints_file(args.joints_file) if args.joints_file else fixed_joints
            if joints is not None:
                draw_tip(view, camera, tool_point_in_base(joints, tool_point))
            if probe:
                probe.run(frame, view)
            draw_text(view, [f"grid every {config['validate']['grid_step_m'] * 100:.0f} cm on the table, labels every 10 cm",
                             "tool point: magenta ring, its foot on the table: dot" if joints is not None else
                             f"no joints ({'is joint_feed.py running?' if args.joints_file else 'press j to type them'})"])
            key = window.show(view)
            if key in QUIT_KEYS:
                break
            if key == ord("j") and not args.joints_file:
                fixed_joints = ask_joints()
                print("tool point:", np.round(tool_point_in_base(fixed_joints, tool_point), 4).tolist(), "m;",
                      ", ".join(f"{n} {v:.1f}" for n, v in zip(ARM_JOINTS, fixed_joints)))
    finally:
        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
