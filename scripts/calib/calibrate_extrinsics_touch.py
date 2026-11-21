import argparse
from pathlib import Path

import cv2
import numpy as np

from common import CONFIG_PATH, QUIT_KEYS, WORK_DIR, LatestFrame, Window, camera_settings, draw_text, load_config, open_camera, to_point
from joints import JOINTS_FILE, ask_joints, read_joints_file, tool_point_in_base
from pixel_to_base import (EXTRINSICS_PATH, INTRINSICS_PATH, CALIB_DIR, TopCamera, from_rvec_tvec, invert, pose_difference,
                           read_yaml, timestamp, write_yaml)

PAIRS_PATH = WORK_DIR / "touch_pairs.yaml"
TOUCH_EXTRINSICS_PATH = CALIB_DIR / "extrinsics_touch.yaml"
COMPARISON_PIXELS = [(u, v) for u in (80, 240, 400, 560) for v in (60, 180, 300, 420)]
# Synthetic test, 2026-11-21, 8 points, 2 px clicks, 5 mm tip error: hand-plane error 39-54 mm (median) with all
# points on the table, 7-11 mm with points spread over 0-25 cm in height.
MIN_HEIGHT_SPREAD_M = 0.10


def solve_touch(points_base, pixels, K, dist):
    # Points that lie almost in one plane make every single solver fail now and then (SQPNP returned no pose in a
    # synthetic run); keep the best of several, each refined.
    points, pixels = np.asarray(points_base, float), np.asarray(pixels, float)
    best = None
    for method in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
        try:
            found, rvec, tvec = cv2.solvePnP(points, pixels, K, dist, flags=method)
            rvec, tvec = cv2.solvePnPRefineLM(points, pixels, K, dist, rvec, tvec)
        except cv2.error:
            continue
        errors = np.linalg.norm(cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1, 2) - pixels, axis=1)
        if found and np.isfinite(errors).all() and (best is None or errors.mean() < best[1].mean()):
            best = (invert(from_rvec_tvec(rvec, tvec)), errors)
    if best is None:
        raise RuntimeError("No camera pose fits these points.")
    return best


def complete(pairs):
    return [pair for pair in pairs if pair["pixel"] is not None]


def solve_pairs(pairs, tool_point, intrinsics):
    K, dist = np.array(intrinsics["K"]), np.array(intrinsics["dist"])
    points = np.array([tool_point_in_base(pair["joints"], tool_point) for pair in pairs])
    T_base_cam, errors = solve_touch(points, [pair["pixel"] for pair in pairs], K, dist)
    return T_base_cam, errors, points


def hand_plane_difference_mm(camera_a, camera_b, plane_z):
    differences = [np.linalg.norm(camera_a.pixel_to_base(u, v, plane_z) - camera_b.pixel_to_base(u, v, plane_z))
                   for u, v in COMPARISON_PIXELS]
    return 1000 * np.median(differences), 1000 * np.max(differences)


def report(T_base_cam, errors, points, compare_path, intrinsics, plane_z):
    for index, (point, error) in enumerate(zip(points, errors)):
        print(f"  point {index}: tool point {np.round(point, 3).tolist()} m, reprojection {error:.1f} px")
    print(f"reprojection RMS {np.sqrt(np.mean(errors ** 2)):.1f} px over {len(errors)} points")
    if np.ptp(points[:, 2]) < MIN_HEIGHT_SPREAD_M:
        print(f"WARNING: the points span only {100 * np.ptp(points[:, 2]):.0f} cm in height, which leaves the camera tilt "
              "loose; add points with the gripper held 10-25 cm above the table")
    x, y, z = T_base_cam[:3, 3]
    print(f"camera at x {x:.3f}  y {y:.3f}  z {z:.3f} m")
    if not 0.2 < z < 2.0 or T_base_cam[2, 2] >= 0:
        print("WARNING: the camera does not come out above the table looking down: wrong clicks, wrong joints or kinematics")
    if not Path(compare_path).is_file():
        return
    other = read_yaml(compare_path, "Extrinsics to compare")
    moved_m, turned_deg = pose_difference(T_base_cam, np.array(other["T_base_cam"]))
    size = (intrinsics["image_width"], intrinsics["image_height"])
    median_mm, max_mm = hand_plane_difference_mm(TopCamera(intrinsics["K"], intrinsics["dist"], size, T_base_cam),
                                                 TopCamera(intrinsics["K"], intrinsics["dist"], size, other["T_base_cam"]), plane_z)
    print(f"against {compare_path} ({other['method']}): camera {1000 * moved_m:.0f} mm and {turned_deg:.2f} deg apart; "
          f"at the hand plane z = {plane_z} m the two put a pixel {median_mm:.0f} mm apart (median), {max_mm:.0f} mm at most")
    print("This difference is calibration error plus kinematics error: joint signs and zero offsets in kinematics.py are not "
          "verified on the arm yet, so a large difference can come from either.")


def save(out_path, T_base_cam, errors, pairs, tool_point, intrinsics, pairs_path):
    write_yaml(out_path, {
        "created": timestamp(), "method": "touch", "intrinsics_created": intrinsics["created"],
        "image_width": intrinsics["image_width"], "image_height": intrinsics["image_height"],
        "T_base_cam": T_base_cam, "camera_position_in_base_m": T_base_cam[:3, 3],
        "reprojection_rms_px": float(np.sqrt(np.mean(errors ** 2))), "points": len(pairs),
        "tool_point_m": tool_point, "pairs_file": str(pairs_path),
    })
    print(f"written {out_path}")


class TouchSession:
    def __init__(self, pairs, pairs_path, tool_point, joints_file):
        self.pairs, self.pairs_path, self.tool_point, self.joints_file = pairs, pairs_path, tool_point, joints_file
        self.message = "SPACE: record the joints of a new point"

    def keep(self, message):
        write_yaml(self.pairs_path, {"tool_point_m": self.tool_point, "pairs": self.pairs})
        self.message = message

    def record_joints(self):
        if self.pairs and self.pairs[-1]["pixel"] is None:
            self.message = "click the tool point of the last point first (u undoes it)"
            return
        joints = read_joints_file(self.joints_file) if self.joints_file else ask_joints()
        if joints is None:
            self.message = f"no fresh joints in {self.joints_file}: is joint_feed.py running?"
            return
        self.pairs.append({"joints": joints.round(3).tolist(), "pixel": None})
        self.keep(f"point {len(self.pairs) - 1}: joints recorded, now click the tool point in the image")

    def click(self, u, v):
        if not self.pairs or self.pairs[-1]["pixel"] is not None:
            self.message = "press SPACE first to record the joints"
            return
        self.pairs[-1]["pixel"] = [round(u, 1), round(v, 1)]
        self.keep(f"point {len(self.pairs) - 1} complete. SPACE: next point, s: solve and save")

    def undo(self):
        if self.pairs:
            self.pairs.pop()
            self.keep("last point removed")

    def draw(self, frame, min_points):
        view = frame.copy()
        for index, pair in enumerate(complete(self.pairs)):
            cv2.circle(view, to_point(pair["pixel"]), 4, (0, 255, 0), -1)
            cv2.putText(view, str(index), to_point(np.add(pair["pixel"], (6, -6))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return draw_text(view, [f"points: {len(complete(self.pairs))} (need {min_points})   u: undo   s: solve and save   q: quit",
                                self.message])


def main():
    parser = argparse.ArgumentParser(description="Camera pose from gripper touches (method B): record the joints at each "
                                                 "touched point, click the tool point in the image.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--out", type=Path, default=TOUCH_EXTRINSICS_PATH)
    parser.add_argument("--compare", type=Path, default=EXTRINSICS_PATH, help="extrinsics to compare with (method A)")
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH, help="joints and clicks, kept after every change")
    parser.add_argument("--joints-file", type=Path, nargs="?", const=JOINTS_FILE,
                        help="read the joints from joint_feed.py's file instead of typing them")
    parser.add_argument("--solve-only", action="store_true", help="solve from the saved pairs, no camera")
    parser.add_argument("--camera", help="index or device path, instead of the config's")
    parser.add_argument("--scale", type=float, default=1.5, help="window size factor, for precise clicks")
    args = parser.parse_args()

    config = load_config(args.config)
    intrinsics = read_yaml(args.intrinsics, "Intrinsics")
    tool_point, min_points = config["touch"]["tool_point_m"], config["touch"]["min_points"]
    plane_z = config["validate"]["hand_plane_z_m"]
    pairs = read_yaml(args.pairs, "Pairs")["pairs"] if args.pairs.is_file() else []

    def solve_and_save():
        usable = complete(pairs)
        if len(usable) < min_points:
            print(f"{len(usable)} points, need {min_points}")
            return
        T_base_cam, errors, points = solve_pairs(usable, tool_point, intrinsics)
        report(T_base_cam, errors, points, args.compare, intrinsics, plane_z)
        save(args.out, T_base_cam, errors, usable, tool_point, intrinsics, args.pairs)

    if args.solve_only:
        solve_and_save()
        return
    print(f"{len(pairs)} pairs loaded from {args.pairs}" if pairs else f"new pairs file {args.pairs}")
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    session = TouchSession(pairs, args.pairs, tool_point, args.joints_file)
    camera = LatestFrame(open_camera(camera_settings(config, args.camera)))
    window = Window("touch calibration", args.scale, on_click=session.click)
    number = 0
    try:
        while True:
            frame, number = camera.latest(after=number)
            key = window.show(session.draw(frame, min_points))
            if key in QUIT_KEYS:
                break
            if key == ord(" "):
                session.record_joints()
            elif key == ord("u"):
                session.undo()
            elif key == ord("s"):
                solve_and_save()
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
