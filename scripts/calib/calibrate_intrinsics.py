import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from board import Board, detect_grid, object_points
from capture_intrinsics import INTRINSICS_FRAMES_DIR
from common import CONFIG_PATH, QUIT_KEYS, Coverage, LatestFrame, Window, camera_settings, draw_text, load_config, open_camera
from pixel_to_base import INTRINSICS_PATH, read_yaml, timestamp, write_yaml

OUTLIER_FACTOR = 2.0
# Below this a view is kept whatever the median: with very clean views 2x the median would drop good ones.
OUTLIER_FLOOR_PX = 0.25
GOOD_RMS_PX, BAD_RMS_PX = 0.5, 1.0
MIN_VIEWS = 10


def detect_views(paths, board, image_size):
    views, skipped = {}, []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if (image.shape[1], image.shape[0]) != image_size:
            sys.exit(f"{path.name} is {image.shape[1]}x{image.shape[0]}, the config says {image_size[0]}x{image_size[1]}. "
                     "Intrinsics are valid only at the runtime size; capture the frames again at that size.")
        grid = detect_grid(image, board)
        if grid is None:
            skipped.append(path.name)
        else:
            views[path.name] = grid
    return views, skipped


def calibrate(views, board, image_size):
    names = list(views)
    points = [object_points(board).astype(np.float32) for _ in names]
    corners = [views[name].reshape(-1, 1, 2).astype(np.float32) for name in names]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(points, corners, image_size, None, None)
    errors = {}
    for name, p, c, rvec, tvec in zip(names, points, corners, rvecs, tvecs):
        projected = cv2.projectPoints(p, rvec, tvec, K, dist)[0]
        errors[name] = float(np.sqrt(np.mean(np.sum((projected - c) ** 2, axis=2))))
    return float(rms), K, dist.ravel(), errors


def outliers(errors):
    limit = max(OUTLIER_FACTOR * float(np.median(list(errors.values()))), OUTLIER_FLOOR_PX)
    return [name for name, error in errors.items() if error > limit]


def warnings_for(rms, errors, dropped, total, coverage):
    warnings = []
    if rms > BAD_RMS_PX:
        warnings.append(f"RMS {rms:.2f} px is above {BAD_RMS_PX} px: do not use this calibration, capture again.")
    elif rms > GOOD_RMS_PX:
        warnings.append(f"RMS {rms:.2f} px is usable but above the {GOOD_RMS_PX} px target.")
    if len(dropped) > 0.2 * total or outliers(errors):
        warnings.append("Per-image errors are inconsistent, as a bent or wavy board makes them. "
                        "Check the board lies flat on its backing; if not, fix it and capture again.")
    if len(errors) < 15:
        warnings.append(f"Only {len(errors)} images used; 20-30 are better.")
    if coverage.empty_cells():
        warnings.append(f"{coverage.empty_cells()} of {coverage.views.size} image cells were never covered by the board; "
                        "add frames with the board in the image corners and edges.")
    return warnings


def run_calibration(images_dir, out_path, config):
    board = Board.from_config(config)
    image_size = (config["camera"]["width"], config["camera"]["height"])
    paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    views, skipped = detect_views(paths, board, image_size)
    if skipped:
        print(f"board not found in {len(skipped)} images, not used: {', '.join(skipped)}")
    if len(views) < MIN_VIEWS:
        sys.exit(f"Only {len(views)} images with the board in {images_dir}; need at least {MIN_VIEWS}.")

    first_rms, *_, first_errors = calibrate(views, board, image_size)
    dropped = outliers(first_errors)
    kept = {name: grid for name, grid in views.items() if name not in dropped}
    rms, K, dist, errors = calibrate(kept, board, image_size)
    coverage = Coverage(*image_size)
    for grid in kept.values():
        coverage.add(grid)

    print(f"\nfirst pass: {len(views)} images, RMS {first_rms:.3f} px")
    for name in views:
        note = "DROPPED" if name in dropped else f"{errors[name]:.3f} px after dropping"
        print(f"  {name}: {first_errors[name]:.3f} px   {note}")
    print(f"final: {len(kept)} images, RMS {rms:.3f} px")
    print(f"K: fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
    print(f"dist (k1 k2 p1 p2 k3): {np.round(dist, 4).tolist()}")
    warnings = warnings_for(rms, errors, dropped, len(views), coverage)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if rms <= GOOD_RMS_PX and not warnings:
        print("GOOD")

    write_yaml(out_path, {
        "created": timestamp(),
        "image_width": image_size[0], "image_height": image_size[1],
        "K": K, "dist": dist, "rms_px": rms,
        "images_used": len(kept), "images_dropped": dropped,
        "per_image_error_px": {name: round(error, 4) for name, error in errors.items()},
        "square_x_m": board.square_x_m, "square_y_m": board.square_y_m,
        "inner_corners": [board.long_corners, board.short_corners],
        "camera": {key: config["camera"][key] for key in ("index_or_path", "fourcc", "fps")},
        "warnings": warnings,
    })
    print(f"written {out_path}")
    return rms, K, dist, errors, dropped


def undistort_maps(intrinsics):
    K, dist = np.array(intrinsics["K"]), np.array(intrinsics["dist"])
    size = (intrinsics["image_width"], intrinsics["image_height"])
    return cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_16SC2)


def draw_guides(image, step=80):
    for x in range(step, image.shape[1], step):
        cv2.line(image, (x, 0), (x, image.shape[0]), (255, 255, 0), 1)
    for y in range(step, image.shape[0], step):
        cv2.line(image, (0, y), (image.shape[1], y), (255, 255, 0), 1)
    return image


def preview(intrinsics_path, settings, scale):
    intrinsics = read_yaml(intrinsics_path, "Intrinsics")
    if (settings["width"], settings["height"]) != (intrinsics["image_width"], intrinsics["image_height"]):
        sys.exit("The config's camera size differs from the size the intrinsics were made at.")
    map_x, map_y = undistort_maps(intrinsics)
    camera = LatestFrame(open_camera(settings))
    window = Window("raw | undistorted: straight edges must look straight on the right", scale)
    number = 0
    try:
        while True:
            frame, number = camera.latest(after=number)
            undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            side_by_side = np.hstack([draw_text(draw_guides(frame), ["raw"]), draw_text(draw_guides(undistorted), ["undistorted   q quits"])])
            if window.show(side_by_side) in QUIT_KEYS:
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Intrinsic calibration from the frames saved by capture_intrinsics.py.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--images", type=Path, default=INTRINSICS_FRAMES_DIR)
    parser.add_argument("--out", type=Path, default=INTRINSICS_PATH, help="intrinsics file to write (read with --preview)")
    parser.add_argument("--preview", action="store_true", help="live raw and undistorted view side by side instead")
    parser.add_argument("--camera", help="index or device path for --preview, instead of the config's")
    parser.add_argument("--scale", type=float, default=1.0, help="window size factor for --preview")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preview:
        preview(args.out, camera_settings(config, args.camera), args.scale)
    else:
        run_calibration(args.images, args.out, config)


if __name__ == "__main__":
    main()
