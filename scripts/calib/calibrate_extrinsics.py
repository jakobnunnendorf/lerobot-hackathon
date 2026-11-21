import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from board import Board, align_to, board_pose_in_camera, detect_grid, faces_camera, orientations
from common import CONFIG_PATH, LatestFrame, Window, camera_settings, draw_text, load_config, open_camera, to_point
from pixel_to_base import (EXTRINSICS_PATH, INTRINSICS_PATH, TopCamera, invert, read_yaml, rotation_z, timestamp,
                           to_rvec_tvec, transform, write_yaml)

CORNERS = (1, 2, 3, 4)
# Directions in the image, not positions: "the end nearest that image corner" picked the wrong end for a board lying
# in the middle of the image (synthetic test, 2026-11-21); "the end towards that corner, seen from the board's
# centre" works wherever the board lies.
IMAGE_CORNER_DIRECTIONS = {"top-left": (-1, -1), "top-right": (1, -1), "bottom-left": (-1, 1), "bottom-right": (1, 1)}
AMBIGUOUS_FRACTION = 0.15
RED_HSV_RANGES = (((0, 120, 120), (12, 255, 255)), ((168, 120, 120), (180, 255, 255)))
# The arm is orange, so the tape is looked for only where it can be: in a window around the point diagonally outside
# each pattern end where the tape's centre sits on board.png (2.1 squares along the long side, 1.6 along the short
# side), and only as a patch lying wholly in such a window. Searching everything within three squares of the board took
# an orange patch next to the robot base for the tape when the board lay near the base (synthetic test, 2026-11-21).
TAPE_OFFSET_SQUARES = (2.1, 1.6)
TAPE_WINDOW_RADIUS_SQUARES = 1.2
TAPE_INSIDE_FRACTION = 0.9
TAPE_MIN_AREA_PX, TAPE_AREA_RATIO, TAPE_DISTANCE_RATIO = 40, 3.0, 1.5
# The lens position is measured to +-5 cm, so it decides only when the two candidates put the camera this much
# further apart from it.
CAMERA_VOTE_MARGIN_M = 0.10
CAMERA_MISMATCH_WARN_M = 0.10
CAMERA_HEIGHT_RANGE_M = (0.2, 2.0)
CORNER_RESIDUAL_WARN_M = 0.005
FRAME_SPREAD_WARN_PX = 0.5


class CalibrationError(RuntimeError):
    pass


@dataclass
class MeasuredBoard:
    corner1_m: np.ndarray
    yaw_deg: float
    z_up: bool | None
    residuals_m: dict | None
    description: str
    hint: str | None = None

    def T_base_board(self, z_up):
        # Board frame: origin corner 1, x towards corner 2, y towards corner 4, z = x cross y, which points up or
        # down depending on which board corner carries the tape.
        flip = 1.0 if z_up else -1.0
        return transform(rotation_z(self.yaw_deg) @ np.diag([1.0, flip, flip]), self.corner1_m)


@dataclass
class Candidate:
    grid: np.ndarray
    T_cam_board: np.ndarray
    rms_px: float
    T_base_cam: np.ndarray | None = None

    @property
    def z_up(self):
        return faces_camera(self.T_cam_board)

    def corner_pixel(self, number):
        return {1: self.grid[0, 0], 2: self.grid[0, -1], 3: self.grid[-1, -1], 4: self.grid[-1, 0]}[number]


def corner_model(board, z_up):
    length_x = (board.long_corners - 1) * board.square_x_m
    length_y = (board.short_corners - 1) * board.square_y_m * (1.0 if z_up else -1.0)
    return np.array([(0.0, 0.0), (length_x, 0.0), (length_x, length_y), (0.0, length_y)])


def fit_rigid_2d(source, target):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    H = (source - source_mean).T @ (target - target_mean)
    angle = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R, target_mean - R @ source_mean


def corner_residuals(model, measured):
    R, t = fit_rigid_2d(model, measured)
    return np.linalg.norm(model @ R.T + t - measured, axis=1), R, t


def z_points_up(measured):
    long_edge, short_edge = measured[1] - measured[0], measured[3] - measured[0]
    return bool(long_edge[0] * short_edge[1] - long_edge[1] * short_edge[0] > 0)


def numbering_hint(measured, board):
    def worst(order):
        reordered = measured[list(order)]
        return corner_residuals(corner_model(board, z_points_up(reordered)), reordered)[0].max()

    order = min(itertools.permutations(range(4)), key=worst)
    if order != (0, 1, 2, 3) and worst(order) < CORNER_RESIDUAL_WARN_M:
        return (f"the values fit within {1000 * worst(order):.1f} mm if corners 1, 2, 3, 4 are the values given for "
                f"{', '.join(str(i + 1) for i in order)}: check the numbering")
    return None


def fit_corners(corners_mm, board):
    # Only x, y and yaw: the board lies flat, and a 3D fit turns ruler noise into tilt (synthetic test, 2026-11-21:
    # 5 mm of height noise moved the camera by 32 mm).
    measured = np.array([corners_mm[number] for number in CORNERS], float) / 1000
    z_up = z_points_up(measured)
    residuals, R, t = corner_residuals(corner_model(board, z_up), measured)
    hint = numbering_hint(measured, board) if residuals.max() > CORNER_RESIDUAL_WARN_M else None
    return t, float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), z_up, dict(zip(CORNERS, residuals)), hint


def measured_board(extrinsics, board):
    corners = extrinsics.get("board_corners_in_base_mm")
    if corners:
        corner1, yaw, z_up, residuals, hint = fit_corners(corners, board)
        return MeasuredBoard(np.array([*corner1, board.thickness_m]), yaw, z_up, residuals,
                             f"four measured corners, fitted: corner 1 at ({corner1[0]:.4f}, {corner1[1]:.4f}) m, yaw {yaw:.1f} deg, "
                             f"z = x cross y points {'up' if z_up else 'down'}", hint)
    origin, yaw = extrinsics.get("board_origin_in_base_xyz_m"), extrinsics.get("board_yaw_in_base_deg")
    if origin is None or yaw is None:
        return None
    return MeasuredBoard(np.add(origin, (0.0, 0.0, board.thickness_m)), yaw, None, None,
                         f"measured corner 1 at {origin} m, yaw {yaw} deg; up or down of z taken from the image")


def average_grid(grids):
    aligned = np.array([align_to(grid, grids[0]) for grid in grids])
    return aligned.mean(axis=0), float(aligned.std(axis=0).max()) if len(grids) > 1 else 0.0


def candidates_from(grid, board, K, dist):
    return [Candidate(oriented, *board_pose_in_camera(oriented, board, K, dist)) for oriented in orientations(grid)]


def tape_windows(grid, board, shape):
    rows, columns = np.mgrid[0:board.short_corners, 0:board.long_corners]
    H = cv2.findHomography(np.column_stack([columns.ravel(), rows.ravel()]).astype(np.float32), grid.reshape(-1, 2).astype(np.float32))[0]
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circle = TAPE_WINDOW_RADIUS_SQUARES * np.column_stack([np.cos(angles), np.sin(angles)])
    mask = np.zeros(shape[:2], np.uint8)
    for column in (0, board.long_corners - 1):
        for row in (0, board.short_corners - 1):
            outward = np.array([-1.0 if column == 0 else 1.0, -1.0 if row == 0 else 1.0])
            window = (np.array([column, row]) + outward * TAPE_OFFSET_SQUARES + circle).astype(np.float32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [cv2.perspectiveTransform(window, H).round().astype(np.int32)], 255)
    return mask


def red_patches(image, grid, board):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, *RED_HSV_RANGES[0]) | cv2.inRange(hsv, *RED_HSV_RANGES[1])
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(red)
    windows = tape_windows(grid, board, image.shape) > 0
    # A patch that reaches out of the windows is something larger than the tape, such as the arm at a board corner.
    return sorted(((int(stats[i, cv2.CC_STAT_AREA]), centroids[i]) for i in range(1, len(stats))
                   if stats[i, cv2.CC_STAT_AREA] >= TAPE_MIN_AREA_PX and windows[labels == i].mean() >= TAPE_INSIDE_FRACTION),
                  key=lambda patch: -patch[0])


def red_tape_corner(image, grid, board):
    patches = red_patches(image, grid, board)
    if not patches:
        return None, "red tape: none found at the board's corners, no decision"
    if len(patches) > 1 and patches[0][0] < TAPE_AREA_RATIO * patches[1][0]:
        return None, f"red tape: two red patches at the board's corners ({patches[0][0]} and {patches[1][0]} px), no decision"
    area, centre = patches[0]
    ends = sorted((grid[r, c] for r in (0, -1) for c in (0, -1)), key=lambda end: np.linalg.norm(end - centre))
    nearest, second = (float(np.linalg.norm(end - centre)) for end in ends[:2])
    if nearest * TAPE_DISTANCE_RATIO > second:
        return None, f"red tape at pixel {to_point(centre)} is not clearly nearer one corner, no decision"
    return ends[0], f"red tape: {area} px patch at pixel {to_point(centre)}, corner 1 is the pattern end nearest it"


def tape_vote(candidates, image, grid, board):
    pixel, verdict = red_tape_corner(image, grid, board)
    if pixel is None:
        return None, verdict
    for candidate in candidates:
        if np.allclose(candidate.corner_pixel(1), pixel):
            return candidate, verdict
    raise CalibrationError("The red tape is next to a pattern end that cannot be corner 1 with the measured corners "
                           "(z would point the other way): check the numbering, are 2 and 4 swapped?")


def rule_vote(candidates, rule):
    direction = np.array(IMAGE_CORNER_DIRECTIONS[rule], float) / np.sqrt(2)
    centre = candidates[0].grid.reshape(-1, 2).mean(axis=0)
    reach = [float((c.corner_pixel(1) - centre) @ direction) for c in candidates]
    ranked = np.argsort(reach)[::-1]
    if reach[ranked[0]] - reach[ranked[1]] < AMBIGUOUS_FRACTION * np.linalg.norm(candidates[ranked[0]].corner_pixel(1) - centre):
        return None, f"origin_image_corner {rule}: two candidate ends lie almost equally towards that image corner, no decision"
    return candidates[ranked[0]], f"origin_image_corner {rule}: corner 1 is the end towards that image corner"


def camera_vote(candidates, camera_xyz):
    distances = [float(np.linalg.norm(c.T_base_cam[:3, 3] - camera_xyz)) for c in candidates]
    ranked = np.argsort(distances)
    verdict = (f"camera_position_in_base_m: the candidates put the camera {distances[ranked[0]]:.2f} m and "
               f"{distances[ranked[1]]:.2f} m from it")
    if distances[ranked[1]] - distances[ranked[0]] < CAMERA_VOTE_MARGIN_M:
        return None, verdict + ", too close to decide"
    return candidates[ranked[0]], verdict + ", the nearer one wins"


def choose_corner1(candidates, measured, extrinsics, image, grid, board):
    notes = []
    if measured.z_up is not None:
        candidates = [c for c in candidates if c.z_up == measured.z_up]
        notes.append(f"the measured corners say z = x cross y points {'up' if measured.z_up else 'down'}; that leaves two "
                     "pattern ends 180 degrees apart, which the measurements cannot tell apart")
    for candidate in candidates:
        candidate.T_base_cam = measured.T_base_board(candidate.z_up) @ invert(candidate.T_cam_board)
    votes = [tape_vote(candidates, image, grid, board)]
    if extrinsics.get("origin_image_corner"):
        votes.append(rule_vote(candidates, extrinsics["origin_image_corner"]))
    if extrinsics.get("camera_position_in_base_m") is not None:
        votes.append(camera_vote(candidates, np.asarray(extrinsics["camera_position_in_base_m"], float)))
    notes += [why for _, why in votes]
    decided = [candidate for candidate, _ in votes if candidate is not None]
    if not decided:
        raise CalibrationError("Cannot tell which pattern end is corner 1 (no clear red tape). Set origin_image_corner "
                               "(--origin-image-corner) or camera_position_in_base_m (--camera-position-mm).")
    if any(candidate is not decided[0] for candidate in decided):
        raise CalibrationError("The red tape, origin_image_corner and camera_position_in_base_m do not agree on corner 1. "
                               "Look at the overlay and correct the one that is wrong.")
    return decided[0], notes


def check_camera_pose(T_base_cam, extrinsics):
    height = T_base_cam[2, 3]
    if not CAMERA_HEIGHT_RANGE_M[0] < height < CAMERA_HEIGHT_RANGE_M[1] or T_base_cam[2, 2] >= 0:
        raise CalibrationError(f"The camera comes out {height:.2f} m above the table, looking "
                               f"{'down' if T_base_cam[2, 2] < 0 else 'up'}: the board pose or the measurements are wrong.")
    measured = extrinsics.get("camera_position_in_base_m")
    if measured is not None and np.linalg.norm(T_base_cam[:3, 3] - measured) > CAMERA_MISMATCH_WARN_M:
        return [f"the camera comes out {np.linalg.norm(T_base_cam[:3, 3] - measured):.2f} m from the measured lens position"]
    return []


def board_extrinsics(images, board, K, dist, extrinsics):
    grids = [grid for grid in (detect_grid(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), board) for image in images) if grid is not None]
    if not grids:
        raise CalibrationError("The board was not found in the image.")
    grid, spread_px = average_grid(grids)
    result = {"image_size": (images[0].shape[1], images[0].shape[0]), "frames": len(grids), "spread_px": spread_px,
              "warnings": [], "candidates": candidates_from(grid, board, K, dist), "measured": measured_board(extrinsics, board)}
    if spread_px > FRAME_SPREAD_WARN_PX:
        result["warnings"].append(f"the corners moved by up to {spread_px:.1f} px between frames: board or camera not still")
    measured = result["measured"]
    if measured is None:
        return result
    if measured.residuals_m is not None and max(measured.residuals_m.values()) > CORNER_RESIDUAL_WARN_M:
        result["warnings"].append("a measured corner is more than 5 mm off the fitted board: mismeasured corner, swapped "
                                  "numbers or coordinates, or the board moved" + (f"; {measured.hint}" if measured.hint else ""))
    try:
        chosen, notes = choose_corner1(result["candidates"], measured, extrinsics, images[-1], grid, board)
        result["warnings"] += check_camera_pose(chosen.T_base_cam, extrinsics)
    except CalibrationError as error:
        result["error"] = str(error)
        return result
    result.update(chosen=chosen, notes=notes, T_base_cam=chosen.T_base_cam)
    return result


def draw_arrow(image, start, end, colour, label):
    if np.all(np.isfinite([start, end])):
        cv2.arrowedLine(image, to_point(start), to_point(end), colour, 2, tipLength=0.08)
        cv2.putText(image, label, to_point(np.add(end, (4, -4))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def draw_label(image, pixel, text, colour):
    cv2.putText(image, text, to_point(np.add(pixel, (8, -8))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def draw_overlay(image, result, board, K, dist):
    view = image.copy()
    for corner in result["candidates"][0].grid.reshape(-1, 2):
        cv2.circle(view, to_point(corner), 2, (0, 255, 255), -1)
    chosen = result.get("chosen")
    if chosen is None:
        for candidate in result["candidates"]:
            cv2.circle(view, to_point(candidate.corner_pixel(1)), 10, (0, 0, 255), 2)
        return draw_text(view, ["NOT CALIBRATED. Red circles: the pattern ends; corner 1 is the one nearest the red tape.",
                                result.get("error", "Give the four measured corners, then run again.")[:95]])
    cv2.circle(view, to_point(chosen.corner_pixel(1)), 10, (0, 0, 255), 2)
    draw_label(view, chosen.corner_pixel(1), "1 (red tape)", (0, 0, 255))
    for number in (2, 3, 4):
        draw_label(view, chosen.corner_pixel(number), str(number), (255, 0, 255))
    axes = np.array([[0, 0, 0], [3 * board.square_x_m, 0, 0], [0, 3 * board.square_y_m, 0]], float)
    origin, x_end, y_end = cv2.projectPoints(axes, *to_rvec_tvec(chosen.T_cam_board), K, dist)[0].reshape(-1, 2)
    draw_arrow(view, origin, x_end, (0, 0, 255), "board x")
    draw_arrow(view, origin, y_end, (0, 200, 0), "board y")
    camera = TopCamera(K, dist, result["image_size"], result["T_base_cam"])
    base = camera.base_to_pixel([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    for end, colour, label in zip(base[1:], ((0, 0, 255), (0, 200, 0), (255, 0, 0)), ("robot x", "robot y", "robot z")):
        draw_arrow(view, base[0], end, colour, label)
    x, y, z = result["T_base_cam"][:3, 3]
    return draw_text(view, ["Is '1' next to the red tape? Are the robot axes (10 cm) on the robot base?",
                            f"camera at x {x:.3f}  y {y:.3f}  z {z:.3f} m"])


def report(result):
    print(f"board found in {result['frames']} frame(s), corner spread between frames {result['spread_px']:.2f} px")
    measured = result["measured"]
    if measured is None:
        return
    print(measured.description)
    for number, residual in (measured.residuals_m or {}).items():
        print(f"  corner {number}: {1000 * residual:.1f} mm off the fitted board")
    if "chosen" not in result:
        return
    for note in result["notes"]:
        print(f"corner 1: {note}")
    chosen, T = result["chosen"], result["T_base_cam"]
    print(f"corner 1 at pixel {to_point(chosen.corner_pixel(1))}, z = x cross y points {'up' if chosen.z_up else 'down'}, "
          f"board reprojection {chosen.rms_px:.2f} px")
    print(f"camera at x {T[0, 3]:.3f}  y {T[1, 3]:.3f}  z {T[2, 3]:.3f} m, optical axis {np.degrees(np.arccos(-T[2, 2])):.1f} deg from vertical")


def confirmed(view, use_window, assume_yes):
    if assume_yes:
        return True
    if use_window:
        return Window("extrinsics: y saves, n cancels").wait_for(view, (ord("y"), ord("n"))) == ord("y")
    return input("Is '1' next to the red tape, and are the robot axes on the robot base? [y/N] ").strip().lower() == "y"


def frames_from_camera(settings, board, count, use_window):
    camera = LatestFrame(open_camera(settings))
    window = Window("extrinsics: collecting frames") if use_window else None
    frames, number = [], 0
    try:
        while len(frames) < count:
            frame, number = camera.latest(after=number)
            if detect_grid(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board) is not None:
                frames.append(frame)
            if window:
                window.show(draw_text(frame.copy(), [f"board found in {len(frames)}/{count} frames; do not touch the board"]))
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return frames


def extrinsics_record(result, board, extrinsics, intrinsics, source):
    chosen, measured = result["chosen"], result["measured"]
    return {
        "created": timestamp(), "method": "board", "source": source,
        "intrinsics_created": intrinsics["created"],
        "image_width": result["image_size"][0], "image_height": result["image_size"][1],
        "T_base_cam": result["T_base_cam"], "camera_position_in_base_m": result["T_base_cam"][:3, 3],
        "board_reprojection_rms_px": chosen.rms_px,
        "board": {"frame": "origin corner 1 (red tape), x towards corner 2, y towards corner 4, z = x cross y",
                  "z_points": "up" if chosen.z_up else "down",
                  "corner1_in_base_m": measured.corner1_m, "yaw_in_base_deg": measured.yaw_deg,
                  "thickness_m": board.thickness_m, "square_x_m": board.square_x_m, "square_y_m": board.square_y_m,
                  "corner_pixels": {number: chosen.corner_pixel(number) for number in CORNERS},
                  "measured_with": measured.description,
                  "corner_residuals_mm": {n: round(1000 * r, 1) for n, r in (measured.residuals_m or {}).items()}},
        "corner1_decided_by": result["notes"],
        "config": {key: extrinsics.get(key) for key in ("board_corners_in_base_mm", "board_origin_in_base_xyz_m",
                                                        "board_yaw_in_base_deg", "origin_image_corner", "camera_position_in_base_m")},
        "warnings": result["warnings"],
    }


def parse_corner(text):
    match = re.fullmatch(r"\s*([1-4])\s*:\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected 'N: x, y' with N in 1-4, got {text!r}")
    return int(match[1]), [float(match[2]), float(match[3])]


def apply_overrides(config, args):
    extrinsics = config["extrinsics"]
    if args.corners_mm:
        corners = dict(args.corners_mm)
        if sorted(corners) != list(CORNERS):
            sys.exit("--corners-mm needs corners 1, 2, 3 and 4 once each")
        extrinsics["board_corners_in_base_mm"] = corners
    if args.thickness_mm is not None:
        config["board"]["board_thickness_m"] = args.thickness_mm / 1000
    if args.camera_position_mm is not None:
        extrinsics["camera_position_in_base_m"] = [v / 1000 for v in args.camera_position_mm]
    if args.origin_image_corner:
        extrinsics["origin_image_corner"] = args.origin_image_corner


def main():
    parser = argparse.ArgumentParser(description="Camera pose in the robot frame from the board lying on the table (method A).")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_PATH)
    parser.add_argument("--out", type=Path, default=EXTRINSICS_PATH)
    parser.add_argument("--image", type=Path, help="saved frame of the board on the table, instead of the live camera")
    parser.add_argument("--camera", help="index or device path, instead of the config's")
    parser.add_argument("--frames", type=int, default=10, help="live frames to average")
    parser.add_argument("--corners-mm", type=parse_corner, nargs=4, metavar="'N: x, y'",
                        help="ruler values of corners 1-4 in the robot frame, mm, e.g. '1: 215, 82' (instead of the config)")
    parser.add_argument("--thickness-mm", type=float, help="board plus backing, instead of board.board_thickness_m")
    parser.add_argument("--camera-position-mm", type=float, nargs=3, metavar=("X", "Y", "Z"), help="rough lens position")
    parser.add_argument("--origin-image-corner", choices=sorted(IMAGE_CORNER_DIRECTIONS))
    parser.add_argument("--no-window", action="store_true", help="confirm in the terminal after looking at the overlay PNG")
    parser.add_argument("--yes", action="store_true", help="save without asking")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_overrides(config, args)
    board = Board.from_config(config)
    intrinsics = read_yaml(args.intrinsics, "Intrinsics")
    K, dist = np.array(intrinsics["K"]), np.array(intrinsics["dist"])
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            sys.exit(f"cannot read {args.image}")
        frames, source = [image], str(args.image)
    else:
        frames, source = frames_from_camera(camera_settings(config, args.camera), board, args.frames, not args.no_window), "live camera"
    if (frames[0].shape[1], frames[0].shape[0]) != (intrinsics["image_width"], intrinsics["image_height"]):
        sys.exit(f"The frame is {frames[0].shape[1]}x{frames[0].shape[0]}, the intrinsics are for "
                 f"{intrinsics['image_width']}x{intrinsics['image_height']}.")

    try:
        result = board_extrinsics(frames, board, K, dist, config["extrinsics"])
    except CalibrationError as error:
        sys.exit(f"FAILED: {error}")
    report(result)
    overlay_path = args.out.with_name(args.out.stem + "_overlay.png")
    view = draw_overlay(frames[-1], result, board, K, dist)
    cv2.imwrite(str(overlay_path), view)
    print(f"overlay written to {overlay_path}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    if result["measured"] is None:
        sys.exit("Not saved: give the four corners (--corners-mm or board_corners_in_base_mm), then run again.")
    if "error" in result:
        sys.exit(f"FAILED: {result['error']}")
    if not confirmed(view, not args.no_window, args.yes):
        sys.exit("Not saved.")
    write_yaml(args.out, extrinsics_record(result, board, config["extrinsics"], intrinsics, source))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
