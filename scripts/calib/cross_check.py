"""Independent cross-check of the top camera's calibration (docs/camera-calibration.md).

Recomputes intrinsics from the same frames and the camera pose from the same table frame and ruler values, without
importing anything from Tian's scripts/calib/*.py, and prints the differences to the YAML files those scripts wrote
(intrinsics.yaml, extrinsics.yaml; formats of PR #21). Opens no camera and sends nothing to the robot.

Verdict: the worst level of all checks. Thresholds, and why:
- intrinsics RMS < 0.5 px GOOD, < 1 px CHECK, else BAD. A 640x480 webcam with 25 views reaches 0.2-0.5 px; at about
  60 cm and fx about 550, 1 px is about 1 mm on the table, so above 1 px the corner detection or the lens model is wrong.
- image size must be 640x480 (the size the pick policies were trained at), else BAD. Fewer than 15 used frames CHECK,
  fewer than 10 BAD. More than 20 % of the frames flagged as outliers (error > max(2 x median, 0.25 px)) CHECK.
- fx/fy within 2 % of each other, principal point within 40 px of the image centre, else CHECK (webcam lenses are
  round and centred; a larger offset usually means a bad fit, not a real lens).
- table board reprojection RMS: same 0.5 / 1 px.
- per-corner residual between ruler and camera, after the best rigid fit in the table plane (no scale): max < 5 mm
  GOOD, < 10 mm CHECK, else BAD. A ruler reading is good to 1-2 mm and one pixel is about 1 mm, so 5 mm is more than
  two standard deviations of honest noise; 10 mm is the tolerance of a hand-over to an open hand.
- ruler side lengths against the board (6 and 4 squares): the square size they imply must be within 2 % of the one
  used (CHECK), since a wrong square size scales the camera height.
- camera position against the roughly measured lens position (+-5 cm): within 10 cm GOOD, else CHECK. Never BAD (it is a rough measurement).
- jaw-tip touches against the camera's corner positions, same 5 / 10 mm.
- against Tian's files: fx, fy within 0.5 % (CHECK 2 %), cx, cy within 2 px (CHECK 6 px), the same K reprojecting our
  corners within the RMS thresholds, camera position within 5 mm (CHECK 15 mm), rotation within 0.3 deg (CHECK 1 deg),
  same pixel for each corner label within 3 px. The same inputs and the same algorithm should agree far tighter than
  these; the margins allow for frames Tian's script dropped as outliers.

What one flat board cannot show: a wrong focal length. The pose from a board of known size moves with it, so the
board-plane residual stays small; only the multi-view intrinsics (RMS) and the lens position expose it. Say so in the
report rather than claim more.

Exit code 0 GOOD, 1 CHECK, 2 BAD.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

GOOD, CHECK, BAD = 0, 1, 2
LEVEL_NAMES = {GOOD: "GOOD", CHECK: "CHECK", BAD: "BAD"}
EXPECTED_SIZE = (640, 480)
DEFAULT_CORNERS = (7, 5)
DEFAULT_SQUARE_MM = (28.75, 29.0)
SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
OUTLIER_FACTOR, OUTLIER_FLOOR_PX = 2.0, 0.25
RMS_LIMITS_PX = (0.5, 1.0)
RESIDUAL_LIMITS_MM = (5.0, 10.0)
LENS_LIMIT_MM = 100.0
SQUARE_TOLERANCE = 0.02
TIAN_LIMITS = {"focal": (0.005, 0.02), "centre_px": (2.0, 6.0), "position_mm": (5.0, 15.0), "rotation_deg": (0.3, 1.0),
               "label_px": 3.0}
HANDOVER_DIR = Path(__file__).resolve().parents[1] / "handover"
LABELS = ("1", "2", "3", "4")


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, text):
        self.rows.append((level, text))
        print(f"[{LEVEL_NAMES[level]:>5}] {text}")

    def graded(self, value, limits, text):
        self.add(GOOD if value < limits[0] else CHECK if value < limits[1] else BAD, text)

    @property
    def worst(self):
        return max((level for level, _ in self.rows), default=GOOD)


def find_grid(gray, long_corners, short_corners):
    # Same detector flags as Tian's script on purpose: a different detector would change the corner numbers.
    for pattern in ((long_corners, short_corners), (short_corners, long_corners)):
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=SB_FLAGS)
        if found:
            grid = corners.reshape(pattern[1], pattern[0], 2).astype(np.float64)
            return grid if pattern[0] == long_corners else grid.transpose(1, 0, 2)
    return None


def board_points(long_corners, short_corners, square_mm):
    rows, cols = np.mgrid[0:short_corners, 0:long_corners]
    return np.column_stack([cols.ravel() * square_mm[0], rows.ravel() * square_mm[1], np.zeros(rows.size)])


def view_errors(points, grids, K, dist, rvecs, tvecs):
    return [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(points, r, t, K, dist)[0].reshape(-1, 2) - g.reshape(-1, 2)) ** 2, axis=1))))
            for g, r, t in zip(grids, rvecs, tvecs)]


def calibrate(grids, points, size):
    obj = [points.astype(np.float32)] * len(grids)
    img = [g.reshape(-1, 1, 2).astype(np.float32) for g in grids]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None)
    return float(rms), K, dist.ravel(), view_errors(points, grids, K, dist, rvecs, tvecs)


def reprojection_rms_with(grids, points, K, dist):
    errors = []
    for grid in grids:
        _, rvec, tvec = cv2.solvePnP(points, grid.reshape(-1, 2), K, dist, flags=cv2.SOLVEPNP_IPPE)
        errors += view_errors(points, [grid], K, dist, [rvec], [tvec])
    return float(np.sqrt(np.mean(np.square(errors))))


def check_intrinsics(folder, long_corners, short_corners, square_mm, report):
    paths = sorted([*folder.glob("*.png"), *folder.glob("*.jpg")])
    if not paths:
        sys.exit(f"no .png or .jpg frames in {folder}")
    grids, names, sizes = [], [], set()
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        sizes.add((gray.shape[1], gray.shape[0]))
        grid = find_grid(gray, long_corners, short_corners)
        if grid is not None:
            grids.append(grid)
            names.append(path.name)
    print(f"intrinsics: {len(grids)} of {len(paths)} frames show the {long_corners}x{short_corners} board")
    if sizes != {EXPECTED_SIZE}:
        report.add(BAD, f"image size {sorted(sizes)}, must be {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    else:
        report.add(GOOD, f"image size {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    if len(grids) < 10:
        report.add(BAD, f"only {len(grids)} usable frames, need at least 10")
        return None
    if len(grids) < 15:
        report.add(CHECK, f"only {len(grids)} usable frames, 20-30 are better")
    size = sizes.pop() if len(sizes) == 1 else EXPECTED_SIZE
    points = board_points(long_corners, short_corners, square_mm)
    first_rms, _, _, first_errors = calibrate(grids, points, size)
    limit = max(OUTLIER_FACTOR * float(np.median(first_errors)), OUTLIER_FLOOR_PX)
    flagged = [n for n, e in zip(names, first_errors) if e > limit]
    print(f"per-image error, first pass (RMS {first_rms:.3f} px, flag above {limit:.2f} px):")
    for name, error in zip(names, first_errors):
        print(f"  {name}: {error:.3f} px{'   OUTLIER' if name in flagged else ''}")
    kept = [g for g, n in zip(grids, names) if n not in flagged]
    rms, K, dist, _ = calibrate(kept, points, size)
    print(f"after dropping {len(flagged)}: {len(kept)} frames, RMS {rms:.3f} px")
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  dist(k1 k2 p1 p2 k3) {np.round(dist, 4).tolist()}")
    report.graded(rms, RMS_LIMITS_PX, f"intrinsics RMS {rms:.3f} px")
    if len(flagged) > 0.2 * len(grids):
        report.add(CHECK, f"{len(flagged)} of {len(grids)} frames are outliers: a board that is not flat, or blurred frames")
    if abs(K[0, 0] / K[1, 1] - 1) > 0.02:
        report.add(CHECK, f"fx/fy differ by {abs(K[0, 0] / K[1, 1] - 1):.1%}")
    centre_offset = float(np.hypot(K[0, 2] - size[0] / 2, K[1, 2] - size[1] / 2))
    if centre_offset > 40:
        report.add(CHECK, f"principal point {centre_offset:.0f} px from the image centre")
    return {"K": K, "dist": dist, "rms": rms, "grids": kept, "points": points}


def compare_intrinsics(mine, tian, report):
    K_tian, dist_tian = np.array(tian["K"], float), np.array(tian["dist"], float).ravel()
    for name, i in (("fx", 0), ("fy", 1)):
        rel = abs(mine["K"][i, i] / K_tian[i, i] - 1)
        report.graded(rel, TIAN_LIMITS["focal"], f"{name}: ours {mine['K'][i, i]:.1f}, Tian's {K_tian[i, i]:.1f} ({rel:.2%} apart)")
    for name, i in (("cx", 2), ("cy", 5)):
        diff = abs(mine["K"].ravel()[i] - K_tian.ravel()[i])
        report.graded(diff, TIAN_LIMITS["centre_px"], f"{name}: ours {mine['K'].ravel()[i]:.1f}, Tian's {K_tian.ravel()[i]:.1f}")
    print(f"        dist ours {np.round(mine['dist'], 4).tolist()}  Tian's {np.round(dist_tian, 4).tolist()}  (rms Tian's {tian.get('rms_px', float('nan')):.3f} px)")
    rms = reprojection_rms_with(mine["grids"], mine["points"], K_tian, dist_tian)
    report.graded(rms, RMS_LIMITS_PX, f"Tian's K and dist on our corner detections: RMS {rms:.3f} px (ours {mine['rms']:.3f} px)")


def undistort(pixels, K, dist):
    # Plain undistortPoints stops after 5 iterations, which leaves an error towards the image corners.
    return cv2.undistortPointsIter(np.asarray(pixels, float).reshape(-1, 1, 2), K, dist, None, None, UNDISTORT_CRITERIA).reshape(-1, 2)


def fit_rigid_2d(source, target):
    s_mean, t_mean = source.mean(axis=0), target.mean(axis=0)
    H = (source - s_mean).T @ (target - t_mean)
    angle = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R, t_mean - R @ s_mean


def outer_cells(long_corners, short_corners):
    last_c, last_r = long_corners - 1, short_corners - 1
    return [(0, 0), (0, last_c), (last_r, last_c), (last_r, 0)]


def label_assignments(long_corners, short_corners):
    # Corners 1-4 sit in a fixed order around the rectangle; the four symmetries that keep long sides long are the
    # four ways the detector's numbering can lie against the physical labels (two of them are mirror images).
    last_c, last_r = long_corners - 1, short_corners - 1
    flips = [lambda r, c: (r, c), lambda r, c: (last_r - r, last_c - c), lambda r, c: (r, last_c - c), lambda r, c: (last_r - r, c)]
    return [[flip(*cell) for cell in outer_cells(long_corners, short_corners)] for flip in flips]


def find_tape(image, grid):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    low, high = grid.reshape(-1, 2).min(axis=0), grid.reshape(-1, 2).max(axis=0)
    margin = 0.4 * (high - low)
    window = np.zeros_like(red)
    x0, y0 = np.maximum(low - margin, 0).astype(int)
    x1, y1 = (high + margin).astype(int)
    window[y0:y1, x0:x1] = 255
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(red & window)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or areas.max() < 30:
        return None
    return centroids[1 + int(areas.argmax())]


def table_pose(K, dist, grid, points):
    image_points = grid.reshape(-1, 2)
    _, rvec, tvec = cv2.solvePnP(points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(points, image_points, K, dist, rvec, tvec)
    rms = float(np.sqrt(np.mean(np.sum((cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1, 2) - image_points) ** 2, axis=1))))
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), rms


def back_project(pixels, K, dist, R, t):
    camera = -R.T @ t
    rays = np.column_stack([undistort(pixels, K, dist), np.ones(len(pixels))]) @ R
    return camera + (-camera[2] / rays[:, 2:3]) * rays


def extrinsics_from_table(K, dist, grid, image, ruler_mm, thickness_mm, square_mm, lens_mm, tape_px):
    long_corners, short_corners = grid.shape[1], grid.shape[0]
    points = board_points(long_corners, short_corners, square_mm)
    R, t, board_rms = table_pose(K, dist, grid, points)
    camera_in_board = -R.T @ t
    flip = 1.0 if camera_in_board[2] > 0 else -1.0
    cells = outer_cells(long_corners, short_corners)
    pixels = np.array([grid[r, c] for r, c in cells])
    seen = back_project(pixels, K, dist, R, t) * [1, flip, 1]
    ruler = np.array([ruler_mm[label] for label in LABELS])
    tape = tape_px if tape_px is not None else (find_tape(image, grid) if image is not None else None)

    candidates = []
    for assignment in label_assignments(long_corners, short_corners):
        order = [cells.index(cell) for cell in assignment]
        R2, t2 = fit_rigid_2d(seen[order][:, :2], ruler)
        residuals = np.linalg.norm(seen[order][:, :2] @ R2.T + t2 - ruler, axis=1)
        R_base_board = np.block([[R2, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]]) @ np.diag([1.0, flip, flip])
        t_base_board = np.array([*t2, thickness_mm])
        candidates.append({"order": order, "residuals": residuals, "R2": R2, "t2": t2,
                           "R_base_cam": R_base_board @ R.T, "position": R_base_board @ camera_in_board + t_base_board,
                           "label1_pixel": pixels[order[0]]})
    pool = sorted(candidates, key=lambda c: c["residuals"].max())[:2]
    notes, level = [], GOOD
    by_tape = None
    if tape is not None:
        by_tape = min(candidates, key=lambda c: np.linalg.norm(c["label1_pixel"] - tape))
        notes.append(f"red tape at pixel ({tape[0]:.0f}, {tape[1]:.0f}) -> corner 1 is the board end nearest it")
    by_lens = None
    if lens_mm is not None and np.linalg.norm(pool[0]["position"] - pool[1]["position"]) > 2 * LENS_LIMIT_MM:
        by_lens = min(pool, key=lambda c: np.linalg.norm(c["position"] - lens_mm))
    if by_tape is not None:
        chosen = by_tape
        if by_lens is not None and by_lens is not by_tape:
            level = CHECK
            notes.append("the lens position points to the other board orientation than the tape: check the tape and the numbering")
    elif by_lens is not None:
        chosen = by_lens
        notes.append("no red tape found: orientation taken from the lens position")
    else:
        chosen, level = pool[0], CHECK
        notes.append("orientation unresolved (no red tape found, lens position missing or too close to the board's centre): "
                     "pass --tape-px X Y or --lens-mm X Y Z; corner 1 and 3 may be swapped")
    optical_axis = chosen["R_base_cam"][:, 2]
    return {"board_rms_px": board_rms, "chosen": chosen, "pool": pool, "notes": notes, "level": level, "flip": flip,
            "position_mm": chosen["position"], "R_base_cam": chosen["R_base_cam"],
            "tilt_deg": float(np.degrees(np.arccos(np.clip(-optical_axis[2], -1, 1)))),
            "pixels": pixels, "label_pixels": pixels[chosen["order"]], "seen_base_mm": seen[chosen["order"]][:, :2] @ chosen["R2"].T + chosen["t2"]}


def report_extrinsics(result, ruler_mm, lens_mm, square_mm, thickness_mm, report, label):
    report.graded(result["board_rms_px"], RMS_LIMITS_PX, f"[{label}] table board reprojection RMS {result['board_rms_px']:.3f} px")
    for note in result["notes"]:
        print(f"        {note}")
    if result["level"]:
        report.add(result["level"], f"[{label}] board orientation not confirmed by two sources")
    residuals = result["chosen"]["residuals"]
    report.graded(float(residuals.max()), RESIDUAL_LIMITS_MM,
                  f"[{label}] corner residual ruler vs camera, mm: " + ", ".join(f"{n}: {r:.1f}" for n, r in zip(LABELS, residuals)))
    points = np.array([ruler_mm[n] for n in LABELS])
    sides = {"1-2": 6, "2-3": 4, "4-3": 6, "1-4": 4}
    lengths = {k: float(np.linalg.norm(points[int(k[0]) - 1] - points[int(k[2]) - 1])) for k in sides}
    long_implied = np.mean([lengths["1-2"], lengths["4-3"]]) / 6
    short_implied = np.mean([lengths["2-3"], lengths["1-4"]]) / 4
    off = max(abs(long_implied / square_mm[0] - 1), abs(short_implied / square_mm[1] - 1))
    print(f"        ruler side lengths mm: " + ", ".join(f"{k} {v:.0f}" for k, v in lengths.items()))
    report.add(GOOD if off < SQUARE_TOLERANCE else CHECK,
               f"[{label}] ruler implies squares of {long_implied:.2f} x {short_implied:.2f} mm, used {square_mm[0]:.2f} x {square_mm[1]:.2f}")
    x, y, z = result["position_mm"]
    print(f"        camera in the robot frame: x {x:.0f}  y {y:.0f}  height above the table {z:.0f} mm, optical axis {result['tilt_deg']:.1f} deg from straight down")
    if lens_mm is not None:
        gap = float(np.linalg.norm(result["position_mm"] - lens_mm))
        report.add(GOOD if gap < LENS_LIMIT_MM else CHECK,
                   f"[{label}] camera {gap:.0f} mm from the reported lens position (x {lens_mm[0]:.0f}, y {lens_mm[1]:.0f}, z {lens_mm[2]:.0f}; +-50 mm expected)")
    elif not z > 0:
        report.add(BAD, f"[{label}] camera below the table")


def compare_extrinsics(result, tian_ext, report, label, tolerance_note):
    T = np.array(tian_ext["T_base_cam"], float)
    theirs = T[:3, 3] * 1000
    delta = float(np.linalg.norm(result["position_mm"] - theirs))
    cos = (np.trace(result["R_base_cam"].T @ T[:3, :3]) - 1) / 2
    angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    report.graded(delta, TIAN_LIMITS["position_mm"], f"[{label}] camera position {delta:.1f} mm from Tian's extrinsics {tolerance_note}"
                  f" (ours {np.round(result['position_mm']).tolist()}, Tian's {np.round(theirs).tolist()})")
    report.graded(angle, TIAN_LIMITS["rotation_deg"], f"[{label}] camera rotation {angle:.2f} deg from Tian's")
    pixels = (tian_ext.get("board") or {}).get("corner_pixels")
    if pixels:
        theirs = {str(k): np.array(v, float) for k, v in pixels.items()}
        worst = max(float(np.linalg.norm(result["label_pixels"][i] - theirs[n])) for i, n in enumerate(LABELS))
        report.add(GOOD if worst < TIAN_LIMITS["label_px"] else BAD,
                   f"[{label}] corner labels 1-4 sit {worst:.1f} px from Tian's: " + ("same corners" if worst < TIAN_LIMITS["label_px"] else "a different corner got a label"))


def parse_ruler(lines):
    ruler, jaw_lines = {}, {}
    for line in lines:
        match = re.match(r"\s*([1-4])\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(.*)", line)
        if match and not line.lstrip().startswith("#"):
            ruler[match.group(1)] = np.array([float(match.group(2)), float(match.group(3))])
            if "|" in match.group(4):
                jaw_lines[match.group(1)] = line
    return ruler, jaw_lines


def jaw_points(jaw_lines, tool_point):
    sys.path.insert(0, str(HANDOVER_DIR))
    try:
        from check_corners import parse_line
        from read_pose import FIXED_JAW_END, jaw_point
    except ImportError as error:
        sys.exit(f"jaw-tip lines need scripts/handover/check_corners.py and read_pose.py (PR #13): {error}")
    return {label: jaw_point(parse_line(line)[2], tool_point or FIXED_JAW_END)[:2] * 1000 for label, line in jaw_lines.items()}


def compare_jaw(jaw_mm, ruler_mm, seen_mm, report):
    for label, jaw in jaw_mm.items():
        print(f"        corner {label}: jaw tip {np.round(jaw, 1).tolist()}  ruler {ruler_mm[label].tolist()}  camera {np.round(seen_mm[int(label) - 1], 1).tolist()}")
    worst_camera = max(np.linalg.norm(jaw - seen_mm[int(label) - 1]) for label, jaw in jaw_mm.items())
    worst_ruler = max(np.linalg.norm(jaw - ruler_mm[label]) for label, jaw in jaw_mm.items())
    report.graded(float(worst_camera), RESIDUAL_LIMITS_MM, f"jaw-tip touches vs camera: worst {worst_camera:.1f} mm (vs ruler {worst_ruler:.1f} mm)")


def draw_overlay(image, result, ruler_mm, path):
    out = image.copy()
    for pixel, label in zip(result["label_pixels"], LABELS):
        cv2.circle(out, (int(pixel[0]), int(pixel[1])), 7, (0, 255, 255), 2)
        cv2.putText(out, label, (int(pixel[0]) + 9, int(pixel[1]) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(path), out)


def corner_value(text):
    match = re.fullmatch(r"\s*([1-4])\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected 'N: x, y' with N in 1-4, got {text!r}")
    return text


def table_frame_path(path):
    if path.is_dir():
        frames = sorted([*path.glob("*.png"), *path.glob("*.jpg")])
        if not frames:
            sys.exit(f"no frame in {path}")
        return frames[0]
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--intrinsics-frames", type=Path, help="folder with the 25 board frames (calib_frames/intrinsics)")
    p.add_argument("--table-frame", type=Path, help="frame of the board on the table, or its folder (calib_frames/table)")
    p.add_argument("--corners-mm", type=corner_value, nargs=4, metavar="'N: x, y'", help="ruler values, the same four arguments as Tian's calibrate_extrinsics.py")
    p.add_argument("--corners-file", type=Path, help="file with one '<1-4>: x, y' line per corner; a '| read_pose.py line' after it adds the jaw-tip touch")
    p.add_argument("--thickness-mm", type=float, help="board plus backing; default: Tian's extrinsics.yaml, else 3")
    p.add_argument("--lens-mm", type=float, nargs=3, metavar=("X", "Y", "Z"), help="rough lens position in the robot frame")
    p.add_argument("--tape-px", type=float, nargs=2, metavar=("X", "Y"), help="pixel of the red tape in the table frame, if it is not found automatically")
    p.add_argument("--square-mm", type=float, nargs=2, metavar=("LONG", "SHORT"), help="square size along the long and the short side; default: Tian's intrinsics.yaml, else 28.75 x 29.0")
    p.add_argument("--tool-point", type=float, nargs=3, help="jaw-tip point in metres, as read_pose.py --tool-point")
    p.add_argument("--tian-intrinsics", type=Path, help="scripts/calib/intrinsics.yaml")
    p.add_argument("--tian-extrinsics", type=Path, help="scripts/calib/extrinsics.yaml")
    p.add_argument("--out-dir", type=Path, default=Path("calib_check"), help="overlay image and result.yaml go here")
    args = p.parse_args()

    tian_intrinsics = yaml.safe_load(args.tian_intrinsics.read_text()) if args.tian_intrinsics else None
    tian_extrinsics = yaml.safe_load(args.tian_extrinsics.read_text()) if args.tian_extrinsics else None
    long_corners, short_corners = (tian_intrinsics or {}).get("inner_corners", DEFAULT_CORNERS)
    square_mm = tuple(args.square_mm) if args.square_mm else (
        (tian_intrinsics["square_x_m"] * 1000, tian_intrinsics["square_y_m"] * 1000) if tian_intrinsics else DEFAULT_SQUARE_MM)
    thickness_mm = args.thickness_mm if args.thickness_mm is not None else (
        ((tian_extrinsics or {}).get("board") or {}).get("thickness_m", 0.003) * 1000)
    report = Report()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mine = None
    if args.intrinsics_frames:
        print(f"== intrinsics (board {long_corners}x{short_corners}, squares {square_mm[0]:.2f} x {square_mm[1]:.2f} mm)")
        mine = check_intrinsics(args.intrinsics_frames, long_corners, short_corners, square_mm, report)
        if mine and tian_intrinsics:
            compare_intrinsics(mine, tian_intrinsics, report)

    summary = {}
    if args.table_frame:
        print("== extrinsics")
        lines = (list(args.corners_mm or [])) + (args.corners_file.read_text().splitlines() if args.corners_file else [])
        ruler_mm, jaw_lines = parse_ruler(lines)
        if set(ruler_mm) != set(LABELS):
            sys.exit("ruler values for all four corners are needed (--corners-mm or --corners-file)")
        frame_path = table_frame_path(args.table_frame)
        image = cv2.imread(str(frame_path))
        if (image.shape[1], image.shape[0]) != EXPECTED_SIZE:
            report.add(BAD, f"table frame is {image.shape[1]}x{image.shape[0]}, must be 640x480")
        grid = find_grid(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), long_corners, short_corners)
        if grid is None:
            report.add(BAD, f"board not found in {frame_path}")
        else:
            points = board_points(long_corners, short_corners, square_mm)
            sources = []
            if mine:
                sources.append(("own K", mine["K"], mine["dist"]))
            if tian_intrinsics:
                sources.append(("Tian's K", np.array(tian_intrinsics["K"], float), np.array(tian_intrinsics["dist"], float).ravel()))
            if not sources:
                sys.exit("extrinsics need intrinsics: give --intrinsics-frames and/or --tian-intrinsics")
            results = {}
            for label, K, dist in sources:
                results[label] = extrinsics_from_table(K, dist, grid, image, ruler_mm, thickness_mm, square_mm,
                                                       np.array(args.lens_mm) if args.lens_mm else None, args.tape_px)
                report_extrinsics(results[label], ruler_mm, np.array(args.lens_mm) if args.lens_mm else None, square_mm, thickness_mm, report, label)
            primary = results[sources[0][0]]
            if tian_extrinsics:
                same_K = "Tian's K" if "Tian's K" in results else sources[0][0]
                compare_extrinsics(results[same_K], tian_extrinsics, report, same_K, "(same K as theirs, so only the method differs)" if same_K == "Tian's K" else "")
            if jaw_lines:
                compare_jaw(jaw_points(jaw_lines, args.tool_point), ruler_mm, primary["seen_base_mm"], report)
            draw_overlay(image, primary, ruler_mm, args.out_dir / "table_overlay.png")
            summary["camera_position_mm"] = primary["position_mm"].round(1).tolist()
            summary["camera_height_mm"] = float(primary["position_mm"][2])

    if not (args.intrinsics_frames or args.table_frame):
        p.error("give --intrinsics-frames and/or --table-frame")
    summary |= {"verdict": LEVEL_NAMES[report.worst], "checks": [{"level": LEVEL_NAMES[lv], "text": text} for lv, text in report.rows]}
    (args.out_dir / "result.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, width=140))
    print("\nOne flat board cannot expose a wrong focal length; only the intrinsics RMS and the lens position can.")
    print(f"VERDICT: {LEVEL_NAMES[report.worst]}")
    sys.exit(report.worst)


if __name__ == "__main__":
    main()
