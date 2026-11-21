import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def board_points(pattern, square):
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float64)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def find_corners(gray, pattern):
    # Always the SB detector: on a board with an odd and an even corner count (9x6) it numbers the corners
    # from the same physical corner whatever the board's rotation; the classic detector starts at the
    # opposite corner, so mixing the two flips the board frame by 180 degrees.
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 2).astype(np.float64) if found else None


def calibrate(image_paths, pattern, square):
    object_points, image_points, used, size = [], [], [], None
    for path in image_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        corners = find_corners(gray, pattern)
        if corners is not None:
            object_points.append(board_points(pattern, square).astype(np.float32))
            image_points.append(corners.astype(np.float32))
            used.append(Path(path).name)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, size, None, None)
    per_view = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, dist)[0].reshape(-1, 2) - i) ** 2, axis=1))))
                for o, i, r, t in zip(object_points, image_points, rvecs, tvecs)]
    return {"image_size": list(size), "K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms),
            "views": dict(zip(used, np.round(per_view, 3).tolist())), "skipped": len(image_paths) - len(used)}


def board_pose(gray, K, dist, pattern, square):
    corners = find_corners(gray, pattern)
    if corners is None:
        raise RuntimeError("board not found")
    ok, rvec, tvec = cv2.solvePnP(board_points(pattern, square), corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    R_cam_board, _ = cv2.Rodrigues(rvec)
    projected = cv2.projectPoints(board_points(pattern, square), rvec, tvec, K, dist)[0].reshape(-1, 2)
    return R_cam_board, tvec.ravel(), corners, float(np.linalg.norm(projected - corners, axis=1).mean())


def board_in_base_from_ruler(origin_xy, yaw_deg, board_z, camera_in_board):
    yaw = np.radians(yaw_deg)
    x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    # The board's z axis (x cross y) points away from the camera or towards it depending on the corner
    # order; the camera is above the table, so the camera's side of the board is up.
    z_axis = np.array([0.0, 0.0, 1.0 if camera_in_board[2] > 0 else -1.0])
    return np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis]), np.array([origin_xy[0], origin_xy[1], board_z])


def fit_rigid_2d(source, target):
    # Least-squares rotation angle and translation with target ~ R @ source + t (2D Kabsch).
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    H = (source - source_mean).T @ (target - target_mean)
    angle = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R, target_mean - R @ source_mean


def board_in_base_from_touches(touches, pattern, square, board_z, camera_in_board):
    # The board lies flat, so only x, y and yaw are fitted. A full 3D fit let 5 mm of gripper-height
    # noise tilt the board enough to move the camera by 32 mm in the selftest.
    grid = board_points(pattern, square)
    indices = [int(i) for i in touches]
    measured = np.array([touches[k] for k in touches], float)[:, :2]
    flip = 1.0 if camera_in_board[2] > 0 else -1.0
    R2, t2 = fit_rigid_2d(grid[indices][:, :2] * [1.0, flip], measured)
    R_base_board = np.eye(3)
    R_base_board[:2, :2] = R2
    R_base_board = R_base_board @ np.diag([1.0, flip, flip])
    t_base_board = np.array([t2[0], t2[1], board_z])
    residual = np.linalg.norm((grid[indices] @ R_base_board.T + t_base_board)[:, :2] - measured, axis=1)
    return R_base_board, t_base_board, residual


def compose_camera_in_base(R_base_board, t_base_board, R_cam_board, t_cam_board):
    return R_base_board @ R_cam_board.T, R_base_board @ (-R_cam_board.T @ t_cam_board) + t_base_board


def pixel_to_base(uv, K, dist, R_base_cam, t_base_cam, z):
    normalized = cv2.undistortPoints(np.array([[uv]], np.float64), np.asarray(K, float), np.asarray(dist, float)).ravel()
    ray = np.asarray(R_base_cam) @ np.array([normalized[0], normalized[1], 1.0])
    origin = np.asarray(t_base_cam, float)
    return origin + (z - origin[2]) / ray[2] * ray


def draw_board_axes(image, corners, cols, path):
    out = image.copy()
    for p in corners:
        cv2.circle(out, tuple(int(v) for v in p), 2, (0, 255, 255), -1)
    origin = tuple(int(v) for v in corners[0])
    cv2.arrowedLine(out, origin, tuple(int(v) for v in corners[0] + 3 * (corners[1] - corners[0])), (0, 0, 255), 2)
    cv2.arrowedLine(out, origin, tuple(int(v) for v in corners[0] + 3 * (corners[cols] - corners[0])), (0, 255, 0), 2)
    cv2.circle(out, origin, 8, (0, 0, 255), 2)
    cv2.putText(out, "0", (origin[0] + 10, origin[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(path), out)


def look_down_camera(position, look_at):
    forward = (look_at - position) / np.linalg.norm(look_at - position)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def selftest(pattern=(9, 6), square=0.025):
    rng = np.random.default_rng(1)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    dist = np.array([0.12, -0.26, 0.0, 0.0, 0.10])
    camera_pos = np.array([0.30, -0.25, 0.70])
    R_base_cam_true = look_down_camera(camera_pos, np.array([0.30, 0.0, 0.0]))
    R_base_board_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    t_base_board_true = np.array([0.20, 0.10, 0.0])

    def project(points_base):
        in_cam = (np.atleast_2d(points_base) - camera_pos) @ R_base_cam_true
        return cv2.projectPoints(in_cam, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)

    board_in_base = board_points(pattern, square) @ R_base_board_true.T + t_base_board_true
    corners = project(board_in_base) + rng.normal(0, 0.3, (len(board_in_base), 2))
    ok, rvec, tvec = cv2.solvePnP(board_points(pattern, square), corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    R_cam_board, _ = cv2.Rodrigues(rvec)
    t_cam_board = tvec.ravel()

    R_rb, t_rb = board_in_base_from_ruler(t_base_board_true[:2], 90.0, 0.0, -R_cam_board.T @ t_cam_board)
    extreme = [0, pattern[0] - 1, pattern[0] * (pattern[1] - 1), pattern[0] * pattern[1] - 1]
    touches = {i: board_in_base[i] + rng.normal(0, 0.005, 3) for i in extreme}
    R_tb, t_tb, residual = board_in_base_from_touches(touches, pattern, square, 0.0, -R_cam_board.T @ t_cam_board)
    print(f"touch fit, 4 corners with 5 mm noise each: residual {1000 * residual.mean():.1f} mm")

    for label, (R_bb, t_bb) in {"ruler": (R_rb, t_rb), "touches": (R_tb, t_tb)}.items():
        R_bc, t_bc = compose_camera_in_base(R_bb, t_bb, R_cam_board, t_cam_board)
        print(f"[{label}] camera position error {1000 * np.linalg.norm(t_bc - camera_pos):.1f} mm")
        for hand in ([0.25, 0.05, 0.15], [0.35, -0.10, 0.20], [0.30, 0.15, 0.10]):
            hand = np.array(hand)
            uv = project(hand)[0]
            err = lambda p: 1000 * np.linalg.norm((p - hand)[:2])
            print(f"  hand {hand} at px {uv.round(0)}: xy error, true height {err(pixel_to_base(uv, K, dist, R_bc, t_bc, hand[2])):5.1f} mm"
                  f" | table plane {err(pixel_to_base(uv, K, dist, R_bc, t_bc, 0.0)):5.1f} mm"
                  f" | 15 cm plane {err(pixel_to_base(uv, K, dist, R_bc, t_bc, 0.15)):5.1f} mm"
                  f" | true height, distortion ignored {err(pixel_to_base(uv, K, np.zeros(5), R_bc, t_bc, hand[2])):5.1f} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=9, help="inner corners along the long side")
    parser.add_argument("--rows", type=int, default=6, help="inner corners along the short side")
    parser.add_argument("--square", type=float, default=0.025, help="measured square size, metres")
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("intrinsics")
    cal.add_argument("--out", type=Path, required=True)
    cal.add_argument("images", type=Path, nargs="+")
    ext = sub.add_parser("extrinsics")
    ext.add_argument("--intrinsics", type=Path, required=True)
    ext.add_argument("--image", type=Path, required=True)
    ext.add_argument("--origin", type=float, nargs=2, metavar=("X", "Y"), help="ruler: corner 0 in the robot base frame, metres")
    ext.add_argument("--yaw", type=float, help="ruler: direction of the red arrow (corner 0 to corner 1) in the base frame, degrees; 0 = robot +x, 90 = robot +y")
    ext.add_argument("--board-z", type=float, default=0.0, help="ruler: height of the board surface above the table, metres")
    ext.add_argument("--touches", type=Path, help='touch points instead of ruler: JSON {"corner index": [x, y, z], ...} in the base frame, metres')
    ext.add_argument("--out", type=Path, required=True)
    px = sub.add_parser("pixel")
    px.add_argument("--intrinsics", type=Path, required=True)
    px.add_argument("--extrinsics", type=Path, required=True)
    px.add_argument("--z", type=float, required=True, help="height of the plane to intersect, metres above the table")
    px.add_argument("uv", type=float, nargs=2)
    sub.add_parser("selftest")
    args = parser.parse_args()
    pattern = (args.cols, args.rows)

    if args.command == "selftest":
        selftest(pattern, args.square)
    elif args.command == "intrinsics":
        result = calibrate(args.images, pattern, args.square)
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
    elif args.command == "extrinsics":
        intr = json.loads(args.intrinsics.read_text())
        image = cv2.imread(str(args.image))
        R_cb, t_cb, corners, reprojection = board_pose(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.array(intr["K"]), np.array(intr["dist"]), pattern, args.square)
        draw_board_axes(image, corners, pattern[0], args.out.with_suffix(".png"))
        result = {"board_reprojection_px": reprojection}
        if args.touches:
            touches = json.loads(args.touches.read_text())
            R_bb, t_bb, residual = board_in_base_from_touches(touches, pattern, args.square, args.board_z, -R_cb.T @ t_cb)
            result["touch_residual_mm"] = (1000 * residual).round(1).tolist()
        else:
            R_bb, t_bb = board_in_base_from_ruler(args.origin, args.yaw, args.board_z, -R_cb.T @ t_cb)
        R_bc, t_bc = compose_camera_in_base(R_bb, t_bb, R_cb, t_cb)
        result |= {"R_base_cam": R_bc.tolist(), "t_base_cam": t_bc.tolist()}
        args.out.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(f"camera at {np.round(t_bc, 3).tolist()} m in the base frame; check this against a tape measure")
    elif args.command == "pixel":
        intr, extr = json.loads(args.intrinsics.read_text()), json.loads(args.extrinsics.read_text())
        print(pixel_to_base(args.uv, intr["K"], intr["dist"], extr["R_base_cam"], extr["t_base_cam"], args.z).round(4).tolist())


if __name__ == "__main__":
    main()
