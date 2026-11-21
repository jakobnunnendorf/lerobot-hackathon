import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "calib"))
from board import Board  # noqa: E402
from calibrate_extrinsics import board_extrinsics  # noqa: E402
from calibrate_extrinsics_touch import solve_touch  # noqa: E402
from calibrate_intrinsics import run_calibration  # noqa: E402
from pixel_to_base import TopCamera, invert, pose_difference, rotation_z, transform, undistort_to_normalized  # noqa: E402

SIZE = (640, 480)
TRUE_K = np.array([[610.0, 0.0, 318.5], [0.0, 607.0, 243.2], [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.25, 0.09, 0.0007, -0.0005, -0.01])
BOARD = Board(7, 5, 0.02875, 0.0290, 0.003)
CONFIG = {"camera": {"width": 640, "height": 480, "index_or_path": "synthetic", "fourcc": "MJPG", "fps": 30},
          "board": {"inner_corners": [7, 5], "square_x_m": 0.02875, "square_y_m": 0.0290, "board_thickness_m": 0.003}}
SUPERSAMPLE = 3
PAPER_PX_PER_M = 4000
# As on the team's board (board.png): wide white margin at the short ends, narrower along the long sides, cardboard
# around the paper with white round stickers, and a square of red tape on one cardboard corner.
MARGIN_M = np.array([0.034, 0.021])
CARD_M = 0.006
STICKER_RADIUS_M = 0.0065
TAPE_M = 0.018
PAPER_BGR, BLACK_BGR, CARD_BGR, STICKER_BGR, TAPE_BGR = (235, 235, 235), (25, 25, 25), (90, 140, 180), (240, 240, 240), (30, 50, 230)
ARM_BGR = (20, 70, 240)
CARD_CORNERS = {"top-left": (0, 0), "top-right": (1, 0), "bottom-left": (0, 1), "bottom-right": (1, 1)}


def look_at(position, target, image_up_hint):
    forward = np.subtract(target, position) / np.linalg.norm(np.subtract(target, position))
    down = np.subtract(0, image_up_hint) - forward * (forward @ np.subtract(0, image_up_hint))
    down /= np.linalg.norm(down)
    return transform(np.column_stack([np.cross(down, forward), down, forward]), position)


def card_size_m(board):
    squares = np.array([(board.long_corners + 1) * board.square_x_m, (board.short_corners + 1) * board.square_y_m])
    return squares + 2 * MARGIN_M + 2 * CARD_M


def inner_corner_on_card(board, column, row):
    return np.array([*(CARD_M + MARGIN_M + ((column + 1) * board.square_x_m, (row + 1) * board.square_y_m)), 0.0])


def numbering(tape_corner, board=BOARD):
    fx, fy = CARD_CORNERS[tape_corner]
    last_column, last_row = board.long_corners - 1, board.short_corners - 1
    column, row = fx * last_column, fy * last_row
    return {1: (column, row), 2: (last_column - column, row), 3: (last_column - column, last_row - row), 4: (column, last_row - row)}


def card_texture(board, tape_corner):
    size = card_size_m(board)
    width, height = (size * PAPER_PX_PER_M).round().astype(int)
    texture = np.empty((height, width, 3), np.uint8)
    texture[:] = CARD_BGR
    to_px = lambda metres: np.round(np.asarray(metres) * PAPER_PX_PER_M).astype(int)  # noqa: E731
    paper = to_px([CARD_M, CARD_M, size[0] - CARD_M, size[1] - CARD_M])
    texture[paper[1]:paper[3], paper[0]:paper[2]] = PAPER_BGR
    for row in range(board.short_corners + 1):
        for column in range(board.long_corners + 1):
            if (row + column) % 2 == 0:
                x0, y0 = CARD_M + MARGIN_M + (column * board.square_x_m, row * board.square_y_m)
                box = to_px([x0, y0, x0 + board.square_x_m, y0 + board.square_y_m])
                texture[box[1]:box[3], box[0]:box[2]] = BLACK_BGR
    inset = CARD_M + 0.003
    for fx, fy in ((0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)):
        if (fx, fy) != CARD_CORNERS.get(tape_corner):
            centre = to_px([inset + fx * (size[0] - 2 * inset), inset + fy * (size[1] - 2 * inset)])
            cv2.circle(texture, (int(centre[0]), int(centre[1])), int(STICKER_RADIUS_M * PAPER_PX_PER_M), STICKER_BGR, -1)
    if tape_corner:
        fx, fy = CARD_CORNERS[tape_corner]
        x0, y0 = to_px([fx * (size[0] - TAPE_M), fy * (size[1] - TAPE_M)])
        texture[y0:y0 + to_px(TAPE_M), x0:x0 + to_px(TAPE_M)] = TAPE_BGR
    return texture


def table_texture(rng):
    texture = cv2.cvtColor(cv2.GaussianBlur(rng.integers(70, 150, (900, 900)).astype(np.uint8), (0, 0), 3), cv2.COLOR_GRAY2BGR)
    texture[430:470, 310:350] = ARM_BGR  # an orange-red patch near the robot base, larger than the tape, like the arm
    return texture


class Renderer:
    def __init__(self, board, K=TRUE_K, dist=TRUE_DIST, size=SIZE, seed=0):
        self.board, self.size = board, size
        fine_K = K.copy()
        fine_K[:2] *= SUPERSAMPLE
        fine_K[:2, 2] += (SUPERSAMPLE - 1) / 2
        u, v = np.meshgrid(np.arange(size[0] * SUPERSAMPLE), np.arange(size[1] * SUPERSAMPLE))
        normalized = undistort_to_normalized(np.column_stack([u.ravel(), v.ravel()]), fine_K, dist)
        self.rays = np.column_stack([normalized, np.ones(len(normalized))])
        self.cards = {}
        self.rng = np.random.default_rng(seed)
        self.table = table_texture(self.rng)

    def _plane_lookup(self, T_cam_plane, texture, origin_m, px_per_m):
        q = self.rays @ np.linalg.inv(T_cam_plane[:3, [0, 1, 3]]).T
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = (q[:, :2] / q[:, 2:3] - origin_m) * px_per_m - 0.5
        shape = (self.size[1] * SUPERSAMPLE, self.size[0] * SUPERSAMPLE)
        maps = [np.where(q[:, 2] > 0, pixels[:, i], -1e6).reshape(shape).astype(np.float32) for i in (0, 1)]
        image = cv2.remap(texture, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside = cv2.remap(np.full(texture.shape[:2], 255, np.uint8), *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image.astype(np.float32), (inside.astype(np.float32) / 255)[..., None]

    def render(self, T_cam_card, T_cam_table=None, warp_px=0.0, tape_corner=None):
        if tape_corner not in self.cards:
            self.cards[tape_corner] = card_texture(self.board, tape_corner)
        image = np.full((self.size[1] * SUPERSAMPLE, self.size[0] * SUPERSAMPLE, 3), 60.0, np.float32)
        if T_cam_table is not None:
            table, inside = self._plane_lookup(T_cam_table, self.table, np.array([-0.6, -0.9]), 500)
            image = inside * table + (1 - inside) * image
        card, inside = self._plane_lookup(T_cam_card, self.cards[tape_corner], np.zeros(2), PAPER_PX_PER_M)
        image = cv2.resize(inside * card + (1 - inside) * image, self.size, interpolation=cv2.INTER_AREA)
        if warp_px:
            u, v = np.meshgrid(np.arange(self.size[0], dtype=np.float32), np.arange(self.size[1], dtype=np.float32))
            bend = warp_px * np.sin(np.pi * u / self.size[0]) * np.sin(np.pi * v / self.size[1])
            image = cv2.remap(image, u + bend, v + 0.6 * bend, cv2.INTER_LINEAR)
        image = cv2.GaussianBlur(image, (0, 0), 0.5) + self.rng.normal(0, 1.5, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)


def project_card_points(T_cam_card, points):
    in_camera = points @ T_cam_card[:3, :3].T + T_cam_card[:3, 3]
    return cv2.projectPoints(in_camera, np.zeros(3), np.zeros(3), TRUE_K, TRUE_DIST)[0].reshape(-1, 2), in_camera


def random_held_pose(rng, board, centre_px):
    card_centre = np.append(card_size_m(board) / 2, 0.0)
    ends = np.array([inner_corner_on_card(board, *ij) for ij in numbering("top-left").values()])
    outline = ends + np.sign(ends - card_centre) * [board.square_x_m * 1.5, board.square_y_m * 1.5, 0]
    for _ in range(1000):
        ray = np.append(undistort_to_normalized([centre_px], TRUE_K, TRUE_DIST)[0], 1.0)
        tilt = cv2.Rodrigues(rotation_z(rng.uniform(0, 360))[:, 0] * np.radians(rng.uniform(0, 45)))[0]
        R = tilt @ rotation_z(rng.uniform(0, 360))
        T = transform(R, ray / np.linalg.norm(ray) * rng.uniform(0.33, 0.75) - R @ card_centre)
        pixels, in_camera = project_card_points(T, outline)
        if invert(T)[2, 3] < 0 and (in_camera[:, 2] > 0.1).all() and (pixels > 2).all() and (pixels < np.subtract(SIZE, 2)).all():
            return T
    raise RuntimeError("no visible pose found")


def board_on_table(board, tape_corner, corner1_xy, yaw_deg):
    corner1 = numbering(tape_corner, board)[1]
    phi = np.radians(yaw_deg if corner1[0] == 0 else yaw_deg + 180)
    x_axis, y_axis = np.array([np.cos(phi), np.sin(phi), 0.0]), np.array([np.sin(phi), -np.cos(phi), 0.0])
    R_base_card = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
    return transform(R_base_card, np.array([*corner1_xy, board.thickness_m]) - R_base_card @ inner_corner_on_card(board, *corner1))


def corner_in_base(T_base_card, tape_corner, number):
    return (T_base_card @ np.append(inner_corner_on_card(BOARD, *numbering(tape_corner)[number]), 1.0))[:3]


def direction_name(vector):
    return f"{'top' if vector[1] < 0 else 'bottom'}-{'left' if vector[0] < 0 else 'right'}"


def opposite(corner_name):
    vertical, horizontal = corner_name.split("-")
    return f"{'bottom' if vertical == 'top' else 'top'}-{'right' if horizontal == 'left' else 'left'}"


TRUE_T_BASE_CAM = look_at([0.30, -0.03, 0.72], [0.27, 0.0, 0.0], image_up_hint=[1.0, 0.0, 0.0])
TRUE_CAMERA = TopCamera(TRUE_K, TRUE_DIST, SIZE, TRUE_T_BASE_CAM)
HAND_PIXELS = [(u, v) for u in (100, 320, 540) for v in (80, 240, 400)]


def hand_plane_error_mm(T_estimate, z=0.15, K=TRUE_K, dist=TRUE_DIST):
    estimate = TopCamera(K, dist, SIZE, T_estimate)
    errors = [np.inf if (hit := estimate.pixel_to_base(u, v, z)) is None else 1000 * np.linalg.norm(TRUE_CAMERA.pixel_to_base(u, v, z) - hit)
              for u, v in HAND_PIXELS]
    return max(errors)


def model_difference_px(K, dist):
    u, v = np.meshgrid(np.linspace(0, SIZE[0], 17), np.linspace(0, SIZE[1], 13))
    normalized = undistort_to_normalized(np.column_stack([u.ravel(), v.ravel()]), TRUE_K, TRUE_DIST)
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    estimated = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    return float(np.linalg.norm(estimated - np.column_stack([u.ravel(), v.ravel()]), axis=1).max())


def check_intrinsics(renderer, folder, views=14):
    rng = np.random.default_rng(1)
    centres = [(u, v) for u in (130, 320, 510) for v in (110, 240, 370)] + [tuple(rng.uniform([130, 110], [510, 370])) for _ in range(views - 9)]
    for index, centre in enumerate(centres):
        cv2.imwrite(str(folder / f"board_{index:02d}.png"), renderer.render(random_held_pose(rng, BOARD, centre), tape_corner="top-right"))
    cv2.imwrite(str(folder / "board_bent.png"), renderer.render(random_held_pose(rng, BOARD, (320, 240)), warp_px=6.0))
    rms, K, dist, errors, dropped = run_calibration(folder, folder / "intrinsics.yaml", CONFIG)
    print(f"intrinsics: RMS {rms:.3f} px, dropped {dropped}, fx {K[0, 0] - TRUE_K[0, 0]:+.2f} fy {K[1, 1] - TRUE_K[1, 1]:+.2f} "
          f"cx {K[0, 2] - TRUE_K[0, 2]:+.2f} cy {K[1, 2] - TRUE_K[1, 2]:+.2f} px against the truth, "
          f"largest pixel difference of the whole model over the image {model_difference_px(K, dist):.2f} px")
    return rms < 0.5 and dropped == ["board_bent.png"], K, dist


def extrinsics_cases(T_base_card, tape_corner, yaw):
    corner1, corner3 = (corner_in_base(T_base_card, tape_corner, n) for n in (1, 3))
    pixel1, centre = TRUE_CAMERA.base_to_pixel([corner1, (corner1 + corner3) / 2])
    right_rule = direction_name(pixel1 - centre)
    corners = {"board_corners_in_base_mm": {n: (1000 * corner_in_base(T_base_card, tape_corner, n)[:2]).round(1).tolist() for n in range(1, 5)}}
    origin_and_yaw = {"board_origin_in_base_xyz_m": [corner1[0], corner1[1], 0.0], "board_yaw_in_base_deg": yaw}
    return {
        "corners, red tape only": (True, corners, "right"),
        "corners, right rule, no tape": (False, {**corners, "origin_image_corner": right_rule}, "right"),
        "corners, wrong rule, no tape": (False, {**corners, "origin_image_corner": opposite(right_rule)}, "mirrored"),
        "corners, wrong rule, red tape": (True, {**corners, "origin_image_corner": opposite(right_rule)}, "refused"),
        "corners, camera position, no tape": (False, {**corners, "camera_position_in_base_m": [0.32, -0.05, 0.70]}, "right"),
        "origin and yaw, red tape": (True, origin_and_yaw, "right"),
        "origin and yaw, right rule, no tape": (False, {**origin_and_yaw, "origin_image_corner": right_rule}, "right"),
    }


PLACEMENTS = (("top-right", 0.20, 0.08, 90.0), ("top-right", 0.20, -0.08, 270.0), ("top-left", 0.24, -0.05, 0.0), ("top-left", 0.30, 0.10, 180.0))


def check_extrinsics(renderer, K, dist, placements=PLACEMENTS):
    ok = True
    T_cam_table = invert(TRUE_T_BASE_CAM)
    for tape_corner, x, y, yaw in placements:
        T_base_card = board_on_table(BOARD, tape_corner, (x, y), yaw)
        images = {tape: renderer.render(T_cam_table @ T_base_card, T_cam_table, tape_corner=tape_corner if tape else None) for tape in (True, False)}
        for name, (tape, extrinsics, expected) in extrinsics_cases(T_base_card, tape_corner, yaw).items():
            result = board_extrinsics([images[tape]], BOARD, K, dist, extrinsics)
            label = f"tape {tape_corner:9s} corner 1 at ({x:.2f}, {y:+.2f}) yaw {yaw:5.1f}, {name:35s}"
            if "error" in result:
                print(f"{label}: refused: {result['error'][:60]}")
                ok &= expected == "refused"
                continue
            moved_m, turned_deg = pose_difference(result["T_base_cam"], TRUE_T_BASE_CAM)
            hand_mm = hand_plane_error_mm(result["T_base_cam"], K=K, dist=dist)
            print(f"{label}: z {'up  ' if result['chosen'].z_up else 'down'} camera off by {1000 * moved_m:6.1f} mm, {turned_deg:6.2f} deg, "
                  f"hand plane {hand_mm:6.1f} mm")
            ok &= hand_mm < 10 if expected == "right" else (expected == "mirrored" and moved_m > 0.05)
    return ok


def check_touch():
    rng = np.random.default_rng(2)
    points = np.column_stack([rng.uniform(0.15, 0.38, 8), rng.uniform(-0.15, 0.15, 8), rng.uniform(0.0, 0.25, 8)])
    pixels = TRUE_CAMERA.base_to_pixel(points) + rng.normal(0, 2.0, (8, 2))
    T_base_cam, errors = solve_touch(points + rng.normal(0, 0.005, points.shape), pixels, TRUE_K, TRUE_DIST)
    moved_m, turned_deg = pose_difference(T_base_cam, TRUE_T_BASE_CAM)
    print(f"touch, 8 points at 0-25 cm, 2 px click and 5 mm tip noise: camera off by {1000 * moved_m:.1f} mm, {turned_deg:.2f} deg, "
          f"reprojection {np.sqrt(np.mean(errors ** 2)):.1f} px, hand plane {hand_plane_error_mm(T_base_cam):.1f} mm")
    return hand_plane_error_mm(T_base_cam) < 30


def check_round_trip():
    worst = 0.0
    for u, v in [(0, 0), (639, 479), (320, 240), (12, 400)]:
        for z in (0.0, 0.15, 0.3):
            worst = max(worst, float(np.linalg.norm(TRUE_CAMERA.base_to_pixel(TRUE_CAMERA.pixel_to_base(u, v, z))[0] - (u, v))))
    sideways = TopCamera(TRUE_K, TRUE_DIST, SIZE, look_at([0, 0, 0.3], [1, 0, 0.3], [0, 0, 1]))
    none_when_no_hit = sideways.pixel_to_base(318.5, 243.2, 0.3) is None and TRUE_CAMERA.pixel_to_base(320, 240, 1.0) is None
    print(f"round trip pixel -> base -> pixel: worst {worst:.1e} px; parallel ray and plane above the camera give None: {none_when_no_hit}")
    return worst < 1e-3 and none_when_no_hit


def main():
    renderer = Renderer(BOARD)
    with tempfile.TemporaryDirectory() as folder:
        intrinsics_ok, K, dist = check_intrinsics(renderer, Path(folder))
        results = [intrinsics_ok, check_extrinsics(renderer, K, dist), check_touch(), check_round_trip()]
    print("PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
