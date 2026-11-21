from dataclasses import dataclass

import cv2
import numpy as np

from pixel_to_base import from_rvec_tvec

SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
CLASSIC_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)


@dataclass(frozen=True)
class Board:
    long_corners: int
    short_corners: int
    square_x_m: float
    square_y_m: float
    thickness_m: float

    @classmethod
    def from_config(cls, config):
        board = config["board"]
        long_corners, short_corners = board["inner_corners"]
        if long_corners <= short_corners:
            raise ValueError("board.inner_corners must list the long side first, e.g. [7, 5]")
        return cls(long_corners, short_corners, board["square_x_m"], board["square_y_m"], board["board_thickness_m"])

    @property
    def patterns(self):
        return (self.long_corners, self.short_corners), (self.short_corners, self.long_corners)


def find_corners(gray, pattern):
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=SB_FLAGS)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=CLASSIC_FLAGS)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
    return corners.reshape(-1, 2).astype(np.float64) if found else None


def detect_grid(gray, board):
    for pattern in board.patterns:
        corners = find_corners(gray, pattern)
        if corners is not None:
            return long_side_rows(corners, pattern, board)
    return None


def long_side_rows(corners, pattern, board):
    # OpenCV lists the corners row by row with pattern[0] per row. Transposing a (5, 7) detection makes every row
    # run along the long side, so square_x_m is always used along the long side, whichever pattern size was found.
    grid = corners.reshape(pattern[1], pattern[0], 2)
    return grid if pattern[0] == board.long_corners else grid.transpose(1, 0, 2)


def object_points(board):
    rows, columns = np.mgrid[0:board.short_corners, 0:board.long_corners]
    return np.column_stack([columns.ravel() * board.square_x_m, rows.ravel() * board.square_y_m, np.zeros(rows.size)])


def orientations(grid):
    return [np.ascontiguousarray(grid[rows, columns]) for rows in (slice(None), slice(None, None, -1))
            for columns in (slice(None), slice(None, None, -1))]


def align_to(grid, reference):
    # The pattern looks the same turned by 180 degrees, so the detector may number it from either end between frames.
    return min(orientations(grid), key=lambda oriented: np.linalg.norm(oriented - reference, axis=2).max())


def grid_motion(grid, previous):
    return float(np.linalg.norm(align_to(grid, previous) - previous, axis=2).max())


def board_pose_in_camera(grid, board, K, dist):
    image_points, points = grid.reshape(-1, 2), object_points(board)
    _, rvec, tvec = cv2.solvePnP(points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(points, image_points, K, dist, rvec, tvec)
    projected = cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1, 2)
    return from_rvec_tvec(rvec, tvec), float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))


def faces_camera(T_cam_board):
    return bool(T_cam_board[:3, 2] @ T_cam_board[:3, 3] < 0)
