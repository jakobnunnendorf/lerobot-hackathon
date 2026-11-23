from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

CALIB_DIR = Path(__file__).resolve().parent
INTRINSICS_PATH = CALIB_DIR / "intrinsics.yaml"
EXTRINSICS_PATH = CALIB_DIR / "extrinsics.yaml"
RUNBOOK = "docs/camera-calibration.md"

# Plain undistortPoints runs 5 fixed iterations, which leaves an error towards the image corners with strong distortion.
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
# Below this |sin| of the angle between ray and plane, a small pixel error moves the intersection by metres.
MIN_RAY_SLOPE = 0.02


def transform(rotation=np.eye(3), translation=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.ravel(translation)
    return T


def invert(T):
    R, t = T[:3, :3], T[:3, 3]
    return transform(R.T, -R.T @ t)


def rotation_z(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def from_rvec_tvec(rvec, tvec):
    return transform(cv2.Rodrigues(np.asarray(rvec, float))[0], tvec)


def to_rvec_tvec(T):
    return cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1)


def pose_difference(T_a, T_b):
    translation_m = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    cos_angle = (np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1) / 2
    return translation_m, float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def undistort_to_normalized(pixels, K, dist):
    points = np.asarray(pixels, float).reshape(-1, 1, 2)
    if hasattr(cv2, "undistortPointsIter"):
        return cv2.undistortPointsIter(points, K, dist, None, None, UNDISTORT_CRITERIA).reshape(-1, 2)
    return cv2.undistortPoints(points, K, dist, criteria=UNDISTORT_CRITERIA).reshape(-1, 2)


def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_yaml(path, data):
    Path(path).write_text(yaml.safe_dump(plain(data), sort_keys=False, default_flow_style=None, width=120))


def read_yaml(path, what):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}. Make it first, see {RUNBOOK}.")
    return yaml.safe_load(path.read_text())


class TopCamera:
    def __init__(self, K, dist, image_size, T_base_cam):
        self.K = np.asarray(K, float)
        self.dist = np.asarray(dist, float).ravel()
        self.image_size = tuple(int(v) for v in image_size)
        self.T_base_cam = np.asarray(T_base_cam, float)
        self.T_cam_base = invert(self.T_base_cam)
        self.field_radius = self._field_radius()

    @classmethod
    def load(cls, intrinsics_path=INTRINSICS_PATH, extrinsics_path=EXTRINSICS_PATH, image_size=None):
        intrinsics = read_yaml(intrinsics_path, "Intrinsics")
        extrinsics = read_yaml(extrinsics_path, "Extrinsics")
        calibrated_size = (intrinsics["image_width"], intrinsics["image_height"])
        if image_size is not None and tuple(image_size) != calibrated_size:
            raise ValueError(f"Frames are {image_size[0]}x{image_size[1]} but the camera was calibrated at "
                             f"{calibrated_size[0]}x{calibrated_size[1]}; intrinsics are valid only at that size.")
        if extrinsics["intrinsics_created"] != intrinsics["created"]:
            raise ValueError(f"{extrinsics_path} was computed with the intrinsics of {extrinsics['intrinsics_created']}, "
                             f"{intrinsics_path} is from {intrinsics['created']}. Run calibrate_extrinsics.py again.")
        return cls(intrinsics["K"], intrinsics["dist"], calibrated_size, extrinsics["T_base_cam"])

    @property
    def position(self):
        return self.T_base_cam[:3, 3]

    def pixel_to_base(self, u, v, plane_z_m):
        direction = self.rays([(u, v)])[0]
        if abs(direction[2]) < MIN_RAY_SLOPE:
            return None
        distance = (plane_z_m - self.position[2]) / direction[2]
        if distance <= 0:
            return None
        return self.position + distance * direction

    def base_to_pixel(self, points_base):
        points = np.atleast_2d(np.asarray(points_base, float))
        in_camera = points @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        pixels = cv2.projectPoints(in_camera, np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.linalg.norm(in_camera[:, :2] / in_camera[:, 2:3], axis=1)
        # The distortion polynomial folds points far outside the view back into the image; drop them.
        pixels[~((in_camera[:, 2] > 0) & (radius <= self.field_radius))] = np.nan
        return pixels

    def rays(self, pixels):
        normalized = undistort_to_normalized(pixels, self.K, self.dist)
        directions = np.column_stack([normalized, np.ones(len(normalized))]) @ self.T_base_cam[:3, :3].T
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def _field_radius(self):
        w, h = self.image_size
        border = [(0, 0), (w / 2, 0), (w, 0), (w, h / 2), (w, h), (w / 2, h), (0, h), (0, h / 2)]
        return 1.1 * float(np.linalg.norm(undistort_to_normalized(border, self.K, self.dist), axis=1).max())
