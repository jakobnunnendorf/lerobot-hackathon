from dataclasses import dataclass

import cv2
import numpy as np

HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class CameraSettings:
    core: HsvRange
    edges: tuple[HsvRange, ...]
    min_core_area: int
    tracked_core_area: int
    min_piece_area: int
    seed_close: int
    max_grow: int
    min_roundness: float | None = None
    max_pale_hue: int | None = None


# Core ranges are (hue, chroma, value), edge ranges (hue, saturation, value). Why these
# numbers, and why only the wrist camera checks roundness and paleness: decisions/other-ball-colours.md.
CAMERAS = {
    "top": CameraSettings(
        core=((28, 45, 80), (48, 255, 255)),
        edges=(((21, 35, 60), (60, 255, 255)),),
        min_core_area=40, tracked_core_area=12, min_piece_area=10, seed_close=5, max_grow=4,
    ),
    "wrist": CameraSettings(
        core=((30, 32, 40), (86, 255, 255)),
        edges=(((21, 20, 50), (95, 255, 255)), ((16, 100, 60), (21, 255, 200))),
        min_core_area=200, tracked_core_area=60, min_piece_area=80, seed_close=15, max_grow=12,
        min_roundness=0.55, max_pale_hue=33,
    ),
}

# Orange arm and gripper, including its overexposed shade. Lit cream on a ball held close to the
# wrist camera falls into the last range too and is cut out: see decisions/other-ball-colours.md.
GRIPPER = (((0, 80, 0), (15, 255, 255)), ((170, 80, 0), (180, 255, 255)), ((15, 50, 200), (21, 255, 255)))
PALE_SATURATION = 110
TRACKING_RADIUS = 40
DISTANCE_DISCOUNT = 50.0


def _disc(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _in_any_range(hsv: np.ndarray, ranges: tuple[HsvRange, ...]) -> np.ndarray:
    hit = cv2.inRange(hsv, *ranges[0])
    for low, high in ranges[1:]:
        hit |= cv2.inRange(hsv, low, high)
    return hit


# Chroma (S*V) rather than saturation: the black bowl has noisy, highly saturated dark pixels.
def _hue_chroma_value(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = cv2.split(hsv)
    return cv2.merge([hue, cv2.multiply(saturation, value, scale=1 / 255), value])


# The ball is convex, so the hull of its stripes closes the notches that dark or blown-out
# cream leaves between them.
def _hull(mask: np.ndarray) -> np.ndarray:
    hull = np.zeros_like(mask)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(mask)), 255)
    return hull


def _filled(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _core_area(labels: np.ndarray, half_core: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(labels.ravel(), weights=half_core.ravel(), minlength=count) * 4 / 255


# Hull area over the part of the enclosing circle inside the image, so a ball cut by the image
# border still counts as round while stripes of mat or tape do not.
def _roundness(component: np.ndarray) -> float:
    points = cv2.findNonZero(component.astype(np.uint8))
    (x, y), radius = cv2.minEnclosingCircle(points)
    circle = np.zeros(component.shape, np.uint8)
    cv2.circle(circle, (round(x), round(y)), round(radius) + 1, 255, cv2.FILLED)
    return cv2.contourArea(cv2.convexHull(points)) / max(cv2.countNonZero(circle), 1)


# Overexposed white table tops and labels come out pale yellow; the ball is greener or, where
# it is as yellow, more saturated.
def _pale(pixels: np.ndarray, max_hue: int) -> bool:
    return len(pixels) > 0 and np.median(pixels[:, 0]) <= max_hue and np.median(pixels[:, 1]) < PALE_SATURATION


class TargetMasker:
    def __init__(self, camera: str):
        self.settings = CAMERAS[camera]
        self.last_centre: np.ndarray | None = None

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        core = cv2.inRange(_hue_chroma_value(hsv), *self.settings.core)
        mask = np.zeros(rgb.shape[:2], np.uint8)
        # Seeds are searched at half resolution to keep two cameras within a 30 Hz budget;
        # INTER_AREA keeps thin stripes that nearest-neighbour downscaling would drop.
        seed = self._seed(cv2.resize(core, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), hsv[::2, ::2])
        if seed is not None:
            box = self._box(seed, mask.shape)
            full_seed = cv2.resize(seed, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)[box]
            mask[box] = self._outline(full_seed, hsv[box], core[box])
        return mask

    def _outline(self, seed: np.ndarray, hsv: np.ndarray, core: np.ndarray) -> np.ndarray:
        ball_coloured = core | _in_any_range(hsv, self.settings.edges)
        grown = ball_coloured & cv2.dilate(seed, _disc(2 * self._grow(seed) + 1))
        grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, _disc(3)) | _hull(seed)
        return _filled(grown) & ~_in_any_range(hsv, GRIPPER)

    # Cream rims only show when the stripes are resolved; around a distant ball the same edge
    # colours are the ball's reflection on the foil, so growth scales with the ball.
    def _grow(self, seed: np.ndarray) -> int:
        _, _, w, h = cv2.boundingRect(seed)
        return int(np.clip(0.05 * max(w, h), 1, self.settings.max_grow))

    def _box(self, half_seed: np.ndarray, shape: tuple[int, int]) -> tuple[slice, slice]:
        x, y, w, h = cv2.boundingRect(half_seed)
        pad = self.settings.max_grow + 2
        return (slice(max(2 * y - pad, 0), min(2 * (y + h) + pad, shape[0])),
                slice(max(2 * x - pad, 0), min(2 * (x + w) + pad, shape[1])))

    def _pieces(self, core: np.ndarray) -> np.ndarray:
        closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _disc(self.settings.seed_close // 2 | 1))
        count, labels = cv2.connectedComponents(closed)
        area = _core_area(labels, core, count)
        return np.where(area[labels] >= self.settings.min_piece_area, closed, 0).astype(np.uint8)

    # Pieces are grouped by a dilation, not joined by a larger closing: a shadowed half of the
    # ball or a jaw across it splits the stripes, and a larger closing would bridge into the foil.
    def _seed(self, core: np.ndarray, hsv: np.ndarray) -> np.ndarray | None:
        pieces = self._pieces(core)
        grouped = cv2.dilate(pieces, _disc(self.settings.seed_close | 1))
        count, labels, _, centres = cv2.connectedComponentsWithStats(grouped)
        centres = 2 * centres
        core_area = _core_area(labels, core, count)
        distance = self._distance_to_last(centres)
        candidates = [label for label in self._large_enough(core_area, distance)
                      if self._looks_like_ball(labels == label, core, hsv)]
        if not candidates:
            self.last_centre = None
            return None
        score = core_area / (1.0 + np.where(np.isfinite(distance), distance, 0) / DISTANCE_DISCOUNT)
        best = max(candidates, key=lambda label: score[label])
        self.last_centre = centres[best]
        return np.where((labels == best) & (pieces > 0), 255, 0).astype(np.uint8)

    def _distance_to_last(self, centres: np.ndarray) -> np.ndarray:
        if self.last_centre is None:
            return np.full(len(centres), np.inf)
        return np.linalg.norm(centres - self.last_centre, axis=1)

    # Near the last known position a smaller seed is accepted, so a ball mostly hidden by the
    # gripper stays tracked.
    def _large_enough(self, core_area: np.ndarray, distance: np.ndarray) -> list[int]:
        s = self.settings
        large = (core_area >= s.min_core_area) | ((core_area >= s.tracked_core_area) & (distance < TRACKING_RADIUS))
        large[0] = False
        return np.flatnonzero(large).tolist()

    # Only the wrist camera sweeps the room, with green mats, yellow labels and white tables.
    def _looks_like_ball(self, component: np.ndarray, core: np.ndarray, hsv: np.ndarray) -> bool:
        s = self.settings
        if s.min_roundness is not None and _roundness(component) < s.min_roundness:
            return False
        return s.max_pale_hue is None or not _pale(hsv[component & (core > 127)], s.max_pale_hue)


def target_mask(rgb: np.ndarray, camera: str) -> np.ndarray:
    return TargetMasker(camera)(rgb)
