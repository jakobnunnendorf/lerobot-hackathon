import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

# (hue 0-179, chroma = saturation * value / 255, value)
HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]
CONFIG_PATH = Path(__file__).with_name("colours.json")


@dataclass(frozen=True)
class ColourSpec:
    name: str
    ranges: tuple[HsvRange, ...]
    min_area: int
    min_roundness: float = 0.0
    max_area: int = 1_000_000
    close_px: int = 5
    # One gray ball is about this large; `pieces` is area over it. Zero means "do not estimate".
    piece_area: int = 0
    use_roi: bool = True
    # Opening with this diameter removes thin things: the red cable along the arm is about 5 px wide.
    min_thickness: int = 9


@dataclass(frozen=True)
class Detection:
    present: bool
    centre: tuple[float, float] | None = None
    area: int = 0
    pieces: int = 0


# Colour ranges use chroma (S*V/255) and not saturation: the black bowl has noisy, highly saturated dark
# pixels. Gray foil is the opposite case, low chroma and bright; its tinted reflections stay under the
# chroma ceiling and its blown-out highlights (S near 0) stay inside it. See decisions/colour-choice.md.
BLUE = (((100, 50, 60), (130, 255, 255)),)
# Red wraps around hue 0; the orange arm sits at hue 8-20, so red stops at 6 (and starts at 172).
RED = (((0, 80, 60), (6, 255, 255)), ((172, 80, 60), (179, 255, 255)))
DEFAULT_SPECS = (
    ColourSpec("striped", (((28, 45, 80), (48, 255, 255)),), min_area=150, min_roundness=0.25, max_area=2200),
    ColourSpec("red", RED, min_area=150, min_roundness=0.25, max_area=2200),
    ColourSpec("blue", BLUE, min_area=150, min_roundness=0.25, max_area=2200),
    # Containers: the same colours, but large and anywhere in the frame (they stand outside the bowl).
    ColourSpec("red_container", RED, min_area=4000, close_px=9, use_roi=False),
    ColourSpec("blue_container", BLUE, min_area=2500, close_px=9, use_roi=False),
    ColourSpec("gray", (((0, 0, 110), (179, 40, 255)),), min_area=300, close_px=7, piece_area=1700),
)
# Ellipse (centre x, centre y, radius x, radius y) as frame fractions around the bowl opening in the top
# camera of molnrb/so101_pick_blue_ball; the camera moved a little between its episodes. Outside it lie the
# blue and gray tape rolls and the white table. Re-measure at the scene freeze.
DEFAULT_ROI = (0.60, 0.38, 0.23, 0.24)
Roi = tuple[float, float, float, float]


def _disc(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _roundness(blob: np.ndarray) -> float:
    points = cv2.findNonZero(blob)
    (x, y), radius = cv2.minEnclosingCircle(points)
    circle = np.zeros(blob.shape, np.uint8)
    cv2.circle(circle, (round(x), round(y)), round(radius) + 1, 255, cv2.FILLED)
    return cv2.contourArea(cv2.convexHull(points)) / max(cv2.countNonZero(circle), 1)


def hue_chroma_value(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    return cv2.merge([hue, cv2.multiply(saturation, value, scale=1 / 255), value])


def roi_mask(shape: tuple[int, ...], roi: Roi | None) -> np.ndarray:
    height, width = shape[:2]
    if roi is None:
        return np.full((height, width), 255, np.uint8)
    cx, cy, rx, ry = roi
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (round(cx * width), round(cy * height)), (round(rx * width), round(ry * height)), 0, 0, 360, 255, cv2.FILLED)
    return mask


def detect_one(hcv: np.ndarray, spec: ColourSpec, roi: Roi | None = None) -> Detection:
    mask = np.zeros(hcv.shape[:2], np.uint8)
    for low, high in spec.ranges:
        mask |= cv2.inRange(hcv, low, high)
    mask &= roi_mask(hcv.shape, roi)
    # Closing joins the stripes of the target ball (cream gaps) and the facets of a foil ball into one blob.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disc(spec.close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disc(spec.min_thickness))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = 0
    best: tuple[int, tuple[float, float]] | None = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not spec.min_area <= area <= spec.max_area:
            continue
        if spec.min_roundness and _roundness((labels == label).astype(np.uint8)) < spec.min_roundness:
            continue
        total += area
        if best is None or area > best[0]:
            centre = centroids[label]
            best = (area, (float(centre[0]), float(centre[1])))
    if best is None:
        return Detection(False)
    pieces = max(1, round(total / spec.piece_area)) if spec.piece_area else 0
    return Detection(True, best[1], total if spec.piece_area else best[0], pieces)


@dataclass
class ColourDetector:
    specs: tuple[ColourSpec, ...] = DEFAULT_SPECS
    roi: Roi | None = DEFAULT_ROI
    _by_name: dict[str, ColourSpec] = field(init=False, repr=False)

    def __post_init__(self):
        self._by_name = {spec.name: spec for spec in self.specs}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def detect(self, rgb: np.ndarray) -> dict[str, Detection]:
        hcv = hue_chroma_value(rgb)
        return {spec.name: detect_one(hcv, spec, self.roi if spec.use_roi else None) for spec in self.specs}

    def area(self, rgb: np.ndarray, name: str) -> int:
        spec = self._by_name[name]
        return detect_one(hue_chroma_value(rgb), spec, self.roi if spec.use_roi else None).area


def load_detector(path: Path = CONFIG_PATH) -> ColourDetector:
    if not path.exists():
        return ColourDetector()
    data = json.loads(path.read_text())
    specs = tuple(
        ColourSpec(
            name=item["name"], ranges=tuple((tuple(low), tuple(high)) for low, high in item["ranges"]),
            min_area=item["min_area"], min_roundness=item.get("min_roundness", 0.0),
            max_area=item.get("max_area", 1_000_000), close_px=item.get("close_px", 5),
            piece_area=item.get("piece_area", 0), use_roi=item.get("use_roi", True), min_thickness=item.get("min_thickness", 9))
        for item in data["colours"])
    roi = tuple(data["roi"]) if data.get("roi") else None
    return ColourDetector(specs, roi)


def save_detector(detector: ColourDetector, path: Path = CONFIG_PATH) -> None:
    colours = [{"name": s.name, "ranges": [[list(low), list(high)] for low, high in s.ranges],
                "min_area": s.min_area, "min_roundness": s.min_roundness, "max_area": s.max_area,
                "close_px": s.close_px, "piece_area": s.piece_area, "use_roi": s.use_roi, "min_thickness": s.min_thickness} for s in detector.specs]
    path.write_text(json.dumps({"roi": list(detector.roi) if detector.roi else None, "colours": colours}, indent=2) + "\n")


def calibrate(present: list[np.ndarray], boxes: list[tuple[int, int, int, int]], absent: list[np.ndarray],
              name: str, base: ColourSpec | None = None, margin_hue: int = 3,
              min_pixel_chroma: int = 35) -> ColourSpec:
    """Thresholds for one colour from frames where the ball lies inside `box` (x0, y0, x1, y1 pixels)
    and, optionally, frames where it is absent (used only to raise the chroma floor and the minimum area)."""
    samples = []
    for rgb, (x0, y0, x1, y1) in zip(present, boxes):
        window = hue_chroma_value(rgb)[y0:y1, x0:x1].reshape(-1, 3)
        samples.append(window[window[:, 1] >= min_pixel_chroma])
    pixels = np.concatenate(samples)
    if len(pixels) < 50:
        raise ValueError(f"only {len(pixels)} coloured pixels in the boxes: wrong box, or a gray ball")
    # The box also holds bowl and background: keep the hue peak (histogram mode +- 8), not the percentiles of
    # everything. Red wraps around hue 0, so the hues are turned by 90 first.
    hues = pixels[:, 0].astype(int)
    turn = 90 if min(np.mean(hues < 15), 1) + np.mean(hues > 165) > 0.3 else 0
    hues = (hues + turn) % 180
    histogram = np.bincount(hues, minlength=180).astype(float)
    smooth = np.convolve(np.concatenate([histogram[-8:], histogram, histogram[:8]]), np.ones(9), "same")[8:-8]
    peak = int(np.argmax(smooth))
    keep = np.abs(hues - peak) <= 8
    near, near_hues = pixels[keep], hues[keep]
    hue_low, hue_high = (int(v) for v in np.percentile(near_hues, [2, 98]))
    chroma_floor = int(np.percentile(near[:, 1], 5))
    value_floor = max(int(np.percentile(near[:, 2], 2)) - 15, 0)
    low = (hue_low - margin_hue - turn) % 180
    high = (hue_high + margin_hue - turn) % 180
    ranges = [((low, chroma_floor, value_floor), (high, 255, 255))]
    if low > high:
        ranges = [((low, chroma_floor, value_floor), (179, 255, 255)), ((0, chroma_floor, value_floor), (high, 255, 255))]
    spec = base or ColourSpec(name, (), min_area=150, min_roundness=0.25, max_area=2200)
    spec = replace(spec, name=name, ranges=tuple(ranges))
    for rgb in absent:
        while spec.ranges[0][0][1] < 200 and detect_one(hue_chroma_value(rgb), spec).present:
            spec = replace(spec, ranges=tuple(((l[0], l[1] + 10, l[2]), h) for l, h in spec.ranges))
    return spec


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotate(rgb: np.ndarray, detections: dict[str, Detection]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    palette = {"striped": (0, 255, 0), "blue": (255, 0, 0), "red": (0, 0, 255), "gray": (0, 255, 255)}
    for index, (name, hit) in enumerate(detections.items()):
        colour = palette.get(name, (255, 0, 255))
        label = f"{name}: {hit.area}px x{hit.pieces}" if hit.present else f"{name}: -"
        cv2.putText(out, label, (8, 20 + 18 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if hit.present:
            cv2.circle(out, (round(hit.centre[0]), round(hit.centre[1])), 14, colour, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Colour balls in a top-camera frame. Reads image files only; never touches the robot.")
    sub = parser.add_subparsers(dest="mode", required=True)
    detect = sub.add_parser("detect")
    detect.add_argument("images", nargs="+")
    detect.add_argument("--config", default=str(CONFIG_PATH))
    detect.add_argument("--save-annotated", help="directory for annotated copies")
    cal = sub.add_parser("calibrate", help="thresholds of one colour from frames with a box around the ball")
    cal.add_argument("--name", required=True)
    cal.add_argument("--present", nargs="+", required=True, metavar="IMAGE")
    cal.add_argument("--box", nargs="+", required=True, metavar="X0,Y0,X1,Y1", help="one per --present image, pixels")
    cal.add_argument("--absent", nargs="*", default=[], metavar="IMAGE", help="frames without this colour in the bowl")
    cal.add_argument("--config", default=str(CONFIG_PATH))
    cal.add_argument("--write", action="store_true", help="store the result in the config (default: print only)")
    args = parser.parse_args()

    config = Path(args.config)
    detector = load_detector(config)
    if args.mode == "detect":
        out_dir = Path(args.save_annotated) if args.save_annotated else None
        for path in args.images:
            rgb = _read_rgb(path)
            detections = detector.detect(rgb)
            print(path, {n: (h.area, h.pieces) for n, h in detections.items() if h.present} or "nothing")
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / Path(path).name), annotate(rgb, detections))
        return
    boxes = [tuple(int(v) for v in b.split(",")) for b in args.box]
    if len(boxes) != len(args.present):
        raise SystemExit("--box needs one box per --present image")
    base = next((s for s in detector.specs if s.name == args.name), None)
    spec = calibrate([_read_rgb(p) for p in args.present], boxes, [_read_rgb(p) for p in args.absent], args.name, base)
    print(json.dumps({"name": spec.name, "ranges": spec.ranges, "min_area": spec.min_area}))
    if args.write:
        others = tuple(s for s in detector.specs if s.name != spec.name)
        save_detector(ColourDetector(others + (spec,), detector.roi), config)
        print(f"written to {config}")


if __name__ == "__main__":
    main()
