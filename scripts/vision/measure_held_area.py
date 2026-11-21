"""Wrist-camera area of each ball colour while the gripper holds the target (at rest) or a covering ball.

Offline, from a recorded dataset; nothing touches the robot. Uses #24's detector set up the way the
supervisor (#17, run_supervisor.py:held_ball_area) does: no size limit, optional ellipse ROI.

    python scripts/vision/measure_held_area.py --vision-dir <colour-pick>/scripts/vision \
        --roi 0.5 0.5 0.25 0.3 --out held_area.json --overlay held_area_overlay.png
"""
import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

GRIPPER = 5
HOLD_RANGE = (6.0, 20.0)
OPEN_ABOVE = 25.0
COLOURS = ("striped", "red", "blue")


def held_frames(gripper: np.ndarray, episode: np.ndarray):
    """(episode, index, kind): the last second of every episode holds the target at rest; 10 frames before
    each release that follows >= 15 held frames is a covering ball over its container."""
    for e in np.unique(episode):
        ix = np.where(episode == e)[0]
        g = gripper[ix]
        for k in (-25, -15, -5):
            yield int(e), int(ix[k]), "target"
        i = 15
        while i < len(g) - 1:
            held_before = (g[i - 15:i - 2] >= HOLD_RANGE[0]) & (g[i - 15:i - 2] <= HOLD_RANGE[1])
            if g[i] > OPEN_ABOVE and held_before.all():
                yield int(e), int(ix[i - 10]), "covering"
                i += 30
            i += 1


def wrist_rgb(dataset: LeRobotDataset, index: int) -> np.ndarray:
    frame = dataset[index]["observation.images.wrist"].permute(1, 2, 0).numpy()
    return (frame * 255).astype(np.uint8)


def summary(values: list[float]) -> dict:
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * len(v)))]
    return {"n": len(v), "min": v[0], "p10": pick(0.10), "p25": pick(0.25), "median": pick(0.5), "p95": pick(0.95), "max": v[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="molnrb/so101_pick_rb_mixed")
    p.add_argument("--vision-dir", type=Path, required=True)
    p.add_argument("--roi", type=float, nargs=4, metavar=("CX", "CY", "RX", "RY"))
    p.add_argument("--thresholds", type=int, nargs="+", default=[50, 100, 120, 150, 200, 300])
    p.add_argument("--out", type=Path, default=Path("held_area.json"))
    p.add_argument("--overlay", type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(args.vision_dir))
    import colour_detector as cd

    specs = tuple(replace(s, use_roi=args.roi is not None, max_area=sys.maxsize) for s in cd.load_detector().specs)
    detector = cd.ColourDetector(specs, roi=tuple(args.roi) if args.roi else None)

    dataset = LeRobotDataset(args.repo_id, video_backend="pyav")
    episode = np.array(dataset.hf_dataset["episode_index"])
    gripper = np.stack([np.asarray(s) for s in dataset.hf_dataset["observation.state"]])[:, GRIPPER]

    rows, centres = [], []
    for e, index, kind in held_frames(gripper, episode):
        rgb = wrist_rgb(dataset, index)
        detections = detector.detect(rgb)
        rows.append({"ep": e, "frame": index, "kind": kind, **{c: detections[c].area for c in COLOURS}})
        if kind == "target" and detections["striped"].present:
            centres.append(detections["striped"].centre)

    target_per_episode = {}
    for r in rows:
        if r["kind"] == "target":
            target_per_episode.setdefault(r["ep"], []).append(r["striped"])
    target_median = [statistics.median(v) for v in target_per_episode.values()]
    covering = [r for r in rows if r["kind"] == "covering"]

    result = {
        "repo_id": args.repo_id, "roi": args.roi,
        "striped_target_per_episode_median": summary(target_median),
        "striped_while_holding_covering_ball": summary([r["striped"] for r in covering]),
        "red_while_holding_covering_ball": summary([r["red"] for r in covering]),
        "blue_while_holding_covering_ball": summary([r["blue"] for r in covering]),
        "red_while_holding_target": summary([r["red"] for r in rows if r["kind"] == "target"]),
        "blue_while_holding_target": summary([r["blue"] for r in rows if r["kind"] == "target"]),
        "thresholds": {t: {"target_rejected": sum(m < t for m in target_median) / len(target_median),
                           "covering_accepted": sum(r["striped"] >= t for r in covering) / len(covering)}
                       for t in args.thresholds},
        "striped_centre_px_while_holding_target": summary([c[0] for c in centres]) | {"axis": "x"},
        "striped_centre_py_while_holding_target": summary([c[1] for c in centres]) | {"axis": "y"},
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=1))
    for key, value in result.items():
        if key != "rows":
            print(key, value)

    if args.overlay:
        picks = [r for r in rows if r["kind"] == "target"][::40][:4] + covering[::20][:4]
        tiles = []
        for r in picks:
            bgr = cv2.cvtColor(wrist_rgb(dataset, r["frame"]), cv2.COLOR_RGB2BGR)
            if args.roi:
                cx, cy, rx, ry = args.roi
                h, w = bgr.shape[:2]
                cv2.ellipse(bgr, (round(cx * w), round(cy * h)), (round(rx * w), round(ry * h)), 0, 0, 360, (0, 255, 255), 2)
            label = f"{r['kind'][0].upper()} ep{r['ep']} s={r['striped']} r={r['red']} b={r['blue']}"
            tiles.append(cv2.putText(cv2.resize(bgr, (320, 240)), label, (5, 20), 0, 0.5, (0, 255, 255), 2))
        cv2.imwrite(str(args.overlay), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])]))


if __name__ == "__main__":
    main()
