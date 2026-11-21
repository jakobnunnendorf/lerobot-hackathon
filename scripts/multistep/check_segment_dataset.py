"""Check a dataset from build_segment_dataset.py against its source and, optionally, against Jakob's step labels.

    python scripts/multistep/check_segment_dataset.py --built <root> --source molnrb/so101_pick_rb_mixed=<root> \
        --hole red [--labels jakob.jsonl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from build_segment_dataset import HOLE_PAN, drops, ends_in_one_drop
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LABEL_HOLE = {"red": "r_hole", "blue": "b_hole"}
CHUNK = 100


def jakob_holes(path: Path, dataset: str) -> dict[int, list[tuple[float, str]]]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["dataset"].split("/")[-1] == dataset:
            latest[record["episode"]] = record
    return {
        ep: sorted((s["t"], s["code"]) for s in r["steps"] if s["code"] in LABEL_HOLE.values())
        for ep, r in latest.items()
    }


def stack(rows, key: str) -> np.ndarray:
    return torch.stack(list(rows[key])).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--built", type=Path, required=True)
    parser.add_argument("--source", required=True, help="repo_id=local_root")
    parser.add_argument("--hole", choices=list(HOLE_PAN), required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--pixel-episodes", type=int, default=6, help="episodes whose first and last frames are compared")
    args = parser.parse_args()

    repo_id, source_root = args.source.split("=", 1)
    segments = pd.read_csv(args.built / "meta" / "segment_sources.csv")
    built = LeRobotDataset("local/segments", root=args.built)
    fps = built.meta.fps
    windowed = LeRobotDataset("local/segments", root=args.built, delta_timestamps={"action": [i / fps for i in range(CHUNK)]})
    source = LeRobotDataset(repo_id, root=source_root, episodes=sorted(segments["episode"].unique().tolist()))
    to_rel = source.absolute_to_relative_idx
    labels = jakob_holes(args.labels, repo_id.split("/")[-1]) if args.labels else {}
    problems, label_checked, label_ok, regrasps = [], 0, 0, 0

    for seg in segments.itertuples():
        name = f"ep {seg.new_episode} (src {seg.episode} {seg.t0}-{seg.t1})"
        ep = built.meta.episodes[seg.new_episode]
        lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
        state = stack(built.hf_dataset.select(range(lo, hi)), "observation.state")
        first = source.meta.episodes[seg.episode]["dataset_from_index"] + round(seg.t0 * fps)
        src_rel = [to_rel[first + i] if to_rel else first + i for i in range(hi - lo)]
        if not np.array_equal(state, stack(source.hf_dataset.select(src_rel), "observation.state")):
            problems.append(f"{name}: state differs from the source")

        regrasps += sum(c == "bowl" for _, c in drops(state, fps))
        if not ends_in_one_drop(state, fps, args.hole):
            problems.append(f"{name}: joint-rule drops {drops(state, fps)}, want one {args.hole} drop at the end")

        if seg.episode in labels:
            label_checked += 1
            expected = [t for t, code in labels[seg.episode] if code == LABEL_HOLE[args.hole]]
            inside = [t for t, _ in labels[seg.episode] if seg.t0 < t < seg.t1]
            if any(abs(t - (seg.t1 - 0.5)) < 0.3 for t in expected) and len(inside) == 1:
                label_ok += 1
            else:
                problems.append(f"{name}: Jakob's holes {labels[seg.episode]}")

        pad = windowed[hi - 1]["action_is_pad"]
        if pad[0] or not pad[1:].all():
            problems.append(f"{name}: action chunk at the last frame is not padded")

        if seg.new_episode < args.pixel_episodes or seg.holdout:
            for offset in (0, hi - lo - 1):
                mine, theirs = built[lo + offset], source[src_rel[offset]]
                problems += [f"{name} frame {offset}: {k} differs" for k in built.meta.video_keys if not torch.equal(mine[k], theirs[k])]

    print(f"{len(segments)} segments ({int(segments['holdout'].sum())} held out), {built.meta.total_frames} frames, "
          f"{regrasps} re-grasps in the bowl")
    if args.labels:
        print(f"Jakob's labels: {label_ok}/{label_checked} segments end at his {LABEL_HOLE[args.hole]} + 0.5 s (within 0.3 s), no other hole inside")
    print("\n".join(problems) if problems else "PASS")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
