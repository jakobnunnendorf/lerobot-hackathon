"""Data-side checks for "the SmolVLA skips clearing": scene mix in episodes and frames, phase durations
and clearing orders from Jakob's step labels.

    git show origin/jakobnunnendorf/curation-table:curation/episodes.csv > episodes.csv
    git show origin/labels:labels/jakob.jsonl > jakob.jsonl
    python scripts/analysis/clear_first_stats.py episodes.csv jakob.jsonl
"""

import json
import sys
from collections import Counter

import pandas as pd

RB_MIXED = "so101_pick_rb_mixed"
V2_TRAIN_EPISODES = range(0, 114)  # eval_split=0.05 holds out the last ceil(120*0.05)=6 episodes, 114-119
BURIED_SCENES = {"rb_buried", "foil_buried"}


def scene_mix(table: pd.DataFrame, title: str) -> None:
    by_scene = table.groupby("scene").agg(episodes=("episode_index", "size"), seconds=("length_s", "sum"))
    by_scene["ep_share"] = by_scene.episodes / by_scene.episodes.sum()
    by_scene["frame_share"] = by_scene.seconds / by_scene.seconds.sum()
    buried = by_scene.index.isin(BURIED_SCENES)
    print(f"\n## {title}")
    print(by_scene.round(3).to_string())
    print(
        f"buried starts: {by_scene.episodes[buried].sum()}/{by_scene.episodes.sum()} episodes, "
        f"{by_scene.frame_share[buried].sum():.0%} of frames"
    )


def latest_labels(path: str) -> dict:
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            latest[(row["dataset"].split("/")[-1], row["episode"])] = row
    return {key: row for key, row in latest.items() if row.get("steps")}


def events(row: dict) -> list[tuple[float, str]]:
    return sorted((step["t"], step["code"]) for step in row["steps"])


def first_time(evs, predicate):
    return next((t for t, code in evs if predicate(code)), None)


def phases(row: dict, length_s: float) -> dict:
    evs = events(row)
    first_target = first_time(evs, lambda c: c.endswith("_target"))
    first_striped_target = first_time(evs, lambda c: c == "s_target")
    striped_grab = max((t for t, c in evs if c == "s_grab"), default=None)
    grabs = [c[0].upper() for t, c in evs if c in ("r_grab", "b_grab") and (striped_grab is None or t < striped_grab)]
    clears = [c[0].upper() for t, c in evs if c in ("r_hole", "b_hole")]
    return {
        "start": row.get("start_state"),
        "first_target": next((c[0].upper() for t, c in evs if c.endswith("_target")), "-"),
        "order": "".join(clears) + "S",
        "n_clears": len(clears),
        "rb_grabs": len(grabs),
        "wait_s": first_target,
        "clearing_s": (first_striped_target - first_target) if clears and first_target is not None else 0.0,
        "striped_s": (striped_grab - first_striped_target) if striped_grab and first_striped_target else None,
        "length_s": length_s,
    }


def label_stats(labels: dict, table: pd.DataFrame) -> None:
    lengths = table.set_index(["dataset", "episode_index"]).length_s
    rows = pd.DataFrame(
        [{"episode": ep, **phases(row, lengths.get((ds, ep)))} for (ds, ep), row in sorted(labels.items())]
    )
    print(f"\n## Step labels: {len(rows)} labelled episodes ({rows.episode.min()}-{rows.episode.max()})")
    print(rows.to_string(index=False))
    print("\nstart state x first target:")
    print(pd.crosstab(rows.start, rows.first_target, margins=True).to_string())
    print("\nstart state x cleared anything first:")
    print(pd.crosstab(rows.start, rows.n_clears > 0, margins=True).to_string())

    clearing = rows[rows.n_clears > 0]
    total = rows.length_s.sum()
    print(
        f"\nframes: waiting at home before the first target {rows.wait_s.sum() / total:.0%}, "
        f"clearing {rows.clearing_s.sum() / total:.0%}, striped approach+grasp {rows.striped_s.sum() / total:.0%}"
    )
    print(
        f"clear-first episodes: {len(clearing)}, clears per episode mean {clearing.n_clears.mean():.1f} "
        f"(min {clearing.n_clears.min()}, max {clearing.n_clears.max()})"
    )
    orders = Counter(clearing.order)
    print(f"distinct clearing orders: {len(orders)} among {len(clearing)} episodes")
    for order, n in orders.most_common():
        print(f"  {order}: {n}")
    print(f"first ball cleared: {dict(Counter(o[0] for o in clearing.order))}")


def main(episodes_csv: str, labels_jsonl: str) -> None:
    table = pd.read_csv(episodes_csv)
    keep = table[table.verdict == "KEEP"]
    scene_mix(keep, "All KEEP episodes in the curation table (all three datasets)")
    rb = table[table.dataset == RB_MIXED]
    scene_mix(rb[rb.episode_index.isin(V2_TRAIN_EPISODES)], "smolvla_redblue_v2 training episodes (rb_mixed 0-113, every verdict)")
    scene_mix(rb[~rb.episode_index.isin(V2_TRAIN_EPISODES)], "smolvla_redblue_v2 held-out episodes (rb_mixed 114-119)")
    label_stats(latest_labels(labels_jsonl), table)


if __name__ == "__main__":
    main(*sys.argv[1:3])
