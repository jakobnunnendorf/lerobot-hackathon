"""Cut labelled segments out of LeRobot v3 datasets into a new dataset with one episode per segment.

    python scripts/multistep/build_segment_dataset.py --kind clear_r \
        --source molnrb/so101_pick_rb_mixed=<local dataset root> \
        --out <new root> --task "Put one red ball into the red hole"
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.io_utils import load_stats, to_parquet_one_row_group_per_episode, write_stats

SEGMENTS_CSV = Path(__file__).resolve().parents[2] / "research" / "label-patterns" / "segments.csv"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

# Drop rule from decisions/multistep-voice-policy.md: the gripper was holding, then opens at a hole's pan.
HOLDING, OPEN = 16.5, 18.0
HOLE_PAN = {"red": lambda pan: pan < -25, "blue": lambda pan: pan > 18}
KIND_HOLE = {"clear_r": "red", "clear_b": "blue"}
DROP_NEAR_END_S = 1.5


def drops(state: np.ndarray, fps: int) -> list[tuple[float, str]]:
    found, held = [], False
    for i, (pan, gripper) in enumerate(zip(state[:, 0], state[:, 5])):
        if gripper < HOLDING:
            held = True
        elif held and gripper >= OPEN:
            held = False
            colour = next((c for c, at_hole in HOLE_PAN.items() if at_hole(pan)), "bowl")
            found.append((i / fps, colour))
    return found


def ends_in_one_drop(state: np.ndarray, fps: int, hole: str) -> bool:
    holes = [(t, c) for t, c in drops(state, fps) if c != "bowl"]
    return [c for _, c in holes] == [hole] and holes[0][0] >= len(state) / fps - DROP_NEAR_END_S


class Source:
    def __init__(self, repo_id: str, root: Path):
        self.repo_id = repo_id
        self.name = repo_id.split("/")[-1]
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text())
        self.fps = self.info["fps"]
        self.video_keys = [k for k, f in self.info["features"].items() if f["dtype"] == "video"]
        self.episodes = pd.concat(
            [pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("*/*.parquet"))]
        ).set_index("episode_index", drop=False)
        self.data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))])
        self.stats = load_stats(root)

    def frames(self, episode: int, t0: float, t1: float) -> pd.DataFrame:
        ep = self.data[self.data["episode_index"] == episode].sort_values("frame_index")
        first = round(t0 * self.fps)
        last = min(round(t1 * self.fps), len(ep) - 1)
        return ep.iloc[first : last + 1]

    def video_file(self, episode: int, key: str) -> tuple[int, int]:
        row = self.episodes.loc[episode]
        return int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"])


def select_segments(segments: pd.DataFrame, kinds: list[str], sources: dict, clean_only: bool) -> pd.DataFrame:
    chosen = segments[segments["kind"].isin(kinds) & segments["dataset"].isin(sources)]
    if clean_only:
        chosen = chosen[chosen["clean"] == 1]
    order = {name: i for i, name in enumerate(sources)}
    return chosen.assign(_src=chosen["dataset"].map(order)).sort_values(["_src", "episode", "t0"]).drop(columns="_src")


def joint_checked(segments: pd.DataFrame, sources: dict) -> pd.DataFrame:
    # On rb_mixed this drops 8 of 224 clean clearing segments: 3 AI-labelled starts that still contain the
    # previous ball's drop (eps 142-144) and 5 partial opens the rule cannot see (eps 11, 28, 51, 160, 174).
    keep = []
    for row in segments.itertuples():
        hole = KIND_HOLE.get(row.kind)
        src = sources[row.dataset]
        state = np.stack(src.frames(row.episode, row.t0, row.t1)["observation.state"].to_numpy())
        keep.append(hole is None or ends_in_one_drop(state, src.fps, hole))
    dropped = segments[~np.array(keep)]
    for row in dropped.itertuples():
        print(f"joint check: skipping {row.dataset} ep {row.episode} {row.kind} {row.t0}-{row.t1}")
    return segments[np.array(keep)]


def holdout_last(segments: pd.DataFrame, fraction: float) -> pd.DataFrame:
    # lerobot-train's eval_split holds out the *last* episodes of each task, and the recordings come in
    # scenario blocks (research/label-patterns.md, section 5), so the hold-out is spread evenly over
    # recording order and moved to the end.
    n = len(segments)
    n_eval = math.ceil(n * fraction) if fraction > 0 else 0
    picks = {min(n - 1, int((k + 0.5) * n / n_eval)) for k in range(n_eval)}
    is_holdout = np.array([i in picks for i in range(n)])
    return pd.concat([segments[~is_holdout].assign(holdout=0), segments[is_holdout].assign(holdout=1)])


def place_videos(sources: dict, segments: pd.DataFrame, out: Path, link: bool) -> dict:
    """Give every source video file a slot in the new dataset; returns (source, episode, key) -> (chunk, file)."""
    slots, placed = {}, {}
    for row in segments.itertuples():
        src = sources[row.dataset]
        for key in src.video_keys:
            origin = (row.dataset, key, *src.video_file(row.episode, key))
            if origin not in placed:
                placed[origin] = (0, sum(1 for o in placed if o[1] == key))
                target = out / VIDEO_PATH.format(video_key=key, chunk_index=placed[origin][0], file_index=placed[origin][1])
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file = src.root / VIDEO_PATH.format(video_key=key, chunk_index=origin[2], file_index=origin[3])
                copy_or_link(source_file, target, link)
            slots[(row.dataset, row.episode, key)] = placed[origin]
    return slots


def copy_or_link(source_file: Path, target: Path, link: bool) -> None:
    if link:
        try:
            os.link(source_file, target)
            return
        except OSError:
            pass
    shutil.copyfile(source_file, target)


def build(sources: dict, segments: pd.DataFrame, out: Path, repo_id: str, task: str, link: bool) -> None:
    first = next(iter(sources.values()))
    out.mkdir(parents=True)
    slots = place_videos(sources, segments, out, link)
    data_parts, episode_rows, index = [], [], 0
    for new_ep, row in enumerate(segments.itertuples()):
        src = sources[row.dataset]
        frames = src.frames(row.episode, row.t0, row.t1).copy()
        start_ts = float(frames["timestamp"].iloc[0])
        length = len(frames)
        frames["timestamp"] = (frames["timestamp"] - np.float32(start_ts)).astype(np.float32)
        frames["frame_index"] = np.arange(length, dtype=np.int64)
        frames["episode_index"] = new_ep
        frames["index"] = np.arange(index, index + length, dtype=np.int64)
        frames["task_index"] = 0
        data_parts.append(frames)
        source_ep = src.episodes.loc[row.episode]
        episode = {
            "episode_index": new_ep,
            "tasks": [task],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": index,
            "dataset_to_index": index + length,
        }
        for key in src.video_keys:
            from_ts = float(source_ep[f"videos/{key}/from_timestamp"]) + start_ts
            chunk, file = slots[(row.dataset, row.episode, key)]
            episode |= {
                f"videos/{key}/chunk_index": chunk,
                f"videos/{key}/file_index": file,
                f"videos/{key}/from_timestamp": from_ts,
                f"videos/{key}/to_timestamp": from_ts + length / src.fps,
            }
        episode |= {"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        episode_rows.append(episode)
        index += length

    data = pd.concat(data_parts, ignore_index=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    to_parquet_one_row_group_per_episode(data, out / "data" / "chunk-000" / "file-000.parquet")
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(episode_rows).to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task")).to_parquet(out / "meta" / "tasks.parquet")
    write_stats(segment_stats(data, first, sources), out)
    info = dict(first.info, total_episodes=len(episode_rows), total_frames=len(data), total_tasks=1)
    info["splits"] = {"train": f"0:{len(episode_rows)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    segments.assign(new_episode=range(len(segments)), source_repo=segments["dataset"].map(lambda n: sources[n].repo_id)).to_csv(
        out / "meta" / "segment_sources.csv", index=False
    )
    print(f"{repo_id}: {len(episode_rows)} episodes ({int(segments['holdout'].sum())} held out last), "
          f"{len(data)} frames, {len(data) / first.fps / 60:.1f} min -> {out}")


def segment_stats(data: pd.DataFrame, first: Source, sources: dict) -> dict:
    stats = {}
    for key, feature in first.info["features"].items():
        if feature["dtype"] in ("video", "image"):
            # Frames are not decoded here: the source stats cover a superset of these frames.
            stats[key] = aggregate_stats([{key: s.stats[key]} for s in sources.values()])[key]
        else:
            values = np.stack(data[key].to_numpy()).astype(np.float64).reshape(len(data), -1)
            stats[key] = get_feature_stats(values, axis=0, keepdims=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", action="append", required=True, help="clear_r, clear_b or pick_s; repeatable")
    parser.add_argument("--source", action="append", required=True, help="repo_id=local_root; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/segments")
    parser.add_argument("--task", required=True)
    parser.add_argument("--segments", type=Path, default=SEGMENTS_CSV)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--episodes", help="comma-separated source episodes to keep (for local tests)")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--link", action="store_true", help="hard-link video files instead of copying")
    parser.add_argument("--skip-joint-check", action="store_true", help="keep clear segments the drop rule rejects")
    parser.add_argument("--dry-run", action="store_true", help="only print which segments would be used")
    args = parser.parse_args()

    sources = {}
    for spec in args.source:
        repo_id, root = spec.split("=", 1)
        src = Source(repo_id, Path(root))
        sources[src.name] = src
    segments = select_segments(pd.read_csv(args.segments), args.kind, sources, not args.include_failures)
    if args.episodes:
        segments = segments[segments["episode"].isin({int(e) for e in args.episodes.split(",")})]
    if not args.skip_joint_check:
        segments = joint_checked(segments, sources)
    segments = holdout_last(segments, args.holdout)
    if args.dry_run:
        print(f"{len(segments)} segments, {int(segments['holdout'].sum())} held out: "
              f"source episodes {sorted(segments[segments['holdout'] == 1]['episode'].tolist())}")
        return
    build(sources, segments, args.out, args.repo_id, args.task, args.link)


if __name__ == "__main__":
    main()
