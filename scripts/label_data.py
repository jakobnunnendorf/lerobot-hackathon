"""Labels and a low-resolution frame cache of ds/so101_pick_rb_mixed, shared by scene_clf and label_encoder."""

import glob
import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd

DS = Path(os.path.expanduser("~/so101/curation/ds/so101_pick_rb_mixed"))
CURATION = Path(os.path.expanduser("~/so101/curation"))
DATASET = "ds/so101_pick_rb_mixed"
HOLDOUT = [6, 23, 65, 88, 99, 110, 126, 157, 177, 191]
FPS_SAMPLE = 3
SIZE = (160, 120)
VISIBILITY = ["buried", "partly", "visible"]
NEXT_ACTION = ["clear_red", "clear_blue", "pick_striped"]
TARGET = ["red", "blue", "striped", "none"]
PHASE = ["approach", "grab", "carry", "home"]
CAMERAS = ["top", "wrist"]
CARRY_DELAY_S = 0.7
BLIND_MAX_S = 2.5
SEGMENT_END = ("_hole", "_putback")


def load_labels() -> dict:
    # prelabels first, labels.jsonl second: the human line wins, and within a file the latest line wins.
    labels = {}
    for name in ("prelabels.jsonl", "labels.jsonl"):
        for line in open(CURATION / name):
            r = json.loads(line)
            if r.get("dataset") == DATASET and "steps" in r:
                labels[r["episode"]] = r
    return labels


def events(record: dict) -> list:
    return sorted(((s["t"], s["code"]) for s in record["steps"]), key=lambda e: e[0])


def visibility_at(record: dict, t: float) -> int:
    state = record.get("start_state")
    for te, code in events(record):
        if te <= t and code in ("s_vis", "s_partly", "s_buried"):
            state = {"s_vis": "visible", "s_partly": "partly", "s_buried": "buried"}[code]
    return VISIBILITY.index(state)


def next_action_at(record: dict, t: float) -> int | None:
    for te, code in events(record):
        if te >= t - 1e-6 and code.endswith("_target"):
            return NEXT_ACTION[{"r": 0, "b": 1, "s": 2}[code[0]]]
    return None


def target_and_phase_at(record: dict, t: float) -> tuple[int, int]:
    """Target of the running segment (target..hole/home), else of the next one; phase from grab time."""
    evs = events(record)
    current = None  # (colour letter, target time)
    grab = None
    for te, code in evs:
        if te > t:
            break
        if code.endswith("_target"):
            current, grab = (code[0], te), None
        elif code.endswith("_grab") and current:
            grab = te
        elif code.endswith("_miss"):
            grab = None
        elif code.endswith(SEGMENT_END) or code == "home":
            current, grab = None, None
    if current is None:
        nxt = next((c for te, c in evs if te > t and c.endswith("_target")), None)
        return ("rbs".index(nxt[0]) if nxt else 3), PHASE.index("home")
    phase = "approach" if grab is None else "grab" if t < grab + CARRY_DELAY_S else "carry"
    return "rbs".index(current[0]), PHASE.index(phase)


def wrist_blind_at(record: dict, t: float) -> int:
    evs = events(record)
    for te, code in evs:
        if code == "wrist_blind" and te <= t:
            end = next((e for e, c in evs if e > te and c.endswith("_grab")), te + BLIND_MAX_S)
            if t <= min(end, te + BLIND_MAX_S):
                return 1
    return 0


def cache_path(cache_dir: Path) -> Path:
    return cache_dir / "frames.npz"


def build_cache(cache_dir: Path) -> Path:
    """Decode every episode at FPS_SAMPLE frames per second into 160x120 uint8 frames for both cameras."""
    episodes = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DS / "meta/episodes/*/*.parquet")))])
    data = pd.concat([pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"])
                      for f in sorted(glob.glob(str(DS / "data/*/*.parquet")))])
    states = {e: np.stack(g.sort_values("frame_index")["observation.state"].to_numpy()) for e, g in data.groupby("episode_index")}
    out = {"ep": [], "t": [], "state": []}
    frames = {cam: [] for cam in CAMERAS}
    ep_rows = {}
    for _, row in episodes.iterrows():
        e, length = int(row.episode_index), int(row.length)
        times = np.arange(0, length / 30 - 1e-6, 1 / FPS_SAMPLE)
        ep_rows[e] = (times, row)
        for t in times:
            out["ep"].append(e)
            out["t"].append(t)
            out["state"].append(states[e][min(int(round(t * 30)), length - 1)])
    row_of = {}
    n = 0
    for e in sorted(ep_rows):
        for j, t in enumerate(ep_rows[e][0]):
            row_of[(e, j)] = n
            n += 1
    for cam in CAMERAS:
        arr = np.zeros((n, SIZE[1], SIZE[0], 3), dtype=np.uint8)
        key = f"videos/observation.images.{cam}"
        by_file = {}
        for e, (times, row) in ep_rows.items():
            by_file.setdefault((int(row[f"{key}/chunk_index"]), int(row[f"{key}/file_index"])), []).append(e)
        for (chunk, fidx), eps in by_file.items():
            wanted = []  # (absolute time, row index)
            for e in eps:
                times, row = ep_rows[e]
                wanted += [(float(row[f"{key}/from_timestamp"]) + t, row_of[(e, j)]) for j, t in enumerate(times)]
            wanted.sort()
            k = 0
            container = av.open(str(DS / f"videos/observation.images.{cam}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"))
            for frame in container.decode(video=0):
                if k >= len(wanted):
                    break
                while k < len(wanted) and frame.time >= wanted[k][0] - 1 / 60:
                    arr[wanted[k][1]] = frame.reformat(width=SIZE[0], height=SIZE[1], format="rgb24").to_ndarray()
                    k += 1
            container.close()
            print(cam, chunk, fidx, "done", flush=True)
        frames[cam] = arr
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path(cache_dir), ep=np.array(out["ep"]), t=np.array(out["t"], dtype=np.float32),
             state=np.array(out["state"], dtype=np.float32), top=frames["top"], wrist=frames["wrist"])
    return cache_path(cache_dir)


def load_cache(cache_dir: Path) -> dict:
    return dict(np.load(cache_path(cache_dir)))


if __name__ == "__main__":
    import sys

    build_cache(Path(sys.argv[1]))
