"""Usage: python check_dataset.py <user/dataset> [--from N] [--sheet out.jpg]"""
import argparse
import glob
import json
import os

import av
import numpy as np
import pandas as pd

GRIPPER = 5
HOLDING_MIN = 8  # gripper state when closed on the ball is ~11-16; ~0-2 means it closed on nothing


def load(root):
    info = json.load(open(f"{root}/meta/info.json"))
    episodes = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")))
    frames = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
    return info, episodes.set_index("episode_index"), frames


def episode_report(ep, g):
    action = np.stack(g["action"].to_numpy())
    state = np.stack(g["observation.state"].to_numpy())
    problems = []
    if state[-1, GRIPPER] < HOLDING_MIN:
        problems.append("ENDS WITH EMPTY GRIPPER")
    if action[:, GRIPPER].max() - state[:, GRIPPER].max() > 15:
        problems.append("FOLLOWER GRIPPER NOT FOLLOWING (motor off?)")
    start, end = np.round(state[0]).astype(int).tolist(), np.round(state[-1]).astype(int).tolist()
    print(f"ep{ep:3d} {len(g):5d} fr {len(g) / 30:5.1f}s start {start} end {end} {' | '.join(problems)}")
    return bool(problems)


def count_video_frames(root, cam):
    return sum(sum(1 for _ in av.open(f).decode(video=0)) for f in sorted(glob.glob(f"{root}/videos/{cam}/*/*.mp4")))


def frame_at(root, episodes, ep, cam, frac):
    e = episodes.loc[ep]
    path = f"{root}/videos/{cam}/chunk-{int(e[f'videos/{cam}/chunk_index']):03d}/file-{int(e[f'videos/{cam}/file_index']):03d}.mp4"
    t0, t1 = e[f"videos/{cam}/from_timestamp"], e[f"videos/{cam}/to_timestamp"]
    target = t0 + frac * (t1 - t0 - 0.1)
    container = av.open(path)
    stream = container.streams.video[0]
    container.seek(int(max(0, target - 1) / stream.time_base), stream=stream)
    for frame in container.decode(stream):
        if frame.time >= target:
            return frame.to_ndarray(format="rgb24")


def contact_sheet(root, episodes, eps, cams, out):
    from PIL import Image
    rows = [np.concatenate([np.array(Image.fromarray(frame_at(root, episodes, ep, cam, frac)).resize((240, 180)))
                            for cam in cams for frac in (0.0, 0.5, 1.0)], axis=1) for ep in eps]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out, quality=80)
    print(f"sheet: {out}  (rows = episodes {eps[0]}..{eps[-1]}, columns = start/middle/end per camera {cams})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--from", dest="first", type=int, default=0)
    parser.add_argument("--sheet")
    args = parser.parse_args()
    root = os.path.expanduser(f"{os.environ.get('HF_LEROBOT_HOME', '~/.cache/huggingface/lerobot')}/{args.repo_id}")

    info, episodes, frames = load(root)
    cams = [k for k in info["features"] if k.startswith("observation.images")]
    print(f"{args.repo_id}: {info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps, cameras {cams}")
    flagged = [ep for ep, g in frames.groupby("episode_index") if ep >= args.first and episode_report(ep, g)]
    for cam in cams:
        n = count_video_frames(root, cam)
        print(f"{cam}: {n} video frames, expected {info['total_frames']} {'OK' if n == info['total_frames'] else 'MISMATCH'}")
    print(f"flagged episodes: {flagged or 'none'}")
    if args.sheet:
        eps = [ep for ep in episodes.index if ep >= args.first][:12]
        contact_sheet(root, episodes, eps, cams, args.sheet)


if __name__ == "__main__":
    main()
