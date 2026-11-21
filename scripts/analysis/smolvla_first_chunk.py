"""Open-loop check: at the moment a demo starts heading for its first ball, does the SmolVLA's predicted
chunk head for the covering ball (as the demo does) or for the striped ball underneath?

Runs on the CPU, one forward pass per (episode, query time, seed). Episodes need step labels and locally
cached video. Writes one CSV row per prediction and, with --dump-inputs, the 512x512 network input frames.

    python scripts/analysis/smolvla_first_chunk.py --checkpoint <dir with config.json> \
        --dataset-root <rb_mixed snapshot> --labels jakob.jsonl --episodes 4 6 7 1 2 --out first_chunk.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.common.vla_utils import resize_with_pad
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CAMERAS = ("top", "wrist")
RENAME_MAP = {f"observation.images.{c}": f"observation.images.camera{i + 1}" for i, c in enumerate(CAMERAS)}
ARM_JOINTS = slice(0, 5)
FPS = 30


class Episode:
    def __init__(self, root: Path, index: int, labels: dict):
        meta = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet")))
        self.meta = meta.set_index("episode_index").loc[index]
        data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
        frames = data[data.episode_index == index].sort_values("frame_index")
        self.states = np.stack(frames["observation.state"].to_numpy())
        self.actions = np.stack(frames["action"].to_numpy())
        self.task = pd.read_parquet(root / "meta/tasks.parquet").index[int(frames.task_index.iloc[0])]
        self.root, self.index = root, index
        self.events = sorted((s["t"], s["code"]) for s in labels["steps"])
        self.start_state = labels.get("start_state")

    def frame(self, t: float) -> int:
        return min(int(round(t * FPS)), len(self.states) - 1)

    def first_time(self, codes) -> float | None:
        return next((t for t, c in self.events if c in codes), None)

    def pose_at(self, t: float | None) -> np.ndarray | None:
        return None if t is None else self.states[self.frame(t), ARM_JOINTS]

    def image(self, camera: str, t: float) -> torch.Tensor:
        key = f"observation.images.{camera}"
        file = self.root / "videos" / key / f"chunk-{self.meta[f'videos/{key}/chunk_index']:03d}" / (
            f"file-{self.meta[f'videos/{key}/file_index']:03d}.mp4"
        )
        timestamp = self.meta[f"videos/{key}/from_timestamp"] + self.frame(t) / FPS
        return decode_video_frames(file, [timestamp], tolerance_s=1 / FPS / 2, backend="pyav")[0]


def query_times(ep: Episode) -> dict[str, float]:
    return {
        "first_target": ep.first_time({"r_target", "b_target", "s_target"}),
        "first_grab": ep.first_time({"r_grab", "b_grab", "s_grab"}),
    }


def reference_poses(ep: Episode) -> dict[str, np.ndarray | None]:
    # The grasp poses of the covering ball and the striped ball lie within ~10 degrees of each other, so the
    # destination after the first grasp (colour hole vs home) is the clearer signal of which skill is running.
    return {
        "clear": ep.pose_at(ep.first_time({"r_grab", "b_grab"})),
        "striped": ep.pose_at(max((t for t, c in ep.events if c == "s_grab"), default=None)),
        "hole": ep.pose_at(ep.first_time({"r_hole", "b_hole"})),
        "home": ep.pose_at(max((t for t, c in ep.events if c == "home"), default=None)),
    }


def closest(chunk: np.ndarray, pose: np.ndarray | None) -> float:
    return float("nan") if pose is None else float(np.linalg.norm(chunk[:, ARM_JOINTS] - pose, axis=1).min())


def save_network_input(image: torch.Tensor, path: Path) -> None:
    padded = resize_with_pad(image[None], 512, 512, pad_value=0)[0]
    Image.fromarray((padded.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-inputs", type=Path)
    args = parser.parse_args()

    labels = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        labels[row["episode"]] = row

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cpu").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )

    with args.out.open("w", newline="") as f:
        writer = None
        for index in args.episodes:
            ep = Episode(args.dataset_root, index, labels[index])
            poses = reference_poses(ep)
            for name, t in query_times(ep).items():
                images = {c: ep.image(c, t) for c in CAMERAS}
                if args.dump_inputs:
                    args.dump_inputs.mkdir(parents=True, exist_ok=True)
                    for c, img in images.items():
                        save_network_input(img, args.dump_inputs / f"ep{index:03d}_{name}_{c}.png")
                i = ep.frame(t)
                demo = ep.actions[i : i + policy.config.chunk_size]
                observation = {f"observation.images.{c}": img for c, img in images.items()}
                observation["observation.state"] = torch.from_numpy(ep.states[i]).float()
                observation["task"] = ep.task
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    started = time.time()
                    with torch.inference_mode():
                        chunk = policy.predict_action_chunk(preprocessor(dict(observation)))
                    pred = postprocessor(chunk)[0].numpy()[: len(demo)]
                    row = {
                        "episode": index,
                        "start_state": ep.start_state,
                        "query": name,
                        "t": round(t, 2),
                        "seed": seed,
                        **{
                            f"{source}_to_{pose_name}": closest(chunk_or_now, pose)
                            for source, chunk_or_now in (("now", ep.states[i : i + 1]), ("pred", pred), ("demo", demo))
                            for pose_name, pose in poses.items()
                        },
                        "pred_gripper_end": float(pred[-1, -1]),
                        "demo_gripper_end": float(demo[-1, -1]),
                        "pred_demo_mae": float(np.abs(pred - demo).mean()),
                        "pred_moves": float(np.linalg.norm(pred[-1, ARM_JOINTS] - ep.states[i, ARM_JOINTS])),
                        "demo_moves": float(np.linalg.norm(demo[-1, ARM_JOINTS] - ep.states[i, ARM_JOINTS])),
                        "seconds": round(time.time() - started, 1),
                    }
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})
                    f.flush()
                    print(row, flush=True)


if __name__ == "__main__":
    main()
