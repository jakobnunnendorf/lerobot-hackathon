"""Usage: python upload_dataset.py <user/dataset> [--private]"""
import argparse
import json
import os

from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser()
parser.add_argument("repo_id")
parser.add_argument("--private", action="store_true")
args = parser.parse_args()

root = os.path.expanduser(f"{os.environ.get('HF_LEROBOT_HOME', '~/.cache/huggingface/lerobot')}/{args.repo_id}")
dataset = LeRobotDataset(args.repo_id, root=root)
print(f"uploading {dataset.num_episodes} episodes, {dataset.num_frames} frames", flush=True)
dataset.push_to_hub(private=args.private, tags=["so101", "lerobot", "hackathon"])

info = HfApi().dataset_info(args.repo_id, files_metadata=True)
hub_episodes = json.load(open(hf_hub_download(args.repo_id, "meta/info.json", repo_type="dataset", force_download=True)))["total_episodes"]
size_mb = round(sum(s.size or 0 for s in info.siblings) / 1e6)
print(f"ON HUB: https://huggingface.co/datasets/{args.repo_id}  private={info.private}  episodes={hub_episodes}  {size_mb} MB")
