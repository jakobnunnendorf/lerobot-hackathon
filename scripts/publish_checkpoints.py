import argparse
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from safetensors import safe_open

VOLUME = "lerobot-smolvla"
REQUIRED_FILES = ["config.json", "model.safetensors", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"]


def modal(*args: str) -> str:
    return subprocess.run(["modal", *args], capture_output=True, text=True, check=True).stdout


def remote_steps(run: str) -> list[str]:
    try:
        listing = modal("volume", "ls", VOLUME, f"outputs/{run}/checkpoints")
    except subprocess.CalledProcessError:
        return []
    return sorted(Path(line).name for line in listing.split() if re.fullmatch(r".*/\d{6}", line))


def fetch_checkpoint(run: str, step: str, workdir: Path) -> Path | None:
    target = workdir / step
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    # EMA runs save the averaged weights separately; held-out loss is measured on those, so publish them
    try:
        modal("volume", "get", "--force", VOLUME, f"outputs/{run}/checkpoints/{step}/pretrained_model_ema", str(target))
        model_dir = target / "pretrained_model_ema"
    except subprocess.CalledProcessError:
        modal("volume", "get", "--force", VOLUME, f"outputs/{run}/checkpoints/{step}/pretrained_model", str(target))
        model_dir = target / "pretrained_model"
    if not all((model_dir / name).exists() for name in REQUIRED_FILES):
        return None
    try:
        with safe_open(model_dir / "model.safetensors", "pt") as f:
            list(f.keys())
    except Exception:  # noqa: BLE001
        # the Volume is committed every 5 min while a checkpoint may still be mid-write; retry next poll
        return None
    return model_dir


def fetch_log(run: str, workdir: Path) -> str:
    path = workdir / "run.log"
    try:
        modal("volume", "get", "--force", VOLUME, f"logs/{run}.log", str(path))
        return path.read_text(errors="ignore")
    except subprocess.CalledProcessError:
        return ""


def model_card(repo_id: str, run: str, dataset: str, policy: str, settings: str, published: list[str], eval_losses: dict[int, float], finished: bool) -> str:
    rows = "\n".join(
        f"| `{step}/` | {eval_losses[int(step)]:.4f} |" if int(step) in eval_losses else f"| `{step}/` | n/a |"
        for step in published
    )
    return f"""---
library_name: lerobot
tags:
- {policy.lower()}
- robotics
- lerobot
- so101
datasets:
- {dataset}
---

# {policy} policy, SO-101, striped ball in a bowl of red and blue balls

{policy} trained with LeRobot 0.7.0 on [`{dataset}`](https://huggingface.co/datasets/{dataset})
(80 episodes, 2 cameras `top` and `wrist` at 640x480, 30 fps), run `{run}`.
{settings}
Training status: {"finished" if finished else "running, new checkpoints are added automatically"}.

Each folder is one checkpoint and is a complete `pretrained_model` directory. Held-out loss is the
mean loss on the held-out episodes; with so few episodes differences of a few percent are noise.

| checkpoint | held-out loss |
|---|---|
{rows}

## Use

```bash
hf download {repo_id} --include "010000/*" --local-dir pick_rb
lerobot-rollout --policy.path=pick_rb/010000 ...   # same robot id, camera names and resolution as the recording
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="act_redblue_l40s_b32_38k")
    parser.add_argument("--dataset", default="molnrb/so101_pick_rb_mixed")
    parser.add_argument("--repo-name", default="act_so101_pick_rb_mixed_b32_38k")
    parser.add_argument("--policy", default="ACT")
    parser.add_argument("--settings", default="Batch 32, lr 2e-5, bf16, 38,000 steps planned, 5% of episodes held out (4 episodes, seed 1000).")
    parser.add_argument("--poll-s", type=int, default=120)
    # the run is capped at 172 min; when the cap stops it, LeRobot never logs "End of training"
    parser.add_argument("--max-hours", type=float, default=3.5)
    args = parser.parse_args()

    api = HfApi(token=dotenv_values(".env")["HF_TOKEN"])
    repo_id = f"{api.whoami()['name']}/{args.repo_name}"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    print(f"publishing to https://huggingface.co/{repo_id}", flush=True)

    existing = {Path(f).parts[0] for f in api.list_repo_files(repo_id) if re.fullmatch(r"\d{6}/.*", f)}
    published = sorted(existing)
    workdir = Path(tempfile.mkdtemp(prefix="publish_"))
    deadline = time.time() + args.max_hours * 3600
    while True:
        log = fetch_log(args.run, workdir)
        finished = "End of training" in log or time.time() > deadline
        eval_losses = {int(s): float(v) for s, v in re.findall(r"step (\d+): eval_loss=([\d.]+)", log)}
        new = [s for s in remote_steps(args.run) if s not in published]
        for step in new:
            model_dir = fetch_checkpoint(args.run, step, workdir)
            if model_dir is None:
                print(f"{step}: incomplete on the Volume, retrying next poll", flush=True)
                continue
            api.upload_folder(repo_id=repo_id, folder_path=model_dir, path_in_repo=step, commit_message=f"checkpoint {step}")
            published = sorted({*published, step})
            print(f"{step}: uploaded", flush=True)
            shutil.rmtree(workdir / step, ignore_errors=True)
        if new or finished:
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=model_card(repo_id, args.run, args.dataset, args.policy, args.settings, published, eval_losses, finished).encode(),
                path_in_repo="README.md",
                commit_message="update model card",
            )
        if finished and not [s for s in remote_steps(args.run) if s not in published]:
            print("training finished and every checkpoint is published", flush=True)
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
