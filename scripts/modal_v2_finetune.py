"""Fine-tune Tomi's act_redblue_from_foil_v2 (checkpoint 20000) on Modal and push each checkpoint to the Hub.

    py -3.12 -m modal run --detach scripts/modal_v2_finetune.py::run_a      # +10K steps on all of rb_mixed
    py -3.12 -m modal run --detach scripts/modal_v2_finetune.py::run_b      # 3K steps, clean hidden/partly starts
    python scripts/modal_v2_finetune.py                                     # free: print both commands and lists
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import modal

DATASET = "molnrb/so101_pick_rb_mixed"
DATASET_REVISION = "8d36c36c357e7ae6a78b3a844ef3caa1093d7c59"
START_REPO, START_FOLDER, START_STEP = "Space0code/act_redblue_from_foil_v2", "checkpoints/20000", 20000

# Tomi's recipe (scripts/modal_train.py on Space0code/modal-act-from-foil: REDBLUE_ACT_ARGS, REDBLUE_AUGMENTATION),
# the one v1 and v2 were trained with.
AUGMENTATION = json.dumps(
    {
        "brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
        "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
        "saturation": {"weight": 0.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
        "hue": {"weight": 0.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
        "sharpness": {"type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
        "affine": {"type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}},
    }
)
RECIPE = [
    "--policy.optimizer_lr=2e-5",
    "--policy.optimizer_lr_backbone=2e-5",
    "--dataset.image_transforms.enable=true",
    f"--dataset.image_transforms.tfs={AUGMENTATION}",
    "--dataset.eval_split=0.05",
    "--seed=1000",
    "--batch_size=32",
    "--accelerator.mixed_precision=bf16",
    "--log_freq=100",
]

# Run B: Jakob's clean training episodes (decisions/act-clean.md on jakobnunnendorf/modal-act-clean, 93 of 201) whose
# start is buried or partly covered (research/label-patterns/episodes.csv). His hold-out 6, 23, 65, 88, 99, 110, 126,
# 157, 177, 191 is never in it.
HIDDEN_BURIED = [7, 13, 14, 15, 17, 20, 25, 27, 29, 53, 55, 59, 72, 133, 143, 150, 162, 165, 166, 167, 168, 171, 184,
                 197, 198]
HIDDEN_PARTLY = [0, 5, 11, 19, 33, 105, 108, 113, 124, 128, 130, 140, 153, 155, 200]
HIDDEN_EPISODES = sorted(HIDDEN_BURIED + HIDDEN_PARTLY)
JAKOB_HOLDOUT = {6, 23, 65, 88, 99, 110, 126, 157, 177, 191}

RUNS = {
    # L40S: on H100 the video loader is the bottleneck (253 vs 227 steps/min). Saves every 2,500 so a late start or
    # the deadline stop still leaves a checkpoint to test.
    "a": dict(hub_repo="TianK003/act_redblue_from_foil_v3", steps=10_000, save_freq=2_500, eval_steps=2_500,
              episodes=None, step_offset=START_STEP, gpu="L40S"),
    "b": dict(hub_repo="TianK003/act_hidden_from_v2", steps=3_000, save_freq=1_000, eval_steps=1_000,
              episodes=HIDDEN_EPISODES, step_offset=0, gpu="L40S"),
}
DEADLINE_CET = "07:45"
UPLOAD_MARGIN_MIN = 4

CPU = 12.0
MEMORY = 32 * 1024
WORK = "/root/work"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[training]==0.7.0")
    .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
app = modal.App("lerobot-v2-finetune", image=image)


def train_command(run: dict, start_dir: str, dataset_root: str, output_dir: str) -> list:
    command = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET}",
        f"--dataset.root={dataset_root}",
        f"--policy.path={start_dir}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        *RECIPE,
        f"--steps={run['steps']}",
        f"--save_freq={run['save_freq']}",
        f"--eval_steps={run['eval_steps']}",
        f"--num_workers={int(CPU) - 2}",
        f"--output_dir={output_dir}",
        f"--job_name={Path(output_dir).name}",
    ]
    if run["episodes"]:
        command.insert(2, f"--dataset.episodes={json.dumps(run['episodes'])}")
    return command


def hub_folder(run: dict, checkpoint: str) -> str:
    return f"checkpoints/{run['step_offset'] + int(checkpoint):06d}"


class CheckpointPusher:
    def __init__(self, run: dict, output_dir: str):
        from huggingface_hub import HfApi

        self.run, self.root = run, Path(output_dir) / "checkpoints"
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.api.create_repo(run["hub_repo"], repo_type="model", exist_ok=True, private=False)
        self.pushed: set[str] = set()

    def push_new(self) -> None:
        if not self.root.is_dir():
            return
        for folder in sorted(p for p in self.root.iterdir() if p.name.isdigit() and p.name not in self.pushed):
            model = folder / "pretrained_model"
            if not (model / "model.safetensors").exists():
                continue
            target = hub_folder(self.run, folder.name)
            self.api.upload_folder(repo_id=self.run["hub_repo"], folder_path=str(model), path_in_repo=target,
                                   commit_message=f"{target} ({folder.name} steps from {START_REPO}@{START_STEP})")
            self.pushed.add(folder.name)
            print(f"[hub] pushed {self.run['hub_repo']}/{target}", flush=True)

    def watch(self, stop: threading.Event) -> None:
        while not stop.wait(60):
            try:
                self.push_new()
            except Exception as error:  # noqa: BLE001 - a failed push is retried next minute and after training
                print(f"[hub] push failed, will retry: {error!r}", flush=True)


def prepare(run: dict) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    start = snapshot_download(START_REPO, allow_patterns=[f"{START_FOLDER}/*"], local_dir=f"{WORK}/start")
    dataset = snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION, local_dir=f"{WORK}/dataset")
    return f"{start}/{START_FOLDER}", dataset


@app.function(cpu=CPU, memory=MEMORY, gpu="H100", timeout=2 * 60 * 60, secrets=[modal.Secret.from_name("hf-write")])
def train(name: str, deadline_epoch: float) -> dict:
    run = RUNS[name]
    start_dir, dataset_root = prepare(run)
    output_dir = f"{WORK}/outputs/{Path(run['hub_repo']).name}"
    command = train_command(run, start_dir, dataset_root, output_dir)
    print("+", " ".join(command), flush=True)
    pusher, stop = CheckpointPusher(run, output_dir), threading.Event()
    threading.Thread(target=pusher.watch, args=(stop,), daemon=True).start()
    proc = subprocess.Popen(command)
    budget_s = deadline_epoch - UPLOAD_MARGIN_MIN * 60 - time.time()
    try:
        code = proc.wait(timeout=max(budget_s, 60))
    except subprocess.TimeoutExpired:
        print(f"[deadline] {DEADLINE_CET} CET minus {UPLOAD_MARGIN_MIN} min reached, stopping training", flush=True)
        proc.terminate()
        code = proc.wait(60)
    stop.set()
    pusher.push_new()
    return {"run": name, "code": code, "pushed": sorted(hub_folder(run, c) for c in pusher.pushed)}


def deadline_epoch() -> float:
    hour, minute = map(int, DEADLINE_CET.split(":"))
    now = datetime.now(timezone(timedelta(hours=1)))
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()


def launch(name: str) -> None:
    run, deadline = RUNS[name], deadline_epoch()
    minutes_left = (deadline - time.time()) / 60
    if minutes_left < 20:
        sys.exit(f"only {minutes_left:.0f} min to {DEADLINE_CET} CET: not starting")
    print(f"run {name}: {run['steps']} steps on {run['gpu']}, pushes to {run['hub_repo']}, stops by {DEADLINE_CET} CET")
    print(train.with_options(gpu=run["gpu"]).remote(name, deadline))


@app.local_entrypoint()
def run_a() -> None:
    launch("a")


@app.local_entrypoint()
def run_b() -> None:
    launch("b")


if __name__ == "__main__":
    assert not JAKOB_HOLDOUT & set(HIDDEN_EPISODES), "run B contains a hold-out episode"
    print(f"run B: {len(HIDDEN_EPISODES)} episodes ({len(HIDDEN_BURIED)} buried, {len(HIDDEN_PARTLY)} partly covered)")
    for name, run in RUNS.items():
        print(f"\nrun {name} ({run['gpu']}), hub {run['hub_repo']}, checkpoints "
              f"{[hub_folder(run, f'{s:06d}') for s in range(run['save_freq'], run['steps'] + 1, run['save_freq'])]}")
        print(" ".join(train_command(run, f"{WORK}/start/{START_FOLDER}", f"{WORK}/dataset", f"{WORK}/outputs/x")))
