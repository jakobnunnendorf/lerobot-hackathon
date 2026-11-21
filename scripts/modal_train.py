"""Train a LeRobot policy on Modal under a cost cap: prepare (CPU), bench (short GPU run), train.

    uv run modal run scripts/modal_train.py::prepare --recipe act_redblue
    uv run modal run scripts/modal_train.py::bench --recipe act_redblue --gpus L40S,A100-80GB --budget-usd 10
    uv run modal run --detach scripts/modal_train.py::train --recipe act_redblue --gpu L40S --steps N --max-minutes M
    uv run modal volume get lerobot-smolvla outputs/<run_name> outputs/
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import modal

SMOLVLA_RENAME_MAP = json.dumps(
    {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
)

# Hue and saturation stay off: the target is told apart from red and blue balls by colour.
REDBLUE_AUGMENTATION = json.dumps(
    {
        "brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
        "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
        "saturation": {"weight": 0.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
        "hue": {"weight": 0.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
        "sharpness": {"type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
        "affine": {"type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}},
    }
)

REDBLUE_AUGMENTATION_V2 = json.dumps(
    {
        "brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
        "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.7, 1.3]}},
        "sharpness": {"type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
        "affine": {"type": "RandomAffine", "kwargs": {"degrees": [-10.0, 10.0], "translate": [0.1, 0.1]}},
        "shadow": {"type": "RandomShadow", "kwargs": {}},
        "patch_brightness": {"type": "GaussianPatchBrightness", "kwargs": {}},
        "dropout": {"type": "CoarseDropout", "kwargs": {}},
        "noise": {"type": "GaussianNoise", "kwargs": {}},
        "blur": {"type": "MotionBlur", "kwargs": {}},
    }
)

VOL = "/vol"

# The source datasets said "blue ball"; the target is a striped green and yellow ball. Edited copies
# live on the Volume, the molnrb repos are never touched.
PICK_SOURCES = {
    "molnrb/so101_pick_ball": (
        "Pick up the ball and put it in the box",
        "Pick up the aluminium ball and put it in the box",
    ),
    "molnrb/so101_pick_blue_ball": (
        "Pick up the blue ball from the bowl and return to the start position",
        "Pick up the striped green and yellow ball from the bowl and return to the start position",
    ),
    "molnrb/so101_pick_buried_ball": (
        "Find the buried blue ball in the bowl, uncover it, pick it up and return to the start position",
        "Find the buried striped green and yellow ball in the bowl, uncover it, pick it up and return to the start position",
    ),
    "molnrb/so101_pick_rb_mixed": (
        "Pick up the striped ball from the bowl of red and blue balls, uncovering it first if it is buried, and return to the start position",
        "Pick up the striped green and yellow ball from the bowl of red and blue balls, uncovering it first if it is buried, and return to the start position",
    ),
}
MERGED_DATASET = "brouthen/so101_pick_striped_all"
PRETRAIN_RUN = "act_pretrain_all_h100_50k"


def dataset_root(repo_id: str) -> str:
    return f"{VOL}/datasets/{repo_id.split('/')[-1]}"


RECIPES = {
    "smolvla_combined": {
        "dataset": "molnrb/so101_pick_ball_combined",
        "frames": 91930,
        "batch_size": 64,
        "prefetch_models": ["lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"],
        "args": [
            "--policy.path=lerobot/smolvla_base",
            f"--rename_map={SMOLVLA_RENAME_MAP}",
        ],
        # smolvla_base decays its LR over 30k steps; a shorter run must decay over its own length.
        "schedule_follows_steps": True,
    },
    "act_redblue": {
        "dataset": "molnrb/so101_pick_rb_mixed",
        "frames": 57183,
        "batch_size": 32,
        "prefetch_models": [],
        "args": [
            "--policy.type=act",
            # 2e-5 at batch 32 instead of ACT's 1e-5 at batch 8: a cautious step below linear scaling.
            "--policy.optimizer_lr=2e-5",
            "--policy.optimizer_lr_backbone=2e-5",
            "--dataset.image_transforms.enable=true",
            f"--dataset.image_transforms.tfs={REDBLUE_AUGMENTATION}",
            # Same split and seed as the stopped local batch-8 run, so eval losses compare.
            "--dataset.eval_split=0.05",
            "--seed=1000",
            "--eval_steps=5000",
        ],
        "schedule_follows_steps": False,
    },
    # Starts from the morning's blue-ball SmolVLA (pretraining on related data helped our ACT runs).
    # Not --resume, so the normaliser takes this dataset's stats instead of the blue-ball ones.
    "smolvla_redblue_from_combined": {
        "dataset": "molnrb/so101_pick_rb_mixed",
        "frames": 57183,
        "batch_size": 64,
        "prefetch_models": ["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"],
        "args": [
            f"--policy.path={VOL}/outputs/smolvla_pick_ball_h100_23k/checkpoints/023000/pretrained_model",
            f"--rename_map={SMOLVLA_RENAME_MAP}",
            # half the from-base LR: the start point is already tuned to this arm and bowl
            "--policy.optimizer_lr=5e-5",
            "--dataset.image_transforms.enable=true",
            "--dataset.image_transforms.backend=gpu",
            "--dataset.image_transforms.max_num_transforms=4",
            f"--dataset.image_transforms.tfs={REDBLUE_AUGMENTATION_V2}",
            "--dataset.eval_split=0.1",
            "--seed=1000",
            "--eval_steps=1000",
        ],
        "schedule_follows_steps": True,
    },
    # LeRobot ACT defaults (batch 8, lr 1e-5, chunk 100, fp32, no augmentation). eval_split 0.1 holds
    # out the last 10% of each task, so pretrain and fine-tune hold out the same 12 red/blue episodes.
    "act_pretrain_all": {
        "dataset": MERGED_DATASET,
        "root": dataset_root(MERGED_DATASET),
        "frames": 180134,
        "batch_size": 8,
        "mixed_precision": "no",
        "prefetch_models": [],
        "args": ["--policy.type=act", "--dataset.eval_split=0.1", "--seed=1000", "--eval_steps=5000"],
        "schedule_follows_steps": False,
    },
    # not --resume: the normaliser switches to the red/blue stats
    "act_finetune_rb": {
        "dataset": "molnrb/so101_pick_rb_mixed",
        "root": dataset_root("molnrb/so101_pick_rb_mixed"),
        "frames": 71301,
        "batch_size": 8,
        "mixed_precision": "no",
        "prefetch_models": [],
        "args": [
            f"--policy.path={VOL}/outputs/{PRETRAIN_RUN}/checkpoints/050000/pretrained_model",
            "--dataset.eval_split=0.1",
            "--seed=1000",
            "--eval_steps=2000",
        ],
        "schedule_follows_steps": False,
    },
    # act_redblue held-out loss went flat after 10k steps while train loss kept falling; this run
    # targets that gap. See decisions/act-redblue-generalisation.md.
    "act_redblue_v2": {
        "dataset": "molnrb/so101_pick_rb_mixed",
        "frames": 57183,
        "batch_size": 32,
        "prefetch_models": [],
        "args": [
            "--policy.type=act",
            "--policy.chunk_size=50",
            "--policy.n_action_steps=25",
            "--policy.dropout=0.2",
            "--policy.optimizer_lr=2e-5",
            "--policy.optimizer_lr_backbone=2e-5",
            "--ema.type=constant",
            "--ema.decay=0.999",
            "--dataset.image_transforms.enable=true",
            # GPU backend: nine transform types would make the CPU data loader the bottleneck.
            "--dataset.image_transforms.backend=gpu",
            "--dataset.image_transforms.max_num_transforms=4",
            f"--dataset.image_transforms.tfs={REDBLUE_AUGMENTATION_V2}",
            "--dataset.eval_split=0.1",
            "--seed=1000",
            "--eval_steps=2000",
        ],
        "schedule_follows_steps": False,
    },
}

CPU = 12.0
MEMORY = 32 * 1024
# Modal prices read from https://modal.com/pricing on 2026-11-21 (USD per hour).
GPU_USD_H = {"B200": 6.25, "H200": 4.54, "H100": 3.95, "A100-80GB": 2.50, "A100-40GB": 2.10, "L40S": 1.95, "A10": 1.10, "L4": 0.80}
CPU_USD_H = CPU * 0.0472 + (MEMORY / 1024) * 0.0080
STARTUP_MINUTES = 8

volume = modal.Volume.from_name("lerobot-smolvla", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[smolvla,training]==0.7.0")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            # Trackio's SQLite is kept off the network Volume and copied there after the run.
            "TRACKIO_DIR": "/root/trackio",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App("lerobot-train", image=image)


def usd(gpu: str, minutes: float) -> float:
    return (GPU_USD_H[gpu] + CPU_USD_H) * minutes / 60


def train_command(recipe: str, run_name: str, steps: int, batch_size: int, save_freq: int, log_freq: int, save: bool) -> list:
    r = RECIPES[recipe]
    command = [
        "lerobot-train",
        f"--dataset.repo_id={r['dataset']}",
        *([f"--dataset.root={r['root']}"] if r.get("root") else []),
        *r["args"],
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        # "no" is the default; passing it would be parsed as the boolean False
        *([f"--accelerator.mixed_precision={r.get('mixed_precision', 'bf16')}"] if r.get("mixed_precision", "bf16") != "no" else []),
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--num_workers={int(CPU) - 2}",
        "--prefetch_factor=4",
        f"--save_checkpoint={str(save).lower()}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--trackio.enable=true",
        "--trackio.project=hackathon",
        f"--trackio.run_name={run_name}",
        f"--job_name={run_name}",
        f"--output_dir={VOL}/outputs/{run_name}",
    ]
    if r["schedule_follows_steps"]:
        command += [
            f"--policy.scheduler_decay_steps={steps}",
            f"--policy.scheduler_warmup_steps={min(1000, max(steps // 20, 10))}",
        ]
    return command


def commit_periodically(stop: threading.Event) -> None:
    while not stop.wait(300):
        try:
            volume.commit()
        except Exception as e:  # noqa: BLE001 - a failed commit must not kill training
            print(f"[commit] failed: {e}", flush=True)


def sample_gpu(stop: threading.Event, samples: list) -> None:
    while not stop.wait(5):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out:
            samples.append(tuple(float(x) for x in out.split(",")))


def run_training(command: list, max_minutes: float, log_path: str) -> tuple[int, float, list]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(" ".join(command), flush=True)
    stop = threading.Event()
    samples: list = []
    for target, args in ((commit_periodically, (stop,)), (sample_gpu, (stop, samples))):
        threading.Thread(target=target, args=args, daemon=True).start()
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump() -> None:
            for line in proc.stdout:
                log.write(line)
                # tqdm redraws share a line with LeRobot's INFO metrics; keep only the INFO part
                if "Training:" in line:
                    line = line[line.find("INFO") :] if "INFO" in line else ""
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()

        pumper = threading.Thread(target=pump, daemon=True)
        pumper.start()
        while proc.poll() is None:
            if time.time() - start > max_minutes * 60:
                print(f"[budget] {max_minutes} min reached, stopping training", flush=True)
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(5)
        pumper.join(10)
    stop.set()
    if os.path.isdir("/root/trackio"):
        shutil.copytree("/root/trackio", f"{VOL}/trackio", dirs_exist_ok=True)
    volume.commit()
    return proc.returncode, (time.time() - start) / 60, samples


@app.function(cpu=4.0, memory=16 * 1024, volumes={VOL: volume}, timeout=60 * 60)
def prepare(recipe: str = "act_redblue") -> None:
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for repo in RECIPES[recipe]["prefetch_models"]:
        snapshot_download(repo)
    ds = LeRobotDataset(RECIPES[recipe]["dataset"])
    print(f"dataset ready: {ds.num_episodes} episodes, {ds.num_frames} frames", flush=True)
    volume.commit()


@app.function(
    cpu=8.0,
    memory=32 * 1024,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=90 * 60,
)
def prepare_merged() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for repo_id, (old_task, new_task) in PICK_SOURCES.items():
        if os.path.isdir(dataset_root(repo_id)) and new_task in LeRobotDataset(repo_id, root=dataset_root(repo_id)).meta.tasks.index:
            continue
        edit = ["lerobot-edit-dataset", f"--repo_id={repo_id}", f"--root={dataset_root(repo_id)}"]
        subprocess.run(
            [*edit, "--operation.type=modify_tasks", f"--operation.task_replacements={json.dumps({old_task: new_task})}"],
            check=True,
        )
    volume.commit()
    shutil.rmtree(dataset_root(MERGED_DATASET), ignore_errors=True)
    subprocess.run(
        [
            "lerobot-edit-dataset",
            f"--new_repo_id={MERGED_DATASET}",
            f"--new_root={dataset_root(MERGED_DATASET)}",
            "--operation.type=merge",
            f"--operation.repo_ids={json.dumps(list(PICK_SOURCES))}",
            f"--operation.roots={json.dumps([dataset_root(r) for r in PICK_SOURCES])}",
            # joining AV1 files from different recordings fails on non-monotonic DTS; keep one file per source
            "--operation.concatenate_videos=false",
            "--push_to_hub=true",
        ],
        check=True,
    )
    volume.commit()
    merged = LeRobotDataset(MERGED_DATASET, root=dataset_root(MERGED_DATASET))
    print(f"merged: {merged.num_episodes} episodes, {merged.num_frames} frames", flush=True)
    for task in merged.meta.tasks.index:
        print(f"  task: {task}", flush=True)


@app.function(cpu=CPU, memory=MEMORY, volumes={VOL: volume}, gpu="L40S", timeout=30 * 60)
def bench_remote(recipe: str, gpu: str, batch_size: int, steps: int) -> dict:
    run_name = f"bench_{recipe}_{gpu}_{batch_size}_{int(time.time())}"
    log_path = f"{VOL}/logs/{run_name}.log"
    command = train_command(recipe, run_name, steps, batch_size, steps, 25, save=False)
    command = [a for a in command if not a.startswith("--eval_steps")]
    code, minutes, samples = run_training(command, 25, log_path)
    text = open(log_path, errors="ignore").read()
    rows = [
        tuple(map(float, m))
        for m in re.findall(r"data_s:([\d.]+) prep_s:[\d.]+ updt_s:([\d.]+) step_s:([\d.]+)", text)
    ]
    rows = rows[len(rows) // 3 :]
    if code != 0 or not rows:
        return {"ok": False, "gpu": gpu, "code": code, "tail": text[-3000:]}
    data_s, updt_s, step_s = (sum(c) / len(c) for c in zip(*rows))
    util = [s[0] for s in samples[len(samples) // 3 :]]
    return {
        "ok": True,
        "gpu": gpu,
        "data_s": data_s,
        "updt_s": updt_s,
        "step_s": step_s,
        "gpu_util_pct": sum(util) / max(len(util), 1),
        "gpu_mem_gb": max((s[1] for s in samples), default=0) / 1024,
        "minutes": minutes,
    }


@app.function(cpu=CPU, memory=MEMORY, volumes={VOL: volume}, gpu="L40S", timeout=4 * 60 * 60)
def train_remote(recipe: str, run_name: str, steps: int, batch_size: int, save_freq: int, max_minutes: float) -> dict:
    command = train_command(recipe, run_name, steps, batch_size, save_freq, 100, save=True)
    code, minutes, samples = run_training(command, max_minutes, f"{VOL}/logs/{run_name}.log")
    checkpoint_dir = f"{VOL}/outputs/{run_name}/checkpoints"
    util = [s[0] for s in samples]
    return {
        "code": code,
        "minutes": minutes,
        "checkpoints": sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else [],
        "gpu_util_pct": sum(util) / max(len(util), 1),
    }


@app.local_entrypoint()
def bench(
    recipe: str = "act_redblue",
    gpus: str = "L40S",
    batch_size: int = 0,
    steps: int = 400,
    budget_usd: float = 10.0,
    hours: float = 3.0,
) -> None:
    r = RECIPES[recipe]
    batch_size = batch_size or r["batch_size"]
    calls = [bench_remote.with_options(gpu=g).spawn(recipe, g, batch_size, steps) for g in gpus.split(",")]
    results = [c.get() for c in calls]
    spent = sum(usd(x["gpu"], x.get("minutes", 10) + 3) for x in results if x["ok"])
    for x in results:
        if not x["ok"]:
            print(f"\n{x['gpu']} bench FAILED (exit {x['code']}):\n{x['tail']}")
            continue
        left = budget_usd - spent
        minutes = min(hours * 60, left / (GPU_USD_H[x["gpu"]] + CPU_USD_H) * 60) - STARTUP_MINUTES
        fit = int(minutes * 60 / x["step_s"] / 1000) * 1000
        bound = "DATA LOADER" if x["data_s"] > 0.3 * x["step_s"] else "GPU"
        print(
            f"\n{x['gpu']} batch {batch_size}: {x['step_s']:.3f} s/step (data {x['data_s']:.3f}, "
            f"update {x['updt_s']:.3f}), GPU util {x['gpu_util_pct']:.0f}%, peak mem {x['gpu_mem_gb']:.1f} GB "
            f"-> bottleneck: {bound}\n"
            f"  {minutes:.0f} min for {usd(x['gpu'], minutes + STARTUP_MINUTES):.2f} USD = ~{fit} steps "
            f"({fit * batch_size / r['frames']:.1f} epochs)\n"
            f"  uv run modal run --detach scripts/modal_train.py::train --recipe {recipe} --gpu {x['gpu']} "
            f"--batch-size {batch_size} --steps {fit} --max-minutes {int(minutes)}"
        )
    print(f"\nbench cost ~{spent:.2f} USD")


@app.local_entrypoint()
def train(
    recipe: str = "act_redblue",
    gpu: str = "L40S",
    steps: int = 20000,
    batch_size: int = 0,
    max_minutes: float = 170,
    save_freq: int = 5000,
    run_name: str = "",
) -> None:
    batch_size = batch_size or RECIPES[recipe]["batch_size"]
    run_name = run_name or f"{recipe}_{gpu}_b{batch_size}_{steps // 1000}k"
    print(f"{run_name}: cap {max_minutes} min, worst case ~{usd(gpu, max_minutes + STARTUP_MINUTES):.2f} USD")
    result = train_remote.with_options(gpu=gpu, timeout=int((max_minutes + 20) * 60)).remote(
        recipe, run_name, steps, batch_size, save_freq, max_minutes
    )
    print(result)
    print(f"download: uv run modal volume get lerobot-smolvla outputs/{run_name} outputs/")
