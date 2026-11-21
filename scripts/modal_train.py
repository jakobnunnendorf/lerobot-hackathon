"""Train a LeRobot policy on Modal under a cost cap: prepare (CPU), bench (short GPU run), train.

    uv run modal run scripts/modal_train.py::prepare --recipe act_redblue
    uv run modal run scripts/modal_train.py::bench --recipe act_redblue --gpus L40S,A100-80GB --budget-usd 10
    uv run modal run --detach scripts/modal_train.py::train --recipe act_redblue --gpu L40S --steps N --max-minutes M
    uv run modal volume get lerobot-smolvla outputs/<run_name> outputs/

SmolVLA on the red/blue data, pushed to a new Hugging Face model repo (needs the Modal secret
huggingface-secret with HF_TOKEN, a write token of the repo's owner):

    uv run modal run scripts/modal_train.py::prepare --recipe smolvla_redblue
    uv run modal run scripts/modal_train.py::train --recipe smolvla_redblue --gpu H100 --smoke --hub-repo USER/smolvla_redblue
    uv run modal run --detach scripts/modal_train.py::train --recipe smolvla_redblue --gpu H100 \
        --steps 20000 --max-minutes 115 --hub-repo USER/smolvla_redblue
    uv run modal run scripts/modal_train.py::push --run-name <run_name> --hub-repo USER/smolvla_redblue  # after a stop
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

REDBLUE_COMMON_ARGS = [
    "--dataset.image_transforms.enable=true",
    f"--dataset.image_transforms.tfs={REDBLUE_AUGMENTATION}",
    # Same split and seed as the stopped local batch-8 run, so eval losses compare.
    "--dataset.eval_split=0.05",
    "--seed=1000",
    "--eval_steps=5000",
]

REDBLUE_ACT_ARGS = [
    # 2e-5 at batch 32 instead of ACT's 1e-5 at batch 8: a cautious step below linear scaling.
    "--policy.optimizer_lr=2e-5",
    "--policy.optimizer_lr_backbone=2e-5",
    *REDBLUE_COMMON_ARGS,
]

FOIL_CHECKPOINT = "molnrb/act_pick_ball_combined"

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
        "args": ["--policy.type=act", *REDBLUE_ACT_ARGS],
        "schedule_follows_steps": False,
    },
    # SmolVLA docs' fine-tune settings (batch 64, 20k steps, lr 1e-4 from smolvla_base) on the red/blue data,
    # with act_redblue's augmentation, split and seed so the eval losses compare. Kamel's smolvla_combined:
    # 0.26 s/step on an H100 at batch 64, so 20k steps take about 90 min there.
    "smolvla_redblue": {
        "dataset": "molnrb/so101_pick_rb_mixed",
        "frames": 57183,
        "batch_size": 64,
        "prefetch_models": ["lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"],
        "args": [
            "--policy.path=lerobot/smolvla_base",
            f"--rename_map={SMOLVLA_RENAME_MAP}",
            *REDBLUE_COMMON_ARGS,
        ],
        "schedule_follows_steps": True,
    },
    # Same as act_redblue except the start: the foil-scene policy's weights. See decisions/act-redblue-init.md.
    "act_redblue_from_foil": {
        "dataset": "molnrb/so101_pick_rb_mixed",
        "frames": 57183,
        "batch_size": 32,
        "prefetch_models": [FOIL_CHECKPOINT],
        "args": [f"--policy.path={FOIL_CHECKPOINT}", *REDBLUE_ACT_ARGS],
        "schedule_follows_steps": False,
    },
}

CPU = 12.0
MEMORY = 32 * 1024
# Modal prices read from https://modal.com/pricing on 2026-11-21 (USD per hour).
GPU_USD_H = {"H100": 3.95, "A100-80GB": 2.50, "A100-40GB": 2.10, "L40S": 1.95, "A10": 1.10, "L4": 0.80}
CPU_USD_H = CPU * 0.0472 + (MEMORY / 1024) * 0.0080
STARTUP_MINUTES = 8
SMOKE = {"steps": 50, "save_freq": 50, "eval_steps": 25, "log_freq": 10, "max_minutes": 15}
HF_SECRET = "huggingface-secret"  # Modal secret holding HF_TOKEN; only attached when --hub-repo is given

VOL = "/vol"
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


def train_command(
    recipe: str, run_name: str, steps: int, batch_size: int, save_freq: int, log_freq: int, save: bool,
    eval_steps: int | None = None,
) -> list:
    r = RECIPES[recipe]
    args = r["args"]
    if eval_steps is not None:
        args = [a for a in args if not a.startswith("--eval_steps")] + [f"--eval_steps={eval_steps}"]
    command = [
        "lerobot-train",
        f"--dataset.repo_id={r['dataset']}",
        *args,
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--accelerator.mixed_precision=bf16",
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


def check_hub_repo(hub_repo: str) -> None:
    """Fail before training if the token cannot write to hub_repo or the repo is someone else's."""
    from huggingface_hub import HfApi

    user = HfApi().whoami()["name"]
    owner = hub_repo.split("/")[0]
    if owner != user:
        raise SystemExit(f"--hub-repo {hub_repo}: the token belongs to {user}, not {owner}; refusing to push there")
    HfApi().create_repo(hub_repo, repo_type="model", exist_ok=True)
    print(f"[hub] {hub_repo} ready for {user}", flush=True)


def push_checkpoints(run_name: str, hub_repo: str, prefix: str = "") -> list:
    """Newest checkpoint at the repo root (or under prefix), every checkpoint under checkpoints/<step>/."""
    from huggingface_hub import HfApi

    check_hub_repo(hub_repo)
    checkpoint_dir = f"{VOL}/outputs/{run_name}/checkpoints"
    steps = sorted(d for d in os.listdir(checkpoint_dir) if d.isdigit()) if os.path.isdir(checkpoint_dir) else []
    if not steps:
        print(f"[hub] no checkpoints in {checkpoint_dir}, nothing pushed", flush=True)
        return []
    api = HfApi()
    for step in steps:
        api.upload_folder(
            repo_id=hub_repo,
            folder_path=f"{checkpoint_dir}/{step}/pretrained_model",
            path_in_repo=f"{prefix}checkpoints/{step}",
            commit_message=f"{run_name}: checkpoint {step}",
        )
    api.upload_folder(
        repo_id=hub_repo,
        folder_path=f"{checkpoint_dir}/{steps[-1]}/pretrained_model",
        path_in_repo=prefix or None,
        commit_message=f"{run_name}: newest checkpoint {steps[-1]}",
    )
    print(f"[hub] pushed {steps} to https://huggingface.co/{hub_repo}", flush=True)
    return steps


@app.function(cpu=4.0, memory=16 * 1024, volumes={VOL: volume}, timeout=60 * 60)
def prepare(recipe: str = "act_redblue") -> None:
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for repo in RECIPES[recipe]["prefetch_models"]:
        snapshot_download(repo)
    ds = LeRobotDataset(RECIPES[recipe]["dataset"])
    print(f"dataset ready: {ds.num_episodes} episodes, {ds.num_frames} frames", flush=True)
    volume.commit()


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
def train_remote(
    recipe: str, run_name: str, steps: int, batch_size: int, save_freq: int, max_minutes: float,
    log_freq: int = 100, eval_steps: int | None = None, hub_repo: str = "", hub_prefix: str = "",
) -> dict:
    if hub_repo:
        check_hub_repo(hub_repo)
    command = train_command(recipe, run_name, steps, batch_size, save_freq, log_freq, save=True, eval_steps=eval_steps)
    log_path = f"{VOL}/logs/{run_name}.log"
    code, minutes, samples = run_training(command, max_minutes, log_path)
    checkpoint_dir = f"{VOL}/outputs/{run_name}/checkpoints"
    util = [s[0] for s in samples]
    text = open(log_path, errors="ignore").read()
    step_s = [float(m) for m in re.findall(r"step_s:([\d.]+)", text)]
    step_s = step_s[len(step_s) // 3 :]
    eval_lines = [line for line in text.splitlines() if "eval" in line.lower() and "loss" in line.lower()]
    pushed: list = []
    if hub_repo:
        try:
            pushed = push_checkpoints(run_name, hub_repo, hub_prefix)
        except Exception as e:  # noqa: BLE001 - checkpoints stay on the Volume; retry with ::push
            print(f"[hub] push failed: {e}; checkpoints stay on the Volume, retry with "
                  f"modal run scripts/modal_train.py::push --run-name {run_name} --hub-repo {hub_repo}", flush=True)
    return {
        "code": code,
        "minutes": minutes,
        "step_s": sum(step_s) / len(step_s) if step_s else None,
        "checkpoints": sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else [],
        "pushed": pushed,
        "gpu_util_pct": sum(util) / max(len(util), 1),
        "eval_lines": eval_lines[-3:],
    }


@app.function(cpu=2.0, memory=8 * 1024, volumes={VOL: volume}, timeout=60 * 60)
def push_remote(run_name: str, hub_repo: str) -> list:
    volume.reload()
    return push_checkpoints(run_name, hub_repo)


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
    hub_repo: str = "",
    smoke: bool = False,
) -> None:
    check_hub_repo_name(hub_repo)
    batch_size = batch_size or RECIPES[recipe]["batch_size"]
    log_freq, eval_steps, hub_prefix = 100, None, ""
    if smoke:
        steps, save_freq, max_minutes = SMOKE["steps"], SMOKE["save_freq"], SMOKE["max_minutes"]
        log_freq, eval_steps, hub_prefix = SMOKE["log_freq"], SMOKE["eval_steps"], "smoke/"
    run_name = run_name or f"{recipe}_{gpu}_b{batch_size}_{steps // 1000}k" + (f"_smoke{int(time.time())}" if smoke else "")
    print(f"{run_name}: cap {max_minutes} min, worst case ~{usd(gpu, max_minutes + STARTUP_MINUTES):.2f} USD")
    options = {"gpu": gpu, "timeout": int((max_minutes + 20) * 60)}
    if hub_repo:
        options["secrets"] = [modal.Secret.from_name(HF_SECRET)]
    result = train_remote.with_options(**options).remote(
        recipe, run_name, steps, batch_size, save_freq, max_minutes, log_freq, eval_steps, hub_repo, hub_prefix
    )
    print(result)
    if result.get("step_s"):
        full = 20000 * result["step_s"] / 60
        print(f"20000 steps at {result['step_s']:.3f} s/step: ~{full:.0f} min, "
              f"~{usd(gpu, full + STARTUP_MINUTES):.2f} USD on {gpu}")
    print(f"download: uv run modal volume get lerobot-smolvla outputs/{run_name} outputs/")


def check_hub_repo_name(hub_repo: str) -> None:
    if hub_repo and (hub_repo.count("/") != 1 or "<" in hub_repo or hub_repo.startswith("USER/")):
        raise SystemExit(f"--hub-repo {hub_repo!r}: give your own Hugging Face user and a new repo, e.g. jdoe/smolvla_redblue")


@app.local_entrypoint()
def push(run_name: str, hub_repo: str) -> None:
    """Push the checkpoints of a stopped or finished run from the Volume to the Hub (CPU only)."""
    check_hub_repo_name(hub_repo)
    print(push_remote.with_options(secrets=[modal.Secret.from_name(HF_SECRET)]).remote(run_name, hub_repo))
