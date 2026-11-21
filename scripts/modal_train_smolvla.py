"""Fine-tune SmolVLA on Modal under a hard cost cap.

Three stages, each a separate `modal run` so nothing paid starts by surprise:

    # 1. CPU only, a few cents: download dataset + smolvla_base into the Volume once,
    #    so no GPU time is spent downloading.
    uv run modal run scripts/modal_train_smolvla.py::prepare

    # 2. ~10 min on the GPU (~0.8 USD on H100): measure s/step and whether the
    #    data loader or the GPU is the bottleneck; prints how many steps fit the budget.
    uv run modal run scripts/modal_train_smolvla.py::bench --gpu H100

    # 3. Full run. Stops itself at --max-minutes, keeps checkpoints every --save-freq.
    uv run modal run scripts/modal_train_smolvla.py::train --gpu H100 --steps 14000 --max-minutes 95

    # Download the result (and the Trackio log) afterwards:
    uv run modal volume get lerobot-smolvla outputs/<run_name> outputs/
    uv run modal volume get lerobot-smolvla trackio ~/.cache/huggingface/

Utilisation choices: bf16 autocast, batch 64, one data-loader worker per spare core with
prefetch, dataset and base model pre-cached on the Volume, LR schedule shortened to the real
number of steps (the default decays over 30k steps, which a short run never reaches).
"""

import os
import subprocess
import sys
import threading
import time

import modal

DATASET = "molnrb/so101_pick_ball_combined"
BASE_MODEL = "lerobot/smolvla_base"
RENAME_MAP = (
    '{"observation.images.top": "observation.images.camera1", '
    '"observation.images.wrist": "observation.images.camera2"}'
)

CPU = 12.0  # physical cores; enough decode workers for 2 x 480p video streams at batch 64
MEMORY = 48 * 1024
# Modal prices read from https://modal.com/pricing on 2026-11-21 (USD per hour).
GPU_USD_H = {"H100": 3.95, "A100-80GB": 2.50, "A100-40GB": 2.10, "L40S": 1.95}
CPU_USD_H = CPU * 0.0472 + (MEMORY / 1024) * 0.0080

VOL = "/vol"
volume = modal.Volume.from_name("lerobot-smolvla", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[smolvla,training]==0.7.0")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",  # dataset and base model are cached on the Volume
            "TRACKIO_DIR": "/root/trackio",  # SQLite stays on local disk, copied to the Volume after the run
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App("smolvla-pick-ball", image=image)


def _usd(gpu: str, minutes: float) -> float:
    return (GPU_USD_H[gpu] + CPU_USD_H) * minutes / 60


def _committer(stop: threading.Event, every_s: int = 300) -> None:
    """Commit the Volume regularly so checkpoints survive a timeout or crash."""
    while not stop.wait(every_s):
        try:
            volume.commit()
        except Exception as e:  # noqa: BLE001 - never kill training over a commit
            print(f"[commit] failed: {e}", flush=True)


def _gpu_sampler(stop: threading.Event, samples: list) -> None:
    while not stop.wait(5):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if out:
            util, mem = (float(x) for x in out.split(","))
            samples.append((util, mem))


def _train_cmd(run_name: str, steps: int, batch_size: int, save_freq: int, log_freq: int, save: bool) -> list:
    workers = int(CPU) - 2
    return [
        "lerobot-train",
        f"--policy.path={BASE_MODEL}",
        f"--dataset.repo_id={DATASET}",
        f"--rename_map={RENAME_MAP}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--accelerator.mixed_precision=bf16",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--policy.scheduler_decay_steps={steps}",
        f"--policy.scheduler_warmup_steps={min(1000, max(steps // 20, 10))}",
        f"--num_workers={workers}",
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


def _run(cmd: list, max_minutes: float, log_path: str) -> tuple[int, float, list]:
    """Run lerobot-train, tee its log to the Volume, stop it at max_minutes."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(" ".join(cmd), flush=True)
    stop = threading.Event()
    samples: list = []
    threads = [
        threading.Thread(target=_committer, args=(stop,), daemon=True),
        threading.Thread(target=_gpu_sampler, args=(stop, samples), daemon=True),
    ]
    for t in threads:
        t.start()
    t0 = time.time()
    deadline = t0 + max_minutes * 60
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump() -> None:
            for line in proc.stdout:
                log.write(line)
                if "Training:" not in line:  # drop tqdm redraws from the Modal log
                    sys.stdout.write(line)
                    sys.stdout.flush()

        pumper = threading.Thread(target=pump, daemon=True)
        pumper.start()
        while proc.poll() is None:
            if time.time() > deadline:
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
        import shutil

        shutil.copytree("/root/trackio", f"{VOL}/trackio", dirs_exist_ok=True)
    volume.commit()
    return proc.returncode, (time.time() - t0) / 60, samples


@app.function(cpu=4.0, memory=16 * 1024, volumes={VOL: volume}, timeout=60 * 60)
def prepare() -> None:
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    snapshot_download(BASE_MODEL)
    snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    ds = LeRobotDataset(DATASET)
    print(f"dataset ready: {ds.num_episodes} episodes, {ds.num_frames} frames", flush=True)
    volume.commit()


@app.function(cpu=CPU, memory=MEMORY, volumes={VOL: volume}, gpu="H100", timeout=30 * 60)
def bench_remote(gpu: str, batch_size: int, steps: int) -> dict:
    import re

    run = f"bench_{gpu}_{batch_size}_{int(time.time())}"
    log_path = f"{VOL}/logs/{run}.log"
    code, minutes, samples = _run(_train_cmd(run, steps, batch_size, steps, 25, save=False), 25, log_path)
    text = open(log_path).read()
    rows = [
        tuple(map(float, m))
        for m in re.findall(r"data_s:([\d.]+) prep_s:[\d.]+ updt_s:([\d.]+) step_s:([\d.]+)", text)
    ]
    rows = rows[len(rows) // 3 :]  # skip warm-up
    if code != 0 or not rows:
        return {"ok": False, "code": code, "tail": text[-3000:]}
    avg = [sum(c) / len(c) for c in zip(*rows)]
    util = [s[0] for s in samples[len(samples) // 3 :]]
    return {
        "ok": True,
        "gpu": gpu,
        "batch_size": batch_size,
        "data_s": avg[0],
        "updt_s": avg[1],
        "step_s": avg[2],
        "gpu_util_pct": sum(util) / max(len(util), 1),
        "gpu_mem_gb": max((s[1] for s in samples), default=0) / 1024,
        "minutes": minutes,
    }


@app.function(cpu=CPU, memory=MEMORY, volumes={VOL: volume}, gpu="H100", timeout=4 * 60 * 60)
def train_remote(run_name: str, steps: int, batch_size: int, save_freq: int, max_minutes: float) -> dict:
    log_path = f"{VOL}/logs/{run_name}.log"
    code, minutes, samples = _run(
        _train_cmd(run_name, steps, batch_size, save_freq, 100, save=True), max_minutes, log_path
    )
    ckpts = sorted(os.listdir(f"{VOL}/outputs/{run_name}/checkpoints")) if os.path.isdir(
        f"{VOL}/outputs/{run_name}/checkpoints"
    ) else []
    util = [s[0] for s in samples]
    return {"code": code, "minutes": minutes, "checkpoints": ckpts, "gpu_util_pct": sum(util) / max(len(util), 1)}


@app.local_entrypoint()
def bench(gpu: str = "H100", batch_size: int = 64, steps: int = 400, budget_usd: float = 10.0) -> None:
    r = bench_remote.with_options(gpu=gpu).remote(gpu, batch_size, steps)
    if not r["ok"]:
        print(f"bench FAILED (exit {r['code']}):\n{r['tail']}")
        return
    spent = _usd(gpu, r["minutes"] + 3)
    left = budget_usd - spent
    usable_min = left / (GPU_USD_H[gpu] + CPU_USD_H) * 60 - 8  # 8 min for startup, model load, checkpoints
    fit = int(usable_min * 60 / r["step_s"] / 1000) * 1000
    bound = "DATA LOADER" if r["data_s"] > 0.3 * r["step_s"] else "GPU"
    print(
        f"\n{gpu} batch {batch_size}: {r['step_s']:.3f} s/step (data {r['data_s']:.3f}, update {r['updt_s']:.3f}), "
        f"GPU util {r['gpu_util_pct']:.0f}%, peak mem {r['gpu_mem_gb']:.1f} GB -> bottleneck: {bound}\n"
        f"bench cost ~{spent:.2f} USD; remaining {left:.2f} USD buys ~{usable_min:.0f} min = ~{fit} steps "
        f"({fit * batch_size / 91930:.1f} epochs)\n"
        f"next: uv run modal run scripts/modal_train_smolvla.py::train --gpu {gpu} "
        f"--batch-size {batch_size} --steps {fit} --max-minutes {int(usable_min)}"
    )


@app.local_entrypoint()
def train(
    gpu: str = "H100",
    steps: int = 14000,
    batch_size: int = 64,
    max_minutes: float = 95,
    save_freq: int = 2000,
    run_name: str = "",
) -> None:
    run_name = run_name or f"smolvla_pick_ball_{gpu}_{steps}"
    print(f"{run_name}: cap {max_minutes} min, worst case ~{_usd(gpu, max_minutes + 8):.2f} USD")
    r = train_remote.with_options(gpu=gpu, timeout=int((max_minutes + 20) * 60)).remote(
        run_name, steps, batch_size, save_freq, max_minutes
    )
    print(r)
    print(f"download: uv run modal volume get lerobot-smolvla outputs/{run_name} outputs/")
