"""Fine-tune one ACT per clearing colour on Modal: cut rb_mixed into segments, check them, train.

    py -3.12 -m modal run scripts/multistep/modal_clear.py::check
    py -3.12 -m modal run --detach scripts/multistep/modal_clear.py::main --colours red,blue --steps 8000 --max-minutes 35
    py -3.12 -m modal volume get lerobot-multistep outputs/act_clear_red_8k/checkpoints/008000/pretrained_model act_clear_red
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SOURCE_REPO = "molnrb/so101_pick_rb_mixed"
# The revision segments.csv and the joint check were verified against (201 episodes).
SOURCE_REVISION = "8d36c36c357e7ae6a78b3a844ef3caa1093d7c59"
START_POLICY = "Space0code/act_redblue_from_foil"
TASKS = {"red": "Put one red ball into the red hole", "blue": "Put one blue ball into the blue hole"}
KINDS = {"red": "clear_r", "blue": "clear_b"}

# Same augmentation as act_redblue_from_foil, the start checkpoint. Hue and saturation stay off: colour is
# exactly what tells a red ball from a blue one here.
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

CPU = 12.0
MEMORY = 32 * 1024
# Modal prices read from https://modal.com/pricing on 2026-11-21 (USD per hour).
GPU_USD_H = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95}
CPU_USD_H = CPU * 0.0472 + (MEMORY / 1024) * 0.0080
STARTUP_MINUTES = 8

VOL = "/vol"
WORK = "/root/work"
volume = modal.Volume.from_name("lerobot-multistep", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .uv_pip_install("lerobot[training]==0.7.0")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "build_segment_dataset.py", "/root/multistep/build_segment_dataset.py")
    .add_local_file(HERE / "check_segment_dataset.py", "/root/multistep/check_segment_dataset.py")
    .add_local_file(HERE.parents[1] / "research" / "label-patterns" / "segments.csv", "/root/multistep/segments.csv")
)
app = modal.App("lerobot-multistep", image=image)


def usd(gpu: str, minutes: float) -> float:
    return (GPU_USD_H[gpu] + CPU_USD_H) * minutes / 60


def run(command: list) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_and_check(colour: str) -> str:
    from huggingface_hub import snapshot_download

    source = snapshot_download(SOURCE_REPO, repo_type="dataset", revision=SOURCE_REVISION, local_dir=f"{WORK}/source")
    out = f"{WORK}/clear_{colour}"
    scripts = "/root/multistep"
    # Built on the container's own disk: video files are hard-linked from the source instead of copied.
    run([sys.executable, f"{scripts}/build_segment_dataset.py", "--kind", KINDS[colour],
         "--source", f"{SOURCE_REPO}={source}", "--segments", f"{scripts}/segments.csv",
         "--out", out, "--repo-id", f"local/clear_{colour}", "--task", TASKS[colour], "--link"])
    run([sys.executable, f"{scripts}/check_segment_dataset.py", "--built", out,
         "--source", f"{SOURCE_REPO}={source}", "--hole", colour, "--pixel-episodes", "3"])
    return out


@app.function(cpu=CPU, memory=MEMORY, gpu="H100", volumes={VOL: volume}, timeout=60 * 60)
def train_colour(colour: str, steps: int, max_minutes: float, run_name: str) -> dict:
    start = time.time()
    dataset_root = build_and_check(colour)
    command = [
        "lerobot-train",
        f"--dataset.repo_id=local/clear_{colour}",
        f"--dataset.root={dataset_root}",
        f"--policy.path={START_POLICY}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        # 2e-5 at batch 32, as in act_redblue_from_foil.
        "--policy.optimizer_lr=2e-5",
        "--policy.optimizer_lr_backbone=2e-5",
        "--batch_size=32",
        "--dataset.image_transforms.enable=true",
        f"--dataset.image_transforms.tfs={AUGMENTATION}",
        # The builder puts a hold-out spread over the recording blocks last; 0.1 selects exactly it.
        "--dataset.eval_split=0.1",
        "--seed=1000",
        "--eval_steps=1000",
        f"--steps={steps}",
        "--save_freq=2000",
        "--log_freq=100",
        f"--num_workers={int(CPU) - 2}",
        "--accelerator.mixed_precision=bf16",
        f"--job_name={run_name}",
        f"--output_dir={VOL}/outputs/{run_name}",
    ]
    print("+", " ".join(command), flush=True)
    proc = subprocess.Popen(command)
    budget_s = max_minutes * 60 - (time.time() - start)
    try:
        code = proc.wait(timeout=max(budget_s, 60))
    except subprocess.TimeoutExpired:
        print(f"[budget] {max_minutes} min reached, stopping training", flush=True)
        proc.terminate()
        code = proc.wait(60)
    volume.commit()
    checkpoints = Path(f"{VOL}/outputs/{run_name}/checkpoints")
    return {
        "colour": colour,
        "code": code,
        "minutes": round((time.time() - start) / 60, 1),
        "checkpoints": sorted(p.name for p in checkpoints.iterdir()) if checkpoints.is_dir() else [],
    }


@app.function(cpu=8.0, memory=16 * 1024, volumes={VOL: volume}, timeout=30 * 60)
def check_colour(colour: str) -> str:
    build_and_check(colour)
    return f"{colour}: built and checked"


@app.local_entrypoint()
def check(colours: str = "red,blue") -> None:
    print("CPU only, about 0.15 USD: downloads rb_mixed, builds and checks each colour's dataset, no training")
    for result in check_colour.map(colours.split(",")):
        print(result)


@app.local_entrypoint()
def main(colours: str = "red,blue", steps: int = 8000, max_minutes: float = 35) -> None:
    wanted = colours.split(",")
    worst = len(wanted) * usd("H100", max_minutes + STARTUP_MINUTES)
    print(f"{len(wanted)} run(s) on H100, cap {max_minutes} min each, worst case ~{worst:.2f} USD in total")
    calls = [
        train_colour.with_options(timeout=int((max_minutes + 15) * 60)).spawn(c, steps, max_minutes, f"act_clear_{c}_{steps // 1000}k")
        for c in wanted
    ]
    for call in calls:
        print(call.get())
