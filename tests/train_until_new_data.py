# scripts/train_until_new_data.py on the CPU with a tiny ACT and a synthetic dataset: when the watched episode count
# grows, the run saves a checkpoint at the current step (long before --save_freq) and exits 75; a fine-tune from that
# checkpoint then trains on the grown dataset. Needs LeRobot 0.7.0 with its training extras.
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# tests/cuda.py would shadow the cuda package torch imports.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
RUNNER = ROOT / "scripts" / "train_until_new_data.py"
SAVE_FREQ = 100_000
failures = []

# Windows without developer mode cannot create the `last` symlink lerobot-train writes; a junction stands in for it.
JUNCTION = """
import os, pathlib, runpy, sys
if os.name == "nt":
    import _winapi
    def symlink_to(self, target, target_is_directory=False):
        _winapi.CreateJunction(str((self.parent / target).resolve()), str(self))
    pathlib.Path.symlink_to = symlink_to
"""
RUN_RUNNER = JUNCTION + f"sys.argv[0] = {str(RUNNER)!r}\nrunpy.run_path(sys.argv[0], run_name='__main__')\n"
RUN_LEROBOT_TRAIN = JUNCTION + "from lerobot.scripts.lerobot_train import main\nmain()\n"


def check(name: str, ok: bool, detail="") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def make_dataset(root: Path, repo: str, episodes: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    joints = [f"j{i}" for i in range(6)]
    features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": joints},
                "action": {"dtype": "float32", "shape": (6,), "names": joints},
                "observation.images.top": {"dtype": "image", "shape": (32, 32, 3),
                                           "names": ["height", "width", "channels"]}}
    dataset = LeRobotDataset.create(repo, fps=30, features=features, root=root, use_videos=False)
    rng = np.random.default_rng(episodes)
    for _ in range(episodes):
        for _ in range(30):
            dataset.add_frame({"observation.state": rng.normal(size=6).astype(np.float32),
                               "action": rng.normal(size=6).astype(np.float32),
                               "observation.images.top": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
                               "task": "pick"})
        dataset.save_episode()
    dataset.finalize()


def train_args(repo: str, root: Path, out: Path, steps: int, save_freq: int, policy: list[str]) -> list[str]:
    return [f"--dataset.repo_id={repo}", f"--dataset.root={root}", "--dataset.eval_split=0", *policy,
            "--policy.device=cpu", "--policy.push_to_hub=false", f"--batch_size=2", f"--steps={steps}",
            f"--save_freq={save_freq}", "--log_freq=20", "--num_workers=0", f"--output_dir={out}", "--seed=1"]


TINY_ACT = ["--policy.type=act", "--policy.dim_model=32", "--policy.n_heads=2", "--policy.dim_feedforward=64",
            "--policy.n_encoder_layers=1", "--policy.n_decoder_layers=1", "--policy.chunk_size=5",
            "--policy.n_action_steps=5", "--policy.use_vae=false", "--policy.pretrained_backbone_weights=null"]


def run(code: str, args: list[str], log: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code, *args], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            cwd=ROOT)


def wait_for_line(log: Path, text: str, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if log.exists() and text in log.read_text(errors="replace"):
            return True
        time.sleep(0.5)
    return False


def chaining() -> None:
    sys.path.insert(0, str(RUNNER.parent))
    import train_until_new_data as follow

    clock = {"t": 0.0}
    hub = {"episodes": 120}
    segments = []

    def run_segment(segment, checkpoint, minutes):
        segments.append((segment, checkpoint, round(minutes)))
        clock["t"] += 600
        if segment == 1:
            hub["episodes"] = 140
            return {"reason": "new_data", "minutes": 10, "checkpoint": "s01/000700", "episodes_trained": 120}
        if segment == 2:
            return {"reason": "done", "minutes": 10, "checkpoint": "s02/005000", "episodes_trained": 140}
        if segment == 3:
            return {"reason": "failed", "minutes": 1, "checkpoint": None, "episodes_trained": None}

    def wait(seconds):
        clock["t"] += seconds
        if clock["t"] > 3000:
            hub["episodes"] = 160

    log = []
    history = follow.follow_segments(
        run_segment, lambda: hub["episodes"], wait, lambda: clock["t"], start_from="v2/018000", budget_usd=10,
        usd_per_minute=0.08, overhead_minutes=8, deadline=4 * 3600, cap_minutes=60, min_new=20, poll_s=60,
        log=log.append)
    check("follow: each continuation starts from the previous one's newest checkpoint; one that ran out of steps "
          "waits for 20 new episodes; a failure ends the chain",
          [s[:2] for s in segments] == [(1, "v2/018000"), (2, "s01/000700"), (3, "s02/005000")]
          and any(e.get("event") == "waiting" and e["for_episodes"] == 160 for e in log)
          and log[-1] == {"event": "follow_end", "reason": "segment failed", "spent_usd": 3.6}, str(log))

    calls = []
    follow.follow_segments(
        lambda s, c, m: calls.append(round(m)) or {"reason": "new_data", "minutes": m, "checkpoint": c},
        lambda: 0, wait, lambda: 0.0, start_from=None, budget_usd=12, usd_per_minute=0.08, overhead_minutes=8,
        deadline=10 * 3600, cap_minutes=60, min_new=20, poll_s=60, log=log.append)
    check("follow: the USD cap bounds the chain (12 USD at 0.08/min with 8 min overhead each)",
          calls == [60, 60, 6] and log[-1]["reason"] == "budget", f"{calls} {log[-1]}")


def main() -> None:
    chaining()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        work = Path(work)
        make_dataset(work / "a", "local/tiny_a", 3)
        make_dataset(work / "b", "local/tiny_b", 4)
        hub = work / "hub"
        (hub / "meta").mkdir(parents=True)
        (hub / "meta" / "info.json").write_text(json.dumps({"total_episodes": 3}))

        log1 = work / "first.log"
        first = run(RUN_RUNNER, ["--watch-root", str(hub), "--min-new", "1", "--min-steps", "30", "--poll-s", "0.5",
                                 "--", *train_args("local/tiny_a", work / "a", work / "out1", 100_000, SAVE_FREQ,
                                                   TINY_ACT)], log1)
        started = wait_for_line(log1, "[new data] training on 3 episodes", 300)
        wait_for_line(log1, "step:", 120)
        (hub / "meta" / "info.json").write_text(json.dumps({"total_episodes": 4}))
        try:
            code = first.wait(timeout=300)
        except subprocess.TimeoutExpired:
            first.kill()
            code = None
        text = log1.read_text(errors="replace")
        result_file = work / "out1" / "stopped_for_new_data.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        checkpoint = Path(result.get("checkpoint", work / "missing"))
        check("the watcher starts with the episode count the run loaded", started, text[-600:] if not started else "")
        check("new episodes: checkpoint at the current step, long before --save_freq, then exit 75",
              code == 75 and 30 <= (result.get("step") or 0) < SAVE_FREQ and result.get("episodes_trained") == 3
              and result.get("episodes_seen") == 4, f"exit {code}, {result}" + ("" if code == 75 else text[-1500:]))
        check("the checkpoint is complete: weights, processors, optimizer state, and marked last",
              (checkpoint / "pretrained_model" / "config.json").exists()
              and (checkpoint / "pretrained_model" / "model.safetensors").exists()
              and (checkpoint / "training_state").is_dir() and (checkpoint.parent / "last").exists(),
              str(sorted(p.name for p in checkpoint.rglob("*"))[:12]))

        log2 = work / "second.log"
        second = run(RUN_LEROBOT_TRAIN, train_args("local/tiny_b", work / "b", work / "out2", 6, 6,
                                                   [f"--policy.path={checkpoint / 'pretrained_model'}"]), log2)
        code2 = second.wait(timeout=600)
        text2 = log2.read_text(errors="replace")
        check("a fine-tune from that checkpoint trains on the grown dataset (4 episodes)",
              code2 == 0 and "dataset.num_episodes=4" in text2 and (work / "out2" / "checkpoints" / "000006").is_dir(),
              f"exit {code2}" + ("" if code2 == 0 else text2[-1500:]))
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
