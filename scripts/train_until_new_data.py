# lerobot-train that makes way for new episodes: once the dataset it trains on has grown by --min-new episodes on the
# Hub, it saves a checkpoint at the current step (not at the next --save_freq) and exits with NEW_DATA_EXIT, so the
# next continuation can start from that checkpoint on the grown dataset. A running lerobot-train never reloads its
# dataset, so stopping is the only way in. See decisions/continual-training.md.
#
#   python scripts/train_until_new_data.py [--min-new 20] [--min-steps 500] -- <lerobot-train arguments>
import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

NEW_DATA_EXIT = 75
RESULT_FILE = "stopped_for_new_data.json"


class NewData(Exception):
    pass


def hub_episodes(repo: str) -> int:
    from huggingface_hub import hf_hub_download

    # The v3.0 tag is what every lerobot-train start resolves, and push_to_hub moves it after the upload commit.
    path = hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision="v3.0", force_download=True,
                           cache_dir=Path(tempfile.gettempdir()) / "train_until_new_data")
    return int(json.loads(Path(path).read_text())["total_episodes"])


def local_episodes(root: Path) -> int:
    return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])


class Watch:
    def __init__(self, count, min_new: int, min_steps: int, poll_s: float):
        self.count, self.min_new, self.min_steps, self.poll_s = count, min_new, min_steps, poll_s
        self.trained_on: int | None = None
        self.seen: int | None = None
        self.forced_step: int | None = None
        self.requested = threading.Event()

    def start(self, trained_on: int) -> None:
        self.trained_on = trained_on
        print(f"[new data] training on {trained_on} episodes; stops for {trained_on + self.min_new}", flush=True)
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        while not self.requested.is_set():
            time.sleep(self.poll_s)
            try:
                self.seen = self.count()
            except Exception as error:  # noqa: BLE001 - a Hub hiccup must not stop the training
                print(f"[new data] check failed: {error}", flush=True)
                continue
            if self.seen >= self.trained_on + self.min_new:
                print(f"[new data] {self.seen} episodes now: checkpoint at the next step, then stop", flush=True)
                self.requested.set()

    def stop_now(self, step: int) -> bool:
        return self.requested.is_set() and step >= self.min_steps


def install_hooks(train_module, watch: Watch) -> None:
    make_datasets, should_save, mark_last = (train_module.make_train_eval_datasets,
                                             train_module.should_save_checkpoint,
                                             train_module.update_last_checkpoint)

    def make_train_eval_datasets(cfg):
        train, evaluation = make_datasets(cfg)
        watch.start(train.meta.total_episodes)
        return train, evaluation

    def should_save_checkpoint(step, save_freq, total_steps):
        if watch.stop_now(step):
            watch.forced_step = step
            return True
        return should_save(step, save_freq, total_steps)

    # Raised after the checkpoint is complete and marked last, so the next continuation can start from it.
    def update_last_checkpoint(checkpoint_dir):
        result = mark_last(checkpoint_dir)
        if watch.forced_step is not None:
            raise NewData(str(checkpoint_dir))
        return result

    train_module.make_train_eval_datasets = make_train_eval_datasets
    train_module.should_save_checkpoint = should_save_checkpoint
    train_module.update_last_checkpoint = update_last_checkpoint


def segment_reason(exit_code: int) -> str:
    # -15: modal_train.run_training ended it at its time cap; it continues from the newest periodic checkpoint.
    return {NEW_DATA_EXIT: "new_data", 0: "done", -15: "capped"}.get(exit_code, "failed")


# Chains continuations, each a fine-tune from the previous one's newest checkpoint. A continuation that ran out of
# steps waits (on a CPU) for the next batch of episodes instead of training on the same data again.
def follow_segments(run_segment, episodes_now, wait, clock, *, start_from: str | None, budget_usd: float,
                    usd_per_minute: float, overhead_minutes: float, deadline: float, cap_minutes: float, min_new: int,
                    poll_s: float, log) -> list[dict]:
    spent, segment, checkpoint, history = 0.0, 1, start_from, []
    while True:
        budget_left = (budget_usd - spent) / usd_per_minute - overhead_minutes
        time_left = (deadline - clock()) / 60
        minutes = min(cap_minutes, budget_left, time_left)
        if minutes < 5:
            log({"event": "follow_end", "reason": "budget" if budget_left <= time_left else "deadline",
                 "spent_usd": round(spent, 2)})
            return history
        result = run_segment(segment, checkpoint, minutes)
        spent += usd_per_minute * (result["minutes"] + overhead_minutes)
        result = {**result, "segment": segment, "started_from": checkpoint, "spent_usd": round(spent, 2)}
        history.append(result)
        log(result)
        if result["reason"] == "failed":
            log({"event": "follow_end", "reason": "segment failed", "spent_usd": round(spent, 2)})
            return history
        checkpoint = result.get("checkpoint") or checkpoint
        segment += 1
        if result["reason"] == "new_data":
            continue
        trained_on = result.get("episodes_trained") or episodes_now()
        log({"event": "waiting", "for_episodes": trained_on + min_new})
        while episodes_now() < trained_on + min_new:
            if clock() >= deadline:
                log({"event": "follow_end", "reason": "deadline", "spent_usd": round(spent, 2)})
                return history
            wait(poll_s)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="lerobot-train that checkpoints and exits 75 when its dataset grows.")
    p.add_argument("--min-new", type=int, default=20, help="new episodes that stop the training")
    p.add_argument("--min-steps", type=int, default=500, help="train at least this many steps before stopping")
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--watch-root", type=Path, help="watch a local dataset folder instead of the Hub (tests)")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    return p.parse_args(argv[:split]), argv[split + 1:]


def dataset_repo(train_args: list[str]) -> str:
    return next(a.split("=", 1)[1] for a in train_args if a.startswith("--dataset.repo_id="))


def main() -> None:
    args, train_args = parse_args()
    repo = dataset_repo(train_args)
    count = (lambda: local_episodes(args.watch_root)) if args.watch_root else (lambda: hub_episodes(repo))
    watch = Watch(count, args.min_new, args.min_steps, args.poll_s)

    import lerobot.scripts.lerobot_train as train_module

    install_hooks(train_module, watch)
    sys.argv = ["lerobot-train", *train_args]
    try:
        train_module.main()
    except NewData as stopped:
        checkpoint = Path(str(stopped))
        result = {"reason": "new_data", "step": watch.forced_step, "checkpoint": str(checkpoint),
                  "episodes_trained": watch.trained_on, "episodes_seen": watch.seen}
        (checkpoint.parents[1] / RESULT_FILE).write_text(json.dumps(result, indent=1))
        print(f"[new data] stopped: {json.dumps(result)}", flush=True)
        sys.exit(NEW_DATA_EXIT)


if __name__ == "__main__":
    main()
