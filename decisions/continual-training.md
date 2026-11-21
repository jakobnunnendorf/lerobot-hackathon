# How do new episodes reach training while recording goes on?

- Date: 2026-11-21
- Status: decided by Tian: "stop the training when you get new data, finish the work that's currently training as new checkpoint (don't wait for 5k steps) and then add data". Jakob agreed on board 7067885958.
  - Built on `TianK003/train-follow`, made from Kamel's #26.
  - `scripts/train_until_new_data.py` was tested on a CPU with a tiny ACT (`tests/train_until_new_data.py`).
  - The Modal side (`modal_train.py::follow`) has **not run yet**, because that needs a paid run.

## Decision

- **Recording:** sessions of 20 episodes into one growing dataset. Each session uploads at its end: the dashboard's collect mode runs `record_session.py --upload`, the same `push_to_hub` as PR #6's `upload_dataset.py`.
- **Training:** a chain of continuations, `modal_train.py::follow`.
  - Each continuation is `train_until_new_data.py` around `lerobot-train`.
  - When the dataset's `meta/info.json` at the `v3.0` tag on the Hub shows at least 20 more episodes than the run loaded, it saves a checkpoint at the current step and exits 75.
  - The next continuation fine-tunes from that checkpoint on the grown dataset.
  - A continuation that finishes its steps first waits on a CPU for the next batch, instead of training on the same data again.
  - The chain stops at `--budget-usd` or after `--hours`. The account owner starts it; no agent does.

## Why

- **A running `lerobot-train` never sees new episodes.** It loads the episode list, the normaliser stats and the sampler once (`make_train_eval_datasets` in `datasets/factory.py`, `lerobot_train.py`).
- **A new start does see them, with nothing stale:**
  - without `--dataset.root`, it resolves the `v3.0` tag and downloads only the missing files (`dataset_metadata.py:139-151`);
  - `push_to_hub` moves that tag after the upload commit (`lerobot_dataset.py:720-723`).
- **The checkpoint at the current step comes from three hooks on module-level names of `lerobot_train`**, so the training code is the same `lerobot-train` as every other run:
  - `make_train_eval_datasets` gives the loaded episode count;
  - `should_save_checkpoint` forces the save;
  - `update_last_checkpoint` stops the run once the checkpoint is complete and marked `last`.
- **Fine-tune (`--policy.path`), not resume.** A resume keeps the first run's normaliser stats (`lerobot_train.py:686-692`). A fine-tune takes the grown dataset's stats. For ACT, losing the optimizer state is cheap, since ACT has a constant LR. SmolVLA's schedule restarts each continuation, with warmup at 5% of its steps (`schedule_follows_steps`).
- **`eval_split` defaults to 0.** LeRobot holds out the last episodes of each task sentence, which is the newest batch (Jakob, 7067885958).
- **Cost and time:** ACT v2 runs at 0.117 s/step on an H100 at batch 32, so 5K steps take about 10 min and about 1 USD (`decisions/act-redblue-generalisation.md`). Recording 20 episodes takes about 25-40 min, so a continuation usually finishes before the next batch is uploaded. The overhead per continuation is assumed to be 8 min (`STARTUP_MINUTES`); it is not measured.

## Alternatives

- **Reload the dataset inside a running `lerobot-train`:** rejected. It would need a training loop of our own, which would drift from `lerobot-train`.
- **Resume (`--resume=true`):** rejected, because of the stale normaliser stats (above). It would also need `--steps` above the old total.
- **One dataset per batch, merged before each continuation:** rejected.
  - LeRobot 0.7.0 refuses a list of repo ids ("LeRobotMultiDataset is not currently implemented", `configs/train.py`).
  - `lerobot-edit-dataset merge` rewrites the whole dataset each time.
  - A growing copy (`cp -r` of the local dataset, then record into the copy) gets the same result.
- **Upload while the next session records:** rejected. Both read and write the same files, so the next session waits for the upload, which is about 400 MB per 20 episodes.
- **Train on the robot laptop between sessions:** rejected. It needs about 2 h per 20K steps on its RTX 4050 (`scripts/so101/train.sh`), and the training's video decoding would compete with the recording's encoder.

## Revisit if

- The measured overhead per continuation is far above 8 min. A long-lived trainer that reloads its dataset would then pay off.
- A continuation's checkpoint does worse on the arm than the one before it, for example by forgetting older scenes. Then compare it with one fresh fine-tune on all data, which is Jakob's 01:00 plan.
