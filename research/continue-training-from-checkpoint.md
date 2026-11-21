# Continuing ACT training from a checkpoint after recording more episodes

Researched 2026-11-21 by an Opus subagent of the agent of @Space0code, read from the LeRobot v0.7.0 source (commit 410c095). Nothing was run. Paths: `lt` = `scripts/lerobot_train.py`, `tc` = `configs/train.py`, `tu` = `common/train_utils.py`. Why this exists: the demo must handle a deeply buried target; the first run trains on the 80 red/blue episodes, more deep-buried episodes follow, and training continues from a checkpoint (Tomi, board 2026-11-21).

## What to do

Use **fine-tune** (`--policy.path`), not resume, whenever the dataset changes.

| | Fine-tune (`--policy.path=<checkpoint>`) | Resume (`--resume=true --config_path=...`) |
| --- | --- | --- |
| Restored | weights only (`policies/factory.py:395-406`) | step, RNG, optimizer, scheduler (`tu:432-457, 537-545`) |
| Reset | optimizer, step counter starts at 0 | nothing |
| Normaliser stats | replaced by the new dataset's stats (`lt:666-668, 690-692`) | kept from the old run on purpose, so stale if the data changed (`lt:686-692`) |
| `--steps` means | additional steps | total steps, must exceed the old total |

ACT uses a constant learning rate (AdamW 1e-5, no scheduler, no warmup; `configuration_act.py:126-128, 154-162`), so a fresh optimizer is not a problem.

## Steps

1. Record the new episodes as their own dataset with the same cameras and fps as the old one, `--dataset.no_stamp=true`, and a **distinct task sentence** (see pitfalls).
2. Merge: `lerobot-edit-dataset --new_repo_id <merged> --operation.type merge --operation.repo_ids "['<old>', '<new>']" --push_to_hub true`. The datasets must have identical fps, robot_type and features including camera names (`datasets/aggregate.py:140-156`); stats are aggregated from both.
3. Fine-tune from a checkpoint (the person at the table or Kamel runs it; suggested 30,000 steps, untested):

```bash
uv run lerobot-train --policy.path=<old run>/checkpoints/<step>/pretrained_model --dataset.repo_id=<merged> --dataset.image_transforms.enable=true --dataset.image_transforms.tfs='<full JSON from board 7061985752, hue and saturation weight 0>' --dataset.eval_split=0.02 --batch_size=8 --steps=30000 --save_freq=5000 --output_dir=outputs/train/<NEW dir> --job_name=<name> --policy.repo_id=<NEW repo id>
```

**Correction to the subagent's command:** it used `--dataset.image_transforms.tfs.hue.weight=0`. Kamel's agent showed that single-field form fails with "unrecognized arguments" (board 7062043167, 7062218338); the full `tfs` JSON form is the one that works. Check the first lines of the log show hue and saturation with weight 0.0.

## Pitfalls

- **Smoke test first:** run the same command with `--steps=2` before the long run (board 7063234241). A list of repo ids passed directly to `--dataset.repo_id` raises "LeRobotMultiDataset is not currently implemented" (`configs/train.py:344-345`), so merge first.
- **Eval split:** held-out episodes are the last ones (`datasets/factory.py:231,251-253`), so with a merged dataset the new episodes dominate the split. Use `--dataset.eval_split=0.02` (or 0) instead of 5%.
- **New repo id and output dir.** `--policy.path` loads the old `config.json`, including `repo_id` and `push_to_hub=true`, so without a new `--policy.repo_id` the final model is pushed as a new commit to the old repo (`configs/policies.py:73-74`). An existing `output_dir` raises FileExistsError (`tc:333-337`).
- **eval_split** holds out the last 5% of episodes per task sentence (`datasets/factory.py:229-253`). New episodes under the same sentence would make up most of the eval set; ACT never reads the task text, so a distinct sentence is free.
- **Cameras** must keep the same names (checkpoint `input_features`, `policies/factory.py:340-341`), or pass `--rename_map`.
- **Re-pass** image_transforms, eval_split and batch_size: fine-tuning reads the dataset config from the CLI.
- **Normalisation:** ACT normalises state and action with mean/std; images use ImageNet stats. Fine-tuning on the merged data uses merged stats. Fine-tuning on the **new data only** would put the stats of the new distribution only, which may cost some old behaviour. UNVERIFIED in practice; prefer merged old plus new.
