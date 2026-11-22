# Training that follows the recording

New episodes are recorded in batches of 20 and uploaded at the end of each session. A chain of Modal continuations keeps training on the growing dataset. Each continuation checkpoints and makes way as soon as 20 new episodes are on the Hub. Why it works this way: `decisions/continual-training.md`.

## At the table (whoever records)

1. **Once: make the growing copy** on the robot laptop, so the batches go into a dataset that already holds the old episodes:
   ```
   cp -r ~/.cache/huggingface/lerobot/molnrb/so101_pick_rb_mixed ~/.cache/huggingface/lerobot/molnrb/so101_pick_rb_more
   ```
   The first upload sends the whole copy (about 2 GB); later uploads send only new and changed files.
2. **Dashboard, Collect data:** pick `so101_pick_rb_more`, 20 episodes, and keep **Upload to the Hugging Face Hub when the session ends** ticked (`docs/collect-data.md`).
3. When the banner shows **UPLOADED**, the next continuation can see the batch. If it shows **UPLOAD FAILED**, press **Retry upload**.

## On Modal (the account owner, never an agent)

**Smoke test first.** It costs well under 1 USD: one 20-step continuation, then it waits on a CPU for 30 min.
```
uv run modal run --detach scripts/modal_train.py::follow --recipe act_redblue_v2 --dataset molnrb/so101_pick_rb_more --segment-steps 20 --save-freq 10 --budget-usd 2 --hours 0.5
```

**ACT v2**, continuing from its 18K checkpoint, 5K steps per continuation:
```
uv run modal run --detach scripts/modal_train.py::follow --recipe act_redblue_v2 \
  --start-from /vol/outputs/<v2 run>/checkpoints/018000/pretrained_model \
  --dataset molnrb/so101_pick_rb_more --segment-steps 5000 --budget-usd 15 --hours 12
```

**SmolVLA**, from `002000`, with short continuations because it overfits fast:
```
uv run modal run --detach scripts/modal_train.py::follow --recipe smolvla_redblue_from_combined \
  --start-from /vol/outputs/<smolvla run>/checkpoints/002000/pretrained_model \
  --dataset molnrb/so101_pick_rb_more --segment-steps 1500 --save-freq 500 --budget-usd 15 --hours 12
```

**Run directories:** `uv run modal volume ls lerobot-smolvla outputs`.

**Options:**

| Option | Default | What it does |
| --- | --- | --- |
| `--min-new` | 20 | New episodes that stop a continuation. |
| `--min-steps` | 500 | Steps a continuation trains before it may stop. |
| `--cap-minutes` | 60 | Longest continuation. |
| `--eval-split` | 0 | Held-out share. LeRobot holds out the newest episodes, so it is off by default. |
| `--gpu` | H100 | GPU type. |

## Watching it and using a checkpoint

- **Live log:** `uv run modal app logs <app id>`. Find the app id with `uv run modal app list`. Lines starting `[follow]` and `[new data]` show the decisions.
- **History:**
  - get it with `uv run modal volume get lerobot-smolvla outputs/<run>_follow.jsonl .`;
  - one line per continuation: why it ended (`new_data`, `done`, `capped`, `failed`), its checkpoint, the episodes it trained on, and the USD spent so far.
- **Newest checkpoint for the arm:** `uv run modal volume get lerobot-smolvla outputs/<run>_sNN/checkpoints/last/pretrained_model <local dir>`.
- **Stop:** `uv run modal app stop <app id>`.
  - Use this run's id, not the app name: other runs on the same account share the name `lerobot-train`.
  - The last finished checkpoint stays on the volume.
