# SmolVLA on Hugging Face Jobs: pretrain on all pick data, fine-tune on red/blue

For a teammate with their own Hugging Face account and compute credit. It repeats Kamel's ACT setup of 2026-11-21 evening (pretrain on every pick dataset, then fine-tune on the demo task) with SmolVLA, trained on HF Jobs instead of Modal. Why this setup and what was tried before: `decisions/act-redblue-generalisation.md` (PR #26).

Every command runs from the repo root on your laptop after `uv sync`, in bash (Git Bash on Windows). `lerobot-train --job.target=...` only submits the run; training happens on Hugging Face. Nothing here moves the robot.

## 1. Log in (once)

```bash
uv run hf auth login          # paste your token (write access)
uv run hf auth whoami         # prints your user name
export HF_USER=<your user name>
```

LeRobot reads this login, passes the token to the job, and bills the job to your account. Prices per hour are from `hf jobs hardware` (2026-11-21).

## 2. The data

`brouthen/so101_pick_striped_all` (public, 5.1 GB): all four pick datasets merged with corrected task sentences. SmolVLA reads the sentence, so it matters here (ACT ignores it).

| episodes | task sentence |
|---|---|
| 0-48 | Pick up the aluminium ball and put it in the box |
| 49-148 | Pick up the striped green and yellow ball from the bowl and return to the start position |
| 149-193 | Find the buried striped green and yellow ball in the bowl, uncover it, pick it up and return to the start position |
| 194-313 | Pick up the striped green and yellow ball from the bowl of red and blue balls, uncovering it first if it is buried, and return to the start position |

`--dataset.eval_split=0.1` holds out the last 10% of each task (32 episodes). The fine-tune uses only episodes 194-313 with the same split, so its 12 held-out episodes (302-313) are also unseen in pretraining.

## 3. Stage 1: pretrain from `lerobot/smolvla_base` on everything

```bash
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=brouthen/so101_pick_striped_all \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 \
  --policy.scheduler_decay_steps=20000 \
  --accelerator.mixed_precision=bf16 --num_workers=16 \
  --dataset.eval_split=0.1 --seed=1000 --eval_steps=2000 \
  --save_freq=2000 --save_checkpoint_to_hub=true \
  --policy.repo_id=${HF_USER}/smolvla_pick_striped_all_pretrain \
  --job_name=smolvla_pick_striped_all_pretrain \
  --job.target=h200 --job.timeout=3h --job.detach=true
```

- `--rename_map` is required: `smolvla_base` expects cameras `camera1..3`; we have `top` and `wrist`.
- `--policy.scheduler_decay_steps` must equal `--steps`: the default cosine decays over 30,000 steps, which a shorter run never reaches.
- `--save_checkpoint_to_hub=true` uploads every checkpoint to `checkpoints/<step>/` in the model repo, so a timeout or crash still leaves models. The final model is pushed to the repo root.
- 20,000 steps at batch 64 is about 8 passes over the 282 training episodes. Kamel's SmolVLA from the base model on 145 blue-ball episodes stopped improving around 17,000 steps.
- On `a100-large` use `--num_workers=10` and `--job.timeout=5h`.

## 4. Watch the job

```bash
uv run hf jobs ls                 # job id and status
uv run hf jobs logs <job_id>      # step lines and "eval_loss=" lines
uv run hf jobs cancel <job_id>    # stop it (billing stops)
```

What to check in the first minutes: `Train/eval split: 282 train, 32 eval`, then step lines with `step_s` around 0.25-0.3 on `h200` (see the table below). Pick checkpoints by the `eval_loss` lines, not by the training loss.

## 5. Stage 2: fine-tune on red/blue only

Start when stage 1 has finished (its final model is at the root of `${HF_USER}/smolvla_pick_striped_all_pretrain`):

```bash
EPISODES=$(uv run python -c "print(list(range(194, 314)))" | tr -d ' ')
uv run lerobot-train \
  --policy.path=${HF_USER}/smolvla_pick_striped_all_pretrain \
  --dataset.repo_id=brouthen/so101_pick_striped_all \
  --dataset.episodes="${EPISODES}" \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=3000 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=300 --policy.scheduler_decay_steps=3000 \
  --accelerator.mixed_precision=bf16 --num_workers=16 \
  --dataset.eval_split=0.1 --seed=1000 --eval_steps=500 \
  --save_freq=500 --save_checkpoint_to_hub=true \
  --policy.repo_id=${HF_USER}/smolvla_pick_rb_from_all \
  --job_name=smolvla_pick_rb_from_all \
  --job.target=h200 --job.timeout=1h --job.detach=true
```

- Not `--resume`: starting from a model this way recomputes normalisation from this data (checked in the LeRobot 0.7.0 source).
- Short and checked often on purpose: Kamel's SmolVLA fine-tuned from a related checkpoint had its best held-out loss at 1,000-3,000 steps and got worse after (0.223 at 1k, 0.259 at 5k).
- Fine-tuning learning rate is half the default (5e-5), because the start point is already tuned to this arm and bowl.

## 6. Time and cost

Measured: SmolVLA at batch 64 took 0.255-0.28 s per step on a Modal H100 (Kamel, 2026-11-21). The H200 numbers assume the same speed (same GPU generation); the A100 numbers are an estimate between that and the LeRobot docs (about 0.7 s/step). Add about 10 minutes per job for start-up and the 5.1 GB dataset download.

| flavor | price/h | stage 1 (20k) | stage 2 (3k) | total |
|---|---|---|---|---|
| `h200` | $5.00 | ~1 h 45 min, ~$9 | ~25 min, ~$2 | **~2 h 10 min, ~$11** |
| `a100-large` | $2.50 | ~2.5-4 h, ~$6.5-10 | ~25-40 min, ~$1-1.7 | ~3-4.5 h, ~$8-12 |

With the Sunday 11:00 deadline, `h200` is the better choice; `a100-large` costs about the same per step but takes longer.

## 7. Use a checkpoint

Models created by `--job.target` runs are **private** (LeRobot creates the repo that way). To let the robot laptop download without your token, make the repo public in its Settings page on huggingface.co.

```bash
uv run hf download ${HF_USER}/smolvla_pick_rb_from_all --include "checkpoints/001500/pretrained_model/*" --local-dir smolvla_rb
# then, on the robot laptop:
#   --policy.path=smolvla_rb/checkpoints/001500/pretrained_model
#   --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
#   task: "Pick up the striped green and yellow ball from the bowl of red and blue balls, uncovering it first if it is buried, and return to the start position"
```

The task sentence must match the dataset exactly. The first load fetches `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` from the Hub, so the robot laptop needs internet once. Inference took about 1 GB of GPU memory and 0.43 s per 50-action chunk on Kamel's RTX 4070 laptop.

## 8. Report

Post on the board (issue #1): job ids, which checkpoint you picked and why, total cost, and robot results next to the ACT fine-tune (`brouthen/act_so101_pick_rb_mixed_from_all_30k`).
