# ACT on red/blue only: the recipe

How to train the PICK policy on `molnrb/so101_pick_rb_mixed` alone (120 episodes: the striped green and yellow ball among red and blue balls, sometimes buried), using what the team's runs of 2026-11-21 showed. Run it from the repo root after `uv sync`. Training does not move the robot.

## What the earlier runs showed

| run | start from | setup | held-out loss | on the arm |
|---|---|---|---|---|
| `brouthen/act_so101_pick_rb_mixed_b32_38k` | scratch | batch 32, lr 2e-5, colour-safe augmentation, chunk 100 | best 0.329 at 10k, flat after (4 held-out episodes) | "did not perform well" |
| `Space0code/act_redblue_from_foil` | `molnrb/act_pick_ball_combined` | same as above | best 0.331 at 10k, flat after (same 4 episodes) | not reported |
| `brouthen/act_so101_pick_rb_mixed_v2_chunk50_20k` | scratch | chunk 50, `n_action_steps` 25, heavy augmentation, EMA, dropout 0.2 | kept improving to 18k (8 held-out episodes) | "a bit unstable" |
| `brouthen/act_so101_pick_rb_mixed_from_all_30k` | `brouthen/act_so101_pick_striped_all_pretrain_50k` | LeRobot defaults (batch 8, lr 1e-5) | flat from 2k (12 held-out episodes) | not tried yet |

What follows from it (held-out losses only compare within one split):

- **Keep chunk 100.** Chunk 50 was the only change tried on the arm that made it worse.
- **About 10k steps at batch 32 is enough** from scratch; later steps only fit the training episodes more tightly.
- **Pretrained weights do not lower the held-out loss** (scratch 0.329 vs from-foil 0.331 on the same episodes), **but they reach it much faster**: the all-pick fine-tune started at loss 0.19 instead of about 10. With the deadline that speed is the reason to use them.
- **EMA** was part of the only run whose held-out loss kept improving late. It is cheap and cannot make the motion jumpier, so it is in.
- **Colour augmentation stays off** (hue, saturation): colour is how the target is told apart.

## The recipe

Start from the all-pick pretrained ACT, default architecture (chunk 100), colour-safe augmentation, EMA, 10% held out, a checkpoint and held-out check often, then pick the checkpoint on the arm.

### 1. Get the starting weights (once, ~200 MB)

```bash
uv run hf download brouthen/act_so101_pick_striped_all_pretrain_50k --include "050000/*" --local-dir pretrain
```

### 2a. On a laptop GPU with 8 GB (batch 8)

```bash
uv run lerobot-train \
  --policy.path=pretrain/050000 \
  --dataset.repo_id=molnrb/so101_pick_rb_mixed \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness": {"type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"weight": 0.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, "hue": {"weight": 0.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}, "sharpness": {"type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}}, "affine": {"type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}}}' \
  --ema.type=constant --ema.decay=0.999 \
  --batch_size=8 --steps=24000 --save_freq=2000 \
  --dataset.eval_split=0.1 --seed=1000 --eval_steps=2000 \
  --num_workers=4 \
  --policy.device=cuda --policy.push_to_hub=false \
  --job_name=act_redblue_recipe --output_dir=outputs/act_redblue_recipe
```

About 2 hours on an RTX 4070 laptop (0.26-0.28 s/step at batch 8, measured on Kamel's; about 3.7 GB of GPU memory).

### 2b. On a GPU with 16 GB or more (batch 32, faster)

Same command with these lines instead of the batch, steps and worker lines:

```bash
  --batch_size=32 --policy.optimizer_lr=2e-5 --policy.optimizer_lr_backbone=2e-5 \
  --accelerator.mixed_precision=bf16 \
  --steps=10000 --save_freq=1000 --eval_steps=1000 --num_workers=10 \
```

Batch 32 peaked at about 14 GB on an L40S. About 45 min on an L40S (0.256 s/step measured), roughly 20-30 min on an H100. On Hugging Face Jobs the starting model must sit at the root of a Hub repo (the job cannot see your `pretrain/` folder), so first copy it into one of your own:

```bash
uv run hf upload <your user>/act_pick_all_pretrain_50k pretrain/050000 .
```

then use `--policy.path=<your user>/act_pick_all_pretrain_50k` and add `--job.target=l40sx1 --job.timeout=2h --policy.repo_id=<your user>/act_redblue_recipe --save_checkpoint_to_hub=true`. Checkpoints then appear under `checkpoints/<step>/` in that repo (private by default).

Both commands were checked with LeRobot 0.7.0's own config parser; neither has been run yet.

## 3. Pick the checkpoint

- The log prints `step <n>: eval_loss=<x>` at every check. Note the 2-3 lowest; differences of 0.01 are noise.
- With EMA, every checkpoint has two models. **Use `checkpoints/<step>/pretrained_model_ema/`**: the held-out loss is measured on those weights.
- The 12 held-out episodes are the newest ones, probably the deep-buried cases, so the held-out loss is about the hardest situation, not the average one.

## 4. On the arm

```bash
--policy.path=outputs/act_redblue_recipe/checkpoints/<step>/pretrained_model_ema
```

- Default `n_action_steps` (100). Do not shorten it: chunk 50 / 25 was unstable.
- If the motion is jerky, one try with temporal ensembling: `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1`.
- Same `--robot.id`, camera names (`top`, `wrist`), resolution (640x480) and fps (30) as the recordings.
- Count successes out of 10 (`docs/eval-counting-sheet.md`) for the 2-3 picked checkpoints and post them on the board (#1).

## If time is short

Skip training: the all-pick fine-tune `brouthen/act_so101_pick_rb_mixed_from_all_30k` (`002000`, `010000`) is the same idea already trained (without augmentation and EMA).
