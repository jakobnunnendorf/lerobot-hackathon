# How do we train the red/blue ACT pick policy so it works on the robot, not only on the recordings?

- Status: OPEN (2026-11-21)

## Decision

Being tried: run `act_redblue_v2_h100_chunk50_20k` (recipe `act_redblue_v2` in `scripts/modal_train.py`): shorter action chunks, stronger augmentation, EMA and more dropout, 20k steps. Checkpoints at https://huggingface.co/brouthen/act_so101_pick_rb_mixed_v2_chunk50_20k. Not yet tried on the arm.

## Why

- Run 1 (`act_redblue_l40s_b32_38k`, Kamel's agent, 2026-11-21, Modal L40S, batch 32, lr 2e-5, chunk 100, 5% held out): held-out loss flat at about 0.33 from 10k to 30k steps while train loss fell from 0.19 to 0.165, so later steps only widened the gap. Checkpoints: https://huggingface.co/brouthen/act_so101_pick_rb_mixed_b32_38k. Kamel reported it did not perform well on the robot (failure mode not yet written down).
- `n_action_steps=100` plays 3.3 s without looking at the cameras; a buried or moving ball needs re-planning. `chunk_size=50` with `n_action_steps=25` re-plans every 0.8 s (assumed benefit, not measured).
- The added augmentations (shadows, brightness patches, dropout holes, noise, motion blur, ±10° / 10% affine) target lighting and camera shifts between recording and the table; hue, saturation and colour temperature stay off because colour tells the target apart.
- 8 held-out episodes instead of 4 make the held-out loss less noisy. Held-out losses are not comparable across chunk sizes (a shorter chunk is easier to predict), so v2 is compared with run 1 on the robot only.
- H100 was fastest and cheapest per step for v2 (Modal benchmark 2026-11-21: H100 0.117 s/step, A100-80GB 0.186, L40S 0.284 at batch 32, augmentations on the GPU).

## Results so far (held-out loss only, nothing measured on the arm yet)

- **ACT v2** finished 20k steps in 41 min (H100, about 4.2 USD). Held-out loss (8 episodes) 0.350 at 2k, about 0.31 from 4k to 14k, then 0.295 / **0.287** / 0.295 at 16k / 18k / 20k. Unlike run 1 it kept improving late; best candidate is `018000`.
- **SmolVLA fine-tuned from the blue-ball SmolVLA** (`smolvla_redblue_from_blue_h100_10k`, recipe `smolvla_redblue_from_combined`, H100, batch 64, lr 5e-5): held-out loss 0.223 / 0.238 / 0.229 / 0.252 / 0.259 at 1k-5k, so it overfit after about 3k; stopped by hand at about 5.1k (about 2.5 USD). Checkpoints `002000` (start here) and `004000` at https://huggingface.co/brouthen/smolvla_so101_pick_rb_mixed_from_blue_10k. Starting from a related model, the transfer happened within the first 1-3k steps. Next time: save every 500-1000 steps and train about 3k.
- Held-out losses are not comparable between ACT and SmolVLA, nor between ACT chunk sizes.
- **Pretrain on all pick data, then fine-tune on red/blue** (Kamel's agent, 2026-11-21, Modal H100, LeRobot ACT defaults: batch 8, lr 1e-5, chunk 100, fp32, no augmentation). Data: `brouthen/so101_pick_striped_all` (molnrb pick_ball, blue_ball, buried_ball and rb_mixed with 120 episodes; task sentences relabelled; merged with `concatenate_videos=false` because joining the AV1 files failed on non-monotonic DTS). Pretrain 50k steps (45 min): held-out loss on 32 episodes of all tasks 0.430 at 5k, flat at about 0.387 from 25k. Fine-tune 30k steps (40 min) from `050000`: train loss 0.19 at step 100 (from scratch about 10), held-out loss on the last 12 red/blue episodes flat at 0.425-0.447 for all 15 checks (best 2k 0.425, 10k 0.426, 30k 0.426). Those 12 are the newest, probably deep-buried episodes, so the number is not comparable to the runs above. About 7.5 USD in total. Checkpoints: https://huggingface.co/brouthen/act_so101_pick_striped_all_pretrain_50k and https://huggingface.co/brouthen/act_so101_pick_rb_mixed_from_all_30k. To try on the arm: fine-tune `002000`, `010000`, pretrain `050000`.
- **On the arm** (Kamel, 2026-11-21 evening, no success counts recorded): ACT v2 with chunk 50 / `n_action_steps` 25 was "a bit unstable", so we keep ACT's default chunk 100. The SmolVLA fine-tune was "not the best" of the policies tried.

## Alternatives

- **Run 1 settings, longer** - tried (above): held-out loss did not improve after 10k steps.
- **Shorter `n_action_steps` or temporal ensembling at rollout only, no retraining** - not tried yet; cheapest test, run it with run 1's 10k checkpoint at the table.
- **Higher image resolution** - not possible without re-recording; ACT already uses the full 640x480.
- **More and more varied episodes, DAgger corrections** - planned by the team (deep-buried episodes, `kb/dagger-corrections.md`); expected to matter more than any training setting.
- **SmolVLA on the same data** - not tried on red/blue; see `decisions/skill-policies.md`.

## Revisit if

- v2 is tried on the arm: record success counts for v2 and run 1 (same `n_action_steps`) here.
