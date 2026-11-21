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

## Alternatives

- **Run 1 settings, longer** - tried (above): held-out loss did not improve after 10k steps.
- **Shorter `n_action_steps` or temporal ensembling at rollout only, no retraining** - not tried yet; cheapest test, run it with run 1's 10k checkpoint at the table.
- **Higher image resolution** - not possible without re-recording; ACT already uses the full 640x480.
- **More and more varied episodes, DAgger corrections** - planned by the team (deep-buried episodes, `kb/dagger-corrections.md`); expected to matter more than any training setting.
- **SmolVLA on the same data** - not tried on red/blue; see `decisions/skill-policies.md`.

## Revisit if

- v2 is tried on the arm: record success counts for v2 and run 1 (same `n_action_steps`) here.
