# How do we train the red/blue ACT pick policy so it works on the robot, not only on the recordings?

- Status: DECIDED (2026-11-22): none of the training recipes tried here beat fine-tuning from the foil-scene policy; the demo used that line plus a short hidden-start fine-tune

## Decision

Keep LeRobot's ACT defaults apart from the from-foil recipe (`decisions/act-redblue-init.md`), fine-tune briefly on the hard cases (`decisions/hidden-ball-picking.md`), and change behaviour on the arm through `n_action_steps` at run time rather than through the training settings. The recipes below were tried on 2026-11-21 and 22; none reached the demo.

## Why

- The arm decided, not the loss: held-out loss was flat at 0.29-0.45 for every run below and did not predict arm results (`decisions/act-redblue-init.md`).
- In the final 7-model benchmark (2026-11-22, 30 action steps, 4 layouts, board 7076529420): from-foil hidden 4/4 and v3 4/4, v2 3/4; Jakob's curated runs `act_clean_from_all@010000` 3/4 and `act_clean_scratch@030000` 3/4; `molnrb/act_pick_rb_mixed@080000` 2/4; SmolVLA `jakobnunnendorf/smolvla_redblue_v2@003000` 0/4. Kamel's runs were not in it.

## Alternatives

- **Run 1, from scratch** (`act_redblue_l40s_b32_38k`, Kamel's agent, 2026-11-21, Modal L40S, batch 32, lr 2e-5, chunk 100, 5% held out) - held-out loss flat at about 0.33 from 10k to 30k steps while train loss fell from 0.19 to 0.165, so later steps only widened the gap. Checkpoints: https://huggingface.co/brouthen/act_so101_pick_rb_mixed_b32_38k. On the arm: 5/13 against from foil's 10/13 (`decisions/act-redblue-init.md`); `@030000` later scored 2/3 on three visible layouts (board 7072792003).
- **ACT v2 recipe: shorter chunks, stronger augmentation, EMA, more dropout** (`act_redblue_v2_h100_chunk50_20k`, recipe `act_redblue_v2` in `scripts/modal_train.py`, Kamel's agent) - trained 20k steps in 41 min (H100, about 4.2 USD). Chunk 50 with `n_action_steps` 25 re-plans every 0.8 s; augmentations added shadows, brightness patches, dropout holes, noise, motion blur and ±10° / 10% affine, with hue, saturation and colour temperature off because colour tells the target apart; 8 held-out episodes. Held-out loss 0.350 at 2k, about 0.31 from 4k to 14k, then 0.295 / **0.287** / 0.295 at 16k / 18k / 20k. On the arm (Kamel, 2026-11-21 evening, no counts recorded) it was "a bit unstable", so the team kept chunk 100 and shortened execution at run time instead. Checkpoints: https://huggingface.co/brouthen/act_so101_pick_rb_mixed_v2_chunk50_20k.
- **Pretrain on all pick data, then fine-tune on red/blue** (Kamel's agent, 2026-11-21, Modal H100, LeRobot ACT defaults: batch 8, lr 1e-5, chunk 100, fp32, no augmentation) - data `brouthen/so101_pick_striped_all` (molnrb pick_ball, blue_ball, buried_ball and rb_mixed with 120 episodes; task sentences relabelled; merged with `concatenate_videos=false` because joining the AV1 files failed on non-monotonic DTS). Pretrain 50k steps (45 min): held-out loss on 32 episodes of all tasks 0.430 at 5k, flat at about 0.387 from 25k. Fine-tune 30k steps (40 min) from `050000`: train loss 0.19 at step 100 (from scratch about 10), held-out loss on the last 12 red/blue episodes flat at 0.425-0.447. About 7.5 USD in total. Checkpoints: https://huggingface.co/brouthen/act_so101_pick_striped_all_pretrain_50k and https://huggingface.co/brouthen/act_so101_pick_rb_mixed_from_all_30k. Kamel's fine-tune was not counted on the arm; Jakob's `act_clean_from_all` (10K steps from the same pretrain, on 103 curated clean episodes) scored 3/4 in the final benchmark.
- **Curated data from fresh weights** (`jakobnunnendorf/act_clean_scratch`, 30K steps on the same clean episodes) - 3/4 in the final benchmark.
- **SmolVLA fine-tuned from the blue-ball SmolVLA** (`smolvla_redblue_from_blue_h100_10k`, recipe `smolvla_redblue_from_combined`, H100, batch 64, lr 5e-5) - held-out loss 0.223 / 0.238 / 0.229 / 0.252 / 0.259 at 1k-5k, so it overfit after about 3k; stopped by hand at about 5.1k (about 2.5 USD). Starting from a related model, the transfer happened within the first 1-3k steps. On the arm "not the best" of the policies tried (Kamel, 2026-11-21 evening). Jakob's later `smolvla_redblue_v2@003000` scored 0/4 in the final benchmark.
- **Higher image resolution** - not possible without re-recording; ACT already uses the full 640x480.
- **Temporal ensembling at rollout** - not tried on the arm: too slow for the robot laptop's 30 Hz loop (`kb/act-replan-override.md`).
- **Human corrections (DAgger)** - not run (`kb/dagger-corrections.md`).

Held-out losses are not comparable between ACT and SmolVLA, nor between ACT chunk sizes, nor between runs with different held-out episodes.

## Revisit if

- A recipe change is tested on the arm against the hidden model on the same layouts at the same `n_action_steps`.
- Much more data is recorded: a from-scratch run may catch up with the from-foil start.
