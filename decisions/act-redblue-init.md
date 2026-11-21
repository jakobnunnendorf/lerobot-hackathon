# Does the red/blue ACT pick policy train better from the foil-scene policy than from scratch?

- Status: DECIDED, tentatively (2026-11-21): start from the foil-scene policy

## Decision

Fine-tune from the foil-scene policy (`Space0code/act_redblue_from_foil`). On the arm it picked the target on 10 of 13 shared layouts against 5 of 13 from scratch, while held-out loss was equal. The gap is suggestive, not settled (sign test on the 6 vs 1 layouts that only one policy solved: p ≈ 0.13).

## Runs

- **From scratch:** recipe `act_redblue` in `scripts/modal_train.py`, Kamel's Modal account, started 2026-11-21 13:38 UTC, up to 38K steps (board 7063508772). Hub: `brouthen/act_so101_pick_rb_mixed_b32_38k`.
- **From foil:** recipe `act_redblue_from_foil`, Tomi's Modal account, starts from `molnrb/act_pick_ball_combined` (30K steps on 145 foil episodes), 20K steps, then resumed (`resume` entrypoint) and stopped at about 28.4K; last checkpoint 25K. Hub: `Space0code/act_redblue_from_foil`.
- Shared: `molnrb/so101_pick_rb_mixed`, batch 32, lr 2e-5, bf16, augmentation with hue and saturation off, `eval_split=0.05`, seed 1000 (same 4 held-out episodes), eval every 5K, L40S.
- Fine-tuning with `--policy.path` keeps the weights only: fresh optimizer, normaliser stats from the red/blue data (`research/continue-training-from-checkpoint.md`).

## Results

Eval loss on the 4 held-out episodes (plain L1, VAE latent zero, as at inference):

| step | from foil | from scratch |
| --- | --- | --- |
| 5K | 0.3372 | 0.3434 |
| 10K | 0.3305 | 0.3292 |
| 15K | 0.3381 | 0.3339 |
| 20K | 0.3395 | 0.3356 |
| 25K | 0.3380 | - |

- Both runs stop improving on eval loss at about 10K, while train loss keeps falling (board 7065399644). So more steps on the same 76 episodes mostly fit the training layouts.
- **Arm test** (board 7065447840): `eval_pick.sh`, shared layouts in alternating order, `n_action_steps` 100, 90 s cap.
  - From foil @15K: 10/13, returned 11, timeout 2. Its misses grabbed a red or blue ball (3 wrong_ball).
  - From scratch @20K: 5/13 on the shared layouts, plus 1 success on a 14th layout that only scratch ran. Returned 7, timeout 7. Its misses kept digging until the cap. Tomi's table notes (board 7065155762): tip-only grabs, and aiming above the ball.
- **Eval loss did not predict the arm.** Choose checkpoints on the arm, not by loss.

## Revisit if

- More red/blue episodes are recorded (deep-buried set): continue from the foil-scene start, 5-10K steps.
- 5-10 more shared layouts, or a comparison of 10K and 25K on the arm, reverse or close the gap.

Written by Claude
