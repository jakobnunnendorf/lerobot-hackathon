# Does the red/blue ACT pick policy train better from the foil-scene policy than from scratch?

- Status: DECIDED (2026-11-21, confirmed by the demo model on 2026-11-22): start from the foil-scene policy

## Decision

Fine-tune from the foil-scene policy `molnrb/act_pick_ball_combined` instead of training from scratch. Every model the demo ran descends from it: v1 `Space0code/act_redblue_from_foil`, then v2 `Space0code/act_redblue_from_foil_v2@20000` (the fallback), then the demo PICK `TianK003/act_hidden_from_v2@002000` (v2 plus 2K steps on 38 hidden-start episodes, `decisions/hidden-ball-picking.md`).

## Why

- On the arm, from foil picked the target on 10 of 13 shared layouts against 5 of 13 from scratch, while held-out loss was equal. On its own that gap was suggestive, not settled (sign test on the 6 vs 1 layouts that only one policy solved: p ≈ 0.13).
- The later models kept the lead. In the final 7-model benchmark (2026-11-22, 30 action steps, 4 layouts, board 7076529420) the from-foil line scored hidden 4/4, v3 4/4, v2 3/4. Of the others, Jakob's `act_clean_from_all@010000` (initialised from a model pretrained on all pick data) and `act_clean_scratch@030000` scored 3/4, Balazs's `molnrb/act_pick_rb_mixed@080000` 2/4, and the SmolVLA 0/4.

## Runs

- **From scratch:** recipe `act_redblue` in `scripts/modal_train.py`, Kamel's Modal account, started 2026-11-21 13:38 UTC, up to 38K steps. Hub: `brouthen/act_so101_pick_rb_mixed_b32_38k`.
- **From foil:** recipe `act_redblue_from_foil`, Tomi's Modal account, starts from `molnrb/act_pick_ball_combined` (30K steps on 145 foil episodes), 20K steps, then resumed (`resume` entrypoint) and stopped at about 28.4K; last checkpoint 25K. Hub: `Space0code/act_redblue_from_foil`.
- Shared: `molnrb/so101_pick_rb_mixed`, batch 32, lr 2e-5, bf16, augmentation with hue and saturation off, `eval_split=0.05`, seed 1000 (same 4 held-out episodes), eval every 5K, L40S.
- Fine-tuning with `--policy.path` keeps the weights only: fresh optimizer, normaliser stats from the red/blue data (`research/continue-training-from-checkpoint.md`).
- v2 is the same recipe on 190 training episodes (`kb/act-model-facts.md`).

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
- **Arm test** (2026-11-21, board 7065447840): `eval_pick.sh`, shared layouts in alternating order, `n_action_steps` 100, 90 s cap.
  - From foil @15K: 10/13, returned 11, timeout 2. Its misses grabbed a red or blue ball (3 wrong_ball).
  - From scratch @20K: 5/13 on the shared layouts, plus 1 success on a 14th layout that only scratch ran. Returned 7, timeout 7. Its misses kept digging until the cap. Tomi's table notes: tip-only grabs, and aiming above the ball.
- **Eval loss did not predict the arm.** Choose checkpoints on the arm, not by loss.

## Alternatives

- **From scratch** - tried (above): 5/13 on the arm. Later screen of `b32_38k@030000` on 3 visible layouts (2026-11-22, board 7072792003): 2/3; not in the final benchmark.
- **Pretrain on all pick data, then fine-tune on red/blue** - tried by Kamel and by Jakob; see `decisions/act-redblue-generalisation.md`. Jakob's `act_clean_from_all@010000` reached 3/4 in the final benchmark, one below the from-foil hidden model.

## Revisit if

- More red/blue episodes are recorded: continue from the latest from-foil checkpoint, 2-10K steps, and compare on the arm.
- 5-10 more shared layouts reverse the gap between the from-foil line and the clean models.
