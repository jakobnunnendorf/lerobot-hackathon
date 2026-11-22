# How do we make PICK work when the target ball is hidden?

- Status: OPEN (2026-11-22 07:15). Run-time setting chosen; two fine-tunes trained and not yet on the arm.

## Decision

Run the pick policy with fewer executed actions per chunk (`n_action_steps` 25 to 50, 30 as the working value) instead of the checkpoint's 100, with a 90 s cap in the eval and `--pick-timeout 60 --max-pick-attempts 4` in the demo chain. Do not make the model larger and do not retrain with a smaller chunk.

## Why

- **Shorter execution is better on the arm.** `Space0code/act_redblue_from_foil_v2@20000`, 10 layouts with the recorded ball set, 75 s cap, new gripper servo at torque 200 (molnrb, 2026-11-22, board 7074960583, `eval_20261122_060603.csv`):

  | | 100 | 50 | 25 |
  |---|---|---|---|
  | all 10 | 4 | 6 | 6 (+1 other) |
  | hidden (5) | 2 | 3 | 2 (+1 other) |
  | partly covered (3) | 2 | 2 | 2 |
  | visible (2) | 0 | 1 | 2 |
  | mean time of a success | 21.3 s | 28.9 s | 37.0 s |

  50 and 25 tie; a gap of 2 in 10 is noise. Tian at the table: 25 to 50 approaches more carefully and retries sooner, 100 commits to a dive and misses; 25 is the most careful, a value between commits better during the approach.
- **It needs no training.** `n_action_steps` is how many of the 100 predicted actions run before the policy looks again. Any value up to the chunk size works with the checkpoints we have. In the demo chain it is `--n-action-steps`.
- **Shorter is slower**, so the 75 s cap cut good runs on deeply buried balls. Tian set the eval cap to 90 s (board 7074889773).
- **Capacity is not the limit.** All our ACT checkpoints are the stock 51.6M-parameter model; train loss keeps falling while held-out loss is flat (`kb/act-model-facts.md`). The literature agrees (`research/act-capacity-and-hidden-targets.md`).
- **Why the policy dives to a typical depth and repeats it** (assumed, from the architecture and the literature, not measured): the top camera shows almost no height, the policy sees one frame and is deterministic, so the same view after a miss gives the same dive; and 56 of 201 episodes contain a miss followed by lift and retry, which the policy imitates.

## Alternatives

- **More steps on the same data** - trained 2026-11-22, Modal L40S: `TianK003/act_redblue_from_foil_v3`, v2 at 20K plus 10K steps on all 201 episodes with Tomi's recipe, checkpoints `022500`, `025000`, `027500`, `030000`. Held-out loss 0.3309, 0.3401, 0.3351 (flat, as v2's 0.34). Not yet on the arm. Reason to try: v1 at 15K had seen its data 8.9 times, v2 at 20K only 5.0 times.
- **Hidden specialist** - trained 2026-11-22, Modal L40S: `TianK003/act_hidden_from_v2`, v2 at 20K plus 3K steps on 38 clean episodes that start buried or partly covered (Jakob's clean list, no hold-out episode), checkpoints `001000`, `002000`, `003000`; held-out loss 0.3624, 0.3516, 0.3654 on 2 episodes. Not yet on the arm. Script: `scripts/modal_v2_finetune.py` on `TianK003/modal-v2-finetune`.
- **Retrain with chunk size 25 to 50** - considered, not tried: executing the first 25 of 100 predicted actions already uses the most accurate ones; a smaller chunk cannot start from v2's weights (the decoder position table has another shape) and there was no time to train from scratch. Kamel's `brouthen/act_so101_pick_rb_mixed_v2_chunk50_20k@018000` (chunk 50, executes 25, 80 episodes) exists and has not been on the arm.
- **A larger model** - rejected: see Why.
- **Temporal ensembling** (`n_action_steps` 1, coefficient 0.01, as in the ACT paper) - not tried. `eval_pick.py` on `molnrb/eval-pick-multi` has the flag; the supervisor has none.
- **Rebuilding the scenes as recorded** - done as part of the test above; hidden stayed at 2 to 3 of 5, so scene mismatch is not the main cause.
- **Stall watchdog in the supervisor** (one committed 100-step chunk when the arm stops moving) - designed, not built: about 60 lines in the pick path, untested on the arm, and a slow careful approach could be misread as a stall. The timeout flags above give a fresh approach from rest without new code.
- **Kamel's mixed scene** (foil background plus red, blue and target) - deferred: it trains target recognition among distractors, which is already reliable on visible and partly covered scenes; it does not address depth, and foil has no hole.
- **Human corrections (DAgger)** at the short close - deferred: the tooling has not run on our arm (`kb/dagger-corrections.md`) and the gripper servo is new.
- **Per-colour clearing policies before PICK** - not usable yet: `TianK003/act_clear_red` and `act_clear_blue` only made empty closes on the arm (board 7072005462); see `decisions/multistep-voice-policy.md` on `TianK003/multistep-act`.

## Revisit if

- The hidden specialist or v3 beats v2's 3 of 5 on hidden layouts at the same `n_action_steps` and cap: it becomes the hidden-case pick.
- A stalled pick (jaws in front of the target, no motion) still dominates with the timeout flags: build the watchdog.
- There is time to record after the hackathon: hidden-start episodes with varied pile height and one consistent approach, then corrections at the short close.
