# How do we make PICK work when the target ball is hidden?

- Status: DECIDED (2026-11-22 08:32): the hidden-start fine-tune `TianK003/act_hidden_from_v2@002000`, run with fewer executed actions per chunk

## Decision

The demo PICK is `TianK003/act_hidden_from_v2@002000`: v2 (`Space0code/act_redblue_from_foil_v2@20000`) fine-tuned for 2K more steps on 38 clean episodes that start with the ball buried or partly covered. It runs with fewer executed actions per chunk than the checkpoint's 100: Tian says the demo ran it at 25; the benchmark below and Jakob's decision (board 7076580521) used 30, which is also the default in `scripts/supervisor/demo_chain.sh`. The pick budget is 90 s (`--pick-timeout 90`). The number of picks per round can be raised live from the dashboard. v2 stays the fallback and is what `demo_chain.sh` runs without `PICK_POLICY`.

```
PICK_POLICY=TianK003/act_hidden_from_v2 bash scripts/supervisor/demo_chain.sh --step 002000
```

The model was not made larger and was not retrained with a smaller chunk.

## Why

- **The hidden fine-tune beat v2 in both arm tests, and never lost a layout v2 won.** All with `eval_pick.py`, the new gripper servo at torque 200 and the recorded ball counts (molnrb at the table).
  - 50 action steps, 90 s cap, 10 layouts, both models on each layout in rotating order (2026-11-22, board 7075616730): hidden **7/9** against v2 5/9 on the 9 layouts both ran; hidden layouts 4/5 against 3/5; partly covered 2/3 against 1/3. It won both split layouts. But it was slow: mean success 61 s against 34 s, and 3 of its 7 successes took over 75 s.
  - 30 action steps, 90 s cap, the final 7-model benchmark (2026-11-22, board 7076529420; stopped after 4 layouts for time; layouts planned hidden, partly, hidden, visible):

    | model | L1 (H) | L2 (P) | L3 (H) | L4 (V) | total |
    |---|---|---|---|---|---|
    | hidden `TianK003/act_hidden_from_v2@002000` | 43.6 s | 45.7 s | 10.4 s | 15.6 s | **4/4** |
    | v3 `TianK003/act_redblue_from_foil_v3@030000` | 64.7 s | 26.2 s | 24.4 s | 53.0 s | 4/4 |
    | v2 `Space0code/act_redblue_from_foil_v2@20000` | 51.4 s | 35.0 s | digging at 90 s | 14.1 s | 3/4 |
    | `jakobnunnendorf/act_clean_from_all@010000` | 38.5 s | 9.8 s | other at 90 s | 16.7 s | 3/4 |
    | `jakobnunnendorf/act_clean_scratch@030000` | 46.0 s | 16.0 s | digging | 17.0 s | 3/4 |
    | `molnrb/act_pick_rb_mixed@080000` | digging | 56.3 s | digging | 45.4 s | 2/4 |
    | SmolVLA `jakobnunnendorf/smolvla_redblue_v2@003000` | digging | digging | digging | digging | 0/4 |

    At 30 steps the hidden model was no longer slow (10-46 s). v3 also scored 4/4 but took 53 s on the visible ball and had no other arm data. Both tests together, paired: hidden 11/13, v2 8/13.
  - Caveats: small samples (a gap of 2 is noise); the benchmark's `--record` dataset stayed empty because of a bug fixed afterwards in #76, so only the CSV and logs exist (on the unmerged log branch `molnrb/arm-logs`, `logs/arm/2026-11-22_pick-benchmark/`).
- **Shorter execution is better on the arm.** v2 on 10 layouts, 75 s cap (molnrb, 2026-11-22, board 7074960583):

  | | 100 | 50 | 25 |
  |---|---|---|---|
  | all 10 | 4 | 6 | 6 (+1 other) |
  | hidden (5) | 2 | 3 | 2 (+1 other) |
  | partly covered (3) | 2 | 2 | 2 |
  | visible (2) | 0 | 1 | 2 |
  | mean time of a success | 21.3 s | 28.9 s | 37.0 s |

  50 and 25 tie. Tian at the table: 25 to 50 approaches more carefully and retries sooner, 100 commits to a dive and misses; 25 is the most careful, a value between commits better during the approach. Shorter is slower, so the eval cap went from 75 s to 90 s.
- **It needs no training.** `n_action_steps` is how many of the 100 predicted actions run before the policy looks again; any value up to the chunk size works with existing checkpoints (`kb/act-replan-override.md`). In the demo chain it is `--n-action-steps`.
- **Capacity is not the limit.** All our ACT checkpoints are the stock 51.6M-parameter model; train loss keeps falling while held-out loss is flat (`kb/act-model-facts.md`). The literature agrees (`research/act-capacity-and-hidden-targets.md`).
- **Why the policy dives to a typical depth and repeats it** (assumed, from the architecture and the literature, not measured): the top camera shows almost no height, the policy sees one frame and is deterministic, so the same view after a miss gives the same dive; and 56 of 201 episodes contain a miss followed by lift and retry, which the policy imitates.
- **Held-out loss did not pick the winner.** Hidden: 0.3624, 0.3516, 0.3654 at 1K, 2K, 3K on 2 held-out episodes; v3: 0.3309, 0.3401, 0.3351 at 22.5K-27.5K, as flat as v2's 0.34. Only the arm runs above separate the candidates.

## How the two fine-tunes were trained

- **Hidden specialist:** Modal L40S, v2 at 20K plus 3K steps on 38 clean episodes that start buried or partly covered (Jakob's clean list, no hold-out episode), checkpoints `001000`, `002000`, `003000`. Script: `scripts/modal_v2_finetune.py`, on the unmerged branch `TianK003/modal-v2-finetune`.
- **v3, more steps on the same data:** Modal L40S, v2 at 20K plus 10K steps on all 201 episodes with Tomi's recipe, checkpoints `022500`, `025000`, `027500`, `030000`. Reason to try: v1 at 15K had seen its data 8.9 times, v2 at 20K only 5.0 times.

## Alternatives

- **v2 at 25-50 action steps** (the run-time setting alone) - the demo default from 2026-11-22 06:35 until the benchmark; now the fallback. See Why.
- **v3** - tied with the hidden model in the benchmark (4/4) but slower on a visible ball, with no other arm runs. Not chosen.
- **Retrain with chunk size 25 to 50** - considered, not tried: executing the first 25 of 100 predicted actions already uses the most accurate ones; a smaller chunk cannot start from v2's weights (the decoder position table has another shape) and there was no time to train from scratch. Kamel's `brouthen/act_so101_pick_rb_mixed_v2_chunk50_20k` (chunk 50, executes 25) was "a bit unstable" on the arm (`decisions/act-redblue-generalisation.md`).
- **A larger model** - rejected: see Why.
- **Temporal ensembling** (`n_action_steps` 1, coefficient 0.01, as in the ACT paper) - not tried. It does not fit the robot laptop's 30 Hz loop (`kb/act-replan-override.md`); the flag exists only in `eval_pick.py` on the unmerged branch `molnrb/eval-pick-multi`.
- **Rebuilding the scenes as recorded** - done as part of the v2 test above; hidden stayed at 2 to 3 of 5, so scene mismatch is not the main cause.
- **Stall watchdog in the supervisor** (one committed 100-step chunk when the arm stops moving) - designed, not built: about 60 lines in the pick path, and a slow careful approach could be misread as a stall. The pick timeout and the live pick counter give a fresh approach from rest without new code.
- **Kamel's mixed scene** (foil background plus red, blue and target) - not recorded: it trains target recognition among distractors, which was already reliable on visible and partly covered scenes; it does not address depth.
- **Human corrections (DAgger)** at the short close - not run: deferred in favour of testing the trained candidates first, and the gripper servo had just been replaced (`kb/dagger-corrections.md`).
- **Per-colour clearing policies before PICK** - not usable: `TianK003/act_clear_red` and `act_clear_blue` only made empty closes on the arm (`decisions/multistep-step-state.md`).

## Revisit if

- The hidden model behaves worse in the demo chain than in `eval_pick`: work through `kb/demo-chain-vs-eval.md`.
- A stalled pick (jaws in front of the target, no motion) dominates with the timeout: build the watchdog.
- There is time to record: hidden-start episodes with varied pile height and one consistent approach, then corrections at the short close.
