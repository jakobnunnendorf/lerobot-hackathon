# Experiment 1: grasp check with lock after hold, ACT from-foil 15K, n_action_steps 20

Board 7066317565 and the update 7066422484. Run at the robot laptop, 17:29–17:41 UTC (18:29–18:41 CET). The program was stopped after round 5 of the planned 8, and the empty-bowl rounds were not run.

## Files
- `supervisor.jsonl`: the whole `--log`, 5 rounds, 120 `grasp_check` lines.
- `terminal.txt`: the terminal output of the run. The only warnings are the usual ones: no frame_link, container poses not taught.
- `command.txt`: the exact command, with the commit.
- `supervisor_verify_hold_r<round>_p<attempt>_*.jpg`: the wrist frame saved at each VERIFY_HOLD, 7 frames.

## Setup
- Checkout `jakobnunnendorf/supervisor` at **0ce2350**, detached; `tests/supervisor_scenarios.py` 103 PASS, 0 FAIL, 0 SKIP.
- Policy `Space0code/act_redblue_from_foil` `checkpoints/015000` (Hub revision c1e6c6a).
- `--n-action-steps 20 --grasp-lock-margin 4 --grasp-min-load 200 --min-held-target-area 120`, colour detector from colour-pick 85df7ca, `--max-step-deg 1.0`.

## Rounds (from the log)
| round | outcome | picks | wrong balls | seconds |
|---|---|---|---|---|
| 1 | put_down | 1 | 0 | 44.8 |
| 2 | put_down | 3 | 2 | 162.0 |
| 3 | gave_up | 3 | 0 | 144.3 |
| 4 | wrong_ball | 3 | 1 | 84.6 |
| 5 | gave_up | 3 | 2 | 142.9 |

`grasp_check` lines: 13 `settle` checks (7 held, 6 miss) and 107 `release_blocked` lines.

## What was not recorded
- The log has events only: the gripper reading and load at each check, the command and the sent value for `release_blocked`, and the state transitions. It has **no per-tick joint states and no camera video**. The supervisor does not write them, and the wrist JPEGs are saved only at VERIFY_HOLD. The top camera is not saved at all.
- So the motor and camera data for these rounds cannot be recovered afterwards.

## Table notes
_Balazs's notes per check (ball or nothing, where the ball was, what looked odd) are still missing._
