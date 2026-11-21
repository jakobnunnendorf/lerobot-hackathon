# Counting sheet for policy runs at the table

One row per run, filled in by the person at the table. Counts decide the roadmap gates (H5), so write them down even for bad runs. The sheet is for runs of `act_pick_ball_combined` and, later, the hand-handover policy.

## Settings to write at the top of every sheet

- Checkpoint (Hub name or path), date and time, laptop.
- `n_action_steps` (default 100 unless `--policy.n_action_steps` was given), run length cap in seconds.
- Policy: ACT or SmolVLA. For SmolVLA use the same task sentence as in training for every run (an exact copy of one of the dataset's sentences) and `--rename_map`. Note stutter separately from success: SmolVLA took 0.43 s per 50-action chunk on the RTX 4070 and has not been timed on the robot laptop's RTX 4050, so the arm may pause about 0.5 s every 1.7 s.
- Display off (`--display_data=false`): Rerun drawing on the CPU caused earlier camera timeouts (board 7060609632).
- Top camera not moved since recording. Lighting as in the recordings.
- Run length: buried-ball episodes average about 33 s, so a 40 s cap can cut off a successful dig. Use about 60 s.

## PICK runs

| # | Ball at start: visible / partly / hidden | Result: success / fail / aborted | Failure mode (never started, missed ball, grabbed gray, stopped, dropped, camera timeout, other) | Stutter (pauses in the motion: none / some / bad) | Gripper reading at end | Seconds |
| --- | --- | --- | --- | --- | --- | --- |

- Success = ball held at the end. A gripper reading of about 6-20 means holding, near 1 means a missed grasp, up to about 38 means open jaws (145 episodes, board 7061759444). A foil ball is not measured yet.
- **Never started** (the arm stays at its rest pose) is its own failure mode: a preprint reports it in 47% of SmolVLA rollouts (https://arxiv.org/abs/2611.00283, one author, one task, not peer reviewed; via the board). Pick checkpoints by counting on the arm, not by held-out loss; a later SmolVLA checkpoint should not be dropped because its eval loss rose.
- Aborted = the run was stopped by a person or a camera timeout, not by the policy.
- Count visible, partly visible and hidden separately: 17 of the 45 "buried" training episodes start partly visible (board 7061088625), so dataset names are not labels.
- Tally at the bottom: successes out of tries for each of the three start states.

## n_action_steps comparison (only if time allows, 10 runs each)

Same checkpoint, same layouts, alternating runs: default 100, then `--policy.n_action_steps=20`. Record the success count for each. No retraining needed (board 7061213314, PR #9). Temporal ensembling is too slow on the robot laptop (about 20 Hz).

## Hand-handover runs (later)

| # | Hand position (area cell) | Hover reached | Second hand noticed | Ball dropped into the hands | Back at HOME | Notes |
| --- | --- | --- | --- | --- | --- | --- |

Hands 10-15 cm above the table, 15-35 cm in front of the base (`research/hand-handover-policies.md`). Ten tries per policy.

Post the tallies on the message board with the settings line, so the numbers can be compared.

Keep the running paid-compute total on the board (so far: about 9 USD Modal for SmolVLA, plus the Hugging Face job for `act_pick_ball_combined`; each person who starts a paid job adds theirs).
