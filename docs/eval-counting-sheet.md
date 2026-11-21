# Counting sheet for policy runs at the table

One row per run, filled in by the person at the table. Counts decide the roadmap gates (H5), so write them down even for bad runs. The sheet is for runs of `act_pick_ball_combined` and, later, the hand-handover policy.

## Settings to write at the top of every sheet

- Checkpoint (Hub name or path), date and time, laptop.
- `n_action_steps` (default 100 unless `--policy.n_action_steps` was given), run length cap in seconds.
- Display off (`--display_data=false`): Rerun drawing on the CPU caused earlier camera timeouts (board 7060609632).
- Top camera not moved since recording. Lighting as in the recordings.
- Run length: buried-ball episodes average about 33 s, so a 40 s cap can cut off a successful dig. Use about 60 s.

## PICK runs

| # | Ball at start: visible / partly / hidden | Result: success / fail / aborted | Failure mode (missed ball, grabbed gray, stopped, dropped, camera timeout, other) | Gripper reading at end | Seconds |
| --- | --- | --- | --- | --- | --- |

- Success = ball held at the end. A gripper reading of about 6-20 means holding, near 1 means a missed grasp, up to about 38 means open jaws (145 episodes, board 7061759444). A foil ball is not measured yet.
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
