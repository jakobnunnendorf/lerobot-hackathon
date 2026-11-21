# Which learned policy runs the hand handover?

- Status: OPEN, working default ACT (2026-11-21)

## Decision

ACT is the default. SmolVLA is trained in parallel in the cloud and compared on the table. Kamel's agent has already claimed a SmolVLA fine-tune on `molnrb/so101_pick_ball_combined` (Modal, about 10 USD, board 2026-11-21), so that run is not duplicated here; it uses the other dataset, not the ~20 hand episodes, until those exist. Whichever completes hover, second-hand notice, drop and return to HOME more often in ten tries wins. This file is updated with the numbers once they exist.

## Why

- ACT is the only policy the team has trained and run end to end, and it fits the 8 GB demo laptop (`kb/machines.md`).
- SmolVLA's one real advantage is the task sentence. For one fixed behaviour that gives little; for "hand over to the left hand" style variants it would matter.
- ACT's weakness for this task is that it re-plans only every `n_action_steps` (100 by default, 3.3 s). That is a setting, not a reason to switch policy (`research/hand-handover-policies.md`, section 2).

## Alternatives

- **SmolVLA as the main policy** - not tried. Cloud training only, and with about 20 episodes the docs' own guidance (25 episodes "was not enough") and a hackathon report both point to it doing worse than ACT.
- **pi0.5** - tried on Kamel's laptop, out of memory (`decisions/skill-policies.md`).
- **ACT with temporal ensembling** - timed, not tried on the arm: about 20 Hz on the robot laptop's RTX 4050, too slow for 30 fps; fits on Kamel's 4070. Only an option if the demo runs on the 4070 laptop.

## Revisit if

- ACT with a small `n_action_steps` still misses the second hand.
- The classical approach (MediaPipe and inverse kinematics, Tian's agent and Jakob's agent) is clearly more reliable; then the learned policy becomes the comparison, not the demo.
