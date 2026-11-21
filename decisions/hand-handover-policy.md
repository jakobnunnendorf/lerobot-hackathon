# Which learned policy runs the hand handover?

- Status: OPEN, working default ACT (2026-11-21)

## Decision

ACT is the default. SmolVLA is trained in parallel in the cloud on the same ~20 episodes and compared on the table. Whichever completes hover, second-hand notice, drop and return to HOME more often in ten tries wins. This file is updated with the numbers once they exist.

## Why

- ACT is the only policy the team has trained and run end to end, and it fits the 8 GB demo laptop (`kb/machines.md`).
- SmolVLA's one real advantage is the task sentence. For one fixed behaviour that gives little; for "hand over to the left hand" style variants it would matter.
- ACT's weakness for this task is that it re-plans only every `n_action_steps` (100 by default, 3.3 s). That is a setting, not a reason to switch policy (`research/hand-handover-policies.md`, section 2).

## Alternatives

- **SmolVLA as the main policy** - not tried. Cloud training only, and with about 20 episodes the docs' own guidance (25 episodes "was not enough") and a hackathon report both point to it doing worse than ACT.
- **pi0.5** - tried on Kamel's laptop, out of memory (`decisions/skill-policies.md`).
- **ACT with temporal ensembling** - not tried; fallback if a small `n_action_steps` makes the motion jerky.

## Revisit if

- ACT with a small `n_action_steps` still misses the second hand.
- The classical approach (MediaPipe and inverse kinematics, Tian's agent and Jakob's agent) is clearly more reliable; then the learned policy becomes the comparison, not the demo.
