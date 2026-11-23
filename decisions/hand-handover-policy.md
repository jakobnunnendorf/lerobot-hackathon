# Which learned policy runs the hand handover?

- Status: DECIDED (2026-11-21): none. GIVE is rule-based hand following; the learned handover policy was never trained

## Decision

The handover is done by the rule-based hand-following loop (MediaPipe on the top camera, our own kinematics; `decisions/hand-following-geometry.md`, `decisions/hand-following-safety.md`), called as GIVE by the supervisor after a learned PICK (`decisions/supervisor.md`). No learned handover policy was trained or run on the arm.

## Why

- The hand-following loop followed a real hand on the arm at hover height 0.28 m on 2026-11-21 (board 7064670797, 7064702491), and later dropped the ball into a palm in the demo chain. Balazs: it hovers 3-6 cm from straight above the palm centre, "fine for the demo".
- Releasing the ball too early when following a hand is the failure a learned policy would have had to fix with new demonstrations, and about 20 hand episodes would have had to be recorded first. The loop needed none.
- Decided by Tomi's agent on 2026-11-21, once the loop worked on the arm.

## Alternatives

- **ACT on about 20 hand episodes, compared with SmolVLA in ten tries each** - the first plan (2026-11-21 morning), dropped for the reasons above. ACT was the default because it was the only policy the team had trained and run end to end and it fits the 8 GB laptop (`kb/machines.md`); its weakness, re-planning only every 100 steps, is a run-time setting (`research/hand-handover-policies.md`, section 2).
- **SmolVLA as the main policy** - not tried for the handover. With about 20 episodes the docs' own guidance (25 episodes "was not enough") and a hackathon report both point to it doing worse than ACT. A SmolVLA fine-tuned from `lerobot/smolvla_base` on `molnrb/so101_pick_ball_combined` (Kamel's agent, 23,000 steps, Modal H100, about 9 USD, board 7062185763) showed it fits the 8 GB laptop for inference (0.43 s per 50-action chunk on an RTX 4070); it needs `--rename_map` (`top`->`camera1`, `wrist`->`camera2`) and downloads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` on first load.
- **pi0.5** - tried on Kamel's laptop, out of memory (`decisions/skill-policies.md`).
- **ACT with temporal ensembling** - timed, not tried on the arm: about 20 Hz on the robot laptop's RTX 4050, too slow for 30 fps (`kb/act-replan-override.md`).

## Revisit if

- The handover has to cope with hands the rule-based loop cannot handle (moving hands, several people), and there is table time to record hand episodes.
