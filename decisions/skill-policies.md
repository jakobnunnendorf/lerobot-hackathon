# Which policies run the skills?

- Status: DECIDED (2026-11-20; updated 2026-11-22 to what shipped)

## Decision

Only PICK is a learned policy: one ACT, trained in the cloud and run on the robot laptop (at the deadline `TianK003/act_hidden_from_v2@002000`, `decisions/hidden-ball-picking.md`). Everything after the grasp is rule-based in the supervisor (`decisions/supervisor.md`):

- **GIVE** is hand following with MediaPipe and our own kinematics, no learned policy (`decisions/hand-handover-policy.md`, `decisions/hand-following-safety.md`).
- **PUT DOWN** is a closed-loop joint move to a pose taught at the table, then open.
- **DANCE** replays an episode of a public SO-101 dataset (`decisions/dance.md`).
- **BITE** is a rule-based lunge at a pointing fingertip; experimental, not working on the arm at the deadline (`decisions/finger-biting.md`).

A supervisor picks which skill runs next.

## Why

- Inference has to run on a team laptop: the robot laptop has an RTX 4050 laptop GPU, and the only other GPU laptop has 8 GB (`kb/machines.md`). ACT needs about 2-6 GB by the LeRobot hardware guide.
- The task branches (ball visible or not, hand present or not). One end-to-end policy would split 100-200 episodes several ways. Hackathon teams that split reasoning from motion got working demos in 48 hours (`research/ball-picking-and-handover.md`).
- The moves after the grasp do not depend on where the ball lay, so they need no learning. The rule-based GIVE followed a real hand on the arm on 2026-11-21, while a learned GIVE would have needed new demonstrations.

## Alternatives

- **The first plan (2026-11-20): one ACT per learned skill (PICK, CLEAR GRAY, and GIVE if time allows); GIVE, PUT DOWN and SHAKE as replays of one recorded episode each.** What happened to each part:
  - CLEAR GRAY was dropped when the foil balls were replaced by red and blue ones (2026-11-21), and the combined pick policy already dug for a buried ball in one episode. Its successor for spoken multi-step plans, one clearing ACT per colour (`TianK003/act_clear_red`, `act_clear_blue`), made only empty grasps on the arm and was not in the demo (`decisions/multistep-step-state.md`).
  - GIVE became rule-based hand following (above).
  - PUT DOWN became a taught pose; no put-down episode was recorded.
  - SHAKE was never built; the supervisor keeps an empty hook.
- **pi0.5 on the demo laptop** - tried on Kamel's laptop, 2026-11-20: `lerobot/pi05_base` ran out of GPU memory while the model was being built, with about 14 GB allocated (full-precision weights; the config's `dtype` was not applied during construction). On the CPU it stalled the machine, which has 15 GB of RAM. Not tried: building in half precision, which puts the weights near 7 GB and leaves almost nothing else on 8 GB. Details in `kb/machines.md`.
- **SmolVLA** - first considered, not tried (training needs about 10-16 GB; one hackathon team's SmolVLA fine-tuned for only 2,500 steps did worse than their ACT; `research/ball-picking-and-handover.md`). Later trained in the cloud and run on the arm for PICK: it fits the 8 GB laptop for inference (0.43 s per 50-action chunk on Kamel's RTX 4070), but `jakobnunnendorf/smolvla_redblue_v2@003000` scored 0/4 in the final benchmark against ACT's 4/4 (`decisions/act-redblue-generalisation.md`).
- **GR00T N1.5** - considered, not tried: training needs about 24-40 GB, and 0.7.0 accepts only N1.7 (`research/lerobot-training-feasibility.md`).
- **One policy for the whole task** - considered, not tried, for the branching reason above.

## Revisit if

- A team laptop with clearly more than 8 GB of GPU memory becomes available.
- A skill has to be told what to do, for example which colour of ball to pick: ACT ignores the task sentence (`decisions/colour-choice.md`).
