# Which policies run the skills?

- Status: DECIDED (2026-11-20)

## Decision

One ACT policy per learned skill (PICK, CLEAR GRAY, and GIVE if time allows), trained in the cloud and run on a team laptop. GIVE, PUT DOWN and SHAKE start as replays of one recorded episode each. A supervisor picks which skill runs next.

## Why

- Inference has to run on a team laptop, and the only one with a GPU has 8 GB (`kb/machines.md`). ACT needs about 2-6 GB by the LeRobot hardware guide.
- The task branches twice (ball visible or not, hand present or not) and has five behaviours. One end-to-end policy would split 100-200 episodes five ways. Hackathon teams that split reasoning from motion got working demos in 48 hours (`research/ball-picking-and-handover.md`).
- The moves after the grasp do not depend on where the ball lay, so a replay is enough to start with.

## Alternatives

- **pi0.5 on the demo laptop** - tried on Kamel's laptop, 2026-11-20: `lerobot/pi05_base` ran out of GPU memory while the model was being built, with about 14 GB allocated (full-precision weights; the config's `dtype` was not applied during construction). On the CPU it stalled the machine, which has 15 GB of RAM. Not tried: building in half precision, which puts the weights near 7 GB and leaves almost nothing else on 8 GB. Details in `kb/machines.md`.
- **SmolVLA or GR00T N1.5** - considered, not tried. Winning hackathon teams used them, but training needs about 10-16 GB (SmolVLA) or 24-40 GB (GR00T), and one team's SmolVLA, fine-tuned for only 2,500 steps, did worse than their ACT. Source: `research/ball-picking-and-handover.md`.
- **One policy for the whole task** - considered, not tried, for the branching reason above.

## Revisit if

- A team laptop with clearly more than 8 GB of GPU memory becomes available.
- A skill has to be told what to do, for example which colour of ball to pick: ACT ignores the task sentence.
