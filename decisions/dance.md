# How does the arm dance, and how is a borrowed dance kept safe on our arm?

- Status: DECIDED (2026-11-21), not yet run on the arm

## Decision

A DANCE round replays one episode of someone else's public SO-101 dataset (`NikeSong/so101_gestures_dance`, episodes 1 and 2) through the supervisor's existing `Replay`, on the demo chain's one robot connection. It starts only from REST with an empty gripper and ends at REST. The recording is slowed with `--dance-speed` (frames interpolated) and clamped to the joint ranges our own arm was teleoperated through. Each tick's step is capped by `--dance-max-step-deg` and by the global `--max-step-deg`, whichever is smaller. At start, the program refuses any episode whose tip, in our kinematic model, goes below `--dance-min-tip-height` (0.20 m). Runbook: `docs/dance.md`.

## Why

- **Episodes 1 and 2**: the calmest of the 50, measured over the action column on 2026-11-21:

  | Episode | Joint span (pan, lift, elbow, wrist flex, roll) | Peak speed (p95) | Pan span | Lowest tip (model) | Largest reach from the base axis |
  | --- | --- | --- | --- | --- | --- |
  | 1 | 45, 29, 44, 45, 40 deg | 197 (103) deg/s | 45 deg | 0.266 m | 0.16 m |
  | 2 | 62, 58, 94, 116, 53 deg | 213 (155) deg/s | 62 deg | 0.265 m | 0.20 m |

  - The busier episodes reach 300-700 deg/s with pan sweeps of 100-170 deg (for example 48 and 49).
  - Front-camera frames of episodes 1 and 2 (extracted 2026-11-21) show the follower upright over its own base, gripper up, small sways. It never leans towards the table.
  - Episode 1 starts near our present pose (pan -1, lift -58, elbow -8, wrist flex 84 against `PRESENT_POSE` 0, -55, 6, 92). The approach from rest is therefore the same fold-out that GIVE already makes on the arm.
- **Our degrees match theirs**: same joint names, LeRobot `use_degrees` and robot type `so_follower`, and the poses above line up with ours. This is inferred from the data, not tested on the arm. The first run at half speed is the test.
- **Replay instead of a policy**: a dance needs no camera and no decision, and `Replay` already approaches the first frame speed-limited (1.5 deg/tick, or the dance cap if lower). ABORT works the same as in every other skill: `_observe` raises, nothing more is sent, the arm holds with torque on.
- **Gripper held at its rest value (30)**: the recording keeps the gripper closed (0-11). Replaying that would squeeze the empty jaws shut for 40 s and could stall the servo, and an open gripper does not change the look of the dance.
- **Clamp, not reject, for the joint envelope**: the envelope is the action min/max of `molnrb/so101_pick_rb_mixed` (`skills.ARM_ENVELOPE`). Episodes 1 and 2 stay inside it: the largest clamp is 0.1 deg (episode 1's wrist flex), so clamping changes nothing visible. A clamp of 1 deg or more is printed at start.
- **Tip-height refusal at start**: the table hazard is the bowl and the balls in front of the arm. The model's tip at grasps is 0.02-0.16 m. The 0.20 m floor is the one the GIVE loop uses, and it is checked over the approach from rest as well as the frames.
- **Speed cap explicit and logged**: demo_chain.sh's `--max-step-deg 15` (450 deg/s) would not slow anything down. The runbook therefore passes `--dance-max-step-deg 4`. The effective cap is the smaller of the two and is printed and logged in the `start` event as `dance_max_step_deg`. At half speed, episode 1 peaks at 3.3 deg/tick and episode 2 at 3.6, so a cap of 4 never bends their path.
- **Refused while a round runs**: the dashboard already refuses every command while a round runs. A dance asked for over the link or the keyboard during a round is cleared when that round ends (`Commands.clear`), so it never starts a round nobody expects.

## Alternatives

- **Reject an episode that leaves our envelope**: considered. Episodes 1 and 2 never leave it. Episodes such as 48 would be refused for a few degrees of pan, while clamping keeps them playable and still safe.
- **Replay the recorded gripper**: considered. It risks a stalled gripper servo for no visible gain (see Why).
- **`lerobot-replay` in a subprocess**: ruled out. It opens its own robot connection, and the demo chain owns the only one (CLAUDE.md, `decisions/supervisor.md`).
- **Busier episodes (5, 48, 49)**: not tried. Peaks of 305-458 deg/s and pan spans of 99-148 deg would sweep over the bowl.
- **Episode 12 as a third, bigger dance**: lowest tip 0.219 m, peak 342 deg/s. At half speed and cap 4 it is limited on 32 ticks. Try it only after 1 and 2 have run well.

## Revisit if

- The first run shows our degrees do not match theirs (for example the wrist bends the other way).
- The team wants a faster dance: try `--dance-speed 0.75`, then 1, only after half speed looked safe.
