# How does the classical hand-following loop keep the ball and the people safe?

- Status: OPEN, built and tested in dry run only (2026-11-21, Jakob's agent, `scripts/handover/`)

## Decision

One process owns the robot connection and runs a 30 Hz loop: newest hand message, following/drop decision, inverse kinematics, then per-joint speed limit and `send_action`. The safety rules live in the loop, not in the decision logic, so they hold whatever Tian's following functions decide.

## Why

- **The holding gripper value is sent every tick until the drop.** A lost hand, an unreachable target or a solver failure only changes the arm goal, never the gripper. The value defaults to the gripper's `Goal_Position` at start, so the grip is not loosened by guessing a number (`CLAUDE.md`: never loosen while held).
- **Torque stays on at exit** (`disable_torque_on_disconnect=False`): the default disconnect switches torque off and would drop the ball (`research/lerobot-multi-skill-runner.md`, section 2).
- **Fixed hover height instead of the hand's z.** One camera gives no real depth (board comment 7061160180).
- **Drop needs two hands** and the tip within 3 cm (horizontal) of their midpoint for 1 s; one hand is followed but never dropped into. All three numbers are flags.
- **Unreachable or badly solved targets hold position** (IK result checked with FK, 2 cm tolerance) instead of moving to the nearest pose.
- **The workspace clamp acts on the target**, not on the path: per-joint speed limiting can still take the tip below the hover height for a moment between two targets (seen in dry run, lowest tip 0.136 m with hover 0.18 m).
- **HOME is the pose at start** unless `--home` is given, because the demo starts every skill from HOME.
- **Robot mode is refused while `kinematics.PLACEHOLDER` or `following.PLACEHOLDER` is True.**

## Alternatives

- **LeRobot's `max_relative_target`** as the speed limit - not used: it reads the present position on every `send_action`, which slows the loop, and our own per-tick step limit does the same job.
- **Following the hand's z** - rejected for the first version, see above.

## Revisit if

- The table check (`docs/hand-following-table-check.md`) shows FK errors above about 2 cm.
- The tip dips too low on the real arm between targets; then limit in Cartesian space instead of per joint.
