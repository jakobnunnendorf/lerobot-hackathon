# How does the classical hand-following loop keep the ball and the people safe?

- Status: OPEN, built and tested in dry run only (2026-11-21, Jakob's agent, `scripts/handover/`)

## Decision

One process owns the robot connection and runs a 30 Hz loop: newest hand message, following/drop decision, inverse kinematics, then per-joint speed limit and `send_action`. The safety rules live in the loop, not in the decision logic, so they hold whatever Tian's following functions decide.

## Why

- **The holding gripper value is sent every tick until the drop.** A lost hand, an unreachable target or a solver failure only changes the arm goal, never the gripper. The value defaults to the gripper's `Goal_Position` at start, so the grip is not loosened by guessing a number (`CLAUDE.md`: never loosen while held).
- **Torque stays on at exit** (`disable_torque_on_disconnect=False`): the default disconnect switches torque off and would drop the ball (`research/lerobot-multi-skill-runner.md`, section 2).
- **Fixed hover height above the table (`--hover-height`, default 0.22 m) instead of the hand's z.** One camera gives no real depth (board comment 7061160180), and hands have to stay low anyway, see Alternatives.
- **Workspace clamp: tip 0.08-0.30 m above the table, 0.12-0.38 m from the pan axis.** Under about 10 cm from the axis is unusable (self-collision not checked; board comments 7061318846, 7061323029). At start the loop checks that the hover height is reachable over the hand zone and prints the points that are not (with the placeholder: 35 cm out).
- **Drop needs two hands** and the tip within 3 cm (horizontal) of their midpoint for 1 s; one hand is followed but never dropped into. All three numbers are flags.
- **Unreachable or badly solved targets hold position** (IK result checked with FK, 2 cm tolerance) instead of moving to the nearest pose.
- **The tip moves in straight Cartesian waypoints** (8 mm per tick, each clamped to the workspace, then IK, then at most 1.5° per joint and tick). The first version stepped joints straight to the target's IK solution: in dry run the tip dipped to 0.136 m with hover 0.18 m between targets. With waypoints it stayed 0.205-0.221 m at hover 0.22 m. This assumes the IK stays near its seed. The placeholder had to be changed to keep the seed's pitch, otherwise it jumped pitch and dipped the tip to 0.107 m. Tian's IK picks the steepest pitch, so it should be checked for the same jump.
- **HOME is the pose at start** unless `--home` is given, because the demo starts every skill from HOME.
- **Robot mode is refused while `kinematics.PLACEHOLDER` or `following.PLACEHOLDER` is True.**

## Alternatives

- **LeRobot's `max_relative_target`** as the speed limit - not used: it reads the present position on every `send_action`, which slows the loop, and our own per-tick step limit does the same job.
- **Hover a few cm above the hand's z, for hands about 30 cm above the table** (Jakob's eyeball estimate, not measured) - rejected 2026-11-21. Tian's reach table from the URDF (board comment 7061302163) and a URDF scan on Jakob's Mac agree: with the gripper within 45° of straight down the tip reaches at most about 0.21 m. A tip at 0.33-0.38 m needs a roughly horizontal or upward gripper. Straight down is reachable at no hover height. Hands are held 10-15 cm above the table instead, 15-35 cm in front of the base, with the tip at 20-25 cm.

## Revisit if

- The table check (`docs/hand-following-table-check.md`) shows FK errors above about 2 cm.
- The tip dips too low on the real arm between targets; then limit in Cartesian space instead of per joint.
