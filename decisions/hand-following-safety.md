# How does the classical hand-following loop keep the ball and the people safe?

- Status: DECIDED (2026-11-21). Built by Jakob's agent with Tian's kinematics and follower (`scripts/handover/`), tested in dry run, then run on the arm by Balazs (T1, T2) and inside the demo chain's GIVE from the same evening.

## Decision

One process owns the robot connection and runs a 30 Hz loop: newest hand message, following/drop decision, inverse kinematics, then per-joint speed limit and `send_action`. The safety rules live in the loop, not in the decision logic, so they hold whatever Tian's following functions decide.

## Why

- **The holding gripper value is sent every tick until the drop.** A lost hand, an unreachable target or a solver failure only changes the arm goal, never the gripper. The value defaults to the gripper's `Goal_Position` at start, so the grip is not loosened by guessing a number (`CLAUDE.md`: never loosen while held).
- **Torque stays on at exit** (`disable_torque_on_disconnect=False`): the default disconnect switches torque off and would drop the ball (`research/lerobot-multi-skill-runner.md`, section 2).
- **Fixed hover height above the table (`--hover-height`, default 0.28 m since 2026-11-21: run on the arm by Balazs with Tian's branch, "works fine", board 7064755928; was 0.22 m) instead of the hand's z.** One camera gives no real depth (board comment 7061160180), and hands have to stay low anyway, see Alternatives.
- **Workspace clamp: tip 0.20-0.30 m above the table (`--min-height`, default 0.20 m as in the command run on the arm; was 0.08 m), 0.12-0.38 m from the pan axis.** Under about 10 cm from the axis is unusable (self-collision not checked; board comments 7061318846, 7061323029). At start the loop checks that the hover height is reachable over the hand zone and prints the points that are not. With Tian's IK, all points at 0.22 m are reachable.
- **Drop: one open palm held still under the gripper for 1.5 s (Tian, 2026-11-21).**
  - The tip must be within 3 cm (horizontal) of the palm.
  - The dwell restarts if the palm moves more than 3 cm, or is unseen for more than 0.5 s; shorter gaps are tolerated.
  - Settings: `drop_radius`, `drop_dwell_s`, `drop_hands`, `drop_gap_s`.
  - Rejected: the first rule, two palms side by side for 1 s ("one hand is followed but never dropped into"). On the arm the side camera saw two open palms together in only 2 of about 830 tracker lines, while one palm was tracked well (board 7065772934). The drop fired once, after about 15 s.
  - Also rejected: two palms with gaps tolerated, for the same reason.
  - `tests/handover_drop_rule.py` covers: a still palm drops; a palm moving 6 cm does not; 0.3 s gaps still drop; 0.8 s gaps restart the dwell.
  - The tracker sends only open, palm-up hands (`--only-open`), so the loop does not filter `palm_up` itself.
- **Unreachable or badly solved targets hold position** (IK result checked with FK, 2 cm tolerance) instead of moving to the nearest pose.
- **After the drop (Tian, 2026-11-21, asked for by Balazs after T2):** the gripper opens, `--celebrate N` fast close-open cycles (0.3 s strokes) with the arm still, then a plain joint move to `--rest` (Balazs's reading `-3.69 -103.43 97.01 -100.26 1.45`) and the loop stays there. From every hover point in the hand zone that joint move only rises (`tests/handover_celebrate_rest.py`). Both ran on the arm inside the demo chain's GIVE, where the supervisor's REST does the fold (`decisions/supervisor.md`). "Hand lost" still returns to the start pose.
- **Hand lost: hold, then back to the start pose after 5 s (`--lost-to-home`, was 3 s; Tian, 2026-11-21).** In T2 on the arm the tracker printed `no hand` in 63 of 157 lines while the gripper hovered over the palm (board 7064755928). The arm hides the palm from the side camera, so short losses are normal, and 3 s sent the arm home during them.
- **The tip moves in straight Cartesian waypoints** (8 mm per tick, each clamped to the workspace, then IK, then at most 1.5° per joint and tick). The first version stepped joints straight to the target's IK solution: in dry run the tip dipped to 0.136 m with hover 0.18 m between targets. With waypoints it stayed 0.205-0.221 m at hover 0.22 m. This assumes the IK stays near its seed. The placeholder had to be changed to keep the seed's pitch, otherwise it jumped pitch and dipped the tip to 0.107 m. Tian's IK did jump (up to 16° between waypoints). His version keeps 20° of elbow bend, which brings that down to at most 8.5° (`tests/handover_kinematics.py`).
- **Each joint step is guarded:** a step is shortened (1, 1/2, 1/4, 1/8, else no move) if it would bring the tip closer to the table (`--min-height`) or the base (0.12 m) than both where it is and where it is going. While the 1.5° limit spreads an 8° IK step over several ticks, the tip leaves the waypoint line. In the dry run with Tian's files, the tip was at most 7.6 mm from the waypoint the loop asked for (95th percentile 5.8 mm), and never outside the clamp.
- **HOME is the pose at start** unless `--home` is given, because the demo starts every skill from HOME.
- **The robot only moves with an explicit `--robot`** (and `--port`, `--top-camera`); `--dry-run` or `--robot` is required. The first version keyed this on a `PLACEHOLDER` flag in the kinematics, which Tian's files do not have.
- **Joint limits come from our calibration file** after connecting (`kinematics.joint_limits_from_calibration`): wrist_flex hovers about 3° from its URDF limit, and our calibrated range may be smaller.

## Alternatives

- **LeRobot's `max_relative_target`** as the speed limit - not used: it reads the present position on every `send_action`, which slows the loop, and our own per-tick step limit does the same job.
- **Hover a few cm above the hand's z, for hands about 30 cm above the table** (Jakob's eyeball estimate, not measured) - rejected 2026-11-21. Tian's reach table from the URDF (board comment 7061302163) and a URDF scan on Jakob's Mac agree: with the gripper within 45° of straight down the tip reaches at most about 0.21 m. A tip at 0.33-0.38 m needs a roughly horizontal or upward gripper. Straight down is reachable at no hover height. Hands are held 10-15 cm above the table instead, 15-35 cm in front of the base, with the tip at 20-25 cm.

## Revisit if

- A touch check (`docs/hand-tracking.md`) shows FK errors above about 2 cm.
- The tip dips too low on the real arm between targets; then limit in Cartesian space instead of per joint.
