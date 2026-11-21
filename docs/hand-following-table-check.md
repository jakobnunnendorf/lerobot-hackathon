# Table check: does the kinematics match the real arm?

For whoever is at the table. Takes about 15 minutes. It settles, per joint, the sign and the zero offset between the forward kinematics (FK) in `scripts/handover/kinematics.py` and the real SO-101, before `scripts/handover/handover_loop.py` is ever run without `--dry-run`.

Needs: the follower arm, a ruler or tape measure, a pen to mark the table, and the robot laptop. **No ball in the gripper**: the script switches torque off.

## 0. The frame

Origin on the table straight below the shoulder_pan axis. x points forward (where the arm points at shoulder_pan = 0), y to the robot's left, z up. Metres.

Mark on the table, with tape:
- point **A**: x = 0.20, y = 0.00 (20 cm straight in front of the pan axis)
- point **B**: x = 0.20, y = +0.10 (20 cm forward, 10 cm to the robot's left)

## 1. Start the reader

On the robot laptop, in the repo (replace the port):

```bash
uv run python scripts/handover/read_pose.py --port /dev/ttyACM0 --robot-id mi_follower
```

It connects, **switches torque off** (hold the arm, it will sag), and prints the five joint angles, the gripper and the FK tip position twice a second. If it prints the placeholder warning, the tip column is not the real FK yet: still write the joint values down, they are useful once Tian's FK is in. Stop with Ctrl-C.

## 2. Three poses

Hold each pose still for a few seconds and copy one printed line into the table below. Measure the real gripper tip (the point between the fingertips) with the ruler.

| Pose | How to hold it | Expected real tip |
| --- | --- | --- |
| P1 straight up | every link vertical, gripper pointing at the ceiling | x ≈ 0, y ≈ 0, z = measured height of the tip |
| P2 straight forward | upper arm, forearm and gripper horizontal, pointing forward | y ≈ 0, z = measured height, x = measured distance from the pan axis |
| P3 tip on A, then on B | gripper pointing down, fingertips touching the mark | A: (0.20, 0.00, 0.00); B: (0.20, 0.10, 0.00) |

Record:

| Pose | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | FK tip x, y, z | Ruler tip x, y, z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | | | | | | | |
| P2 | | | | | | | |
| P3 on A | | | | | | | |
| P3 on B | | | | | | | |

## 3. Reading the result

- **All poses within about 2 cm**: done. Post the filled table on the board.
- **P3 on B gives FK y negative**: shoulder_pan has the wrong sign.
- **P3 on A gives FK y not near 0**: shoulder_pan has an offset; it is the printed shoulder_pan in degrees.
- **P1 is not straight up in FK** (x or z clearly off): the joint angles in P1 are the offsets of shoulder_lift, elbow_flex and wrist_flex relative to the FK's own "straight up" angles. Compare each with the angle the FK expects for vertical links.
- **P2 has FK z far below or above the ruler, or FK x behind the base**: the sign of shoulder_lift or elbow_flex is flipped. Flip one at a time in the FK and recompute P1 and P2 from the recorded joint values (no need to move the arm again).
- wrist_roll does not change the tip position; ignore it here.

Post the filled table and the conclusions on the board (#1) for Tian's and Jakob's agents. They fix the signs and offsets in code; nobody edits the calibration file for this.
