# Table check: does the kinematics match the real arm?

For whoever is at the table. Takes about 15 minutes. It settles, per joint, the sign and the zero offset between the forward kinematics (FK) in `scripts/handover/kinematics.py` and the real SO-101, before `scripts/handover/handover_loop.py` is ever run without `--dry-run`.

Needs: the follower arm, a ruler or tape measure, a pen to mark the table, and the robot laptop. **No ball in the gripper**: the script switches torque off.

## 0. The frame

Origin on the table straight below the shoulder_pan axis. x points forward (where the arm points at shoulder_pan = 0), y to the robot's left, z up. Metres. In the URDF this is `base_link` shifted: the pan axis is 38.8 mm in front of the `base_link` origin, and the table is 2.4 mm below it (board comments 7061318846, 7061323029). Measure every distance from the point on the table straight below the pan axis, not from the edge of the base.

Mark on the table, with tape:
- point **A**: x = 0.20, y = 0.00 (20 cm straight in front of the pan axis)
- point **B**: x = 0.20, y = +0.10 (20 cm forward, 10 cm to the robot's left)

## 1. Start the reader

On the robot laptop, in the repo (replace the port):

```bash
uv run python scripts/handover/read_pose.py --port /dev/ttyACM0 --robot-id mi_follower
```

It connects, **switches torque off** (hold the arm, it will sag), and prints the five joint angles, the gripper and the FK tip position twice a second. The FK is Tian's URDF kinematics (`scripts/handover/kinematics.py`). Stop with Ctrl-C.

## 2. Three poses

Hold each pose still for a few seconds and copy one printed line into the table below. Measure the real gripper tip (the point between the fingertips) with the ruler.

| Pose | How to hold it | Expected real tip |
| --- | --- | --- |
| P1 straight up | every link vertical, gripper pointing at the ceiling | x ≈ 0, y ≈ 0, z = measured height of the tip |
| P2 straight forward | upper arm, forearm and gripper horizontal, pointing forward | y ≈ 0, z = measured height, x = measured distance from the pan axis |
| P3 tip on A, then on B | gripper tilted down, fingertips touching the mark | A: (0.20, 0.00, 0.00); B: (0.20, 0.10, 0.00) |
| P4 turned left | from P2, turn only the base clearly (about 45°) to the robot's left | y clearly positive, x smaller than in P2 |

Record:

| Pose | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | FK tip x, y, z | Ruler tip x, y, z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | | | | | | | |
| P2 | | | | | | | |
| P3 on A | | | | | | | |
| P3 on B | | | | | | | |
| P4 | | | | | | | |

## 3. Signs, before anything else

From the recorded joint values alone (no FK needed), write down for each joint whether its reading goes up or down:

- **shoulder_pan**: P4 minus P2. Turning to the robot's left gives a positive or negative change? In the URDF a positive shoulder_pan turns the arm to the robot's **right** (negative y); whether LeRobot's calibrated degrees on our arm follow that is not known yet. This is the most important line of the check.
- **shoulder_lift**: P2 minus P1 (the upper arm tips forward).
- **elbow_flex**: P2 minus P1 (the forearm folds forward).
- **wrist_flex**: P3 minus P2 (the gripper tips down).

Post the four signs with the table.

## 4. Reading the result

- **All poses within about 2 cm**: done. Post the filled table on the board.
- **P3 on B or P4 gives FK y negative**: shoulder_pan has the wrong sign.
- **P3 on A gives FK y not near 0**: shoulder_pan has an offset; it is the printed shoulder_pan in degrees.
- **P1 is not straight up in FK** (x or z clearly off): the joint angles in P1 are the offsets of shoulder_lift, elbow_flex and wrist_flex relative to the FK's own "straight up" angles. Compare each with the angle the FK expects for vertical links.
- **P2 has FK z far below or above the ruler, or FK x behind the base**: the sign of shoulder_lift or elbow_flex is flipped. Flip one at a time in the FK and recompute P1 and P2 from the recorded joint values (no need to move the arm again).
- wrist_roll does not change the tip position; ignore it here.

Post the filled table and the conclusions on the board (#1) for Tian's and Jakob's agents. They fix the signs and offsets in code; nobody edits the calibration file for this.

## 5. The hand zone for the demo

The arm cannot hover over a hand held high: hands 30 cm above the table (an eyeball estimate) are out of reach (board comment 7061302163). The person handing over holds their hands **10-15 cm above the table and 15-35 cm in front of the pan axis**; the gripper then hovers at 22 cm, tilted 35-45° down. Mark that zone on the table.

**The drop gesture is two flat open palms side by side, palms up, not cupped hands.** MediaPipe found no hand in 3 of 3 photos of cupped hands (board comment 7061460387), and the tracker runs with `--only-open`, so a fist or the back of a hand is not followed.

Once, with a ruler: have the person hold their hands as they will in the demo and measure the palm height above the table. Post it on the board; if it is above 15 cm, the hover height (`--hover-height`) and the zone need another look.

## 6. Starting the hand following (only after sections 3-4 are settled)

The arm moves. Someone stands at the table with a hand on the power switch. Connecting switches torque off for a moment, so the ball is not in a closed gripper at start: hold it between the open jaws and pass `--gripper-hold` with a closing value a few below the 11-16 the gripper reads when holding (for example 5). The gripper closes at start and keeps that value every tick until the drop. Without `--gripper-hold`, the gripper keeps the goal it had at start.

1. Robot program first; it owns the top camera and shares its frames with the tracker:
   ```bash
   uv run python scripts/handover/handover_loop.py --robot --port /dev/ttyACM0 --top-camera 0 --robot-id mi_follower --gripper-hold 5
   ```
2. Then the tracker in its own environment (`scripts/handtrack/requirements.txt`), with `--only-open` and the calibration files from the checkerboard step; see `research/hand-tracking-and-camera-calibration.md`.

Ctrl-C stops the loop with torque left on, so a held ball stays held.
