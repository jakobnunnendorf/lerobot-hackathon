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

It connects, **switches torque off** (hold the arm, it will sag), and prints a line twice a second: the five joint angles, the gripper, then two positions from Tian's URDF kinematics (`scripts/handover/kinematics.py`). Stop with Ctrl-C.

**Which point of the gripper is measured.** Touch marks with the **end of the fixed jaw** (the jaw that does not move when the gripper opens), on its **inner edge**: the edge on the side facing the moving jaw, in the middle of the jaw's width. Read the **`jaw tip`** numbers.
- `frame` is the URDF's reference point (`gripper_frame_link`). It lies on the inner contact face of the fixed jaw, centred, 6.3 mm back from the jaw's end, so it can never touch the paper itself. This was worked out from the official meshes (SO-ARM100, commit `141aea7`) by Kamel's agent (board comment 7061703346) and independently here.
- `jaw tip` is that point moved 6.3 mm forward along the gripper (`--tool-point 0 0 0.0063`, the default). That is the inner edge of the fixed jaw's end. The sign is checked in the model by `tests/handover_tool_point.py` (gripper 82° down: the jaw tip is 6.2 mm below the frame point) and by Kamel's agent (board comment 7061721093: 5.7-6.1 mm lower in jaw-down poses). `--tool-point 0 0 0` turns the shift off. A quick look at the table is welcome but not required.

## 2. Arm check with torque off (no calibration needed)

Do this while Tian's checkerboard is still taped down, right after its corners were measured with the ruler and before anyone moves it: the board is the ruler. The arm is never powered.

**Corner names** (Tian's board comment 7061830090, which replaces 7061715390): one corner of the cardboard has a square of **red tape**. The four points are the outermost inner corners of the pattern (where four squares meet, one square in from the edge):
- **1**: the one nearest the red tape
- **2**: from 1 along the **long** side, six squares (about 172 mm)
- **3**: diagonally opposite 1
- **4**: from 1 along the **short** side, four squares (about 116 mm)

Steps:
1. Start the reader as in section 1. No ball in the gripper.
2. For corners 1, 2, 3 and 4, one after another: put the **inner edge of the fixed jaw's end** on the corner, touching the paper. Point the gripper as steeply down as the arm allows. Hold it still for two printed lines. Then copy the **whole** printed line.
3. Write a text file `corners.txt` with one line per corner: the corner number, its ruler position `x, y` in **millimetres** in the robot frame (section 0; the same values as in the calibration step), a `|`, then the copied line:
   ```
   1: 320, 86 | shoulder_pan= -15.04  shoulder_lift=  55.48  elbow_flex= -51.47  wrist_flex=  62.98  wrist_roll=   0.00  gripper= 12.00  frame (m) ...
   2: 320, -86 | ...
   3: 204, -86 | ...
   4: 204, 86 | ...
   ```
4. Run (any laptop, needs only numpy):
   ```bash
   python scripts/handover/check_corners.py corners.txt
   ```
   Per corner it prints the ruler position, the kinematics position of the jaw end and the error in cm. Then come the board side lengths from the ruler and from the kinematics, the mean error, and a verdict.
5. Post `corners.txt` and the output on the board (#1).

Reading the result:
- **Every error under 1 cm:** the kinematics matches the arm.
- **About the same error at all four corners:** one joint has a zero offset, or the ruler origin (the point below the pan axis) is off.
  - Kinematics side lengths close to 172 and 116 mm mean the origin is off.
  - Side lengths clearly off mean a joint offset.
  - The four joint readings are what is needed to fit the offsets.
- **Errors pointing in different directions:** a joint has the wrong sign. A flipped shoulder_pan shows up as left and right swapped: y errors of opposite sign on the two sides of the board.
- Tried on the Mac with made-up readings:
  - exact readings: 0.0 cm;
  - a 4° shoulder_lift offset: the same error of 1.5-2.3 cm at all four corners, and the long sides come out 165-167 mm;
  - a flipped shoulder_pan: ±17 cm in y, side lengths correct.

## 3. Three poses

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

## 4. Signs, before anything else

From the recorded joint values alone (no FK needed), write down for each joint whether its reading goes up or down:

- **shoulder_pan**: P4 minus P2. Turning to the robot's left gives a positive or negative change? In the URDF a positive shoulder_pan turns the arm to the robot's **right** (negative y); whether LeRobot's calibrated degrees on our arm follow that is not known yet. This is the most important line of the check.
- **shoulder_lift**: P2 minus P1 (the upper arm tips forward).
- **elbow_flex**: P2 minus P1 (the forearm folds forward).
- **wrist_flex**: P3 minus P2 (the gripper tips down).

Post the four signs with the table.

## 5. Reading the result

- **All poses within about 2 cm**: done. Post the filled table on the board.
- **P3 on B or P4 gives FK y negative**: shoulder_pan has the wrong sign.
- **P3 on A gives FK y not near 0**: shoulder_pan has an offset; it is the printed shoulder_pan in degrees.
- **P1 is not straight up in FK** (x or z clearly off): the joint angles in P1 are the offsets of shoulder_lift, elbow_flex and wrist_flex relative to the FK's own "straight up" angles. Compare each with the angle the FK expects for vertical links.
- **P2 has FK z far below or above the ruler, or FK x behind the base**: the sign of shoulder_lift or elbow_flex is flipped. Flip one at a time in the FK and recompute P1 and P2 from the recorded joint values (no need to move the arm again).
- wrist_roll does not change the tip position; ignore it here.

Post the filled table and the conclusions on the board (#1) for Tian's and Jakob's agents. They fix the signs and offsets in code; nobody edits the calibration file for this.

## 6. The hand zone for the demo

The arm cannot hover over a hand held high: hands 30 cm above the table (an eyeball estimate) are out of reach (board comment 7061302163). The person handing over holds their hands **10-15 cm above the table and 15-35 cm in front of the pan axis**; the gripper then hovers at 22 cm, tilted 35-45° down. Mark that zone on the table.

**The drop gesture is two flat open palms side by side, palms up, not cupped hands.** MediaPipe found no hand in 3 of 3 photos of cupped hands (board comment 7061460387), and the tracker runs with `--only-open`, so a fist or the back of a hand is not followed.

Once, with a ruler: have the person hold their hands as they will in the demo and measure the palm height above the table. Post it on the board; if it is above 15 cm, the hover height (`--hover-height`) and the zone need another look.

## 7. Starting the hand following (only after sections 2-5 are settled)

The arm moves. Someone stands at the table with a hand on the power switch. Connecting switches torque off for a moment, so the ball is not in a closed gripper at start: hold it between the open jaws and pass `--gripper-hold` with a closing value a few below the 11-16 the gripper reads when holding (for example 5). The gripper closes at start and keeps that value every tick until the drop. Without `--gripper-hold`, the gripper keeps the goal it had at start.

1. Robot program first; it owns the top camera and shares its frames with the tracker:
   ```bash
   uv run python scripts/handover/handover_loop.py --robot --port /dev/ttyACM0 --top-camera 0 --robot-id mi_follower --gripper-hold 5
   ```
2. Then the tracker in its own environment (`scripts/handtrack/requirements.txt`), with `--only-open` and the calibration files from the checkerboard step; see `research/hand-tracking-and-camera-calibration.md`.

Ctrl-C stops the loop with torque left on, so a held ball stays held.
