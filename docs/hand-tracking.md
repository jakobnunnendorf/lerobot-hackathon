# Hand tracking and hand following

The handover uses the top camera to find an open palm (MediaPipe Hand Landmarker, `scripts/handtrack/hand_tracker.py`), turns its pixel into a position in the robot frame with the camera calibration (`docs/camera-calibration.md`), and moves the gripper above it with our own inverse kinematics (`scripts/handover/kinematics.py`). In the demo this runs inside the supervisor's GIVE skill (`docs/demo.md`). Why it works this way: `decisions/hand-following-geometry.md` and `decisions/hand-following-safety.md`.

This page covers checks you need after changing the camera, the calibration or the arm. Setup of the `handtrack` env and `hand_landmarker.task`: `docs/setup.md`.

**The hand:** one flat, open hand, palm up, about 20 cm above the table, 15-35 cm in front of the robot base, where the top camera sees it. Cupped hands and fists are not followed: MediaPipe often finds no hand in cupped hands. The arm cannot hover over hands held much higher.

## 1. Tracker check, the arm does not move (about 10 minutes)

Stop any robot program: the tracker opens the top camera itself here.

Terminal 1, the hand-following loop in dry run. It only prints what it would command:

```bash
conda activate lerobot
python scripts/handover/handover_loop.py --dry-run --hover-height 0.28 --print-every 15
```

Terminal 2, the tracker with its own camera:

```bash
conda activate handtrack
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml \
    --hand-z 0.20 --camera
```

Check:

- Terminal 2 prints `camera: 640x480 at 30 fps, backend V4L2`, then twice a second the hand in robot coordinates, e.g. `x +0.251 y -0.048 m open palm`.
- Hold the open palm 20 cm above a point you can measure, e.g. 25 cm straight ahead of the base's turning axis: x reads about 0.25 and y about 0.00, within about 2 cm. Move it 10 cm to the robot's left: y grows by about 0.10.
- Terminal 1 shows `follow` while the hand is seen, with `tip=(x, y, +0.280)` moving with the hand; `hold` when the hand disappears; `home` after 5 s.

`--hand-z` is the palm height the tracker assumes (m). If people hold their hands at another height, measure it and change `--hand-z`.

Ctrl+C in both terminals.

## 2. Kinematics check against the real arm (torque off, about 15 minutes)

Do this after the arm is rebuilt or recalibrated. It confirms that the joint signs and zero offsets in `scripts/handover/kinematics.py` match the arm, at table height and at hover height. A wrong sign moves the arm the wrong way, and the safety limits use the same kinematics.

`read_pose.py` connects, **switches torque off** (hold the arm), and prints twice a second the five joints, the gripper, and `jaw tip (m) x= y= z=`: the end of the fixed jaw, on its inner edge (the side facing the moving jaw). No ball in the gripper.

```bash
conda activate lerobot
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

**At table height, with the calibration board.** While the board from the extrinsics step is still taped down and measured (`docs/camera-calibration.md`, corners 1-4):

1. Put the inner edge of the fixed jaw's end on each corner in turn, gripper as steeply down as the arm allows. Hold it still for two printed lines and copy a whole line.
2. Write `corners.txt`, one line per corner: corner number, ruler x, y in mm, `|`, the copied line:

   ```text
   1: 320, 86 | shoulder_pan= -15.04  shoulder_lift=  55.48  elbow_flex= -51.47  wrist_flex=  62.98  wrist_roll=   0.00  gripper= 12.00  ...
   2: 320, -86 | ...
   ```

3. Run it (any laptop, numpy only):

   ```bash
   python scripts/handover/check_corners.py corners.txt
   ```

   Per corner it prints the ruler position, the kinematics position and the error in cm, then the board's side lengths from both, and a verdict.

Reading it:

- **Every error under 1 cm:** the kinematics matches. Ours came out 4-11 mm after the offsets were fitted.
- **About the same error at all four corners:** a joint zero offset, or the ruler origin is off. Side lengths close to 172 and 116 mm mean the origin; side lengths clearly off mean a joint offset.
- **Errors in different directions:** a joint has the wrong sign. A flipped shoulder_pan swaps left and right: y errors of opposite sign on the two sides of the board.

Fix signs and offsets in `kinematics.py`, never in the calibration file. The same corner lines also feed `cross_check.py --corners-file`.

**At hover height.** Hold the fixed jaw's end about 25 cm above a point you can measure and compare the printed `x`, `y`, `z` with the ruler. Within about 3 cm is fine. At the folded rest pose the model is weaker (jaw end at 35 cm, model 31 cm), which is why handover moves start from the raised present pose.

## 3. Hand following on its own (optional)

The demo runs following inside GIVE. To test it without a pick, `handover_loop.py` runs the same loop on the arm. Someone at the power switch, no ball, nothing else in reach.

Hold the arm raised, about 25 cm above the table (`read_pose.py` leaves it limp), then in the same terminal:

```bash
python scripts/handover/handover_loop.py --robot --port /dev/so101_follower \
    --top-camera /dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0 --robot-id mi_follower \
    --hover-height 0.28 --min-height 0.20 --max-step-m 0.004 --max-step-deg 1.0 --print-every 15 \
    --celebrate 3 --rest -3.69 -103.43 97.01 -100.26 1.45
```

- Once it prints `listening on 127.0.0.1:8766; home=...`, torque is on and the arm holds itself: let go. The pose at start is where it returns when the hand is lost.
- It must connect without a calibration. If it asks `Press ENTER to use provided calibration file`, press Enter.
- `--min-height 0.20` keeps every target at least 20 cm above the table. `--max-step-m 0.004 --max-step-deg 1.0` halve the speed.
- To hold a ball, put it between the open jaws before starting and pass `--gripper-hold 5`: the gripper closes at start and keeps that goal every tick until the drop.

Terminal 2: the tracker as in section 1, **without** `--camera` (it reads the loop's frames).

Expect: no hand, the arm holds. An open palm at 20 cm: the gripper rises to 28 cm, moves over the palm and follows it slowly. Hand gone: it stops, and after 5 s returns to the start pose. One open palm held still under it for 1.5 s: it opens (the drop), closes and opens 3 times, folds back to `--rest` and stays there.

**Stop: Ctrl+C in terminal 1.** Torque stays on, so a held ball stays held. If the arm moves **away from the hand or mirrored** (hand left, arm right), stop at once: a joint sign is wrong (section 2).

If the tracker loses the palm while the gripper is above it (terminal 2 prints `no hand`), the arm hides it from the side-mounted camera: try `--hover-height 0.30`.
