# Top camera calibration

Gives `scripts/calib/intrinsics.yaml` (the lens) and `scripts/calib/extrinsics.yaml` (where the top camera sits in the robot frame). With both, the hand tracker turns a palm in the image into a position in metres that the arm can follow. Nothing here moves the arm. Why it works this way: `decisions/hand-following-geometry.md`.

The files in the repo belong to our camera at our mount. **If the top camera moves or is bumped, redo the extrinsics** (section 2). The intrinsics stay valid unless focus or zoom changed. New intrinsics need new extrinsics too; the tracker refuses an extrinsics file made with older intrinsics.

**Robot frame:** origin on the table straight below the shoulder_pan axis, x forward (where the arm points at shoulder_pan = 0), y to the robot's left, z up, metres. Measure every distance from the point below the pan axis, not from the edge of the base.

## Before you start

- **Any laptop can compute it.** A calibration belongs to the camera, its mode (640x480, MJPG) and its mount, not to the computer. Frames captured on one laptop give files valid on the robot laptop. Move only the USB cable, never the camera.
- **Calibrate at 640x480**, the size the robot program uses. Every live script stops if it gets another size.
- **One program per camera:** stop the robot program, teleop or recordings first.
- **Opening the top camera:**
  - in WSL: its by-id path, `/dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0` (the default in `scripts/calib/config.yaml`);
  - on native Windows: a number (`--camera 1`, or the first argument of `save_board_frames.py`). The laptop's own webcam is usually 0, so look at the first saved frame.
  - Autofocus must be off. Check once with `v4l2-ctl -d <device> --list-ctrls` in WSL: a `focus_auto` or `focus_automatic_continuous` control must be 0.
- **Environments:**
  - Computing from saved frames (`calibrate_intrinsics.py`, `calibrate_extrinsics.py --image ... --no-window`, `cross_check.py`) and capturing without a window (`save_board_frames.py`) need only numpy, OpenCV and pyyaml: the repo's uv env or the `lerobot` env work.
  - Anything with a window (`capture_intrinsics.py`, `--preview`, the confirmation window, method B, `validate.py`) needs the `handtrack` env (`docs/setup.md`); the other two have headless OpenCV.
- **The board:** an 8 x 6 checkerboard (7 x 5 inner corners, squares about 28.75 x 29 mm), taped flat to cardboard. One cardboard corner carries a square of red tape. A bent board shows up as inconsistent per-frame errors.

## 1. Intrinsics (the lens), about 10 minutes

Hold the board under the mounted camera; the camera does not move.

**Capture 20-30 frames.** Without a window, `save_board_frames.py` saves a frame every 2 s when the whole board is found and has moved since the last one, beeps, and prints which image areas are covered so far:

```bash
python scripts/calib/save_board_frames.py /dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0 calib_frames/intrinsics 25   # WSL
python scripts/calib/save_board_frames.py 1 calib_frames/intrinsics 25                                                                 # Windows, camera 1
```

With a window: `python scripts/calib/capture_intrinsics.py --out-dir calib_frames/intrinsics` (add `--camera 1` on Windows). It draws the detected corners and a grid of image cells: red = no saved frame has corners there yet, yellow = one, green = two or more. SPACE saves (only when the whole board is found and still), q quits.

How to move the board: the whole board with some white margin always in the image; centre, all four image corners and all four edges (the corners matter most); flat, and tilted 30-45 degrees towards each side; some poses near the table, some 30-40 cm below the lens; still at each pose. Fill the red cells.

**Calibrate** (no camera needed):

```bash
python scripts/calib/calibrate_intrinsics.py --images calib_frames/intrinsics
```

It drops frames above twice the median error, calibrates again and writes `scripts/calib/intrinsics.yaml` (`--out` for another file).

- **RMS below 0.5 px: good.** 0.5-1 px usable. **Above 1 px: capture again.**
- "Per-image errors are inconsistent": usually a bent board. Flatten it and capture again.
- "image cells never covered": add frames with the board in those image corners.

**Check by eye:** `python scripts/calib/calibrate_intrinsics.py --preview` shows the raw and the undistorted image side by side; table edges and the board's lines must look straight on the right.

## 2. Extrinsics (the camera's pose), board on the table, about 10 minutes

**Name the corners.** The four outermost inner corners (where four squares meet, not the paper corners):

- **1**: the one nearest the red tape. The board's origin.
- **2**: from 1 along the long side (six squares, about 172.5 mm).
- **3**: diagonally opposite 1.
- **4**: from 1 along the short side (four squares, about 116 mm).

**Lay it down** flat, taped, in the area where hands will be held, fully in the camera's view. Keep the arm and anything orange or red away from the board's corners: the script looks for the red tape there, and the arm is orange. `docs/calib-results/table_board_00.png` is a frame from our setup.

**Save one frame:** `python scripts/calib/save_board_frames.py <device or number> calib_frames/table 1`. Or let `calibrate_extrinsics.py` read the live camera (it averages 10 frames).

**Measure in millimetres in the robot frame**, with tape lines on the table along robot x and y:

- x and y of corners 1-4 (y is negative to the robot's right);
- the thickness of board plus cardboard;
- the lens position to about 5 cm (x, y, height).

**Compute:**

```bash
python scripts/calib/calibrate_extrinsics.py --image calib_frames/table/board_00.png \
    --corners-mm '1: 215.0, 82.0' '2: 216.0, -90.5' '3: 331.0, -89.0' '4: 330.0, 83.5' \
    --thickness-mm 6 --camera-position-mm 300 -30 720
```

The numbers are an example. The same values can go into `scripts/calib/config.yaml` under `extrinsics` instead.

It checks the four corners against the board's known size. **Up to 2-3 mm off is good; above 5 mm** means a mismeasured corner, swapped numbers (it says so if renumbering fits) or a board that moved.

**Corner 1 versus corner 3.** The pattern looks the same turned by 180 degrees, and a wrong guess puts the camera at the mirror position. Up to three tie-breakers decide, and all that decide must agree, or it refuses:

1. the red tape, found automatically;
2. `--origin-image-corner top-left|top-right|bottom-left|bottom-right`: the image corner that corner 1 lies towards, seen from the board's middle. Needed only when the tape is not found;
3. `--camera-position-mm X Y Z`: decides unless the camera is almost straight above the board's centre.

**Confirm.** It writes `scripts/calib/extrinsics_overlay.png` and shows it (`--no-window`: look at the PNG, answer in the terminal). Check that '1' sits next to the red tape, 2, 3 and 4 follow the naming, and the robot axes (red x, green y, 10 cm) start below the pan axis and point forward and left. `y` saves.

Good output: board reprojection below 0.5 px, and a camera position and tilt that match the tape measure to a few centimetres. It warns if the camera comes out more than 10 cm from the measured lens position, and stops if the camera would be below the table or looking up.

## 3. Independent cross-check

`scripts/calib/cross_check.py` recomputes intrinsics and pose from the same frames and ruler values with separate code, compares them with the YAML files, and prints one verdict. It opens no camera. Why: `decisions/calibration-cross-check.md`.

```bash
uv run python scripts/calib/cross_check.py \
  --intrinsics-frames calib_frames/intrinsics --table-frame calib_frames/table \
  --corners-mm '1: 215, 82' '2: 216, -90.5' '3: 331, -89' '4: 330, 83.5' \
  --thickness-mm 3 --lens-mm 300 -60 560 \
  --tian-intrinsics scripts/calib/intrinsics.yaml --tian-extrinsics scripts/calib/extrinsics.yaml
```

- `VERDICT: GOOD`: every check passed. `CHECK`: look at the lines marked CHECK. `BAD`: do not use the calibration. Exit code 0, 1 or 2.
- `--corners-file corners.txt` takes the corner lines from a file. A line `1: x, y | <read_pose.py line>` adds a jaw-tip touch of that corner (`docs/hand-tracking.md`).
- "corner labels 1-4 sit ... px from" BAD: the two scripts numbered the corners differently (1 and 3 swapped). Look at `calib_check/table_overlay.png`: corner 1 must be by the red tape.
- Without the red tape in the picture and without `--lens-mm`, the orientation cannot be decided: give `--tape-px X Y` or `--lens-mm X Y Z`.
- A flat board on the table cannot reveal a wrong focal length; only the intrinsics RMS and the lens position can.

Results land in `calib_check/`. Self-tests with synthetic pictures: `tests/calib_cross_check.py`, `tests/calib_synthetic.py`.

## 4. Optional: touching points with the gripper

Method B solves the camera pose from points the gripper touches, so it checks calibration and arm kinematics together. It writes `scripts/calib/extrinsics_touch.yaml` and does not replace section 2 unless you pass `--out scripts/calib/extrinsics.yaml`.

**The point to touch and click is the end of the fixed jaw** (`touch.tool_point_m` in `config.yaml`).

Joints, live, in a second terminal (`lerobot` env). It only reads positions; nothing is sent to the arm:

```bash
python scripts/calib/joint_feed.py --port /dev/so101_follower --robot-id mi_follower
```

Or paste lines from `read_pose.py`, which switches torque off: hold the arm. Joint values are only right with the arm's own calibration file (`docs/setup.md`).

```bash
python scripts/calib/calibrate_extrinsics_touch.py --joints-file    # live joints
python scripts/calib/calibrate_extrinsics_touch.py                  # paste joints in the terminal
```

Per point: hold the jaw's end still on it, SPACE (records the joints), click the jaw's end in the image. u undoes, s solves and saves, q quits. Points are kept in `data/calib/touch_pairs.yaml`; `--solve-only` solves again without the camera.

Take 8-12 points over the hand area, **half on the table and half held 10-25 cm above it**. Points all on the table leave the camera's tilt loose: in a synthetic test, 39-54 mm error at hand height against 7-11 mm with spread heights.

## 5. Check it live

```bash
python scripts/calib/validate.py --joints-file
```

Yellow dots on the table every 5 cm, labelled every 10 cm in robot metres, and the robot axes. With `--joints-file` (from `joint_feed.py`) a magenta ring marks the gripper's tool point. Clicking a pixel prints its position on the table (`--plane-z` for another height) and at hand height. Put a coin on a labelled dot: it should land within about 1 cm. Move the arm by hand: the ring should follow the jaw's end. With `--hand-model hand_landmarker.task` it also prints detected hands.

Then run the hand tracker (`docs/hand-tracking.md`).
