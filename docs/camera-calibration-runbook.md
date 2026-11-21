# Top camera calibration: runbook for the table

What it gives: `scripts/calib/intrinsics.yaml` (lens) and `scripts/calib/extrinsics.yaml` (where the top camera is in the robot frame). With both, `scripts/handtrack/hand_tracker.py` sends hand positions in metres in the robot frame. Nothing here moves the robot.

Robot frame: origin on the table straight below the shoulder_pan axis, x forward (where the arm points at shoulder_pan = 0), y to the robot's left, z up, metres.

## Before you start

- **Any laptop can do it.** A calibration belongs to the camera, its mode (640x480, MJPG) and where it is mounted, not to the computer: frames captured on one laptop give files that are valid on the robot laptop. Move only the USB cable, never the camera.
- **Only one program can use the camera**: stop the robot program (teleop, recording, policy runs) when the camera is on the robot laptop.
- **Opening the top camera.**
  - In WSL: attach it with usbipd (`bash scripts/so101/attach_usb.sh` from Git Bash, on the `molnrb/so101-robot-skill` branch until it is merged). Its path is the config's `/dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0`, the same on every laptop.
  - In native Windows: a number, `0`, `1`, `2`, ... (`--camera 1` for the live scripts, or the first argument of `save_board_frames.py`). The laptop's own webcam is usually 0. **Look at the first saved frame** to make sure it shows the table from above and not your face.
  - Every live script prints the size it got and stops if it is not 640x480. It also tries to switch autofocus off and says whether the driver accepted it; "could not be switched off" is normal for a fixed-focus webcam. Check once with `v4l2-ctl -d <device> --list-ctrls` in WSL: if a `focus_auto` or `focus_automatic_continuous` control exists, it must be 0 whenever the robot program runs too.
- **Which environment.**
  - Computing from saved frames (`calibrate_intrinsics.py` without `--preview`, `calibrate_extrinsics.py --image ... --no-window`) needs only numpy, OpenCV and pyyaml. It runs in the repo's uv environment (`uv run python ...`; LeRobot already brings opencv-python-headless 4.13, numpy 2.2.6 and pyyaml 6.0.3) and in the environment below; both were checked to give the same numbers.
  - Capturing without a window: `save_board_frames.py` needs only cv2 and numpy, so it runs in the uv or the conda `lerobot` environment. **Use it whenever OpenCV is the headless build** (no `cv2.imshow`).
  - Everything with a window (`capture_intrinsics.py`, `--preview`, the confirmation window, method B, `validate.py`) and the hand tracker need their own environment, once:
    ```bash
    python3 -m venv ~/calib-venv
    ~/calib-venv/bin/pip install -r scripts/handtrack/requirements.txt -r scripts/calib/requirements.txt
    ```
    It installs mediapipe 1.2.3, numpy 2.5.5, opencv-contrib-python 5.1.0.95 and pyyaml 6.0.4 (checked on Tian's laptop, `pip check` clean). In WSL, if `import cv2` fails with `libGL.so.1`: `sudo apt install libgl1 libglib2.0-0`. On Windows: `py -3.12 -m venv .venv-calib` and `.venv-calib\Scripts\pip install ...`.
- All commands below run from the repo root.

## 1. Intrinsics (lens), about 10 minutes

The board is held under the mounted top camera; the camera does not move.

**Capture 20-30 frames.** Without a window (headless OpenCV, for example the uv or conda `lerobot` environment), `save_board_frames.py` saves a frame every 2 s when the whole board is found and has moved since the last one, beeps, and prints which image areas are covered so far (top row first):
```bash
python scripts/calib/save_board_frames.py /dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0 calib_frames/intrinsics 25   # WSL
python scripts/calib/save_board_frames.py 1 calib_frames/intrinsics 25                                                                 # Windows, camera 1
```
With a window:
```bash
python scripts/calib/capture_intrinsics.py --out-dir calib_frames/intrinsics            # add --camera 1 on Windows
```
It shows the camera with the detected corners drawn in colour and a grid of image cells: red = no saved frame has put corners there yet, yellow = one, green = two or more. SPACE saves a frame only when the whole board is found and held still ("NOT saved" otherwise), q quits. Running it again adds to the same folder (default `data/calib/intrinsics/`).

How to move the board: the whole board (including some white margin) always in the image; centre, all four image corners and all four edges (the image corners matter most); flat, and tilted 30-45 degrees towards each side; some poses near the table, some 30-40 cm below the lens; hold it still at each pose. Fill the red cells.

**Calibrate** (no camera or window needed; works on any laptop with the saved frames):
```bash
python scripts/calib/calibrate_intrinsics.py --images calib_frames/intrinsics
```
It prints the error of every frame, drops frames above twice the median (and above 0.25 px), calibrates again and writes `scripts/calib/intrinsics.yaml` (`--out` for another file).
- **RMS below 0.5 px: good** ("GOOD" is printed when there is no warning). 0.5-1 px usable. **Above 1 px: capture again.**
- "Per-image errors are inconsistent": usually a bent or wavy board. Flatten it on its backing and capture again.
- "image cells never covered": add frames with the board in those image corners.

**Check by eye** (live): `python scripts/calib/calibrate_intrinsics.py --preview` shows the raw and the undistorted image side by side with straight guide lines; table edges and the board's lines must look straight on the right.

## 2. Extrinsics (camera position), method A: board on the table, about 10 minutes

**The board.** It is taped to cardboard, and one cardboard corner carries a square of red tape. Name the four outermost inner corners (the points where four squares meet, not the paper corners):
- **1**: the one nearest the red tape. This is the board's origin.
- **2**: from 1 along the long side (six squares, about 172.5 mm).
- **3**: diagonally opposite 1.
- **4**: from 1 along the short side (four squares, about 116 mm).

No rotation sense is assumed: the script works out from the four measurements whether 1-2-3-4 runs clockwise or counter-clockwise seen from above.

**Lay it down.** Flat, in the area where hands will be held, fully in the top camera's view, taped down. Keep the arm and anything orange or red away from the board's corners: the script looks for the red tape there, and the arm is orange.

**Save one frame**: `python scripts/calib/save_board_frames.py <device or number> calib_frames/table 1`. Or let `calibrate_extrinsics.py` read the live camera itself (`--camera 1` on Windows; it then averages 10 frames).

**Measure, in millimetres in the robot frame**, with tape lines on the table along robot x (forward from the point below the shoulder_pan axis) and robot y (to the robot's left):
- x and y of corners 1, 2, 3 and 4 (y is negative to the robot's right);
- the thickness of board plus cardboard;
- the camera lens position, to about 5 cm (x, y and height);
- a phone photo of the table with the board and the tape lines.

The script checks the four corners against the board's known size and prints how far each one is off the fitted board. **Up to 2-3 mm is good. Above 5 mm** means a mismeasured corner, swapped numbers (it says so if renumbering fits) or a board that moved.

**Compute** (no camera needed when you give `--image`):
```bash
python scripts/calib/calibrate_extrinsics.py --image calib_frames/table/board_00.png \
    --corners-mm '1: 215.0, 82.0' '2: 216.0, -90.5' '3: 331.0, -89.0' '4: 330.0, 83.5' \
    --thickness-mm 6 --camera-position-mm 300 -30 720
```
The same values can go into `scripts/calib/config.yaml` under `extrinsics` instead (then just `--image ...`, or nothing for the live camera). The old way, corner 1 plus the direction from 1 to 2 (`board_origin_in_base_xyz_m`, `board_yaw_in_base_deg`), still works when the four corners are not available.

**Corner 1 versus corner 3.** The pattern looks the same turned by 180 degrees, so the image alone cannot tell corner 1 from corner 3, and getting it wrong puts the camera at the mirror position (in the synthetic test 17-50 cm off, turned by 180 degrees). The script uses up to three tie-breakers, and every one that decides must agree, or it refuses:
1. the **red tape**, found automatically at the board's corners;
2. `--origin-image-corner top-left|top-right|bottom-left|bottom-right`: look at the frame, and name the image corner that corner 1 lies towards, seen from the middle of the board. Needed only when the tape is not found (out of the image, covered, or the arm next to it);
3. `--camera-position-mm X Y Z`: decides when the camera is not almost straight above the board's centre. It also catches swapped numbers 1 and 3.

**Confirm.** The script writes `scripts/calib/extrinsics_overlay.png` and shows it in a window (`--no-window`: look at the PNG and answer in the terminal). Check that **'1' sits next to the red tape**, 2, 3 and 4 follow the naming above, and **the robot axes (red x, green y, 10 cm long) start at the point below the shoulder_pan axis and point forward and left**. `y` saves; anything else cancels.

What it prints, and what is good: the corner residuals (above); board reprojection below 0.5 px; the camera position and its tilt from vertical, which must match the tape measure to a few centimetres ("WARNING: the camera comes out ... from the measured lens position" if it is more than 10 cm off). It stops with an error if the camera would end up below the table or looking up.

## 3. Method B: touching points with the gripper (validation)

Use it after method A, to check calibration and kinematics together. **The kinematics in `scripts/handover/kinematics.py` is not verified on the arm yet** (joint signs and zero offsets are open), so a disagreement between A and B can come from either.

**The point to touch and click.** The kinematics' tool point is the origin of the URDF's `gripper_frame_link`: on the inner face of the fixed jaw, 6.3 mm short of the jaw's end (read from the URDF by a teammate; `research/so101-inverse-kinematics.md` says 6 mm short of the end and 7.9 mm off the wrist_roll axis). `touch.tool_point_m` in the config is `(0, 0, 0.0063)`, the **end of the fixed jaw**, so touch and click the end of the fixed jaw. The sign of the 6.3 mm is checked in the URDF model but not yet on the arm. Check it in step 4: the magenta ring must sit on the jaw's end. If it sits about 1.3 cm back along the jaw, change it to `-0.0063`.

**Joints.** Either live, in a second terminal in the LeRobot environment:
```bash
python scripts/calib/joint_feed.py --port /dev/so101_follower --robot-id mi_follower
```
It only reads positions: no calibration, no torque change, nothing sent to the arm. Or paste one line per point from the teammate's `read_pose.py` (careful: `read_pose.py` switches the torque off, so the arm goes limp; hold it).

Joint values are only right with the arm's own LeRobot calibration file for `mi_follower` (on the robot laptop under `~/.cache/huggingface/lerobot/calibration/robots/so_follower/mi_follower.json`). On another laptop, **copy that file from the robot laptop; never run a new calibration** (it would change what every recorded dataset and policy means by a joint angle). Or read the joints on the robot laptop and paste them.

**Run:**
```bash
python scripts/calib/calibrate_extrinsics_touch.py --joints-file      # live joints
python scripts/calib/calibrate_extrinsics_touch.py                    # paste joints in the terminal
```
For each point: hold the fixed jaw's end still at the point, press SPACE (records the joints), then click the jaw's end in the image (the window is 1.5x larger for precise clicks). u undoes, s solves and saves, q quits. Every point is kept in `data/calib/touch_pairs.yaml`, so `--solve-only` solves again later without the camera, for example after the kinematics offsets are fixed.

**Take 8-12 points spread over the hand area, half on the table and half held 10-25 cm above it.** Points all on the table leave the camera's tilt loose: in the synthetic test 8 table points gave 39-54 mm error at hand height, 8 points spread over 0-25 cm gave 7-11 mm. The script warns when the heights span less than 10 cm.

It writes `scripts/calib/extrinsics_touch.yaml`, which does not replace method A, prints the reprojection error per point (a few px is normal; a single point far above the rest is a misclick, so undo it), and compares with `extrinsics.yaml`: camera position and angle difference, and how far apart the two put the same pixel at hand height. To make B the active calibration: `--out scripts/calib/extrinsics.yaml`.

## 4. Check it live

```bash
python scripts/calib/validate.py --joints-file
```
It shows yellow dots on the table every 5 cm, labelled every 10 cm in robot metres, and the robot axes. With `--joints-file` (or j to type joints) it adds the magenta ring at the gripper's tool point, with a dot on the table straight below it. Clicking a pixel prints its robot-frame position on the table (`--plane-z` for another height) and at hand height. Put a coin on a labelled dot and check that it lands within about 1 cm, then move the arm by hand and check that the ring follows the jaw's end. With `--hand-model hand_landmarker.task` it also prints the wrist and index fingertip of detected hands at the hand height (0.15 m, `validate.hand_plane_z_m`).

Then the tracker:
```bash
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml --hand-z 0.15
```

## Rules that keep the calibration valid

- **Calibrate at the size the robot program uses (640x480).** Intrinsics are invalid at any other size; the scripts and the tracker refuse a size mismatch.
- **If the top camera moves or is bumped, redo the extrinsics** (section 2). The intrinsics stay valid unless the focus or zoom changed. Redoing the intrinsics also requires redoing the extrinsics; the tracker refuses an extrinsics file made with older intrinsics.
- **The board must be flat.** A bent board shows up as inconsistent per-frame errors in section 1.
