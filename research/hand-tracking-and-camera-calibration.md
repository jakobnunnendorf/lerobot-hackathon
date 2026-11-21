# Hand tracking, hand position from one camera, checkerboard calibration and camera-to-robot transform

Researched 2026-11-21. Web sources were read through an automated page reader; docs.opencv.org refused the reader (403), so OpenCV facts come from the OpenCV source and docs in the `opencv/opencv` GitHub repository. Everything under "Checked on Tian's laptop" was run there (Windows 11, Ryzen 5 5500U, no GPU, Python 3.12.10) on public test photos; no webcam, microphone or robot was used. The laptop was running two other jobs at the same time, so the timings are pessimistic. Kinematics is covered in `research/so101-inverse-kinematics.md`, not here.

**Where the code is.** Tracker and sender are in the repo under `scripts/handtrack/` (`hand_tracker.py`, `hand_features.py`, `frame_link.py`, `hand_link.py`); the camera calibration is in `scripts/calib/` (runbook: `docs/camera-calibration-runbook.md`). Test scripts, photos and outputs named below with `data/handtrack/` are on Tian's laptop in the git-ignored `data/` folder.

## Summary

1. MediaPipe 1.2.3 installs with plain `pip install mediapipe` on Python 3.12 and runs the Hand Landmarker on this CPU at 35-75 ms per 640x480 frame (13-28 frames per second), on one core. With `num_hands=2` and only one hand in view it costs as much as two hands, because the palm detector then runs on every frame.
2. Palm up or palm down can be read from the landmarks (index knuckle, pinky knuckle, wrist and the left/right label); it was right on all 7 photos where the answer was clear. The Gesture Recognizer's `Open_Palm` does not separate palm from back of the hand and missed half of the open hands.
3. **Two cupped hands touching were not detected** in 3 of 3 photos at the default thresholds; two separate open hands were detected every time, in any rotation. Ask for two flat open palms side by side, not cupped hands. Lowering the thresholds to find cupped hands also produced the only false "hand" on photos of orange SO-100 arms.
4. One camera gives no reliable depth. Use the pixel ray and intersect it with an assumed palm height. The error is `(true height - assumed height) x tan(angle of the ray from vertical)`: 1-2 cm for a hand 15-20 cm high and 20-30 cm beside the point under a camera 70 cm up, up to 5-7 cm at the edges of a 10-25 cm height band. MediaPipe's metric "world landmarks" are not a usable depth source: the same hand came out 20% nearer when seen from the back than from the palm.
5. Calibration route: print OpenCV's 9x6 board (25 mm squares, A4), calibrate the top camera where it is mounted, lay the board flat on the table in the handover area and measure four corners with a ruler in the robot frame. About 30-40 minutes. (The team's board is 8 x 6 squares, 7 x 5 inner corners, instead; `docs/camera-calibration-runbook.md`.) Fallback without any geometry: teach 9-12 hover poses and interpolate joint angles from the palm's pixel position.
6. Frames should reach the tracker through shared memory written by the robot program: 0.2 ms per frame for the control loop, about 2 ms transport. Tracker and robot program both run inside WSL2. Sender and tracker are built and tested on photos (see the end).

## Recommended pipeline

1. The robot program keeps owning both cameras. Every second top-camera frame (RGB, as LeRobot delivers it) is copied into a named shared-memory block. Writing never waits for the reader, so a slow or crashed tracker cannot stall the 30 Hz loop.
2. A separate tracker process in its own virtual environment, in the same WSL2 Ubuntu as the robot program, takes the newest frame, runs the Hand Landmarker in `VIDEO` mode with `num_hands=2` and the default thresholds (0.5), and computes per hand: palm centre (mean of landmarks 0, 5, 9, 13, 17), palm up (geometry below), fingers extended.
3. It turns the palm-centre pixel into metres in the robot frame: undistort the pixel, turn it into a ray with the camera pose, intersect the ray with the plane `z = 0.15 m` (tune at the table).
4. It sends one JSON line per frame over one long-lived TCP connection to the robot program on `127.0.0.1:8766` (the contract agreed on the board): `{"t": ..., "hands": [{"x", "y", "z", "palm_up", "score"}]}`, plus additive fields `open`, `handedness`, `u`, `v`. Tested against the receiver on `jakobnunnendorf/handover-loop` (commit `90bb1d3`): accepted unchanged.
5. The robot side follows one hand at a fixed hover height and drops when two hands are seen close together for about a second. Three things to handle there: that receiver ignores `palm_up` and `open`, so a fist or the back of a hand counts as a hand unless the tracker runs with `--only-open` or the follow logic filters; smooth the position (median of the last 5); and expect the top camera to lose the hands when the arm is right above them (section 2, "occlusion").

## What the people at the table do with the checkerboard

Superseded by `docs/camera-calibration-runbook.md`. The team's board is 8 x 6 squares, so **7 x 5 inner corners**, marked with red tape at one corner; where this file says 9 x 6 it means OpenCV's own board, which was not used.

## 1. MediaPipe Hand Landmarker

**Version and install.** `pip install mediapipe` installed `mediapipe 1.2.3` with `numpy 2.5.5` and `opencv-contrib-python 5.1.0.95` (plus `absl-py`, `flatbuffers`, `sounddevice`, `matplotlib`) on Python 3.12 Windows. Run on Tian's laptop. PyPI lists wheels for Windows, Linux x86-64 and ARM64 (glibc 2.28+) and macOS ARM64 (https://pypi.org/project/mediapipe/, see `research/ball-picking-and-handover.md`). The old `mp.solutions.hands` API is gone; use the Tasks API (https://github.com/google-ai-edge/mediapipe/issues/6192).

**Model.** `hand_landmarker.task`, 7,819,105 bytes, a bundle of a palm detector (192x192 input) and a landmark model (224x224), float16, Apache-2.0. Download: https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full/resolve/main/hand_landmarker.task (model card: https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full). The older URL `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task` still serves a byte-identical file (same SHA-256, `fbc2a300...cde1`). Checked on Tian's laptop.

**Running mode for a live camera.** Three modes exist: `IMAGE` (every call stateless), `VIDEO` (`detect_for_video(image, timestamp_ms)`, synchronous, timestamps must increase) and `LIVE_STREAM` (`detect_async`, result in a callback, frames dropped while busy). In `VIDEO` and `LIVE_STREAM` the landmark box of one frame is reused for the next, and the palm detector runs again only when tracking is lost. Source: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker and https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python. Use `VIDEO` in a loop that always takes the newest frame: same tracking, no callback thread, no queue. Measured: with `num_hands=2` and one hand visible, `VIDEO` mode costs as much as `IMAGE` mode (75 ms), which shows the palm detector runs every frame until the second hand is found.

```python
import mediapipe as mp
vision = mp.tasks.vision
landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO, num_hands=2))
result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8_hxwx3), timestamp_ms)
```

Options and defaults: `num_hands` 1, `min_hand_detection_confidence` 0.5, `min_hand_presence_confidence` 0.5, `min_tracking_confidence` 0.5. Source: the Python guide above; the defaults match `inspect.signature` of the installed package.

**What it returns, per hand.** Source: the Python guide above.
- `hand_landmarks`: 21 points, `x` and `y` normalised to 0..1 by image width and height, `z` a relative depth with the wrist as origin. The depth was learned from synthetic images only (https://arxiv.org/abs/2006.10214), so it is not a distance.
- `hand_world_landmarks`: 21 points in metres, origin at the hand's geometric centre. The scale is the model's guess of an average hand (see section 2).
- `handedness`: "Left" or "Right" with a score, for the subject's hand; a mirrored image swaps the labels. Confirmed on the test photos: unmirrored photos of right and left hands got the right labels.
- No per-hand presence score is exposed in the Python result; the handedness score is the only score.

**Palm facing the camera, from the landmarks.** In image coordinates (x right, y down) take `a = p5 - p0` (index knuckle minus wrist) and `b = p17 - p0` (pinky knuckle minus wrist) and `c = a.x * b.y - a.y * b.x`. For a right hand the palm faces the camera when `c < 0`, for a left hand when `c > 0`. A downward-looking camera sees an upward-facing palm, so "palm faces camera" is "palm up". Limitation: a right palm and a left back of the hand look identical in 2D, so this is only as good as the left/right label, which the model takes from appearance. Result on the test photos: correct on all 7 hands where palm or back was clear (one right palm, one right back, one left palm, and two photos with one palm and one back each), and unchanged under 90 and 180 degree rotation. Implemented in `scripts/handtrack/hand_features.py`.

**Open hand.** For each finger, straightness = distance from knuckle to tip divided by the summed bone lengths, from the world landmarks (shape is fine even if scale is not). Above 0.9 counts as extended (thumb 0.85). Open palms gave 4-5 extended fingers, the pointing and victory photos 2-3, the thumbs-up 2. "Open" = palm up and at least 3 extended.

**Gesture Recognizer.** Built-in classes `None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou`, model at https://huggingface.co/litert-community/MediaPipe-GestureRecognizer/resolve/main/gesture_recognizer.task (https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer). On the test photos it said `Open_Palm` for a back of the hand as readily as for a palm (2 photos), and `None` on 5 of 9 clearly open hands. Not useful here; the landmark geometry is better.

**Failure cases.**
- *Two hands touching or cupped:* 0 of 3 photos of cupped hands gave any hand at the default thresholds (monochrome cupped hands seen from above, cupped hands full of sweets, two decorated hands). At threshold 0.1 at most one hand appeared, mostly with the wrong label. Two open hands apart were found in every photo and rotation. The paper says palm detection was chosen partly so that non-maximum suppression copes with two-hand occlusion like handshakes (https://arxiv.org/abs/2006.10214), but cupping hides the palms themselves. Whether tracking in `VIDEO` mode keeps both hands when they come together after being found apart is **[unverified]**: plausible from the tracking design, test it at the table.
- *Camera looking straight down:* a palm-up hand seen from above is a frontal palm view at some in-plane rotation; detection was stable under 90/180/270 degree rotation of the test photos. A hand held edge-on (handshake pose) from above shows little palm and is likely missed **[unverified]**.
- *Hand partly hidden by the arm:* the model card says blur, heavy self-occlusion, low light and camera noise increase the error, and holding occluding objects is out of scope (https://huggingface.co/litert-community/MediaPipe-HandLandmarker-Full). When the gripper hovers right above the hands, a top camera above it may lose them. Plan for it: freeze the last good position when the tip is within about 5 cm of it, and place a dedicated camera off the arm's approach path if needed (section 5).
- *Gloves and heavy decorations:* out of scope by the model card. Two hands with dense henna patterns were not found at the default threshold.
- *The orange robot:* on 6 photos of SO-100/SO-101 arms (orange and yellow arms, a black SO-101, a wrist-camera view of the gripper) from https://github.com/TheRobotStudio/SO-ARM100/tree/main/media there was no detection at thresholds 0.5 or 0.3, and one false "hand" at 0.1 on the orange/yellow pair. Keep the thresholds at 0.5. Live frames with the arm moving are **[unverified]**.

**Speed.** Google reports 17 ms on a Pixel 6 CPU (https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker). On the Ryzen 5 5500U (table under "Checked on Tian's laptop"): palm detector about 37-40 ms, landmark model about 30 ms per hand, all on one core. Input size does not matter (320x240 took the same time as 640x480: both models resize internally).

## 2. Hand position in 3D from one camera

The pixel of the palm centre fixes a ray from the camera. What is missing is how far along the ray the hand is.

**(a) Known plane: intersect the ray with an assumed palm height.** Needs intrinsics and the camera pose. With the camera at height H, the hand at true height h, intersected with an assumed height z0, and X the horizontal distance between the hand and the point straight below the camera, the horizontal error is

`e = X (h - z0) / (H - h)` = `(h - z0) x tan(angle of the ray from vertical)`.

The direction of the error is away from the point below the camera if the hand is higher than assumed. In centimetres (first number: plane at the table, z0 = 0; second: plane at z0 = 17.5 cm):

| H | X | h = 10 | h = 15 | h = 20 | h = 25 |
| --- | --- | --- | --- | --- | --- |
| 60 | 20 | 4.0 / -3.0 | 6.7 / -1.1 | 10.0 / +1.2 | 14.3 / +4.3 |
| 70 | 10 | 1.7 / -1.2 | 2.7 / -0.5 | 4.0 / +0.5 | 5.6 / +1.7 |
| 70 | 20 | 3.3 / -2.5 | 5.5 / -0.9 | 8.0 / +1.0 | 11.1 / +3.3 |
| 70 | 30 | 5.0 / -3.8 | 8.2 / -1.4 | 12.0 / +1.5 | 16.7 / +5.0 |
| 70 | 45 | 7.5 / -5.6 | 12.3 / -2.0 | 18.0 / +2.2 | 25.0 / +7.5 |
| 80 | 30 | 4.3 / -3.2 | 6.9 / -1.2 | 10.0 / +1.2 | 13.6 / +4.1 |

Intersecting with the table itself is useless for a hand in the air (8-12 cm off). With a plane at the expected hand height, hands held 15-20 cm high are within 1-2 cm up to X = 30 cm. A synthetic check of the first geometry code reproduced these numbers; `tests/calib_synthetic.py` now covers `scripts/calib/`. A plain table homography is the z0 = 0 column, so it is not enough; a homography at a second, raised board height would give the same ray without intrinsics (two-plane method), but intrinsic calibration is quicker.

**(b) Depth from apparent hand size.** `distance = f x L / l`, with f the focal length in pixels, L a real length on the hand, l its length in pixels. Index-to-pinky knuckle distance (landmarks 5-17) is about 6-7 cm on adults **[estimate from the world landmarks of open hands]**. Hand breadth varies from 72 to 96 mm between the 5th percentile woman and the 95th percentile man (ANSUR II, https://madegooddesigns.com/tools/hand-size-chart/), so an unknown person gives about +-13 % in distance, +-8 cm at 60 cm. For a known demo person measured once, what is left is pixel noise (about 3 % on a 70 px span) and tilt (a palm tilted 20-25 degrees looks 6-10 % smaller). The resulting horizontal error is the depth error times the sine of the ray angle, so (b) beats (a) only when hand heights vary by more than about 5 cm. On two photos of the same hand from both sides, this size estimate gave the same distance to within 5 %.

**(c) `solvePnP` between world and image landmarks.** Published as a working idea (https://gist.github.com/eldog/9012ce957be26934044131daffc25c73, https://github.com/ntu-rris/google-mediapipe/issues/3). It inherits the world landmarks' scale, which is unreliable: the knuckle span in the world landmarks ranged from 3.2 to 7.4 cm over the test photos, and on the two photos showing one hand from the palm and from the back the palm view came out 0.31-0.33 m away and the back view 0.26 m, about 20 % apart. Not recommended.

**(d) Triangulation with the wrist camera.** Needs wrist-camera intrinsics, the camera-to-gripper transform (hand-eye calibration) and accurate forward kinematics, and the wrist camera sees the hands only when the arm is already near them. Too much for the time left.

**(e) Coarse from the top camera, fine from the wrist camera.** Visual servoing: once hovering, move so the midpoint of the palms sits in a fixed spot of the wrist image. Removes calibration error and solves the occlusion problem of the top camera, but needs a second tracker input and a closed loop with the motion half. A cheaper use of the wrist camera: only confirm "two open hands below me" before opening the gripper, no geometry needed. Stretch goal.

**Use first: (a)** with the plane at the expected palm height and the people told to hold their hands at about that height (for example level with a marked box). The robot side already hovers at a fixed height and uses only x and y, which matches. Add (b) only if hands at varying heights cause misses.

**Occlusion.** With a camera straight above, the gripper over the palms hides their centre. The drop rule on the robot side ("two hands within 3 cm of the tip for 1 s") needs the hands visible exactly then. Either latch the last two-hand position when hands vanish while the tip is within a few centimetres, or confirm with the wrist camera. Test at the table which happens.

## 3. Camera intrinsics with a printed checkerboard

**Pattern.** OpenCV's `pattern.png`: 9 x 6 inner corners, 25 mm squares, A4 landscape (measured on the file: 147.65 px per square at 150 dpi = 25.0 mm). Source: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png and the pattern tutorial, which calls it the "9x6 chessboard, page width: 210 mm, page height: 297 mm (A4)" (https://github.com/opencv/opencv/blob/4.x/doc/tutorials/calib3d/camera_calibration_pattern/camera_calibration_pattern.markdown). The same tutorial: the pattern "should be as flat as possible", matte is better than glossy, and detectors expect a white border. `findChessboardCornersSB` wants a white border about one square wide (https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp). At 70 cm a 25 mm square is about 20 px wide, enough for detection.

**Printing pitfalls.** Printer scaling ("fit to page" shrinks it by a few percent, which scales every distance by the same factor) and a board that is not flat. Measure the squares; pass the measured size.

**Corner numbering.** Checked here: with `findChessboardCornersSB` the 9x6 board is numbered from the same physical corner whatever its rotation (0, 90, 180, 270 degrees, also under perspective). The classic `findChessboardCorners` numbers it from the opposite corner. Never mix the two detectors when computing the board pose. On OpenCV's printed pattern, the SB detector's corner 0 is the inner corner nearest the start of the printed text **[checked on the rendered file, not on a photo of a print]**.

**Views and error.** The OpenCV tutorial asks for "at least 10 test patterns" (https://github.com/opencv/opencv/blob/4.x/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.markdown); use 15-20 and drop views whose error is far above the rest. MathWorks: whether an error is acceptable depends on the application, with an example at 0.18 px (https://www.mathworks.com/help/vision/ug/evaluating-the-accuracy-of-single-camera-calibration.html); a forum rule of thumb calls 0.1-0.5 px very good and up to 1 px acceptable (https://discuss.luxonis.com/d/5558-different-quality-of-depth-from-the-same-cameras, secondary). OpenCV's own 640x480 sample images (`left01..14.jpg`, a 9x6 board) gave 0.24 px here.

**Code path.** `findChessboardCornersSB` (sub-pixel corners directly, "localized radon transformation", flags `CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY`; source: calib3d.hpp above), then `calibrateCamera`. No `cornerSubPix` is needed with SB. Now in `scripts/calib/calibrate_intrinsics.py`; the first version ran on the sample images in both the MediaPipe environment and an environment with LeRobot's `opencv-python-headless 4.13`. SB took about 140 ms per 640x480 frame there.

**ChArUco.** Corners are interpolated from ArUco markers, so partial views and occlusion are allowed, which makes it easier to cover the image corners (https://github.com/opencv/opencv/blob/4.x/doc/tutorials/objdetect/charuco_detection/charuco_detection.markdown). `cv2.aruco.CharucoDetector` exists in `opencv-python-headless 4.13`, so no contrib package is needed (checked). For 1-2 cm at the hand, the plain board is enough; not worth reprinting.

**Is intrinsic calibration needed at all?** For a typical webcam, distortion is small: from the calibration of a Logitech C920 at 640x480 shipped with the FTC SDK (f = 622 px, k1 = 0.121, k2 = -0.262, k3 = 0.103; https://github.com/FIRST-Tech-Challenge/FtcRobotController/blob/master/TeamCode/src/main/res/xml/teamwebcamcalibrations.xml) the shift is at most 5 px in the image corners, 6 mm at 70 cm. The camera in OpenCV's sample images (k1 = -0.31) shifts the image corners by 63 px, 8 cm at 70 cm. Which kind the team's cameras are is unknown, and (a) needs the focal length anyway for any height other than the table. The calibration takes 5 minutes, so do it.

**Autofocus and settings.** LeRobot 0.7.0's `OpenCVCamera` sets width, height, fps and optionally FOURCC, and nothing else (no autofocus, exposure or buffer size); default backend `CAP_ANY`, default colour RGB (https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/opencv/camera_opencv.py, https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/cameras/opencv/configuration_opencv.py). Autofocus changes the focal length, so it must be off or absent.
- Linux/WSL: `v4l2-ctl -d /dev/videoN --list-ctrls` shows the names; older uvcvideo drivers call them `focus_auto` and `focus_absolute`, and `focus_absolute` can only be set while auto is 0. Settings may revert when the camera is replugged or the computer restarts, so set them again before the demo (https://people.eng.unimelb.edu.au/pbeuchat/asclinic/software/building_block_usb_camera_settings.html). Newer kernels name them `focus_automatic_continuous` and `auto_exposure` **[from memory of the kernel docs, unverified]**. Through OpenCV with V4L2, `CAP_PROP_AUTO_EXPOSURE` 1 is manual and 3 aperture priority (https://techoverflow.net/2022/11/01/how-to-set-v4l2-exposure-to-manual-mode-in-opencv-python/).
- Windows: `cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)` is often ignored; reports on MSMF and DirectShow are mixed, and the camera's own settings dialog (`CAP_PROP_SETTINGS` with DirectShow) is the usual workaround (https://answers.opencv.org/question/211355/how-to-disable-autofocus-of-a-webcam-on-windows-10). The robot program runs in WSL, so this only matters for a Windows fallback.
- In the demo program, the camera object is reachable as `robot.cameras["top"].videocapture`, so the same `set` calls can be made after connect **[unverified that the driver honours them while the read thread runs]**.

## 4. Camera to robot base

Frame (agreed on the board): metres, origin on the table straight below the shoulder_pan axis, x forward where the arm points at shoulder_pan = 0, y left, z up.

| | What it needs | Steps at the table | Time | Expected accuracy | What goes wrong |
| --- | --- | --- | --- | --- | --- |
| (a) Board flat on the table + ruler | intrinsics, ruler, tape axes | steps 7-11 above | 10 min | a few mm at the board; 1-2 cm at the hands from a few degrees of error in where x points | pan zero not where the tape says; board moved between photo and measurement; corner numbering (handled by the SB detector and the camera position check) |
| (b) Marker in the gripper, `calibrateHandEye` | forward kinematics, a small board fixed to the gripper, 10-20 poses | teleoperate to poses, record joints and an image each | 45 min+ | limited by FK and by pose estimation of a small marker at 70 cm | needs at least 3 poses with non-parallel rotation axes and "strongly" more (calib3d.hpp above); eye-to-hand needs the inverted gripper poses; FK errors go straight in |
| (c) Touch board corners with the gripper tip | forward kinematics of the tip, board from (a) | teleoperate the closed tip onto 4+ board corners, record joints | 20 min | 5-10 mm at the hands (synthetic test: 5 mm touch noise gives 4-7 mm) | FK tip point must be defined at the jaw tip; slight press moves the board |
| (d) Teach hover poses, no geometry | the tracker's pixel output only | put a hand at 9-12 grid points under the gripper, record joints and palm pixel; interpolate joints from pixel | 20-30 min | 1-2 cm between taught points **[estimate]**, only at the taught hand height | extrapolation outside the grid; needs the robot program to replay joints directly instead of IK |

The ruler measurement in (a) is fitted in the plane (x, y, yaw only, because the board lies flat), now in `scripts/calib/calibrate_extrinsics.py` (four measured corners); (c) became `scripts/calib/calibrate_extrinsics_touch.py`, which fits the camera to touched points directly. A full 3D fit let 5 mm of height noise move the camera by 32 mm in the synthetic test; the planar fit 9 mm. The 2D Kabsch fit is the standard least-squares rigid fit (https://en.wikipedia.org/wiki/Kabsch_algorithm).

**Recommendation.** Main route: (a), with four ruler-measured corners as in the table steps. As soon as the forward kinematics is confirmed on the arm, redo the four corners with (c): it measures in the robot's own kinematic frame, so an offset in shoulder_pan zero or a base that is not square to the table disappears. Fallback: (d), which needs neither intrinsics nor IK and uses the `u`, `v` fields the tracker already sends. Skip (b).

Validation at the table: put a coin at a ruler-measured spot and click it in `scripts/calib/validate.py`; it should land within about 1 cm. Then a palm at a known height.

## 5. Process layout

Constraints: LeRobot uses `opencv-python-headless`, MediaPipe pulls `opencv-contrib-python`, both own the `cv2` module; a webcam can normally be streamed by one process; the robot program needs the top camera for the PICK policies. From the board (not checked by me): the robot program runs inside WSL2 (Ubuntu 24.04) with LeRobot 0.7.0 in a conda environment, the cameras are passed into WSL with usbipd.

| Option | Cost to the control loop | Latency | Verdict |
| --- | --- | --- | --- |
| **Robot program writes frames to shared memory, tracker reads** | 0.21 ms per 640x480 write | 1.75 ms median transport | **Use this.** Standard library plus numpy on both sides; the writer never waits |
| Robot program sends JPEG over a local socket | 1.5 ms encode (34 KB per frame) plus a send that can block if the reader stalls | a few ms | Fallback, for example if tracker and robot program end up on different sides of WSL |
| Tracker as a thread inside the robot program | none measured | lowest | Works technically (below), but puts MediaPipe and its dependencies into the demo environment |
| Tracker owns the top camera | none | lowest | Not possible while PICK needs the same camera; Windows 11 has an "allow multiple apps to use camera" setting (https://pureinfotech.com/allow-multiple-apps-use-camera-windows-11/, **[unverified]**), WSL/Linux would need v4l2loopback **[unverified]** |
| Third camera only for hands | none | lowest | Good if the top camera sees the handover area badly or the arm hides the hands; can be placed freely because no policy uses it; needs its own calibration and USB bandwidth (use MJPG) |

**Which side owns the tracker: WSL.** The frames are in the robot program's memory inside the WSL2 VM, and shared memory does not cross the VM boundary. Running MediaPipe on Windows would add an encode and a socket hop and gain nothing (same CPU). Inside WSL:
- Create a separate virtual environment (not the conda `lerobot` one): `python3 -m venv ~/handtrack-venv && ~/handtrack-venv/bin/pip install mediapipe`. Ubuntu 24.04 has glibc 2.39, above the wheel's 2.28 minimum.
- MediaPipe's CPU inference needs no display. But `import mediapipe` imports `cv2` (through `mediapipe/tasks/python/vision/drawing_utils.py`, checked in the installed package), and the non-headless `opencv-contrib-python` on a bare Ubuntu usually fails with `libGL.so.1: cannot open shared object file` **[unverified on our WSL]**. Fix either with `sudo apt install libgl1 libglib2.0-0`, or inside the tracker venv: `pip uninstall -y opencv-contrib-python && pip install opencv-python-headless`. MediaPipe ran unchanged with `opencv-python-headless` here (checked).
- `sounddevice` is imported too, but inside `try/except OSError` (`mediapipe/tasks/python/audio/core/audio_record.py`), so a missing PortAudio library does not break the import (read in the installed code, not run on Linux).
- Shared memory on Linux with Python 3.12: a reader process started on its own gets its own resource tracker, which deletes the block when the reader exits; Python 3.13 adds `track=False` for this (https://docs.python.org/3/library/multiprocessing.shared_memory.html). The reader in `frame_link.py` unregisters the block for this reason. On Windows the block lives until the last handle closes.

**One environment instead of two?** Checked: MediaPipe 1.2.3 installed with `--no-deps` next to `numpy 2.2.6` and `opencv-python-headless 4.13.0.92` (both inside LeRobot 0.7.0's pins) gave the same detections; `pip check` only complains about the missing `opencv-contrib-python`. With uv the documented way is `[tool.uv] exclude-dependencies = ["opencv-contrib-python"]` in the root `pyproject.toml` (https://docs.astral.sh/uv/reference/settings/#exclude-dependencies), not tried. And MediaPipe releases the GIL during inference: a 30 Hz loop with a tracker thread beside it kept its period (median 34.2 ms, max 35.8 ms, against 34.9 ms alone). So a thread would work, but the robot program on the robot laptop is a conda environment, and a separate process is the change that cannot break the PICK demo. Keep the separate process.

**End-to-end latency** (tested with photos played into shared memory at 30 Hz, tracker in its own process, messages read by a 30 Hz loop): 23-26 messages per second, 68 ms median from frame capture to the robot loop seeing the result with hands in view, 34 ms with none. The camera's own delay comes on top **[not measured]**.

## Scripts in `data/handtrack/` (git-ignored; Tian decides what moves to `scripts/`)

- `hand_features.py`: palm centre, palm up, extended fingers, "open".
- `frame_link.py`: `FrameWriter` (robot program) and `FrameReader` (tracker) over shared memory; `python frame_link.py` measures it.
- `hand_link.py`: `HandLineSender`, the TCP client for the agreed contract, never blocks for long and reconnects once a second; a test receiver.
- `hand_tracker.py`: the tracker process (shared memory in, JSON lines out). `--only-open` sends only open, palm-up hands. Without calibration files `hands` stays empty and the detections go under an extra key `pixels` (`u`, `v`, flags): the receiver would turn null coordinates into NaN instead of rejecting them (`np.array([None], dtype=float)` is `nan`).
- `receiver_check/`: the robot side's `hand_socket.py` and `fake_tracker.py` copied unchanged from `jakobnunnendorf/handover-loop` (`90bb1d3`), and `check_against_hand_socket.py`, which runs the tracker against that receiver.
- The first calibration code (intrinsics, extrinsics from a ruler or touched corners, pixel to robot frame, selftest) and its capture scripts were replaced by `scripts/calib/`.
- `detect_images.py`, `probe_thresholds.py`, `bench.py`, `test_pipeline.py`: the checks below.

## Checked on Tian's laptop

All with `data/handtrack/.venv` (Python 3.12.10, mediapipe 1.2.3, numpy 2.5.5, opencv-contrib-python 5.1.0.95) unless marked "headless env" (`data/handtrack/.venv-headless`: mediapipe 1.2.3 installed with `--no-deps`, numpy 2.2.6, opencv-python-headless 4.13.0.92). Test photos: the MediaPipe sample `woman_hands.jpg` (https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg), the four Gesture Recognizer samples (`thumbs_up`, `thumbs_down`, `victory`, `pointing_up` from `https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/`), eight Wikimedia Commons photos of palms, backs of hands and cupped hands, six SO-100/SO-101 photos from https://github.com/TheRobotStudio/SO-ARM100/tree/main/media, and OpenCV's `left01..14.jpg` and `pattern.png`. Each photo was letterboxed into a 640x480 frame.

**Detection (IMAGE mode, num_hands=2, defaults).**

| Photo | Found | Labels and palm/back | Notes |
| --- | --- | --- | --- |
| woman_hands | 2 | Right palm, Left palm | as expected |
| Right hand palm / Right hand back | 1 / 1 | Right palm / Right back | correct |
| Open palm of left hand | 1 | Left palm | correct |
| Hands front-back, back-front (one right hand, both sides) | 2 / 2 | Right palm + Right back, both photos | correct, also rotated |
| pointing_up, victory, thumbs_up | 1 each | Left palm, Right palm, Right back | plausible |
| thumbs_down | 0 | | found at threshold 0.3, and at 0.5 when rotated 90 or 180 degrees |
| Cupped hands (3 photos) | 0 | | see section 1 |
| SO-100/SO-101 photos (6) | 0 | | one false hand at threshold 0.1 |

**Timing per 640x480 frame** (median of 55 frames after 5 warm-up, other jobs running, CPU time / wall time about 1.0, so one core):

| Mode | 2 hands in view | 1 hand | no hand |
| --- | --- | --- | --- |
| VIDEO, num_hands=1 | 35 ms (finds one) | 34 ms | 40 ms |
| VIDEO, num_hands=2 | 57-66 ms | 75 ms | 37 ms |
| IMAGE, num_hands=2 | 105 ms | 76 ms | 37 ms |
| LIVE_STREAM, num_hands=2, fed at 30 fps | 73 ms, 61 of 96 frames returned | | 46 ms |

In the end-to-end test the tracker ran at 23-26 frames per second with two hands, so the load from the other jobs moved the numbers by about a third.

**Other runs.**
- Shared memory, 640x480x3 at 30 Hz between two processes: write 0.21 ms median (0.82 max), transport latency 1.75 ms median (2.8 ms p95), 117 of 117 frames.
- JPEG encode of a 640x480 frame at quality 85: 1.5 ms, 34,320 bytes.
- GIL test: 30 Hz pure-Python loop alone 34.2/34.5/34.9 ms (median/p95/max period), with a MediaPipe thread beside it 34.2/34.7/35.8 ms; the thread did 114 frames in about 5 s.
- Headless env: same detections and labels as the main env.
- Calibration on OpenCV's samples: 12 of 13 views used, RMS 0.235 px, fx = 532 px, k1 = -0.31 (headless env with SB flags: 11 of 13, 0.238 px). Six frames saved through the capture script from a video made of the samples: RMS 0.229 px.
- Corner order of `findChessboardCornersSB` on the 9x6 pattern: same physical corner 0 under 0/90/180/270 degree rotation and four perspective warps; classic detector consistently the opposite corner.
- The first geometry selftest (synthetic camera 70 cm up, 0.3 px corner noise, C920-like distortion): camera position error 1.7 mm with exact ruler values; palm error 0.2-0.6 mm with the right height; 15-34 mm with a wrong plane height as the formula predicts; 4-7 mm with 5 mm noise on four touched corners (planar fit).
- The first capture script on a video file paced at 30 fps (headless env): saves only frames with the board, prints `640x480`.
- End-to-end `test_pipeline.py`: photos into shared memory at 30 Hz, `hand_tracker.py` in its own process, JSON lines over TCP; 22-29 messages per second, 69-70 ms median capture-to-loop latency with hands, 35 ms without; positions consistent with the synthetic camera.
- Against the robot side's own receiver (`hand_socket.py` from `jakobnunnendorf/handover-loop`, commit `90bb1d3`, copied unchanged), with a synthetic camera 70 cm above (0.30, 0) looking down:
  - tracker started 4 s before the receiver: hands arrived within the first second after the receiver came up (the sender retries once a second);
  - `woman_hands` gave two hands `[0.347, -0.022, 0.15]` and `[0.215, 0.076, 0.15]`; the right palm one; the cupped hands none; 0.7 s after the tracker stopped the receiver reported no hands (its 0.5 s freshness limit);
  - the front-and-back photo gave two hands although one is the back of a hand, and `pointing_up` gave one hand although it is not open: the receiver ignores `palm_up`/`open` (see "Recommended pipeline", point 5);
  - uncalibrated tracker: `hands` stayed empty, no NaN;
  - the exact bytes `{"t": 1763720000.123, "hands": [{"x": 0.3473, "y": -0.0231, "z": 0.15, "u": 345.2, "v": 188.4, "palm_up": true, "score": 0.92, "open": true, "handedness": "Right"}]}\n` went through the receiver's own parser to `[0.3473, -0.0231, 0.15]`.

## Not verified

- Anything with a live camera: real detection rate on our hands from the top camera, the arm hiding the hands while hovering, two flat palms touching, tracking that holds two hands as they come together, motion blur, the venue lighting.
- MediaPipe on Linux/WSL (the `libGL` question, speed on the robot laptop's CPU).
- Whether the team's top camera has autofocus, and whether its settings survive a replug in WSL.
- The newer V4L2 control names (`focus_automatic_continuous`, `auto_exposure`).
- That corner 0 of a printed OpenCV pattern seen by the real camera is the one nearest the start of the text (checked only on the image file).
- Windows 11 multi-app camera access, and v4l2loopback in WSL.
- The accuracy of option (d) between taught poses, and of the ruler route against the robot's kinematic frame.
- Hand-size numbers: the 6-7 cm knuckle span is read off MediaPipe's world landmarks, not measured on people.
- uv `exclude-dependencies` for a single environment (documented, not tried).
