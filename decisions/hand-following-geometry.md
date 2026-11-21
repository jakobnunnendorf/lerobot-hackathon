# How does the arm know where a hand is, and how does it get above it without a learned policy?

- Status: OPEN (2026-11-21). Built and checked in simulation and on photos. Nothing has run on the arm or with a live camera.

## Decision

The classical hand-following path, built in parallel to the learned one (`decisions/hand-handover-policy.md`) to compare them:

- **Seeing** (`scripts/handtrack/`): MediaPipe Hand Landmarker on the top camera, in its own process. The palm centre's pixel becomes a ray through the calibrated camera, and the ray is cut with a plane at an assumed palm height (15 cm). One JSON line per frame goes to the robot program on `127.0.0.1:8766`, in metres in the robot frame (origin on the table below the shoulder_pan axis, x forward at shoulder_pan = 0, y left, z up).
- **Moving** (`scripts/handover/kinematics.py`, `following.py`, used by the loop in `scripts/handover/handover_loop.py`): closed-form kinematics from the official URDF in plain numpy. The tip hovers at a fixed height above the table (22 cm) with the gripper tilted as steeply as the joints allow.
- **Set-up it needs:** hands held 10-15 cm above the table and 15-35 cm in front of the base, shown as two flat open palms side by side.
- **Calibrating the top camera** (`scripts/calib/`, `docs/camera-calibration-runbook.md`): intrinsics from 20-30 frames of the team's printed 8 x 6-square board (7 x 5 inner corners); the camera pose from one frame of that board lying on the table, with its four outermost inner corners measured by ruler in the robot frame and corner 1 marked by red tape. Robot-touch points (method B) check calibration and kinematics together.

## Why

- **Own closed-form kinematics.** LeRobot's kinematics needs `placo`, which has no Windows wheel and is not installed on the robot laptop (`kb/kinematics-on-windows.md`, `research/so101-inverse-kinematics.md`). The arm is a yaw joint, three pitch joints in one plane and a roll, so the inverse is exact: 0.2 ms per call, forward(inverse(p)) within 2e-6 m, forward kinematics equal to ikpy on the same URDF to 3e-16 m (Tian's laptop, 2026-11-21). The reach table from it agrees with `placo` to 1-3 cm and 1-3 degrees (Kamel's agent on the WSL2 laptop, board, 2026-11-21).
- **Tilted gripper, fixed hover height.** With 5 degrees kept free at the joint limits the gripper cannot point straight down at any tip height of 10 cm or more. Hovering over a hand held at 30 cm needs the tip at 40 cm, where the gripper points upwards and reach ends at 32 cm. At a tip height of 20-25 cm the reach is 10-40 cm with the gripper 35-45 degrees below horizontal.
- **The elbow keeps at least 20 degrees of bend.** The steepest reachable tilt is often the one where the arm is fully stretched, and there 8 mm of tip travel took up to 16 degrees at shoulder and elbow. With the bend kept, the largest step is 9 degrees and reach is unchanged to the centimetre (`tests/handover_kinematics.py`).
- **An assumed palm height instead of measured depth.** The position error is (true height - assumed height) x tan(angle of the ray from vertical): 1-2 cm for hands 15-20 cm high under a camera 70 cm up. The robot side uses only x and y.
- **Palm up from the landmarks.** The cross product of wrist-to-index-knuckle and wrist-to-pinky-knuckle with MediaPipe's left/right label was right on all 7 test photos where the answer was clear.
- **Frames reach the tracker through shared memory** written by the robot program, because only one process can open the camera and a socket send could stall the 30 Hz loop: 0.2 ms per write, about 2 ms transport (Tian's laptop, synthetic frames).
- **The 7 x 5 board with a tape mark, not a reprinted 9 x 6 one.** With both corner counts odd the pattern looks the same turned by 180 degrees, so the image cannot tell corner 1 from corner 3, and a wrong pick puts the camera at the mirror position (17-50 cm off in the synthetic test). The board was already printed, measured and mounted. Three tie-breakers that must agree (the red tape found next to a pattern end, an image-corner rule, the rough lens position), an overlay to confirm by eye, and the handedness read from the four ruler-measured corners settle it; every wrong-rule case in `tests/calib_synthetic.py` came out as the mirror image, or was refused when the tape was visible (Tian's agent, 2026-11-21, rendered images and `board.png`).
- **Four measured corners instead of one corner and a yaw.** Fitting x, y and yaw to four corners gives a residual per corner (a mismeasured or renumbered corner shows up) and the handedness of the numbering (z = (2 - 1) x (4 - 1) up or down), which removes the two mirrored candidates. Synthetic, 1 mm ruler noise: camera 3-5 mm off, hand plane under 1 mm.
- **Touch points at several heights.** Synthetic, 8 points, 2 px clicks, 5 mm tip error: 39-54 mm median error at hand height with all points on the table, 7-11 mm with points spread over 0-25 cm.

Details, sources and what was run: `research/so101-inverse-kinematics.md`, `research/hand-tracking-and-camera-calibration.md`.

## Alternatives

- **Steepest tilt without the elbow bend** - tried 2026-11-21 in simulation: joint steps up to 16 degrees between 8 mm waypoints near the edge of reach.
- **A fixed gripper tilt** (30 to 45 degrees below horizontal, nearest reachable tilt elsewhere) - tried 2026-11-21 in simulation: steps up to 13-23 degrees, because outside its band it falls back onto the stretched arm.
- **`placo` through LeRobot's `RobotKinematics`** - considered, used only as a cross-check: no Windows wheel, a numerical solver with 8 steps per call.
- **ikpy** - tried 2026-11-21: 27-37 ms per call and 2 of 30 reachable targets missed by more than 1 mm.
- **MediaPipe's metric world landmarks with `solvePnP` as depth** - tried on photos: the same hand came out 20% nearer seen from the back than from the palm.
- **Depth from the hand's size in the image** - considered: hand breadth varies about +-13% between people, about 8 cm at 60 cm distance. Worth it only for one measured demo person.
- **The Gesture Recognizer's `Open_Palm`** - tried on photos: it fired on the back of a hand and missed 5 of 9 open hands.
- **Two cupped hands as the drop gesture** - tried on photos: no hand detected in 3 of 3 photos at the default thresholds.
- **No geometry at all: teach 9-12 hover poses and interpolate joint angles from the palm's pixel** - not built. It is the fallback if calibration or the joint offsets cannot be settled in time.
- **Reprinting OpenCV's 9 x 6 board** (one odd and one even count; the SB detector then numbers it from the same physical corner) - considered 2026-11-21, not done: the 7 x 5 board was already mounted, and the tie-breakers above handle it.
- **Origin rule "the pattern end nearest a chosen image corner"** - tried 2026-11-21 on rendered images: for a board in the middle of the image the opposite end was nearer (296 against 308 px). Replaced by "the end lying towards that image corner, seen from the board's centre".
- **Red tape searched anywhere within three squares of the board** - tried 2026-11-21 on rendered images: an orange patch next to the robot base was taken for the tape when the board lay near the base. Now only patches lying wholly inside a window at the tape's place next to each pattern end count.
- **Distortion with k3 fixed at 0** - tried 2026-11-21 on rendered frames: no clear difference (central-image model error 2.30 against 2.35 px with 25 views, 3.48 against 3.61 px with 12). OpenCV's default five coefficients kept.
- **Auto exposure switched off for calibration** - not done by default: V4L2 keeps the setting after the script exits, which would change the images the PICK policies see; exposure does not change the geometry. `camera.manual_exposure` in the config.
- **Reading joints by importing LeRobot inside the calibration scripts** - not done: the calibration scripts need OpenCV with windows, LeRobot brings the headless build. `scripts/calib/joint_feed.py` runs in the LeRobot environment, opens only the motor bus (no `connect()`, which writes motor settings and switches torque on) and hands the joints over through a file.

## Revisit if

- The table check (`docs/hand-following-table-check.md`) shows a joint sign or offset that differs from the URDF: set `LEROBOT_SIGNS` and `LEROBOT_OFFSETS_DEG` in `kinematics.py` and repeat.
- The top camera loses the hands while the gripper hovers over them: latch the last two-hand position or confirm with the wrist camera.
- The ball does not fall cleanly from a gripper tilted 35-45 degrees.
- Hands at varying heights cause misses of more than about 3 cm: add depth from hand size for the demo person.
- Method A and method B disagree by more than about 2 cm at hand height after the kinematics offsets are set: find out which one is wrong before the demo.
- The red tape is not found in the real top-camera frame (lighting, size): rely on `origin_image_corner` and the lens position, or retune `RED_HSV_RANGES`.
