# How does the arm know where a hand is, and how does it get above it without a learned policy?

- Status: OPEN (2026-11-21). Built and checked in simulation and on photos. Nothing has run on the arm or with a live camera.

## Decision

The classical hand-following path, built in parallel to the learned one (`decisions/hand-handover-policy.md`) to compare them:

- **Seeing** (`scripts/handtrack/`): MediaPipe Hand Landmarker on the top camera, in its own process. The palm centre's pixel becomes a ray through the calibrated camera, and the ray is cut with a plane at an assumed palm height (15 cm). One JSON line per frame goes to the robot program on `127.0.0.1:8766`, in metres in the robot frame (origin on the table below the shoulder_pan axis, x forward at shoulder_pan = 0, y left, z up).
- **Moving** (`scripts/handover/kinematics.py`, `following.py`, used by the loop in `scripts/handover/handover_loop.py`): closed-form kinematics from the official URDF in plain numpy. The tip hovers at a fixed height above the table (22 cm) with the gripper tilted as steeply as the joints allow.
- **Set-up it needs:** hands held 10-15 cm above the table and 15-35 cm in front of the base, shown as two flat open palms side by side.

## Why

- **Own closed-form kinematics.** LeRobot's kinematics needs `placo`, which has no Windows wheel and is not installed on the robot laptop (`kb/kinematics-on-windows.md`, `research/so101-inverse-kinematics.md`). The arm is a yaw joint, three pitch joints in one plane and a roll, so the inverse is exact: 0.2 ms per call, forward(inverse(p)) within 2e-6 m, forward kinematics equal to ikpy on the same URDF to 3e-16 m (Tian's laptop, 2026-11-21). The reach table from it agrees with `placo` to 1-3 cm and 1-3 degrees (Kamel's agent on the WSL2 laptop, board, 2026-11-21).
- **Tilted gripper, fixed hover height.** With 5 degrees kept free at the joint limits the gripper cannot point straight down at any tip height of 10 cm or more. Hovering over a hand held at 30 cm needs the tip at 40 cm, where the gripper points upwards and reach ends at 32 cm. At a tip height of 20-25 cm the reach is 10-40 cm with the gripper 35-45 degrees below horizontal.
- **The elbow keeps at least 20 degrees of bend.** The steepest reachable tilt is often the one where the arm is fully stretched, and there 8 mm of tip travel took up to 16 degrees at shoulder and elbow. With the bend kept, the largest step is 9 degrees and reach is unchanged to the centimetre (`tests/handover_kinematics.py`).
- **An assumed palm height instead of measured depth.** The position error is (true height - assumed height) x tan(angle of the ray from vertical): 1-2 cm for hands 15-20 cm high under a camera 70 cm up. The robot side uses only x and y.
- **Palm up from the landmarks.** The cross product of wrist-to-index-knuckle and wrist-to-pinky-knuckle with MediaPipe's left/right label was right on all 7 test photos where the answer was clear.
- **Frames reach the tracker through shared memory** written by the robot program, because only one process can open the camera and a socket send could stall the 30 Hz loop: 0.2 ms per write, about 2 ms transport (Tian's laptop, synthetic frames).

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

## Revisit if

- The table check (`docs/hand-following-table-check.md`) shows a joint sign or offset that differs from the URDF: set `LEROBOT_SIGNS` and `LEROBOT_OFFSETS_DEG` in `kinematics.py` and repeat.
- The top camera loses the hands while the gripper hovers over them: latch the last two-hand position or confirm with the wrist camera.
- The ball does not fall cleanly from a gripper tilted 35-45 degrees.
- Hands at varying heights cause misses of more than about 3 cm: add depth from hand size for the demo person.
