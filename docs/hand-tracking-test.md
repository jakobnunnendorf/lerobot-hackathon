# Hand tracking test: the gripper follows a hand, 5 cm above it

For the person at the robot laptop. All commands run in WSL. No ball is needed.

- Hand: about 20 cm above the table.
- Gripper: hovers at 25 cm (`--hover-height 0.25`).
- Stage 1 checks the whole chain with the arm standing still.
- Stage 2 moves the arm. **Run stage 2 only after K, the corner touches (card K, `docs/hand-following-table-check.md` sections 2-5), has passed.** Until then, the joint directions in our kinematics are not confirmed on the real arm. A wrong sign moves the arm the wrong way, and the safety limits are computed with the same kinematics.

The hand: **one flat, open hand, palm up**. Hold it about 20 cm above the table (a box or ruler helps), 15-35 cm in front of the robot base, where the top camera sees it. Cupped hands and fists are not followed.

## 0. Once: code and environments

```bash
cd ~/lerobot-hackathon           # the folder you record from
git fetch origin
git worktree add ~/track origin/TianK003/hand-tracking-test
cd ~/track
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n handtrack python=3.12
conda activate handtrack
pip install -r scripts/handtrack/requirements.txt -r scripts/calib/requirements.txt
python -c "import mediapipe, cv2, yaml; print('ok')"     # libGL.so.1 missing: sudo apt install -y libgl1 libglib2.0-0
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

The robot side runs in the usual `lerobot` environment. Nothing is installed into it.

## 1. Stage 1: the arm does not move (about 10 minutes)

Stop teleop, recording or policy runs first: the tracker opens the top camera itself.

Terminal 1, the control loop in dry-run. It only prints what it would command:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/track
python scripts/handover/handover_loop.py --dry-run --hover-height 0.25 --print-every 15
```

Terminal 2, the tracker:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate handtrack && cd ~/track
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml \
    --hand-z 0.20 --camera
```

Check:
- Terminal 2 prints `camera: 640x480 at 30 fps, backend V4L2`. Then, twice a second, the hand in robot coordinates, for example `x +0.251 y -0.048 m open palm`.
- Hold the open palm 20 cm above a point you can measure, for example 25 cm straight ahead of the base's turning axis. x should read about 0.25 and y about 0.00, within about 2 cm. Move the hand 10 cm to the robot's left: y should grow by about 0.10. Note 2-3 readings next to the ruler values.
- Terminal 1 shows `follow` while the hand is seen, with `tip=(x, y, +0.250)` moving to the hand's x and y. It shows `hold` when the hand disappears, and `home` after 3 s.

Ctrl+C in both terminals.

## 2. Stage 2: the arm follows

K has passed after a fix: the joint signs are right, and the zero offsets fitted from Balazs's corner touches are now in `scripts/handover/kinematics.py` on this branch (board 7064451117). The corners now come out 4-11 mm from the ruler. That holds at table height. 2a checks hover height.

One person stands at the arm with a hand on its power switch. No ball, nothing else within reach.

### 2a. Two-minute check of the arm model at hover height (torque off)

Terminal 1:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/track
git fetch origin && git checkout --detach origin/TianK003/hand-tracking-test   # brings the fitted offsets
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

It switches torque off: hold the arm. It prints the joints and `jaw tip (m) x= y= z=` twice a second.

1. The folded rest pose is already known: the jaw's end is at 35 cm, and the model says 31 cm. The model is weaker there than in the hand area, which is why the arm starts raised (step 3).
2. Hold the fixed jaw's end about **25 cm above** a point you can measure, for example 25 cm straight ahead of the base's turning axis. Compare the printed `x`, `y` and `z` with the ruler. **Within about 3 cm: go on.** If z is more than 3 cm off, stop and post both readings.
3. **Keep holding the arm in that raised pose** and go straight to 2b. The pose at start is where the arm returns when the hand is lost. Starting raised keeps every move at hover height, away from the bowl.

### 2b. Start the loop with the arm held raised

Press Ctrl+C in terminal 1 (`read_pose.py` stops; the arm stays limp, keep holding it), then start the control loop in the same terminal. It owns the top camera and shares the frames with the tracker:

```bash
python scripts/handover/handover_loop.py --robot --port /dev/so101_follower \
    --top-camera /dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0 --robot-id mi_follower \
    --hover-height 0.25 --min-height 0.20 --max-step-m 0.004 --max-step-deg 1.0 --print-every 15
```

- Once it prints `listening on 127.0.0.1:8766; home=...`, the torque is on and the arm holds itself: let go. Before switching the torque on, it sets each servo's target to where the arm is, so the arm does not jump to an old target.
- It must connect without a calibration. If it asks `Press ENTER to use provided calibration file ...`, press ENTER. If it starts a calibration procedure, press Ctrl+C and post here.
- `--min-height 0.20` keeps every target at least 20 cm above the table.
- `--max-step-m 0.004 --max-step-deg 1.0` halves the speed for the first runs, to about 12 cm/s. Leave both out once it behaves.

Terminal 2, the tracker, reading the loop's frames (no `--camera`):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate handtrack && cd ~/track
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml --hand-z 0.20
```

What should happen:
- No hand: the arm holds still.
- Open palm at 20 cm: the gripper first rises to 25 cm, then moves over the palm and follows it slowly.
- Hand gone: the arm stops, and after 3 s it returns to the start pose.
- Two open palms under it for 1 s: it opens the gripper. That is the drop; it is harmless without a ball.

**Stop: Ctrl+C in terminal 1.** The arm stays where it is, with torque on. If it moves **away from the hand or mirrored** (hand to the left, arm to the right), stop at once and post terminal 1's last lines.

## 3. Post on the board (issue #1)

- Stage 1: the 2-3 readings against the ruler.
- Stage 2:
  - Does it go to the hand? How far from above the palm centre does the gripper end, in cm, at 3 places?
  - Is the hand lost while the gripper is above it? The camera looks from the side, so the arm can hide the palm. Terminal 2 then prints `no hand`. If so, try `--hover-height 0.30` (10 cm above a hand at 20 cm).
  - Any `target out of reach` lines.
