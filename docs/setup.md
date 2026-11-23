# Setup

Do this once per laptop. Commands run from the repo root unless a step says otherwise.

## Any laptop: the repo environment

You need [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
git clone https://github.com/jakobnunnendorf/lerobot-hackathon.git
cd lerobot-hackathon
uv sync
uv run python tests/cuda.py            # torch build, CUDA, a GPU matmul
uv run python tests/lerobot_check.py   # builds ACT, one train and one inference step, prints PASS
```

- LeRobot is pinned to 0.7.0. Flags change between releases: check them in the version-matched docs (https://huggingface.co/docs/lerobot/v0.7.0/en/index) or in `.venv/.../site-packages/lerobot`.
- torch comes from the CUDA 12.8 wheel index, so `uv.lock` has torch wheels for Linux and Windows only, not macOS.
- The other files in `tests/` are standalone checks of one part each, without hardware. Run them the same way, `uv run python tests/<file>.py`; a few need the env of the part they test (e.g. `voice_parse.py` needs `rapidfuzz`).
- Hugging Face login, for pushing datasets and models: `uv run hf auth login`.

That is enough to train, curate data and run the fake demo. Driving the arm needs the robot laptop setup below.

## Robot laptop: Windows + WSL2

Our robot laptop runs Ubuntu 24.04 in WSL2 with an RTX 4050 (8 GB). The arms and both cameras are passed into WSL with usbipd. Everything that touches the arm runs inside WSL.

### 1. USB into WSL

Install [usbipd-win](https://github.com/dorssel/usbipd-win) on Windows. Bind each device once, in an admin PowerShell:

```powershell
usbipd list
usbipd bind --busid <busid>     # once per device: both arms, both cameras
```

Then attach them, from Git Bash on Windows or from inside WSL:

```bash
bash scripts/so101/attach_usb.sh
```

It attaches the devices whose USB ids are in `SO101_USB_IDS` (default: our two arms and two cameras). Set that variable if your hardware has other ids.

**WSL loses the devices.** When the last Linux process exits, WSL shuts down and every usbipd device detaches. A camera with a loose plug also drops out alone. Run `attach_usb.sh` again whenever `/dev/so101_*` or `/dev/v4l/by-id/...` is missing. Keep WSL alive from PowerShell:

```powershell
Start-Process wsl.exe -ArgumentList '-e','sleep','infinity' -WindowStyle Hidden
```

### 2. System packages and environments (inside WSL)

```bash
sudo apt install -y ffmpeg v4l-utils libgl1 libglib2.0-0 espeak-ng
```

`v4l-utils` is for the camera exposure lock, `libgl1` for OpenCV with a window, `espeak-ng` for the spoken prompts of the recording scripts.

Three environments, so their OpenCV builds do not clash. We use conda (miniforge):

| Env | For | Install |
| --- | --- | --- |
| `lerobot` | everything that drives the arm | LeRobot 0.7.0, e.g. `pip install "lerobot[feetech]==0.7.0"` |
| `dashboard` | the web dashboard | `pip install -r scripts/dashboard/requirements.txt` |
| `handtrack` | hand tracker and calibration tools with a window | `pip install -r scripts/handtrack/requirements.txt -r scripts/calib/requirements.txt` |

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n dashboard python=3.12 && conda activate dashboard && pip install -r scripts/dashboard/requirements.txt
conda create -y -n handtrack python=3.12 && conda activate handtrack \
  && pip install -r scripts/handtrack/requirements.txt -r scripts/calib/requirements.txt
python -c "import mediapipe, cv2, yaml; print('ok')"
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

The hand model `hand_landmarker.task` goes in the repo root; the demo commands look for it there.

Instead of the conda `lerobot` env you can use the repo's uv `.venv` (`uv sync` inside WSL): see `SO101_ACTIVATE` below.

After installing or reinstalling LeRobot, patch its recorder once (SPACE key, wait before the first episode, on-screen banners). It is idempotent and keeps `.orig` backups:

```bash
conda activate lerobot && python scripts/so101/patch_record.py
```

### 3. Machine settings: `~/.so101.env`

```bash
cp scripts/so101/so101.env.example ~/.so101.env
```

Edit it:

- `SO101_ACTIVATE`: the command that activates the env with LeRobot. Empty means the repo's `.venv`.
- `SO101_FOLLOWER_PORT`, `SO101_LEADER_PORT`: the arms' serial ports.
- `SO101_FOLLOWER_ID`, `SO101_LEADER_ID`: the calibration ids (`mi_follower`, `mi_leader` on our arm). Every dataset and policy was made with these; keep them.
- `SO101_CAM_TOP`, `SO101_CAM_WRIST`: `/dev/v4l/by-id/...` paths. They survive replugging; `/dev/videoN` numbers do not.
- `SO101_HF_USER`: the Hub user that owns the datasets you record.
- `PUT_DOWN_POSE`: added later, see `docs/demo.md`.

Every script in `scripts/so101/`, `scripts/supervisor/demo_chain.sh` and `scripts/eval/eval_pick.sh` reads this file. Without it, the `scripts/so101/` scripts fall back to the example values.

**Finding the ports.** `lerobot-find-port` asks you to unplug and replug each arm. Plain `/dev/ttyACM*` names swap on replug. We gave the arms stable names (`/dev/so101_follower`, `/dev/so101_leader`) with a udev rule on the USB serial number; the rule is not in the repo. The usual form:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -m1 'ATTRS{serial}'
# /etc/udev/rules.d/99-so101.rules
SUBSYSTEM=="tty", ATTRS{serial}=="<follower serial>", SYMLINK+="so101_follower"
SUBSYSTEM=="tty", ATTRS{serial}=="<leader serial>", SYMLINK+="so101_leader"
sudo udevadm control --reload && sudo udevadm trigger
```

Find the camera paths with `ls /dev/v4l/by-id/`.

### 4. Arm calibration

**On our arm, never calibrate again.** The calibration file defines what a joint angle means for every recorded dataset and every trained policy. On another laptop, copy the file instead:

`~/.cache/huggingface/lerobot/calibration/robots/so_follower/mi_follower.json` (the leader's is under `teleoperators/`).

WSL and Windows keep separate calibration files and caches, so drive the arm from WSL only.

On a new arm, calibrate once with the ids from `~/.so101.env`:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=mi_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/so101_leader --teleop.id=mi_leader
```

When a program asks `Press ENTER to use provided calibration file`, press Enter. If it starts a calibration procedure instead, stop it with Ctrl+C: the ids or the file are wrong.

### 5. Gripper torque: 200, never more

**Never set the follower gripper's torque limit above 200 of 1000.** LeRobot 0.7.0 sets 500, and that stripped the gripper servo's gears. 200 holds a ball firmly. Why and how it was measured: `kb/gripper-torque-limit.md`.

`scripts/so101/gripper_limit.py` writes the cap. Our programs call it after every connect: the supervisor, `eval_pick.py`, the hand-following loop, the dashboard recorder, and `record.sh`, `teleop.sh` and `run_policy.sh` before their `lerobot-*` command. The cap is lost at power-up, so after a power cycle run it before any bare `lerobot-*` command:

```bash
python scripts/so101/gripper_limit.py /dev/so101_follower
```

### 6. Check it

```bash
python scripts/preflight.py                 # nothing moves; see docs/demo.md
bash scripts/so101/teleop.sh                # leader drives the follower, Rerun shows both cameras; --no-cams for arms only
python scripts/so101/motor_health.py        # reads every motor of both arms; the arm program must be stopped
```

`teleop.sh`: to stop, hold the follower (it goes limp), click the terminal, press Ctrl+C.

## Known failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/dev/so101_*` or `/dev/v4l/by-id/...` missing, everything "crashed" | WSL shut down and detached the USB devices | `attach_usb.sh`, then keep WSL alive (above) |
| Wrist camera `read failed` / `No such device` mid-run | its USB dropped out of WSL | `attach_usb.sh`. A recording keeps the episodes saved before it |
| `Failed to enable torque ... 'gripper' (motor ID 6): Operation timed out` at connect | one dropped bus message | retry once; if it repeats, check the gripper motor plug at the wrist |
| Follower gripper stops moving | gripper motor overheat (status bit 4, cuts at 70 C) from squeezing the ball | stop, `motor_health.py`, wait below about 45 C (`motor_health.py --cool 45`). When teleoperating, close the leader gripper only until the ball holds |
| Both cameras will not stream at once | two uncompressed 640x480 streams do not fit on one USB bus | keep MJPG (all our scripts set it) |
| `import cv2` fails with `libGL.so.1` | missing system library | `sudo apt install -y libgl1 libglib2.0-0` |
| A camera fails in a script | another program has it open | only one program can use a camera; stop the other |

## Safety rules for every robot program

- Someone stands at the arm with a hand on the power switch.
- Stopping a program that holds torque (Ctrl+C in `teleop.sh`, `run_policy.sh`, a recording) makes the arm go limp. Hold the arm first.
- **Never reset, disconnect or reconnect the robot while it holds the ball.** LeRobot's `connect()` briefly switches torque off and `disconnect()` switches it off; the ball drops. Our demo and eval loops keep one connection for that reason.
