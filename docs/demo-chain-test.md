# Demo chain test: voice → pick → hand over → rest

For the people at the robot laptop. One round:
1. Someone says "give me the ball" (or "put it in my hand").
2. The learned ACT policy (Tomi's `Space0code/act_redblue_from_foil`, 15K by default) digs out and picks the striped ball, then returns to rest.
3. The arm carries the ball to the present pose and follows an open palm at 28 cm.
4. Two flat palms held still under the gripper for 1 s: it drops the ball, opens and closes 3 times fast, then folds back to rest.

**Hand lost:** the arm holds. After 5 s it goes back to the present pose and follows again when the palm comes back. After 45 s without a drop, it folds back to rest and puts the ball down over the bowl.

The program is the supervisor (#17) on branch `TianK003/demo-chain`; why it works this way is in `decisions/supervisor.md`. Nothing here was run on the arm before this test. Hand on the power switch for every round.

## 0. Once: code and environments

**WSL, robot side** (about 2 min):

```bash
cd ~/lerobot-hackathon && git fetch origin
git worktree add --detach ~/demo origin/TianK003/demo-chain
cp ~/track/hand_landmarker.task ~/demo/
```

To update later: `cd ~/demo && git fetch origin && git checkout --detach origin/TianK003/demo-chain`.

The `lerobot` and `handtrack` conda environments from the hand-tracking test are used as they are.

**Windows, voice side** (about 10 min, on good Wi-Fi). First copy the voice code out of WSL, still in WSL:

```bash
mkdir -p /mnt/c/demo-voice/scripts /mnt/c/demo-voice/tests
cp -r ~/demo/scripts/voice /mnt/c/demo-voice/scripts/ && cp ~/demo/tests/voice_parse.py /mnt/c/demo-voice/tests/
rm -rf /mnt/c/demo-voice/scripts/voice/.venv
```

Then in PowerShell:

```powershell
cd C:\demo-voice
py -3.12 -m venv scripts\voice\.venv
scripts\voice\.venv\Scripts\python -m pip install -r scripts\voice\requirements.txt
scripts\voice\.venv\Scripts\python tests\voice_parse.py
```

The last line must print `PASS`.

If `py -3.12` is not found, use uv instead:

```powershell
winget install astral-sh.uv
# open a new PowerShell window
cd C:\demo-voice
uv venv scripts\voice\.venv --python 3.12
uv pip install -r scripts\voice\requirements.txt --python scripts\voice\.venv\Scripts\python.exe
```

Plug in the headset, then:
- `scripts\voice\.venv\Scripts\python scripts\voice\listen.py --list-devices`: note the number of the headset's `MME` line.
- `scripts\voice\.venv\Scripts\python scripts\voice\listen.py --device <n> --dry-run`: the first start downloads the speech model, about 660 MB. Press Enter, say "put it in my hand", press Enter. It must show `>>> HAND (striped ball)`. Ctrl+C.

## 1. Checks without moving the arm (3 min)

WSL, terminal 1:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
python scripts/supervisor/run_supervisor.py --fake all        # one line per scenario (name, outcome, states), no traceback
python3 scripts/voice/link.py --host 0.0.0.0                  # waits for commands
```

In PowerShell, run `scripts\voice\.venv\Scripts\python scripts\voice\listen.py --device <n>`. Press Enter, say "give me the ball", press Enter.
- Terminal 1 must print a JSON line with `"destination": "hand"`.
- If the listener says `NOT SENT`, wait 15 s and try again.

Ctrl+C in terminal 1. Leave the listener running.

## 2. Put-down pose over the bowl (2 min, torque off)

```bash
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

Torque goes off: hold the arm. Put the closed gripper about 10 cm above the middle of the bowl, read the five joint values, Ctrl+C, and save them:

```bash
echo 'PUT_DOWN_POSE="<pan> <lift> <elbow> <wrist_flex> <wrist_roll>"' >> ~/.so101.env
```

The ball is only put down there when a handover fails.

## 3. First round, started from the keyboard

Before starting:
- Fold the arm to its rest pose by hand, gripper **open**. The program refuses to start if the gripper reads 6-20.
- Put the striped ball on top of the pile, visible. Nobody's hands in the bowl.

Terminal 1, the robot program. It owns the arm and both cameras:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
bash scripts/supervisor/demo_chain.sh --step 015000
```

- It downloads the checkpoint the first time, loads it, connects and prints `Enter starts a round...`.
- It must not start a calibration. If it asks `Press ENTER to use provided calibration file`, press Enter.
- Use the step that did best in the P runs (`--step 005000`, `--step 025000`, ...). Any `run_supervisor.py` flag can be appended.

Terminal 2, the hand tracker, reading the frames terminal 1 shares:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate handtrack && cd ~/demo
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml --hand-z 0.20
```

Type `h` + Enter in terminal 1. Expect:

1. **REST**: a small move into the rest pose, gripper 30.
2. **PICK** (up to 90 s): the policy digs and grasps.
   - The grasp check (Jakob's, new) pauses the arm for about 0.4 s at each close and reads the gripper.
   - On a miss it opens and tries again, 3 picks at most.
   - PICK ends when the arm is back at rest holding the ball.
3. **VERIFY_HOLD**: the wrist camera looks at the ball. Logging only, it decides nothing yet.
4. **GIVE**: the arm carries the ball to the present pose. Hold one open palm about 20 cm above the table, 15-35 cm in front of the base. The gripper follows at 28 cm. Bring the second palm next to it and hold still 1 s: drop, 3 fast close-open cycles.
5. **REST**: it folds back to rest. Terminal 1 prints the round's outcome, `given`.

**Watch the first move from rest to the present pose:** in the model it stays 24-40 cm above the table. If it comes near the bowl or the table, press `a` + Enter.

**Stop:** `a` + Enter (or Ctrl+C) in terminal 1. The arm stops where it is with torque on and the gripper is not opened. Hold the arm, then switch the power off.

Use these if the step misbehaves:
- the grasp check pauses or re-picks in a silly way: append `--no-grasp-check`;
- the wrist check errors: append `--no-held-check`;
- you want slow moves to watch the first round: append `--max-step-deg 1.0`. It also slows the policy, which then picks worse.

## 4. Rounds by voice

With terminals 1 and 2 running and the arm at rest: in the PowerShell listener press Enter, say "give me the ball" (or "put it in my hand"), press Enter. The round starts by itself. Keys `1` + Enter in the listener, or `h` + Enter in terminal 1, do the same.

Run 5 rounds:
- 2 with the striped ball visible;
- 2 half covered;
- 1 under 3 or more balls.

In one round, take the hand away for about 6 s during GIVE and bring it back: the arm should go back to the present pose and then follow again.

## 5. Post on the board (issue #1)

Per round: ball start (visible / half / buried), pick OK?, handover OK?, seconds, anything odd. Then the log summary:

```bash
python -c "import json,glob; f=sorted(glob.glob('$HOME/so101/demo/chain_*.jsonl'))[-1]; [print(e['round'], e['outcome'], e['duration_s'], 'picks', e['pick_attempts'], ' -> '.join(e['states'])) for e in map(json.loads, open(f)) if e.get('event') == 'round_end']"
```

**Passed** when:
- at least 4 of 5 rounds run voice → pick → handover → rest with no help beyond speaking and the hands;
- the gripper never goes below about 20 cm during GIVE;
- the hand-lost round recovered.
