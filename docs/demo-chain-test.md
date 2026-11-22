# Demo chain test: voice → pick → hand over → rest

For the people at the robot laptop. One round:
1. Someone says "give me the ball" (or "put it in my hand"), or clicks a tile on the dashboard.
2. The learned ACT policy (Tomi's `Space0code/act_redblue_from_foil`, 15K by default) digs out and picks the striped ball, then returns to rest.
3. The wrist camera checks it is the striped ball. A red or blue ball is put down and picked again.
4. The arm carries the ball to the present pose and follows an open palm at 28 cm.
5. One open palm held still under the gripper for 1.5 s: it drops the ball, opens and closes 3 times fast, then folds back to rest.

**Hand lost:** the arm holds. After 5 s it goes back to the present pose and follows again when the palm comes back. After 45 s without a drop, it folds back to rest and puts the ball down at the put-down pose.

"Put in container" picks the same way and puts the ball down at the put-down pose. Red and blue are greyed out: there is no red or blue pick skill.

The same program can also dance: see `docs/dance.md`.

The program is the supervisor (#17), on `main` since #47; why it works this way is in `decisions/supervisor.md` and `decisions/dashboard.md`. Hand on the power switch for every round.

## 0. Once: code and environments (WSL, about 10 min on good Wi-Fi)

```bash
cd ~/lerobot-hackathon && git fetch origin
git worktree add --detach ~/demo origin/main
cp ~/track/hand_landmarker.task ~/demo/
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n dashboard python=3.12
conda activate dashboard
pip install -r ~/demo/scripts/dashboard/requirements.txt
```

To update later: `cd ~/demo && git fetch origin && git checkout --detach origin/main`.

- The `lerobot` and `handtrack` environments from the hand-tracking test are used as they are.
- Nothing is installed on Windows: the dashboard takes the microphone through the browser.

## 1. Try the dashboard without the arm (3 min)

Terminal 1, a stand-in for the robot program (scripted rounds, test pattern instead of cameras):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate dashboard && cd ~/demo
python scripts/dashboard/fake_robot.py --speed 2
```

Terminal 3, the dashboard. The first start downloads the speech model, about 660 MB:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate dashboard && cd ~/demo
python scripts/dashboard/server.py
```

Open **http://localhost:8080** in Chrome or Edge on Windows. Check:
- The header says `FAKE robot program`, and `Speech: ready` once the model has loaded.
- The **Striped → ✋** tile starts a scripted round. Both camera tiles show a moving test pattern, and the state strip moves Rest → Pick → Check ball → Hand over → Done.
- 🎤: click (allow the microphone), say "give me the ball", click again. The chat shows what it heard and `▶ Starting`. Holding Space works the same.
- ABORT during a round ends it as `aborted`.

Ctrl+C in terminal 1 (leave the dashboard running; it reconnects by itself).

## 2. Put-down pose (2 min, torque off)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

Torque goes off: hold the arm. Put the closed gripper about 10 cm above the container used for "put in container" (or the middle of the bowl if there is none). Read the five joint values, Ctrl+C, and save them:

```bash
echo 'PUT_DOWN_POSE="<pan> <lift> <elbow> <wrist_flex> <wrist_roll>"' >> ~/.so101.env
```

## 3. Rounds on the arm

Before starting:
- Fold the arm to its rest pose by hand, gripper **open**. The program refuses to start if the gripper reads 6-20.
- Put the striped ball in the bowl. Nobody's hands in the bowl.

Terminal 1, the robot program. It owns the arm and both cameras, and shares their frames with the tracker and the dashboard:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
bash scripts/supervisor/demo_chain.sh
```

- It must not start a calibration. If it asks `Press ENTER to use provided calibration file`, press Enter.
- Any `run_supervisor.py` flag can be appended.

Terminal 2, the hand tracker:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate handtrack && cd ~/demo
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml --hand-z 0.20
```

Terminal 3, the dashboard: `python scripts/dashboard/server.py` (env `dashboard`), page at http://localhost:8080. The header must say `Robot program: running`.

What lands in `~/so101/demo/` per start of `demo_chain.sh`:
- `chain_<time>.jsonl`: the events (transitions, grasp checks, round results).
- `chain_<time>.ticks.jsonl`: one line per control tick (policy action, command, goal sent, measured joints, gripper load).
- `chain_<time>_roundNN.mp4` + `.frames.csv`: both cameras for each round, written by the dashboard. The header shows `● REC` while it records. **Keep the dashboard running during rounds**, or there is no video.

Push them to `molnrb/arm-logs` after the session, as for exp1.

Start a round with the **Striped → ✋** tile, the chat, or 🎤 "give me the ball". `h` + Enter in terminal 1 still works. The dashboard refuses a new command while a round runs. Expect:

1. **Rest**: a small move into the rest pose, gripper 30.
2. **Pick** (up to 90 s): the policy digs and grasps.
   - The grasp check pauses the arm for about 0.4 s at each close.
   - On a miss it opens and tries again, 3 picks at most.
3. **Check ball**: the live panel shows the wrist areas, for example `target: striped 274 px (need 120)`.
4. **Hand over**: hold one open palm about 20 cm above the table, 15-35 cm in front of the base.
   - The gripper follows at 28 cm. Hold the palm still under it for 1.5 s: drop, 3 fast close-open cycles.
   - The live panel shows `following the hand`, and `hand lost: home in …` when the tracker loses it.
5. **Rest**: it folds back. The round appears under **Rounds** with its outcome.

**Watch the first move from rest to the present pose.** If it comes near the bowl or the table, press **ABORT**.

**Stop:** ABORT on the page, or `a` + Enter in terminal 1. The arm stops where it is with torque on and the gripper is not opened. Hold the arm, then switch the power off.

If the control loop on the live panel reads well under 30 Hz, say so on the board.

If something misbehaves, append to the `demo_chain.sh` line:
- `--no-grasp-check` if the grasp check pauses or re-picks in a silly way;
- `--no-held-check` if the wrist check errors;
- `--min-held-target-area 0` if the wrong-ball check rejects real striped holds. Post the areas the live panel showed;
- `--max-step-deg 1.0` for slow moves (the policy then picks worse).

## 4. The test

Run 5 rounds:
- 2 with the striped ball visible;
- 2 half covered;
- 1 under 3 or more balls.

In one round, take the hand away for about 6 s during the handover and bring it back: the arm should go back to the present pose and then follow again.

After each round, click your result in its row (success, wrong ball, empty, dropped, digging, other) and add a note.

## 5. Post on the board (issue #1)

Click **Copy board summary** and paste it into a comment. It has every round with its outcome, your result and note, the tally and the grasp checks.

**Passed** when:
- at least 4 of 5 rounds run voice → pick → handover → rest with no help beyond speaking and the hands;
- the gripper never goes below about 20 cm during the handover;
- the hand-lost round recovered.
