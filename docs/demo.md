# Running the demo

One round: you ask for the ball (dashboard tile, voice or keyboard). The arm picks the striped ball out of the bowl with a learned ACT policy, digging if it is buried. The wrist camera checks that it holds the striped ball; a red or blue ball is put down and picked again. The arm carries the ball to the present pose, follows one open palm from 28 cm above, drops the ball into it, and folds back to rest.

"Put in container" picks the same way and puts the ball down at the put-down pose. The same program also dances (`docs/dance.md`) and, experimentally, bites a pointing finger (`docs/finger-biting.md`).

The program is the supervisor, `scripts/supervisor/run_supervisor.py`, started by `scripts/supervisor/demo_chain.sh`. Why it works this way: `decisions/supervisor.md` and `decisions/dashboard.md`.

Before the first time: `docs/setup.md`. The top camera calibration (`scripts/calib/intrinsics.yaml`, `extrinsics.yaml`) is only valid for our camera mount; if the top camera moved, redo it (`docs/camera-calibration.md`).

## Try the dashboard without the arm

Terminal 1, a stand-in for the robot program (scripted rounds, a test pattern instead of the cameras):

```bash
conda activate dashboard
python scripts/dashboard/fake_robot.py --speed 2
```

Terminal 2, the dashboard. The first start downloads the speech model, about 660 MB:

```bash
conda activate dashboard
python scripts/dashboard/server.py
```

Open http://localhost:8080 in Chrome or Edge on Windows. The header says `FAKE robot program`, and `Speech: ready` once the model has loaded. The **Striped → ✋** tile starts a scripted round; ABORT ends it as `aborted`. Ctrl+C in terminal 1; the dashboard reconnects by itself when the real robot program starts.

Without the arm or the dashboard, the supervisor runs scripted scenarios: `python scripts/supervisor/run_supervisor.py --fake all`, and `python tests/supervisor_scenarios.py` prints PASS.

## Once: the put-down pose

```bash
conda activate lerobot
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

Torque goes off: hold the arm. Put the closed gripper about 10 cm above the container (or the middle of the bowl). Read the five joint values, Ctrl+C, and save them:

```bash
echo 'PUT_DOWN_POSE="<pan> <lift> <elbow> <wrist_flex> <wrist_roll>"' >> ~/.so101.env
```

`demo_chain.sh` refuses to start without it.

## Before every session: preflight

In WSL, `lerobot` env, repo root:

```bash
python scripts/preflight.py --policy TianK003/act_hidden_from_v2@002000
```

One line per check, PASS / WARN / FAIL, with the fix under each WARN and FAIL. It never writes to the motors. Fix every FAIL before starting a robot program.

- WARN on the tracker (8766) and voice (8765) lines is normal before those programs run. Run again with `--strict-services` afterwards to make them FAIL.
- The ports and cameras come from `SO101_FOLLOWER_PORT`, `SO101_CAM_TOP` and `SO101_CAM_WRIST` in the environment, else our default paths. It does not read `~/.so101.env` by itself: run `set -a; source ~/.so101.env; set +a` first, or pass `--port`, `--top-camera`, `--wrist-camera`.
- `--read-motors` opens the serial port and reads each motor's position (read only). The arm program must not be running.
- `--load-policy` loads the weights on the GPU; without it only the config is read.
- `--skip camera,hf` skips checks (python, cuda, port, calibration, camera, exposure, policy, tracker, voice, disk, hf, motors).
- Camera lines fail if another program has the camera open.

**Lock the camera exposure** after every replug, under the lamp you demo with. A replug resets the cameras to auto exposure and white balance, and the policy sees a different picture:

```bash
python scripts/preflight.py --lock-exposure     # --unlock-exposure puts the saved controls back
```

It saves the original controls to `~/.so101_camera_controls.json` on the first lock. Cameras only, never the motors.

## Rounds on the arm

Before starting:

- Fold the arm to its rest pose by hand, gripper **open**. The program refuses to start while the gripper reads 6-20 (the holding band).
- Put the striped ball in the bowl. No hands in the bowl. Someone at the power switch.

**Terminal 1, the robot program.** It owns the arm and both cameras and shares their frames with the tracker and the dashboard. The pick model we demo with:

```bash
conda activate lerobot
PICK_POLICY=TianK003/act_hidden_from_v2 bash scripts/supervisor/demo_chain.sh --step 002000 --n-action-steps 25
```

- `TianK003/act_hidden_from_v2@002000` is ACT `Space0code/act_redblue_from_foil_v2` fine-tuned on 38 buried or partly covered episodes. `--n-action-steps 25` makes it look at the cameras again after 25 of its 100 predicted actions, so it approaches more carefully (`decisions/hidden-ball-picking.md`).
- Without `PICK_POLICY` and `--step` the script runs the fallback, `Space0code/act_redblue_from_foil_v2@20000`, at 30 action steps. `--step` must come first; it names the checkpoint folder as the repo spells it (`002000` here, `20000` in the `Space0code` repos).
- Any `run_supervisor.py` flag can be appended; a later flag wins over the script's defaults.
- It runs 10 rounds, then exits. Append `--rounds 50` for more.

**Terminal 2, the hand tracker:**

```bash
conda activate handtrack
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml --hand-z 0.20
```

**Terminal 3, the dashboard:** `python scripts/dashboard/server.py` in the `dashboard` env. Open http://localhost:8080. The header must say `Robot program: running`. Keep the dashboard running during rounds: it films them.

The top camera shows the tracked hand: green with `offering` for an open palm up, amber otherwise. Untick **✋ Hand overlay** to hide it.

## Starting a round

- **Tiles**: **Striped → ✋** (give to hand) or **Striped → 📦** (put in container). Red and blue are greyed out: there is no red or blue pick skill.
- **Voice**: click 🎤 (or hold Space), say "give me the ball" or "put it in the container", click again (or release Space). The chat shows what it heard. More phrases: `docs/voice.md`.
- **Chat**: type the same sentences.
- **Keys** in terminal 1, each followed by Enter: `h` give to hand, `c` container, `d` dance, `f` bite, `a` or `q` abort. Enter alone starts a round with no destination: the arm goes to the present pose, waits 3 s for palms, and gives or puts down.

The dashboard refuses a new command while a round runs. A command is for one round only.

**Max picks:** the Live panel's − / + changes how many pick attempts a round gets (1-20), also during a round. Press + when the ball is buried deep, so the round keeps digging.

## What to expect

1. **Rest**: a small move into the rest pose, gripper at 30.
2. **Pick** (up to 90 s per attempt, 3 attempts): the policy digs and grasps.
   - At each close the grasp check pauses the arm for about 0.4 s and reads the gripper. A miss opens, backs off and picks again.
3. **Check ball**: the live panel shows the wrist camera's colour areas, e.g. `target: striped 274 px (need 120)`. Not the striped ball: it is put down and picked again.
4. **Hand over**: hold one flat, open palm, palm up, about 20 cm above the table and 15-35 cm in front of the base.
   - The gripper follows 8 cm above it (28 cm above the table) and never goes below 20 cm.
   - Hold the palm still under it for 1.5 s: it drops the ball, opens and closes 3 times fast.
   - Hand lost: the arm holds. After 5 s it goes back to the present pose and follows again when the palm comes back. After 45 s without a drop it folds back to rest and puts the ball down at the put-down pose.
5. **Rest**: it folds back. The round appears under **Rounds** with its outcome. Click your own result in its row (success, wrong ball, empty, dropped, ...) if you are keeping count; **Copy board summary** gives all rounds as text.

**Watch the first move from rest to the present pose.** If it comes near the bowl or the table, press ABORT.

## Stopping

ABORT on the page, or `a` + Enter in terminal 1. The arm stops where it is with torque on, and the gripper is not opened. Hold the arm, then switch the power off.

Ctrl+C in terminal 1 also aborts. Never restart the program while the gripper holds the ball: connecting switches torque off for a moment and the ball drops.

## When something misbehaves

Append to the `demo_chain.sh` line:

- `--no-grasp-check`: the grasp check pauses or re-picks in a silly way.
- `--no-held-check`: the wrist check errors.
- `--min-held-target-area 0`: the wrist check rejects real striped holds. It then only logs the areas.
- `--max-step-deg 1.0`: slow moves everywhere (the policy then picks worse).
- `--max-pick-attempts N`, `--pick-timeout S`: more or longer attempts.
- `--grasp-min-load N` (default 120): the grasp check also needs this gripper load. A held ball reads the torque cap, 200; an empty close 28-44.

If the control loop on the live panel reads well under 30 Hz, the policy picks worse: close other programs (Rerun, a second camera viewer).

The full flag list: `python scripts/supervisor/run_supervisor.py --help`.

## What gets written

In `~/so101/demo/`, per start of `demo_chain.sh`:

- `chain_<time>.jsonl`: events (transitions, grasp checks, wrist checks, round results).
- `chain_<time>.ticks.jsonl`: one line per control tick (policy action, command, goal sent, measured joints, gripper load).
- `chain_<time>_verify_hold_r<round>_p<attempt>_<time>.jpg`: the wrist frame of each ball check.
- `chain_<time>_roundNN.mp4` + `.frames.csv`: both cameras for each round, written by the dashboard. The header shows `● REC` while it records.

Count outcomes:

```bash
python -c "import json,collections,sys; print(collections.Counter(json.loads(l)['outcome'] for l in open(sys.argv[1]) if '\"round_end\"' in l))" ~/so101/demo/chain_<time>.jsonl
```

## Multi-step plans (experimental, not reliable)

You can ask for balls to be cleared before the striped pick: "take out two red balls, then give me the striped ball", or in terminal 1 `p 2r 2b s h` (2 red, 2 blue, then striped to the hand). Each colour step runs that colour's own ACT policy until it has dropped that many balls into the colour's hole.

Set the clearing policies in `~/.so101.env` or on the command line, as `PICK_POLICY` (Hub id or local dir, optionally `@step`):

```bash
CLEAR_RED_POLICY=TianK003/act_clear_red CLEAR_BLUE_POLICY=TianK003/act_clear_blue \
  PICK_POLICY=TianK003/act_hidden_from_v2 bash scripts/supervisor/demo_chain.sh --step 002000 --n-action-steps 25
```

A colour left unset refuses its plans. On our arm the clearing policies made empty grasps. How a step runs and ends: `decisions/multistep-step-state.md`.
