# Runbook: 10 pick runs of a checkpoint with `eval_pick.sh`

For the agent on the robot laptop (Balazs's) and the person at the table. First use: checkpoint 15K of `Space0code/act_redblue_from_foil` (ACT, fine-tuned from the foil policy on the red/blue data). Results go into the counting sheet (`docs/eval-counting-sheet.md`) and onto the board.

## What one run does

1. The person sets up the bowl and presses **SPACE**.
2. The arm moves slowly (30°/s) to HOME, the folded rest pose every red/blue episode starts and ends in, with the gripper half open.
3. The policy runs at 30 Hz: find the striped ball, sort red and blue balls into the containers if it is buried, pick it, come back to HOME.
4. The run ends at the first of:
   - **returned**: the arm has left HOME and is back there with the gripper reading 6-20 (something held) for 1 s;
   - **timeout**: 90 s;
   - **operator**: the person pressed SPACE;
   - **aborted**: the person pressed ESC or q.

   After that nothing more is sent. **Torque stays on and the arm holds its pose**, so a held ball stays in the gripper.
5. The script asks two questions, one key each:
   - Ball at start: **v** visible, **p** partly, **h** hidden.
   - Result: **s** success (target ball held at HOME), **w** wrong ball (red or blue held), **e** empty, **d** dropped, **t** still digging at the timeout, **n** never started, **o** other.
6. The person holds a hand under the gripper and presses SPACE. The gripper opens and the ball falls into the hand.
7. The next run waits for SPACE. After 10 runs the script shows the tally: SPACE adds 5 more runs, **q** quits (the arm goes HOME first, then torque goes off).

Every run is one row in `~/so101/evals/eval_<date>_<time>.csv` (inside WSL): checkpoint, `n_action_steps`, seconds, end reason, gripper reading, share of slow control ticks, start state and result. The file is written after every run, so nothing is lost if the script stops.

## Setup (agent, once)

1. In the repo inside WSL, get the scripts without leaving the current branch, because `scripts/so101/` lives on PR #6's branch: `git fetch origin && git checkout origin/Space0code/eval-pick-loop -- scripts/eval docs/eval-pick-runbook.md` (after PR #36 is merged, `origin/main` instead). The scripts are `scripts/eval/eval_pick.py` and `scripts/eval/eval_pick.sh`.
2. It uses `~/.so101.env` and the conda env `lerobot` (LeRobot 0.7.0) as set up for `scripts/so101/` (PR #6). It reads the ports, robot id and camera paths from `SO101_FOLLOWER_PORT`, `SO101_FOLLOWER_ID`, `SO101_CAM_TOP` and `SO101_CAM_WRIST`. Without that file it falls back to the values in `so101.env.example`.
3. The checkpoint downloads from the Hub on first start (about 200 MB, public, no login).
4. Check that the devices are there: `ls /dev/so101_follower /dev/v4l/by-id/`. If not, run `attach_usb.sh` (so101-robot skill, known failures). Stop any other program that holds the arm or the cameras (teleop, recording, `run_loop.sh`).

## Start (the person runs it, or the agent opens the window; the arm only moves after SPACE)

Inside WSL, repo root:

```bash
bash scripts/eval/eval_pick.sh --step 015000
```

From PowerShell, in its own window (the person needs a window to type in):

```powershell
Start-Process -FilePath wsl.exe -ArgumentList '-e','bash','-lc','"bash /mnt/c/<repo>/scripts/eval/eval_pick.sh --step 015000"'
```

Options: `--runs 10` (default), `--seconds 90` (default), `--n-action-steps 20` for the comparison on the counting sheet (default: the checkpoint's 100), `--policy <repo id or local dir>` for another model, no `--step` for the newest checkpoint at the repo root.

## Safety (tell the person before the first SPACE)

- **The arm moves by itself after SPACE.** Hands out of reach. Someone stays at the power plug.
- **ESC is the safe stop:** the policy stops and the arm holds its pose with torque on.
- **Ctrl+C or closing the window switches torque off and the arm goes limp: hold the arm first.**
- Watch the first move to HOME of the session. If the arm starts somewhere unusual it moves for a few seconds.

## Setting up the 10 runs

- Mix the start states and write them down honestly. Suggested: 3 visible, 3 partly covered, 4 hidden under 1-3 balls, with the target in different places (centre, against the rim).
- Red and blue balls go back into the bowl after every run. The containers start empty.
- Lighting and the top camera as in the recordings. Nothing on screen (no Rerun).

## Afterwards (agent)

Post on the board: the settings line (checkpoint, `n_action_steps`, 90 s, laptop), the tally per start state (the last line the script prints), the failure modes, and the end-reason counts from the CSV. Raw counts, for example "hidden 2/4". Mention anything odd: slow ticks above about 10% (stutter), camera errors, the gripper running hot (`motor_health.py`).

## Tested so far

- Tested on 2026-11-21 on a Mac with LeRobot 0.7.0 on the CPU, with a fake robot and scripted keys: checkpoint loading (local and Hub `--step`), the `n_action_steps` override, the control loop with the real policy, the end on SPACE and on timeout, the result questions, the CSV, HOME before every run and HOME before quitting.
- **Not tested:** the real arm, the cameras, and keys typed into a WSL terminal. Watch the first run closely.

Written by Claude
