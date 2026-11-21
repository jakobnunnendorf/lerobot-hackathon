# Runbook: compare pick policies on the arm with `eval_pick.sh`

For the agent on the robot laptop (Balazs's) and the person at the table. Current use: Tomi's `Space0code/act_redblue_from_foil` (ACT fine-tuned from the foil policy) against Kamel's `act_redblue` (ACT from scratch). Both are trained on the same red/blue data with the same settings, so only the starting weights differ. Results go onto the board.

## How a comparison session works

- Each policy loads its **newest checkpoint** at start-up. The script prints which step it picked (`Loading foil: Space0code/act_redblue_from_foil@020000`). Pin a step with `@<step>`, e.g. `Space0code/act_redblue_from_foil@015000`.
- **Layouts are shared.** With two policies, every layout (one arrangement of the target ball and the red and blue balls) is run once by each policy, one right after the other:
  - layout 1: foil, then scratch;
  - layout 2: scratch, then foil;
  - and so on, so neither always goes first.

  The prompt says whether to set up **a NEW layout** or **the SAME layout as the last run**. Rebuild it as closely as possible: target position and depth, which balls cover it.
- `--runs 10` means 10 layouts, so 20 runs with two policies. With one `--policy` it is plain repeated runs.

## What one run does

1. The person sets up the bowl as the prompt says and presses **SPACE**.
2. The arm moves slowly (30°/s) to HOME, the folded rest pose every red/blue episode starts and ends in, with the gripper half open.
3. The policy runs at 30 Hz: find the striped ball, sort red and blue balls into the containers if it is buried, pick it, come back to HOME.
4. The run ends at the first of:
   - **returned**: the arm has left HOME and is back there with the gripper reading 6-20 (something held) for 1 s;
   - **timeout**: 90 s;
   - **operator**: the person pressed SPACE;
   - **aborted**: the person pressed ESC or q.

   After that nothing more is sent. **Torque stays on and the arm holds its pose**, so a held ball stays in the gripper.
5. One key each:
   - Ball at start, asked once per layout: **v** visible, **p** partly, **h** hidden.
   - Result: **s** success (target ball held at HOME), **w** wrong ball (red or blue held), **e** empty, **d** dropped, **t** still digging at the timeout, **n** never started, **o** other.
6. The person holds a hand under the gripper and presses SPACE. The gripper opens and the ball falls into the hand.
7. After the last layout the script prints the tally per policy: SPACE adds 5 more layouts, **q** quits (the arm goes HOME first, then torque goes off).

Every run is one row in `~/so101/evals/eval_<date>_<time>.csv` (inside WSL): run, layout, policy, checkpoint, `n_action_steps`, seconds, end reason, gripper reading, share of slow control ticks, start state and result. The file is written after every run.

## Setup (agent, once per session)

1. In the repo inside WSL, get the scripts from `main` without leaving the current branch, because `scripts/so101/` lives on PR #6's branch: `git fetch origin && git checkout origin/main -- scripts/eval docs/eval-pick-runbook.md`.
2. The scripts use `~/.so101.env` and the conda env `lerobot` (LeRobot 0.7.0), as set up for `scripts/so101/` (PR #6). They read the ports, robot id and camera paths from `SO101_FOLLOWER_PORT`, `SO101_FOLLOWER_ID`, `SO101_CAM_TOP` and `SO101_CAM_WRIST`. Without that file they fall back to the values in `so101.env.example`.
3. Hub checkpoints download on start-up (about 200 MB each, public, no login). A local run folder works too: give the folder that holds `checkpoints/<step>/pretrained_model`. The newest numbered step is used and `last` is ignored.
4. Check that the devices are there: `ls /dev/so101_follower /dev/v4l/by-id/`. If not, run `attach_usb.sh` (so101-robot skill, known failures). Stop any other program that holds the arm or the cameras (teleop, recording, `run_loop.sh`, the hand-following loop).

## Start (the person runs it, or the agent opens the window; the arm only moves after SPACE)

Inside WSL, repo root, with `<KAMEL>` replaced by Kamel's checkpoint location from the board:

```bash
bash scripts/eval/eval_pick.sh --policy foil=Space0code/act_redblue_from_foil --policy scratch=<KAMEL>
```

From PowerShell, in its own window (the person needs a window to type in):

```powershell
Start-Process -FilePath wsl.exe -ArgumentList '-e','bash','-lc','"bash /mnt/c/<repo>/scripts/eval/eval_pick.sh --policy foil=Space0code/act_redblue_from_foil --policy scratch=<KAMEL>"'
```

Options:

- `--runs 10`: layouts (default 10).
- `--seconds 90`: time cap per run (default 90).
- `--n-action-steps 20`: override for all policies (default: each checkpoint's own, 100).
- `--policy name=<repo or folder>[@step]`: repeat for each policy.

## Safety (tell the person before the first SPACE)

- **The arm moves by itself after SPACE.** Hands out of reach. Someone stays at the power plug.
- **ESC is the safe stop:** the policy stops and the arm holds its pose with torque on.
- **Ctrl+C or closing the window switches torque off and the arm goes limp: hold the arm first.**
- Watch the first move to HOME of the session. If the arm starts somewhere unusual it moves for a few seconds.

## Choosing the layouts

- Mix the start states. Suggested for 10 layouts: 3 visible, 3 partly covered, 4 hidden under 1-3 balls, with the target in different places (centre, against the rim).
- Red and blue balls go back into the bowl after every run. The containers start empty.
- Lighting and the top camera as in the recordings. Nothing on screen (no Rerun).

## Reading the result

- 10 layouts per policy only separate big gaps. Two blocks of ten need a gap of about 6 successes before it is more than chance (board 7064727685). Shared layouts help, because each pair of runs faces the same difficulty.
- If the gap is small, add 5-10 more layouts (SPACE at the end) rather than declaring a winner.

## Afterwards (agent)

Post on the board:
- the settings line: checkpoints with their steps as printed at start-up, `n_action_steps`, 90 s, laptop;
- the tally per policy and start state (the last lines the script prints);
- the failure modes and the end-reason counts from the CSV, as raw counts, e.g. "foil hidden 2/4, scratch hidden 1/4";
- anything odd: slow ticks above about 10% (stutter), camera errors, the gripper running hot (`motor_health.py`).

## Tested so far

- Tested on 2026-11-21 on a Mac with LeRobot 0.7.0 on the CPU, with a fake robot and scripted keys:
  - newest-step resolution for a Hub repo and for a LeRobot run folder;
  - two policies alternating on shared layouts;
  - the `n_action_steps` override;
  - the control loop with the real policy;
  - the end on SPACE and on timeout;
  - the questions, the CSV, and HOME before every run and before quitting.
- **Not tested:** the real arm, the cameras, and keys typed into a WSL terminal. Watch the first run closely.

Written by Claude
