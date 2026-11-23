# Evaluating pick policies on the arm

`scripts/eval/eval_pick.sh` runs one or more pick policies on the real arm, run after run, and writes each result to a CSV. It uses its own control loop instead of `lerobot-rollout`, because rollout disconnects after every run, which switches torque off and drops the ball.

It is not the demo: the demo program adds a grasp check, a wrist-camera ball check and retries (`kb/demo-chain-vs-eval.md`). Use the eval to compare checkpoints and settings, then confirm the winner in the demo.

Before the first time: `docs/setup.md`. It reads ports, robot id and camera paths from `~/.so101.env`.

## Start

Inside WSL, repo root, in a terminal you can type in (the arm only moves after SPACE):

```bash
bash scripts/eval/eval_pick.sh --policy hidden=TianK003/act_hidden_from_v2@002000 \
    --policy v2=Space0code/act_redblue_from_foil_v2@20000 --n-action-steps 30
```

From PowerShell, in its own window:

```powershell
Start-Process -FilePath wsl.exe -ArgumentList '-e','bash','-lc','"bash /mnt/c/<repo>/scripts/eval/eval_pick.sh --policy v2=Space0code/act_redblue_from_foil_v2@20000"'
```

Check first that the devices are there (`ls /dev/so101_follower /dev/v4l/by-id/`; else `attach_usb.sh`), and stop any other program that holds the arm or the cameras.

## Options

- `--policy [name=]<Hub repo or local run dir>[@step]`: repeat for each policy. Without `@step` the newest checkpoint is used; the script prints which (`Loading v2: Space0code/act_redblue_from_foil_v2@20000`). Both repo layouts work, `checkpoints/<step>/` and `<step>/` at the top. A local run folder works too.
- `--runs 10`: layouts (default 10). With two policies that is 20 runs.
- `--seconds 90`: time cap per run (default 90). Shorter `n_action_steps` is slower; a 75 s cap cut good runs on deeply buried balls.
- `--n-action-steps N`: override for all policies (default: each checkpoint's own, 100 for our ACT). See "Settings that matter" below.
- `--task "<sentence>"`: task sentence for language-conditioned policies (SmolVLA). ACT ignores it.
- `--record <dir>`: save every run as one LeRobot episode (below).
- `--no-telemetry`, `--push-to-hub <repo_id>`: with `--record` only.
- `--log-dir`: where the CSV goes (default `~/so101/evals`).

Any policy type works; it is read from the checkpoint's config. The script stops at start-up if a checkpoint expects other camera names than `top` and `wrist`. SmolVLA checkpoints trained with the camera rename carry it in their saved preprocessor.

## How a session runs

**Layouts are shared.** A layout is one arrangement of the striped ball and the red and blue balls. With two policies every layout is run once by each, one right after the other, and the order alternates (layout 1: A then B; layout 2: B then A). The prompt says whether to set up **a NEW layout** or **the SAME layout as the last run**; rebuild it as closely as you can: target position and depth, which balls cover it.

One run:

1. Set up the bowl as the prompt says and press **SPACE**.
2. The arm moves slowly to HOME (the folded rest pose every training episode starts and ends in) with the gripper half open.
3. The policy runs at 30 Hz.
4. The run ends at the first of:
   - **returned**: the arm left HOME and is back there with the gripper reading 6-20 (something held) for 1 s;
   - **timeout**: `--seconds`;
   - **operator**: you pressed SPACE;
   - **aborted**: you pressed ESC or q.

   Then nothing more is sent. Torque stays on and the arm holds its pose, so a held ball stays in the gripper.
5. Answer with one key:
   - Ball at the start, once per layout: **v** visible, **p** partly covered, **h** hidden.
   - Result: **s** success (striped ball held at HOME), **w** wrong ball, **e** empty, **d** dropped, **t** still digging at the timeout, **n** never started, **o** other.
6. Hold a hand under the gripper and press SPACE: the gripper opens and the ball falls into your hand.
7. After the last layout it prints the tally per policy and start state. SPACE adds 5 more layouts; **q** quits (the arm goes HOME first, then torque goes off).

Every run is one row in `~/so101/evals/eval_<date>_<time>.csv`: run, layout, policy, checkpoint, `n_action_steps`, seconds, end reason, gripper reading, share of slow control ticks, start state, result. It is written after every run.

## Safety

- **The arm moves by itself after SPACE.** Hands out of reach, someone at the power switch.
- **ESC is the safe stop**: the policy stops and the arm holds its pose with torque on.
- **Ctrl+C or closing the window switches torque off and the arm goes limp. Hold the arm first.**
- Watch the first move to HOME of a session: from an unusual pose it moves for a few seconds.

## Choosing layouts

- Mix the start states, e.g. for 10 layouts: 3 visible, 3 partly covered, 4 hidden under 1-3 balls, with the target in different places (centre, against the rim).
- Put the red and blue balls back into the bowl after every run. The containers start empty.
- Lighting and the top camera as in the recordings, exposure locked. Nothing else on screen (no Rerun).

## Reading the result

- 10 layouts per policy only separate big gaps: two blocks of ten need a gap of about 6 successes before it is more than chance. Shared layouts help, because each pair faces the same difficulty.
- If the gap is small, add 5-10 more layouts (SPACE at the end) rather than declaring a winner.
- Report raw counts per start state, e.g. "hidden 3/5, partly 2/3, visible 2/2", with the checkpoint, `n_action_steps` and time cap.
- Slow ticks above about 10% mean the loop stutters. Check the gripper motor's temperature if runs get worse over a session (`python scripts/so101/motor_health.py`).

## Settings that matter

- **`n_action_steps`** is how many of ACT's 100 predicted actions run before it looks at the cameras again. It is read at run time, so any value up to the chunk size works on an existing checkpoint (`kb/act-replan-override.md`). On our arm, 25-50 picked better than 100, especially on hidden balls (`decisions/hidden-ball-picking.md`). Temporal ensembling is too slow for the RTX 4050.
- **HOME and the gripper** at the start match the training episodes (gripper 17.8). The demo starts at gripper 30.

## Recording the runs

With `--record <dir>`, each run is one episode of a LeRobot dataset with the training features (30 fps, `top` and `wrist` video, state, action, `task`). Use it to look at failures, or as material for corrections.

```bash
bash scripts/eval/eval_pick.sh --record ~/so101/evals/ds_$(date +%Y%m%d) --policy act=TianK003/act_hidden_from_v2@002000
```

- Frames are kept in memory during the run and written after you enter the result, while the arm holds its pose, so the 30 Hz loop is not slowed. Saving takes about 10 s after a 90 s run. A crash loses only the run in progress.
- `<dir>/runs.jsonl` has one line per episode: `episode_index`, run, layout, policy, checkpoint, `policy_type`, `n_action_steps`, `git_commit`, start state, end, result, seconds, gripper reading, tick timing and motor temperatures at start and end. The CSV gets an `episode_index` column.
- **Telemetry** (on by default with `--record`; `--no-telemetry` turns it off): extra per-tick features the policy never sees: `observation.motor_current`, `motor_load`, `motor_temperature`, `motor_voltage` (6 raw servo values each), `observation.wall_time` and `observation.tick_ms`. Current and load are read every tick, temperature and voltage once a second. A failed read gives NaN. Check the slow-tick share in the first run with and without it.
- An existing dataset in `<dir>` gets the new runs appended; it keeps the feature set it was created with, so mixing telemetry on and off stops at start-up.
- Load it with `LeRobotDataset("local/eval_pick", root="<dir>")`.

Without hardware, `python tests/eval_pick_record.py` runs the loop and the recording against a fake robot.
