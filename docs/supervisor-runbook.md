# Supervisor runbook

Runs the demo round: PICK (combined ACT policy) -> check the gripper -> GIVE into a hand or PUT DOWN -> back to rest. Why it works this way: `decisions/supervisor.md`.

## Without the robot (any laptop)

```bash
source .venv/bin/activate
python tests/supervisor_scenarios.py                       # fake scenarios, voice link, policy loader; prints PASS
python tests/supervisor_scenarios.py --handover-dir <dir with #13's handover_loop.py>   # adds GIVE on a simulated arm
python scripts/supervisor/run_supervisor.py --fake all     # one line per scripted scenario
python scripts/supervisor/run_supervisor.py --fake pick_fails_once --verbose   # every transition as JSON
```

## On the robot (a person at the table, power switch in reach)

Before the first run:

1. The arm is at its folded rest pose with the gripper **empty and open** (the program refuses to start if the gripper reads 6-20, the holding band).
2. Hand tracker (#15) and hand-following loop (#13) are on `main` or in `scripts/handover/` and `scripts/handtrack/`. Without #13's `hand_socket.py` there are no hand messages and every round ends in PUT DOWN. Without camera calibration use `--give-mode present`: the arm holds the present pose over a taped spot and drops when two flat palms are seen. This needs #13's `hand_socket.py` with `newest_palm_count()` (it reads the tracker's `pixels` list, presence only, no position); an older receiver only sees calibrated 3D hands, so without calibration there are no palms and every round ends in PUT DOWN. Follow mode always needs the calibration.
3. Teach poses with the arm there (torque off, `read_pose.py` from #13), five joint values each: the **present pose** over the hand spot (`--present-pose`), the generic **put-down pose** (`--put-down-pose`, used for the striped ball) and, for the new table (11 red and 11 blue balls, the striped target, fixed red and blue sorting containers), the **red container** (`--container-pose-red`) and **blue container** (`--container-pose-blue`). The two container poses are placeholders until taught at the table: without them a red or blue ball goes to `--put-down-pose` and the program prints a WARNING. Do not run a colour ball over a container before the pose is taught.

```bash
python scripts/supervisor/run_supervisor.py --robot \
  --port <follower port> --top-camera <index> --wrist-camera <index> --robot-id mi_follower \
  --present-pose <5 joint values> --put-down-pose <5 joint values> \
  --container-pose-red <5 joint values> --container-pose-blue <5 joint values> \
  --voice --look --rounds 10 --log outputs/supervisor.jsonl
```

Then start the hand tracker (the receiver now ignores closed hands itself; `--only-open` is optional, `--any-hand` on #13's loop turns the filter off) (it reads the frames this program shares) and, for voice, `scripts/voice/listen.py` (`docs/voice-runbook.md`).

- **Enter** starts a round. `h` + Enter: give to a hand, `c` + Enter: container, `s`/`r`/`b` + Enter: target striped/red/blue, `x` + Enter: clear both. A voice command also starts a round. **A command is for one round**: destination and target are cleared when the round ends, so the next round starts without a preference.
- **Targets and containers:** `striped` (the default pick policy), `red`, `blue`. Destination `container` means the container of the ball's colour (red -> red, blue -> blue); the striped ball goes to `--put-down-pose`. A put-down after a failed hand decision uses the same container. **Red and blue picks are not wired yet**: a round that asks for a colour ends at once as `unsupported_target` with the arm untouched, until the colour pick (branch `jakobnunnendorf/colour-pick`, `scripts/vision/colour_detector.py`) is plugged into `Skills.pick_colour`. The colour of the ball in the gripper comes from the hook `Sensors.ball_colour` (default: the colour that was asked for); a detector replaces it. No colour detection is built in this supervisor.
- **Held-ball check (target or a red/blue ball?):** the gripper reads the same for both, so after a pick the wrist camera checks the ball at rest with #24's colour detector, inside the region where the jaws hold the ball (`--held-check-roi CX CY RX RY`, an ellipse in frame fractions, default `0.5 0.5 0.25 0.3`: a guess at the frame centre, check it on a few wrist frames with a held ball). Not the target: it is put down (never handed over) and PICK runs again within `--max-pick-attempts`; the round ends `wrong_ball` when the picks run out. Before #24 is merged add `--vision-dir <colour-pick worktree>/scripts/vision`.
  - **Threshold:** without `--min-held-target-area` the check only logs (`held_check: unmeasured`). **Suggested start: `--min-held-target-area 120`**, from offline wrist frames of `molnrb/so101_pick_rb_mixed` (board 7063041111: striped area with the target held at rest p25 236, median 370; a covering ball median 0, p95 564; at 120 about 12% of targets wrongly rejected, about 8% covering balls wrongly accepted, measured over the bowls, so pessimistic; measured on whole frames, before the region). Confirm at the table: read `held_area_px` from the VERIFY_HOLD log lines holding the striped ball and holding a red and a blue ball, and move the threshold between them.
  - **Only the striped ball is judged** (`--held-check-colours`, default `striped`). In a red or blue round the area is logged as `held_check: untrusted` and the ball is accepted as without the check: blue read up to ~218,000 px on whole wrist frames whatever was held (bluish white balance, the camera faces the ceiling at rest), and red was never measured as the held ball. Add `red` or `blue` only after their areas inside the region separate at the table.
  - **Cutting the ~12% missed targets:** #24's striped colour range was tuned on the top camera. Re-tune the striped spec on wrist frames with #24's calibrate mode (`scripts/vision/colour_detector.py`). Until then a rejected target only costs a re-pick (it is put down and picked again), never safety.
  - `--no-held-check` turns it off. If the detector fails or the wrist frame is missing the ball is accepted as before and the log says `held_check: unavailable`.
- **Crash or Ctrl-C:** the shared-memory block for the hand tracker is closed on exit, and a stale block of the same name left by an earlier crash is removed at start.
- **`a` + Enter (or Ctrl-C) = ABORT.** The arm stops where it is with torque on, the gripper is not opened. Hold the arm, then switch the power off.
- **Watch the first move from rest to the present pose** (default `0 -55.3 5.6 91.8 0`): in the model the tip passes about 3 cm above the table. If it comes close to the bowl or the table, stop and change `--present-pose`.
- `--max-step-deg 1.0` caps every arm joint at 1 degree per tick (30 deg/s) in every skill, the policy included. Use it for the first runs with a person's hand; it changes how the policy moves.
- Voice with the robot program in **WSL** and the listener on Windows: add `--voice-host 0.0.0.0`. With 127.0.0.1, 2 of 8 starts were unreachable from Windows for 11-15 s (`decisions/voice-commands.md`).
- `--clear-gray-policy <checkpoint>` turns on CLEAR GRAY. It is for the **old foil-ball setup** (gray tinfoil balls, replaced by red and blue balls on 2026-11-21, board 7062034477) and stays off; LOOK's target mask is also for the striped ball only.
- `--hold-range 6 20` (measured holds 9.3-16.4, open jaws 17-40) and `--max-pick-attempts 3` are flags. Every VERIFY_HOLD reading is in the log: check it after the first rounds, and read a foil ball once.
- `--n-action-steps 20` replans ACT every 0.67 s instead of 3.3 s; `--camera-rename top=camera1 wrist=camera2` for a SmolVLA checkpoint.

## Counting for H5 / H7

```bash
python -c "import json,collections; print(collections.Counter(json.loads(l)['outcome'] for l in open('outputs/supervisor.jsonl') if '\"round_end\"' in l))"
```

Post the raw counts on the board (e.g. "given 6, put_down 2, gave_up 1, aborted 1 of 10"), and which checkpoint ran.
