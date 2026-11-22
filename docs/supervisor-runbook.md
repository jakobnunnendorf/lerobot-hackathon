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
- **Held-ball check (target or a red/blue ball?):** the gripper reads the same for both, so after a pick the wrist camera checks the ball at rest with #24's colour detector, inside the region where the jaws hold the ball (`--held-check-roi CX CY RX RY`, an ellipse in frame fractions). Default `0.52 0.72 0.2 0.25`, the lower middle of the frame: from PR #30 (`kb/held-ball-wrist-areas.md`, board 7063126045), the striped ball's centre at x 0.49-0.62, y 0.64-0.77 over 183 offline detections in `molnrb/so101_pick_rb_mixed`, checked on overlay frames. Not the target: it is put down (never handed over) and PICK runs again within `--max-pick-attempts`; the round ends `wrong_ball` when the picks run out. Before #24 is merged add `--vision-dir <colour-pick worktree>/scripts/vision`. Every VERIFY_HOLD line logs the areas of all three balls (`held_areas_px`), whichever was asked for.
  - **Striped threshold:** without `--min-held-target-area` the check only logs (`held_check: unmeasured`). **Suggested start: `--min-held-target-area 120`** (offline, inside the default region: target held at rest p25 235, median 370; covering ball over a container p95 0, max 1694; at 120 about 12.5% of targets wrongly rejected, 4.5% of covering balls wrongly accepted, the latter pessimistic because over the containers the camera looks into the bowl).
  - **Unseen:** when every ball reads below `--held-unseen-floor` (30 px) the held ball is unseen, not wrong (a deep grasp holds it above the region; board 7065772934): it is kept as the gripper band says, logged `held_check: unseen`. With `--log`, each VERIFY_HOLD saves its wrist frame as `<log name>_verify_hold_r<round>_p<attempt>_<time>.jpg` next to the log (`wrist_frame` in the transition).
  - **Per-ball thresholds:** `--min-held-area red=2000 striped=120` (`striped=` overrides `--min-held-target-area`). Default `red=2000`, a candidate from the same data (red at most 529 px with the target held, median about 15,000 with a red ball held). **Confirm it with one red ball held at rest** (command below), then judge red too with one flag: `--held-check-colours striped red`. Red rounds only exist once the colour pick is plugged into `Skills.pick_colour`.
  - **Blue is not judged** and has no threshold: on the wrist camera it is not usable, up to 27,466 px inside the region with the striped target held (the bluish ceiling at rest falls inside it). A blue round logs `held_check: untrusted` and accepts, as without the check. Listing blue in `--held-check-colours` prints a WARNING and only logs.
  - **Cutting the ~12.5% missed targets:** #24's striped colour range was tuned on the top camera and misses the ball under the wrist camera's cool white balance. Re-tune the striped spec on wrist frames with #24's calibrate mode (`scripts/vision/colour_detector.py calibrate`). Until then a rejected target only costs a re-pick (it is put down and picked again), never safety.
  - `--no-held-check` turns it off. If the detector fails or the wrist frame is missing the ball is accepted as before and the log says `held_check: unavailable`.
  - **Confirm the red threshold at the table:** run striped rounds into the container, logging; any round whose pick ends holding a red ball logs that ball's red area at rest (it is then put down as the wrong ball):

    ```bash
    python scripts/supervisor/run_supervisor.py --robot \
      --port <follower port> --top-camera <index> --wrist-camera <index> --robot-id mi_follower \
      --put-down-pose <5 joint values> --vision-dir <colour-pick worktree>/scripts/vision \
      --min-held-target-area 120 --rounds 10 --log outputs/held_check.jsonl     # c + Enter, then Enter, per round
    python -c "import json; [print(e['held_check'], e['held_areas_px']) for e in map(json.loads, open('outputs/held_check.jsonl')) if 'held_areas_px' in e]"
    ```

    Red should read well above 2000 with a red ball held and below it with the striped ball held. Set `--min-held-area red=<between>` and add `--held-check-colours striped red`.
- **Grasp check inside PICK (on by default, `--no-grasp-check` turns it off; untested on the arm).** ACT often closes on the jaw tips, misses, and the rest of its chunk carries the empty gripper on. Now the moment the policy commands the gripper below `--grasp-trigger 10` (after it had opened above `--grasp-armed-above 20`), the arm is held still, the gripper goal set to `--grasp-close-goal 1`, and after `--grasp-settle 0.4` s the median gripper reading decides: held if inside `--hold-range 6 20` **and** at least `--grasp-min-error 3` above the goal (a ball stalls the jaw; an empty jaw reaches its goal). Held: the policy is re-queried from the real observation and the gripper is watched; if it falls below 6 during the lift (slip), that is a miss too. Miss: open, back off along the last `--grasp-backoff 0.5` s of the commanded path, and PICK again at once with a fresh policy query (counts toward `--max-pick-attempts`). Every check is a `grasp_check` line in the log (reading, goal, error, spread, first reading, gripper `load`, verdict). `--grasp-min-load N` (default 120) also requires `|Present_Load| >= N`. `--grasp-max-tip-height M` calls a close with the FK tip above M m a miss before reading (off: the model's tip z at real holding closes spans 0.02-0.16 m).
  - **After a hold (2026-11-21, `molnrb/arm-logs` 4e20743):** the gripper is locked for the rest of PICK: it gets min(policy command, lock goal), with the lock goal `--grasp-lock-margin 4` below the held reading, at least the close goal and at most 5 (so an empty jaw still reads below 6 for the slip watch, and the servo does not stall at its torque cap as it did at goal 1; logged as `lock_goal` in the held line), so the policy cannot open on the ball (on the arm it did, then closed on nothing and the check called that a miss). Each blocked open is a `grasp_check` line `phase: release_blocked` (at most one a second); the slip watch keeps running, so a real slip is still a miss. `--no-grasp-lock-after-hold` brings back the old flow, with the re-open logged as `phase: release` and the next close a fresh check. `--grasp-min-load` is 120 since the gripper torque cap of 200 (2026-11-22, `kb/gripper-torque-limit.md`; a held ball reads the cap, 200; empty 28-44), and both it and the reading rule must say held; if the load cannot be read, the reading rule decides alone (`load_rule: unavailable`).
  - **Measure once at the table:** close on a ball and on nothing, 5 times each, and read the `grasp_check` lines: `python -c "import json; [print(e['verdict'], e['reading'], e['error'], e['load']) for e in map(json.loads, open('outputs/supervisor.jsonl')) if e.get('event') == 'grasp_check']"`. Ball should read ~11-16 with error >8; nothing ~0.5. If the loads separate too, set `--grasp-min-load` between them.
- **Cameras use MJPG** (`--camera-fourcc MJPG`, as PR #6's `scripts/so101/common.sh`): two uncompressed 640x480/30 fps streams usually do not both fit on one USB bus. `--camera-fourcc none` leaves the format to OpenCV if a camera refuses MJPG. PR #6 also sets the OpenCV backend (`backend: 200`, V4L2); this program does not.
- **Crash or Ctrl-C:** the shared-memory block for the hand tracker is closed on exit, and a stale block of the same name left by an earlier crash is removed at start.
- **`a` + Enter (or Ctrl-C) = ABORT.** The arm stops where it is with torque on, the gripper is not opened. Hold the arm, then switch the power off.
- **Watch the first move from rest to the present pose** (default `0 -55.3 5.6 91.8 0`): from Balazs's folded rest pose the model (Tian's offsets) keeps the tip at 0.243-0.374 m; from the guessed rest `0 20 60 90 0` it passed about 3 cm above the table. If it comes close to the bowl or the table, stop and change `--present-pose`.
- `--max-step-deg 1.0` caps every arm joint at 1 degree per tick (30 deg/s) in every skill, the policy included. Use it for the first runs with a person's hand; it changes how the policy moves.
- Voice with the robot program in **WSL** and the listener on Windows: add `--voice-host 0.0.0.0`. With 127.0.0.1, 2 of 8 starts were unreachable from Windows for 11-15 s (`decisions/voice-commands.md`).
- `--clear-gray-policy <checkpoint>` turns on CLEAR GRAY. It is for the **old foil-ball setup** (gray tinfoil balls, replaced by red and blue balls on 2026-11-21, board 7062034477) and stays off; LOOK's target mask is also for the striped ball only.
- `--hold-range 6 20` (measured holds 9.3-16.4, open jaws 17-40) and `--max-pick-attempts 3` are flags. Every VERIFY_HOLD reading is in the log: check it after the first rounds, and read a foil ball once.
- `--n-action-steps 20` replans ACT every 0.67 s instead of 3.3 s; `--camera-rename top=camera1 wrist=camera2` for a SmolVLA checkpoint.

## Full chain PICK -> GIVE (Tian's hand-following loop) -> REST

The chain the team decided on (Space0code, board 7064763916; Tian, 7064748304): learned PICK, then the hand-following loop that ran on the arm (Balazs, 7064755928, "works fine"), then back to rest, on one connection. **Not yet run on the arm as a chain.**

- GIVE runs the loop with the values of that arm run: hover 0.28 m, no target below 0.20 m, half speed (`--give-hover-height`, `--give-min-height`, `--give-max-step-deg 1.0`, `--give-max-step-m 0.004`), then Tian's 3 close-open cycles after the drop (`--celebrate 3`, 0 turns it off).
- The fold to rest is the supervisor's REST (a joint move, done when the measured joints are there), not the loop's `--rest`. Start with the arm at its folded rest pose (Balazs's reading `-3.69 -103.43 97.01 -100.26 1.45`), gripper empty: the pose at start is REST.
- "Hand lost" for 3 s ends GIVE still holding: back to REST, then PUT DOWN.
- Before torque on, each servo's goal is set to where it is (Tian's fix), so the arm does not jump at start.
- In the model, with Tian's offsets, the tip stays 0.243-0.40 m above the table over the whole round (rest -> present pose -> follow -> rest, `tests/supervisor_scenarios.py`).
- The handover and tracker files come from Tian's checkout (`~/track`, `TianK003/hand-tracking-test` at 3bb344d or later) or #13 (`jakobnunnendorf/handover-loop`); both work.

```bash
cd ~/track && git fetch origin && git checkout --detach origin/TianK003/hand-tracking-test   # tracker, calibration, loop
cd ~/sup && git fetch origin && git checkout --detach origin/jakobnunnendorf/supervisor     # this program
python scripts/supervisor/run_supervisor.py --robot --port /dev/so101_follower \
  --top-camera /dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0 --wrist-camera <index> --robot-id mi_follower \
  --handover-dir ~/track/scripts/handover --handtrack-dir ~/track/scripts/handtrack --give-mode follow \
  --put-down-pose <5 joint values> --no-held-check --max-step-deg 1.0 --rounds 1 --log outputs/chain.jsonl
```

Then terminal 2, the tracker reading this program's frames, as in Tian's runbook (`docs/hand-tracking-test.md` 2b on his branch, no `--camera`). Press Enter, then `h` + Enter to go straight to GIVE once the arm holds the ball at the present pose (or show two palms within 3 s). Ctrl-C or `a` + Enter = ABORT.

## Counting for H5 / H7

```bash
python -c "import json,collections; print(collections.Counter(json.loads(l)['outcome'] for l in open('outputs/supervisor.jsonl') if '\"round_end\"' in l))"
```

Post the raw counts on the board (e.g. "given 6, put_down 2, gave_up 1, aborted 1 of 10"), and which checkpoint ran.
