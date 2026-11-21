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
2. Hand tracker (#15) and hand-following loop (#13) are on `main` or in `scripts/handover/` and `scripts/handtrack/`. Without #13's `hand_socket.py` there are no hand messages and every round ends in PUT DOWN. Without camera calibration use `--give-mode present`: the arm holds the present pose over a taped spot and drops when two flat palms are seen.
3. Teach two poses with the arm there (torque off, `read_pose.py` from #13), five joint values each: the **present pose** over the hand spot (`--present-pose`) and the pose over the **container** (`--put-down-pose`).

```bash
python scripts/supervisor/run_supervisor.py --robot \
  --port <follower port> --top-camera <index> --wrist-camera <index> --robot-id mi_follower \
  --present-pose <5 joint values> --put-down-pose <5 joint values> \
  --voice --look --rounds 10 --log outputs/supervisor.jsonl
```

Then start the hand tracker with `--only-open` (it reads the frames this program shares) and, for voice, `scripts/voice/listen.py` (`docs/voice-runbook.md`).

- **Enter** starts a round. `h` + Enter: give to a hand, `c` + Enter: container, `x` + Enter: no preference. A voice command also starts a round.
- **`a` + Enter (or Ctrl-C) = ABORT.** The arm stops where it is with torque on, the gripper is not opened. Hold the arm, then switch the power off.
- **Watch the first move from rest to the present pose** (default `0 -55.3 5.6 91.8 0`): in the model the tip passes about 3 cm above the table. If it comes close to the bowl or the table, stop and change `--present-pose`.
- `--max-step-deg 1.0` caps every arm joint at 1 degree per tick (30 deg/s) in every skill, the policy included. Use it for the first runs with a person's hand; it changes how the policy moves.
- Voice with the robot program in **WSL** and the listener on Windows: add `--voice-host 0.0.0.0`. With 127.0.0.1, 2 of 8 starts were unreachable from Windows for 11-15 s (`decisions/voice-commands.md`).
- `--clear-gray-policy <checkpoint>` turns on CLEAR GRAY when LOOK does not see the target; only once such a policy exists.
- `--hold-range 6 20` (measured holds 9.3-16.4, open jaws 17-40) and `--max-pick-attempts 3` are flags. Every VERIFY_HOLD reading is in the log: check it after the first rounds, and read a foil ball once.
- `--n-action-steps 20` replans ACT every 0.67 s instead of 3.3 s; `--camera-rename top=camera1 wrist=camera2` for a SmolVLA checkpoint.

## Counting for H5 / H7

```bash
python -c "import json,collections; print(collections.Counter(json.loads(l)['outcome'] for l in open('outputs/supervisor.jsonl') if '\"round_end\"' in l))"
```

Post the raw counts on the board (e.g. "given 6, put_down 2, gave_up 1, aborted 1 of 10"), and which checkpoint ran.
