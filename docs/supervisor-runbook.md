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

1. The arm is at its folded rest pose with the gripper **empty** (the program refuses to start if the gripper reads 10-17).
2. Hand tracker (#15) and hand-following loop (#13) are on `main` or in `scripts/handover/` and `scripts/handtrack/`; otherwise GIVE is off and every round ends in PUT DOWN.
3. You have a PUT DOWN: a recorded episode (`--put-down-episode molnrb/<dataset>:<episode>`) or five joint values over the container (`--put-down-pose`, read them with the arm there).

```bash
python scripts/supervisor/run_supervisor.py --robot \
  --port <follower port> --top-camera <index> --wrist-camera <index> --robot-id mi_follower \
  --put-down-pose <5 joint values> \
  --voice --look --rounds 10 --log outputs/supervisor.jsonl
```

Then start the hand tracker with `--only-open` (it reads the frames this program shares) and, for voice, `scripts/voice/listen.py` (`docs/voice-runbook.md`).

- **Enter** starts a round. `h` + Enter: give to a hand, `c` + Enter: container, `x` + Enter: no preference. A voice command also starts a round.
- **`a` + Enter (or Ctrl-C) = ABORT.** The arm stops where it is with torque on, the gripper is not opened. Hold the arm, then switch the power off.
- **Watch the first move from rest to the GIVE start pose** (`--give-start`, default `0 -55.3 5.6 91.8 0`): in the model the tip passes about 3 cm above the table. If it comes close to the bowl or the table, stop and change `--give-start`.
- Voice with the robot program in **WSL** and the listener on Windows: add `--voice-host 0.0.0.0`. With 127.0.0.1, 2 of 8 starts were unreachable from Windows for 11-15 s (`decisions/voice-commands.md`).
- `--clear-gray-policy <checkpoint>` turns on CLEAR GRAY when LOOK does not see the target; only once such a policy exists.
- `--hold-range 10 17` and `--max-pick-attempts 3` are flags; change them if the gripper reads differently on our arm.

## Counting for H5 / H7

```bash
python -c "import json,collections; print(collections.Counter(json.loads(l)['outcome'] for l in open('outputs/supervisor.jsonl') if '\"round_end\"' in l))"
```

Post the raw counts on the board (e.g. "given 6, put_down 2, gave_up 1, aborted 1 of 10"), and which checkpoint ran.
