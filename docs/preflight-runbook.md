# Preflight: check the setup before anything moves

**Before every table session**, on the robot laptop (WSL, repo root, robot env active), run:

```bash
python scripts/preflight.py
```

One line per check, PASS / WARN / FAIL, with the fix in one phrase under each WARN and FAIL. Exit code 1 if anything FAILs. It never writes to the motors and moves nothing. Fix every FAIL before starting a robot program.

- WARN on the tracker (8766) and voice (8765) lines is normal before the programs are started. Run it again afterwards with `--strict-services` to turn them into FAIL.
- `--read-motors` also opens the arm's serial port and reads each motor's present position (read only, torque untouched). Off by default; the arm program must not be running.
- `--load-policy` loads the checkpoint weights on the GPU; without it only its config is read.
- `--skip camera,hf` skips checks (names: python, cuda, port, calibration, camera, policy, tracker, voice, disk, hf, motors). `--port`, `--top-camera`, `--wrist-camera`, `--policy`, `--robot-id`, `--voice-host` override the defaults (`~/.so101.env` values are read from `SO101_FOLLOWER_PORT`, `SO101_CAM_TOP`, `SO101_CAM_WRIST`).
- Camera lines fail if the top or wrist camera is already open in another program.
- Test without hardware: `python tests/preflight_checks.py` prints `PASS`.
