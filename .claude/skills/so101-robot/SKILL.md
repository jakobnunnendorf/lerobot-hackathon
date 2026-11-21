---
name: so101-robot
description: Use for ANY hands-on work with our SO-101 arm - teleoperate, record episodes, check or clean a dataset, upload a dataset, train an ACT policy, run or test a trained model on the arm, or fix the arm, cameras or USB (motor not answering, gripper stopped, camera dropped, WSL lost the devices). Has ready scripts in scripts/so101/ so you do not rebuild commands or debug known failures again.
---

# SO-101 robot runbook

Everything goes through `scripts/so101/`. Do not hand-build `lerobot-*` command lines: the scripts carry the ports, calibration ids, camera settings and the fixes below. Read a script only if it fails.

## Where it runs

- Robot laptop: Windows + WSL2 (Ubuntu 24.04), conda env `lerobot` (lerobot 0.7.0), arms and cameras passed into WSL with usbipd. Agents call it from Git Bash or PowerShell with `wsl -e bash -lc '...'`; repo path inside WSL is `/mnt/c/...` (`wslpath`).
- Machine settings live in `~/.so101.env` (template: `scripts/so101/so101.env.example`; without it the template values are used). Another machine: copy the template, edit ports/ids/camera paths, set `SO101_ACTIVATE` empty to use the repo `.venv`.
- After reinstalling lerobot run `python scripts/so101/patch_record.py` once (SPACE key, wait-before-first-episode, on-screen banners). Idempotent.

## Safety

- Teleop, recording and policy runs move the arm. CLAUDE.md's rule holds: prepare the command (the table below) and hand it to the person at the table to run.
- Before anything that holds the arm is stopped, the follower must be held: stopping switches torque off and the arm goes limp.
- Policy runs wait for SPACE pressed by the person; never bypass that.

## Starting a robot program

These programs need a window the person can type in (SPACE, LEFT, ESC). From PowerShell, a separate console opens with:

```powershell
Start-Process -FilePath wsl.exe -ArgumentList '-e','bash','-lc','"bash /mnt/c/<repo>/scripts/so101/<script> <args>"'
```

Never put double quotes inside a `wsl -e bash -lc '...'` typed into PowerShell 5.1: it strips them. Make a wrapper script instead (see the record example).

| Goal | Command (inside WSL, repo root) |
| --- | --- |
| Teleop + live cameras (Rerun) | `bash scripts/so101/teleop.sh` (`--no-cams` for arms only) |
| Record episodes | `bash scripts/so101/record.sh <dataset> <num> "<task sentence>" "<on-screen label>"` - same name again ADDS episodes |
| Check a dataset | `python scripts/so101/check_dataset.py <user/dataset> [--from N] [--sheet /tmp/s.jpg]` then look at the sheet image |
| Upload a dataset | `python scripts/so101/upload_dataset.py <user/dataset> [--private]` - ask the person public or private first; run in background |
| Delete bad episodes | `lerobot-edit-dataset --repo_id=<r> --new_repo_id=<r> --operation.type=delete_episodes --operation.episode_indices="[i,j]"` - ask first; keeps `<r>_old` as backup |
| Train ACT | `bash scripts/so101/train.sh <dataset> [steps=20000] [batch=8]` - run in background, ~2 h for 50 episodes on an RTX 4050 |
| Run a model once | `bash scripts/so101/run_policy.sh <.../pretrained_model> [seconds=40]` |
| Run a model repeatedly (success counting) | `bash scripts/so101/run_loop.sh <.../pretrained_model>` - logs each run to `~/so101/logs/` |
| Motor health | `python scripts/so101/motor_health.py` (nothing moves; arm program must be stopped) |
| Wait for gripper to cool | `python scripts/so101/motor_health.py --cool 45` in background |
| Re-attach USB to WSL | `bash scripts/so101/attach_usb.sh` from Git Bash (Windows side) |

Recording keys (person presses them): SPACE start/finish (kept), LEFT throw away, ESC save and end. A wrapper per dataset keeps quoting simple, e.g. `~/so101/record_buried.sh`:
`exec bash /mnt/c/<repo>/scripts/so101/record.sh so101_pick_buried_ball "${1:-50}" "Find the buried blue ball in the bowl, uncover it, pick it up and return to the start position" "BURIED BALL"`

## After every recording session

1. Read the terminal tail: clean "SESSION ENDED" vs a traceback.
2. `check_dataset.py <dataset> --from <first new episode> --sheet ...` and look at the sheet. It flags episodes ending with an empty gripper and episodes where the follower gripper did not follow.
3. Report counts and flagged episodes; ask before deleting any.

## Known failures (all seen on 2026-11-20/21)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Everything "crashed", `/dev/so101_*` or `/dev/v4l/by-id` missing, WSL `uptime` is minutes | WSL shut down when its last Linux process exited; usbipd devices detach | `attach_usb.sh`, then keep WSL alive: `Start-Process wsl.exe -ArgumentList '-e','sleep','infinity' -WindowStyle Hidden` |
| Recording stops: wrist camera `read failed` / `No such device` | Wrist camera USB dropped out of WSL (loose plug or cable tug) | `attach_usb.sh`; episodes saved before it are intact, only the current one is lost |
| `Failed to enable torque ... 'gripper' (motor ID 6): Operation timed out` at connect | One dropped bus message | Retry once; if it repeats check the gripper motor plug at the wrist |
| Follower gripper stops moving mid-session | Gripper motor overheat (Status bit 4, ~67 C, cuts at 70 C) from squeezing the ball while carrying | Stop, `motor_health.py`, cool below ~45 C; tell the person to close the leader gripper only until the ball holds, and pause every ~10 episodes. Throw away the episode where it stopped |
| Ctrl+C does nothing in a recording tab | Recorder reads keys raw | Ask the person to press ESC; else `kill -INT` the `lerobot-record` pid, then `kill -TERM`/`-KILL`. Check the dataset afterwards |
| Episode kept although LEFT was pressed | LEFT during recording then ESC can still save it | Check with `check_dataset.py`; delete the episode (ask first) |
| `pgrep -f pattern` inside `bash -c` finds a process | It matches its own command line | Use `ps -eo args | grep "[p]attern"` |
| Terminal panel refuses new tabs | Tab limit; tabs the person typed in cannot be closed by the agent | Ask them to close old tabs, or use `Start-Process` above |

## Facts the policies depend on

- Camera names `top` and `wrist`, 640x480, 30 fps, MJPG, `/dev/v4l/by-id` paths; robot id `mi_follower`, leader `mi_leader`. A model trained on our datasets expects exactly these.
- Datasets on the Hub (user `molnrb`): `so101_pick_ball` (49, ball into ring), `so101_pick_blue_ball` (100, visible striped ball; ep 33 and 49 end empty), `so101_pick_buried_ball` (45, ball buried under foil).
- The ACT settings in `train.sh` produced a first-try success on `so101_pick_ball` (2026-11-20). A teammate trained `act_pick_blue_ball` (20k steps) elsewhere; it is unpacked at `~/so101/models/act_pick_blue_ball_020000/pretrained_model` on the robot laptop.
- Report robot results as raw counts ("7/10").
