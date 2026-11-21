# Grasp-check calibration, 2026-11-21 17:37–17:48 CET (16:37–16:48 UTC)

Request: board 7065336460 and 7065949329. The log was copied while the program was still running, and no round was started after round 6.

## Files
- `supervisor.jsonl`: the whole `--log` file, every line, 67 lines.
- `terminal.txt`: the terminal output of the run. The warnings at the top are copied verbatim. The JSON lines below them were printed to the terminal and are identical to `supervisor.jsonl`. There were no tracebacks in this run.
- `local_patch.diff`: the local change the run used. Without it, 7bf1482 crashed on start with `NameError: clear_policy`. It is the same fix as Tian's 115ccbb and #17's 7c7364d.
- No wrist JPEGs: 7bf1482 does not save them; that started with 6fdb19d.

## Run
- Checkout `jakobnunnendorf/supervisor` at **7bf1482fcdfd11c7c8ea452e98e31324c1728b61**, plus `local_patch.diff`.
- Command (WSL, conda env `lerobot`; the variables come from `~/.so101.env`):
  ```
  python scripts/supervisor/run_supervisor.py --robot --port $SO101_FOLLOWER_PORT --top-camera $SO101_CAM_TOP \
    --wrist-camera $SO101_CAM_WRIST --robot-id $SO101_FOLLOWER_ID --max-step-deg 1.0 \
    --log outputs/supervisor.jsonl --rounds 10 --put-down-pose $PUT_DOWN_POSE
  ```
- No `--pick-policy`, so the default is used: `molnrb/act_pick_ball_combined`, n_action_steps 100.
- No hand tracker and no colour detector, so there is no GIVE. A held ball goes to PUT_DOWN, and `held_check` is off.

## Checks (from the log)
| time UTC | round, attempt | verdict | reading | first reading | load |
|---|---|---|---|---|---|
| 16:43:18 | r1 p1 | held | 10.94 | 13.42 | 500 |
| 16:43:30 | r1 p1 | miss | 1.31 | 12.39 | 28 |
| 16:43:33 | r1 p2 | miss | 1.24 | 13.01 | 28 |
| 16:43:35 | r1 p3 | held | 11.08 | 14.38 | 500 |
| 16:45:09 | r2 p1 | held | 12.04 | 14.32 | 500 |
| 16:45:53 | r3 p1 | miss | 1.24 | 13.97 | 28 |
| 16:45:56 | r3 p2 | miss | 1.58 | 14.11 | 44 |
| 16:45:58 | r3 p3 | miss | 1.31 | 15.35 | 28 |
| 16:46:23 | r4 p1 | miss | 1.38 | 12.11 | 32 |
| 16:46:25 | r4 p2 | miss | 1.45 | 13.56 | 40 |
| 16:46:27 | r4 p3 | miss | 1.24 | 14.04 | 28 |
| 16:46:45 | r5 p1 | miss | 1.31 | 12.66 | 28 |
| 16:46:47 | r5 p2 | miss | 1.38 | 11.98 | 32 |
| 16:46:48 | r5 p3 | miss | 1.31 | 13.97 | 32 |
| 16:47:18 | r6 p1 | held | 8.33 | 12.66 | 440 |
| 16:47:43 | r6 p1 | miss | 1.31 | 11.36 | 32 |
| 16:47:46 | r6 p2 | miss | 1.51 | 14.45 | 40 |
| 16:47:48 | r6 p3 | miss | 1.38 | 14.52 | 32 |

Round outcomes: r1 put_down (PICK attempt 3 ended on timeout while holding, 82.9 s), r2 put_down, r3–r6 gave_up.

## What the log alone shows (agent's reading, not checked at the table)
- **Held and miss separate cleanly.** Held readings are 8.3–12.0 with load 440–500. Misses read 1.2–1.6 with load 28–44. `--grasp-min-load` anywhere between about 50 and 400 would separate them.
- **`first_reading` is 11–15 for every check, misses included**, so the 0.4 s settle is what tells them apart.
- **r1 and r6: a `held` was followed by a `miss` in the same pick attempt** (r1 12 s later, r6 25 s later), with no slip event between them. After the re-query, the policy seems to have opened on the ball and closed again on nothing. It looks like the hold was lost without the slip watch catching it.
- **r1: after the second `held`** (attempt 3), PICK ran 44 s with the ball until the 90 s pick timeout and never settled at rest.

## Balazs's notes, one line per check
_To be added in Balazs's words: what was in the gripper (ball or nothing), where the ball was, and what looked weird._
