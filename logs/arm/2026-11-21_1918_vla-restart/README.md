# VLA restart, 5K checkpoint, layout 1 — two takes, 2026-11-21 19:18–19:27 CET

Board 7067107016 (restart) and 7067279539, 7067280812 (push everything). Both takes ran from `TianK003/demo-chain` at **d5af78b** (the `4011d7c` dashboard fix was not applied) with `vla2.sh`: `--no-grasp-check --n-action-steps 20 --camera-rename top=camera1 wrist=camera2 --tick-log --min-held-target-area 120 --max-step-deg 1.0`, checkpoint `smolvla_redblue/checkpoints/005000`, dashboard with `--no-speech`. Nobody pressed `c` before the round in either take, so the destination was unset. In take 2 the destination mattered only if a hold had been confirmed.

## Files (each take: supervisor.jsonl, ticks.jsonl, supervisor_round01.mp4 and .frames.csv, wrist JPEG, terminal.txt, command.txt)
- `5k_take1/`: first take, 18:20–18:22 UTC. The video is 8.8 MB.
- `5k_take2/`: second take, 18:26–18:27 UTC. The video is 18 MB.
- `dashboard_local_patch.diff`: my one-line local change to `scripts/dashboard/server.py`, described below.
- The dashboard saves nothing outside the log folders (no results CSV was created), and its terminal showed no output.
- The pilot is in `../2026-11-21_1902_vla-pilot/` and experiment 1 in `../2026-11-21_1829_exp1-nas20/`. Nothing was added to either.

## Take 1 (18:20:43–18:22:25 UTC, 93 s, `put_down`)
- 2 picks: both timed out with no hold; the second ended with the gripper at 11.4 and a wrist reading of all 0, and the log called it a hold, then PUT_DOWN.
- **The video starts late.** Frame 0 is 18:22:07, the VERIFY_HOLD transition, so it has nothing of the pick. The tick log is complete.
- **Cause (from the code, not confirmed on the arm):** the dashboard follows the log named in the supervisor's UDP status, and falls back to the newest `chain_*.jsonl` in `~/so101/demo` when the status is more than 2 s old. The status went stale during PICK, so the dashboard switched logs, lost the round, and rejoined at VERIFY_HOLD. My local patch makes it keep the last log instead.

## Take 2 (18:26:05–18:27:06 UTC, 54 s, `aborted`)
- **Video covers the whole round:** REST, then PICK for 603 frames, then VERIFY_HOLD. This is with the dashboard patch.
- Pick 1 ran to its 40 s timeout and ended at 18:26:52 with the gripper at 15.3 (the policy had closed on something). The arm left PICK at a pose of about [20, -15, 45, 9, -23] and the gripper reading 15.3.
- **VERIFY_HOLD then ran 13 s with the arm standing still at [-0.79, -56.2, 8.0, 90.8, -0.6], gripper 14.6.** At 18:27:06 the program aborted itself with `TimeoutError('OpenCVCamera(.../usb-BSW-V06-200512_Integrated_Camera-video-index0) latest frame is too old: 519.0 ms (max allowed: 500 ms).')`, and the round ended as `aborted`.
- **How the run ended:** the program stopped itself (`ABORT: arm holds its pose with torque on`), it did not crash or hang, and the terminal returned to the prompt. Nobody pressed `a`. The arm was left holding its pose with torque on. The top-camera read failure is the cause of the abort; the log does not show why the frame was 519 ms old.
- Whether the dashboard header showed REC was not watched; the video covering the round shows that recording ran.

## Jakob's observation (board 7067279539) for the 5K round
"The arm picked a blue ball right next to the striped ball and put it in the blue hole. It went back to the start pose, then slowly approached the bowl again with the blue ball still in the gripper, and froze."
The logs for take 2 do not show a put-down or a blue hole. They show one pick that timed out with the gripper at 15.3, the arm standing still in VERIFY_HOLD for 13 s, and the camera-timeout abort. Which take Jakob watched is not in the logs.

## Table notes (Balazs, in his words, relayed by the session)
- **Take 1: "a success, it picked up the striped ball."** The log agrees that it ended holding something (gripper 11.4) and put it down, but the wrist check read all 0 and so could not confirm which ball.
- **Take 2: "not successful, it tried but couldn't grab the ball and then stopped."** This matches the log: pick 1 timed out with no hold, then the program aborted on the top-camera timeout.
- Not noted for either take: what the first approach went for, and whether grasps were deliberate or accidental. Both takes used layout 1, with 5K.
