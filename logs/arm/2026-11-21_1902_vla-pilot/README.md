# VLA pilot (not counted), 2026-11-21 19:02–19:13 CET

Board 7067107016: Jakob restarted the VLA comparison, so these rounds are kept but not counted. This folder was run from #17 at 0ce2350 with `vla.sh`. The same flags were used for both checkpoints: `--n-action-steps 20 --no-grasp-check --min-held-target-area 120 --max-step-deg 1.0 --camera-rename top=camera1 wrist=camera2`. There is no `--tick-log` and no round video.

Files: `5k/` and `20k/`, each with `supervisor.jsonl`, `terminal.txt`, `command.txt` and the wrist JPEGs at each VERIFY_HOLD. The commit line in `command.txt` is empty, and the commit was 0ce2350.

## 5k, layout 1
- A first attempt at 18:03 UTC aborted after 6.8 s with a motor bus `ConnectionError` (all six motors timed out; the WSL log showed USB resets at the same time). It is in the log as `outcome: aborted`.
- The retry ran 18:05–18:07 UTC: pick 1 timed out (no hold), pick 2 timed out with the gripper at 11.5 and an all-0 wrist reading, then put_down (106 s).
- Jakob's notes (board 7067020779 and 7067030881, round 1 as watched at the table): it narrowly missed the striped ball, possibly aimed at it; then went for a red ball and knocked the striped ball to the near edge; it put the red ball in the red hole; it later struggled with a blue ball.

## 20k, layout 1
- 18:10–18:13 UTC: pick 1 and pick 2 did not hold (not holding, retry); pick 3 timed out with the gripper at 10.9 and `held_areas_px` striped 169, so the striped ball was held; put_down (150 s).
- Jakob's note: the arm grabbed next to the striped ball twice, then got it on the third try with a clumsy movement (board 7067107016).

The per-round notes asked for (first approach, each grasp, whether on purpose) are not written for these rounds.
