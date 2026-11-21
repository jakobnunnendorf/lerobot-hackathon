# Held ball in the wrist camera: detector areas and thresholds

- Date: 2026-11-21
- Verified by: Kamel's agent, offline on `molnrb/so101_pick_rb_mixed` (80 episodes, red/blue scene) with `scripts/vision/measure_held_area.py`. #24's `colour_detector.py` was set up the way #17's `held_ball_area` does it: no size limit, ellipse ROI. No robot was involved.

Samples: **target held at rest** = 3 frames in the last second of each episode, median per episode, the way the supervisor takes its median. **Covering ball held** = the frame 0.33 s before each of 66 releases over the containers.

## Where the held ball is

At rest the wrist camera faces the ceiling, and the held ball sits in the lower middle of the frame. The striped ball's centre lies at x 0.49-0.62 and y 0.64-0.77 of the frame (p10-p95, 183 detections). The ellipse `0.5 0.5 0.25 0.3` from #17 is too high: it sees only the upper half of the ball and mostly ceiling. **`--held-check-roi 0.52 0.72 0.2 0.25` covers the ball** (checked on overlay frames).

## Striped area (px) inside `0.52 0.72 0.2 0.25`

| | min | p25 | median | p95 |
|---|---|---|---|---|
| target held at rest, per episode | 0 | 235 | 370 | 1224 |
| covering ball held, over a container | 0 | 0 | 0 | 0 (max 1694) |

| `--min-held-target-area` | target rejected | covering ball accepted |
|---|---|---|
| 50-150 | 12.5% | 4.5% |
| 200 | 15% | 4.5% |
| 300 | 40% | 3% |

**Use 120.**
- The rejected targets are detector misses: the ball is clearly inside the ellipse, but under a cool, bluish white balance the striped colour range, tuned on the top camera, finds nothing. Re-tuning the striped spec on wrist frames should remove most of them.
- The accepted covering balls come from frames over the containers, where the camera looks down into the bowl and sees the target lying there. At rest the camera faces the ceiling, so the rate there should be lower.

## Red and blue in the same ellipse

- **Red:** with the target held at rest, red stays at p95 345 px (max 529); with a red covering ball held, the median is about 15,000. A red check at rest looks feasible with a threshold around 2,000. Confirm it at the table with a red ball held at rest, because no recording holds a red ball at rest.
- **Blue is not usable:** with the striped target held at rest, blue reaches p95 19,651 px (max 27,466) inside the ellipse. The bluish ceiling falls inside it.
