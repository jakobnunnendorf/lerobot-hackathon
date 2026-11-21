# Pre-labelling method v2

Written by Claude. = `METHOD.md` (unchanged: gripper events, zones, grab/hole/home timings, v1 sheets) plus these checks, applied by `scripts/prelabel_v2/` on top of the v1 steps.

1. **Start state.** Keep v1's, except: stripe pixels in the top-camera bowl crop at 0.2-0.8 s (median) under 30 means buried; over 150 while v1 says buried means partly. Partly vs visible stays a judgement call (pixels overlap).
2. **Visibility change.** If the start is not visible and there is a hole release, add `s_partly` at the first moment after that release where stripe pixels stay at 40 or more for two samples (0.2 s apart), at least 1.0 s after the release. Only the first one; `s_vis` and `s_buried` are not predicted.
3. **wrist_blind.** If the episode ends with `s_grab` and the smallest wrist-camera stripe pixel count in [grab - 2.0 s, grab - 0.2 s] is under 3000, add `wrist_blind` at grab - 0.9 s. Jakob only used it from about ep 20 on.
4. **Misses.** A `s_miss` followed within 2.5 s by a red or blue target or grab is a miss of that ball (and the target before it, within 3.5 s).
5. **Grab timing.** A grab labelled while the gripper still reads above 20 moves to the next close plus 0.15 s.
6. **Quick attempts.** Do not add events for unmatched closes (precision 6/15); list the times in `uncertain`.
7. **Foil datasets** (blue_ball, buried_ball): only steps 1, 4 and 5; steps 2 and 3 have no calibration there (Jakob has not labelled them).

Calibration numbers (37 episodes, rb_mixed 13-49, scores against Jakob's final labels): in `scripts/prelabel_v2/evalv2.py`. Thresholds were fitted on these same episodes, so the numbers are optimistic; the held-out checks are stated in `FAILURES.md`.
