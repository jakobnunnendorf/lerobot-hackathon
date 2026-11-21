# Blind calibration of v2 on Jakob's 37 corrected episodes (rb_mixed 13-49)

Pipeline as in `METHOD_v2.md`: rules (`scripts/prelabel_rules_v2.py`), then four Sonnet labellers of 9-10 episodes each, working from `draft.txt` and the review sheets. They never saw `corrections.jsonl`, `prelabels_v1.jsonl` or `labels/`. The output is `calibration_v2.jsonl`, scored with `scripts/score_prelabels.py corrections.jsonl calibration_v2.jsonl`.

**Tolerances.** Grabs, holes, misses, slips and home ±0.7 s; targets ±2 s; visibility changes ±2.5 s; `wrist_blind` ±1.5 s. A visibility step at t=0 counts as the start state.

| | v1 | v2 rules only | v2 rules + Sonnet |
|---|---|---|---|
| episodes needing no edit at all | 4/37 | 1/37 | **8/37** |
| ball events need no edit | 24/37 | 25/37 | **27/37** |
| ball events: recall / extra | 265/303 / 16 | 276/303 / 18 | **285/303 / 9** |
| start state | 26/37 (70%) | - | **28/37 (76%)** |
| visibility changes: recall / precision | 0/24 / - | 0/24 / - | **12/24 / 12/33** |
| `wrist_blind`: recall / precision | 0/21 / - | 0/21 / - | **10/21 / 10/13** |
| timing, median / max error | grab, hole, home 0.0 s | 0.0 / 0.24 s | 0.0 / 0.43 s |

On the independent set rb_mixed 0-12, which Jakob labelled himself without pre-labels, the rules alone get the ball events right in 10/13 episodes, with grab timing median 0.08 s.

**What still goes wrong**
- **`s_vis` is overused: 15 in these 37 episodes; Jakob uses it 10 times in his 50.** Most of the 21 extra visibility events are `s_vis` after a hole drop where Jakob wrote nothing or `s_partly`. Next step: emit `s_partly` for an uncovered ball and `s_vis` only after a slip leaves the ball lying free.
- **Mid-episode `s_partly` is missed in 9 episodes** where the labeller saw no change in the wrist strip after the drop. In most of them the change is the first uncovering in a buried start. Default: in a buried start, add `s_partly` at the first wrist frame after the last clearing that shows the stripe.
- **`wrist_blind` recall is 0.48.** The labellers skip it when a thin sliver of stripe stays visible under the covering ball (20, 26, 30), and Jakob still counts that as blind.
- **Ball events left:**
  - quick-attempt semantics: 38 grab+slip vs miss; 39 miss vs grab+slip; 41 the R pinch at 22.5;
  - put-aside vs miss: 32, 48 (B pushed aside, then a b_miss);
  - the two-ball grab in 22.
- **Start state, 9 wrong:** buried vs partly in 16, 20, 22, 25, 34, 37 and 46; partly vs visible in 24 and 42.

---
Written by Claude (Space0code's agent) for pre-label v2. The full v2 run was not done on this branch. Space0code's other session owns the v2 run: branch `Space0code/prelabel-v2-claude`, board comment 7070170298.
