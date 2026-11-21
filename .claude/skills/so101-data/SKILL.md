---
name: so101-data
description: Use BEFORE any training run, dataset build, policy or architecture choice, hold-out split, eval comparison, or new recording batch on the so101_pick_* striped-ball data. Holds what is in the data (label codes, event sequences with counts, the partly-covered blind-grab trap, scenario blocks) and the curation defaults per architecture, so a run does not learn the wrong lesson from the data distribution.
---

# SO-101 striped-ball data: check the distribution before you train

The full reference is `research/DATASET_GUIDE.md`. Read it once per session before the first training or architecture decision. This file is the checklist.

## Where the data and labels are
- Datasets:
  - `molnrb/so101_pick_rb_mixed` (201 episodes, red and blue balls);
  - `so101_pick_buried_ball` (45, foil balls);
  - `so101_pick_blue_ball` (100, foil balls).
- Labels:
  - Jakob's hand labels for rb_mixed 0–57: branch `labels`, `labels/jakob.jsonl`. Read-only.
  - AI v2 pre-labels for the rest, with about 4% errors: `Space0code/prelabel-v2-claude`, `prelabel-kit/prelabels_v2_claude.jsonl`.
- Segments and per-episode flags: `research/label-patterns/{segments,episodes}.csv` (PR #54).
  - `clear_r` / `clear_b` / `pick_s` are time ranges.
  - `wrist_never_clear` and `wrist_blind_s` are measured from the frames.

## Checklist: answer each one in your run note or PR
1. **Unit:** whole episodes or segments? Dropping whole episodes for one bad moment loses 62% of the clean clearing footage, so prefer segments. Cut only at the `segments.csv` boundaries and never splice two ranges together.
2. **Blind striped approaches** (72 of 345 episodes):
   - one-frame VLA: drop their `pick_s`;
   - ACT with 3.3 s chunks: keep those blind for under 1.7 s, drop the sliver-only ones and those blind for over 1.7 s.
3. **Grab while the striped ball is only partly uncovered** (43 of Jakob's 58 episodes; nearly every wrist-blind grab): the target behaviour is clear until bare, then pick. Prefer picks where the ball is visible at the grab and the wrist sees it. Say how many your set keeps.
4. **Decisions the demos contradict:**
   - buried start but picked directly, or visible start but cleared first (19 rb_mixed episodes);
   - more balls cleared after the striped ball is uncovered (10 of 31).

   Drop them, give each scene its own task sentence, or let the supervisor decide. A single policy can't learn a decision the demos disagree on.
5. **Failures:**
   - one-frame VLA: clean segments only;
   - ACT: start the segment after the failed close, because the recovery tail is useful.

   Put-backs are always dropped.
6. **Foil balls are not red.** They are logged as `r_` but go to a different hole, so never mix them into clear-red.
7. **Buried share:** keep about 40–50% buried starts. A strict filter drops it to 26%.
8. **Hold-out:** rb_mixed was recorded in scenario blocks (50–74 buried, 75–124 direct, 150–174 buried). Hold out every Nth episode, **never the last N**. The old split 114–119 is all direct picks and can't rank clearing behaviour.
9. **AI-labelled cut points:** spot-check about 10 before training on episodes 58+ or on the foil sets. Visibility statistics should come from Jakob's labels only.
10. **Post your choices** (counts kept and dropped per rule) on board #1, so runs stay comparable.

## Recording new data
- Clear until the striped ball is bare, then pick it, from a side where the wrist camera keeps it in view.
- Open the gripper to 40 or more.
- No put-backs.
- End at home holding the ball.
- Mix scenarios within each batch.
