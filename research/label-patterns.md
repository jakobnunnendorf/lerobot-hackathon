# What the step labels say about our pick data, and how to curate it per policy

Analysed 2026-11-22 by Space0code's agent, at Tomi's request. Offline only: no arm, no paid runs.

## Inputs and method
- **Datasets:** `molnrb/so101_pick_rb_mixed` (201 episodes), `so101_pick_buried_ball` (45) and `so101_pick_blue_ball` (100). The last two use grey foil balls instead of red and blue.
- **Labels:**
  - Jakob's hand labels for rb_mixed 0-57: `labels/jakob.jsonl`, branch `labels`, 8705f73.
  - AI pre-labels for the other 288 episodes: `prelabel-kit/prelabels_v2_claude.jsonl`, branch `Space0code/prelabel-v2-claude`. Calibrated ball-event precision about 0.96, start state 78%.
  - Numbers that rest on Jakob's labels alone are marked.
- **Wrist blindness is measured from the frames, not from labels.**
  - Detector: a wrist frame "clearly shows" the striped ball when 13 or more 16×16 px cells are at least 20% mint or yellow-green.
  - The final approach is "blind" when the ball was last clearly seen 0.6 s or more before the striped grab, or never clearly seen during the approach.
  - Against Jakob's `wrist_blind` labels (rb_mixed 0-57): precision 18/19, recall 18/21. Both AI labellers were near 0.5 recall.
  - Frames: 5 fps from `prelabel-kit/scripts/prelabel_prep_v2.py`, branch `Space0code/prelabel-v2`.
- **Outputs:** `scripts/analysis/label_patterns.py` writes `label-patterns/episodes.csv` (one row per episode) and `label-patterns/segments.csv` (one row per clear or pick segment, as time ranges).
- **Policy horizons:**
  - ACT runs `n_action_steps=100`: 3.3 s of open-loop motion per prediction.
  - The SmolVLA runs 50 (1.7 s) and sees one frame, no history (`research/smolvla-clear-first.md`).

## Bottom line
1. **One episode in five has a blind final approach to the striped ball: 72 of 345.**
   - 36 never show more than a sliver of stripe during the approach.
   - 36 lose sight of it 0.6 s or more before the grab: 26 for 0.6-1.7 s, 9 for 1.7-3.3 s, 1 for more than 3.3 s.
   - Blindness is concentrated in buried starts (42/113) and rb_mixed (54/201). It is rare in blue_ball (7/99).
2. **The blind demos are not worse demos.** The operator grabs successfully just as often: a striped miss or slip in 12 of 72 blind episodes against 52 of 273 others. The problem is only what the policy can see.
3. **Dropping whole episodes is wasteful.** A strict VLA filter (no failure anywhere, no blind approach) keeps 103 of 201 rb_mixed episodes. It throws away 138 clean clearing segments (19 min, 62% of all clean clearing footage), and the buried-start share falls from 43% to 26%.
   - **Cutting episodes into segments keeps them.** One segment per cleared ball, plus the final striped pick.
4. **Some demos disagree with each other on a decision**, not on a motion. A per-frame policy then can't learn that decision:
   - when to stop clearing: in 10 of Jakob's 31 buried episodes, the operator keeps clearing after the stripe is uncovered;
   - whether to clear at all: 8 buried starts go direct, 11 visible starts clear first, and partly covered starts split 19 to 16.
5. **The datasets were recorded in scenario blocks.**
   - rb_mixed 50-74 is 21/25 buried starts; 75-124 is almost all direct picks; 150-174 is 18/25 buried.
   - Default "last N episodes" hold-out splits and quick trials therefore test a narrow slice.

## Patterns, with what they mean for training

### 1. Wrist blindness on the striped approach
**What happens.** The striped ball is in the wrist view while the arm descends. Then a red or blue ball fills the view, and the jaws close on the striped ball without the camera seeing it.
- Jakob's `wrist_blind` comes a median 0.9 s before the grab.
- The measured blind time before the grab:

| final approach | rb_mixed | buried_ball | blue_ball | all |
|---|---|---|---|---|
| clear view until under 0.6 s before the grab | 147 | 34 | 92 | 273 |
| blind 0.6-1.7 s (within one SmolVLA chunk) | 19 | 4 | 3 | 26 |
| blind 1.7-3.3 s (within one ACT chunk) | 6 | 2 | 1 | 9 |
| blind longer than 3.3 s | 1 | 0 | 0 | 1 |
| only a sliver of stripe during the whole approach | 28 | 5 | 3 | 36 |

**Why it matters, per policy.** This is inferred, not measured on the arm.
- **Why blind frames confuse a one-frame policy.** A wrist view full of red is what *every* red-clearing grab looks like. In a blind approach the right action is "keep going down to the striped ball under it". In a clearing grab it is "grab this red ball". Only the joint state and the small top-camera view tell the two apart.
- **SmolVLA:** one frame, re-planned every 1.7 s, the striped ball worth a tenth of a visual token. It is most exposed. Training on blind approaches teaches it "grab what fills the wrist view" in exactly the scenes where that is wrong.
- **ACT:** with 3.3 s chunks, all but one blind period is shorter than one chunk.
  - A chunk planned while the ball was in view carries the arm through. A re-plan that falls inside the blind window (probability about blind time / 3.3 s) has to rely on joints and the top camera.
  - A smaller `n_action_steps` (20-50, as `research/lerobot-multi-skill-runner.md` suggests for other reasons) makes that more likely. Keep 100 for the pick skill if blind approaches stay in its data, or drop them.

**Recommendation.**
- **SmolVLA:** leave out the striped-pick segment of all 72 blind episodes, but keep their clearing segments.
- **ACT:** keep blind approaches under 1.7 s (26). Drop the 36 sliver-only ones and the 10 that are blind for longer than 1.7 s.
- The `wrist_blind` column in `episodes.csv` holds both measures: `wrist_never_clear` and `wrist_blind_s`.
- **When recording more:** approach the striped ball from a side where it stays in the wrist view, or clear the ball lying on it first. Both are a recording rule, not a data fix.

### 2. Visibility changes while clearing (Jakob's labels, rb_mixed 0-57)
- **Buried starts follow one path** (27 of 58 episodes): buried > partly after a clear > pick. Three more end visible.
- **Clears needed:**
  - the first drop uncovers the stripe in 20 of 34 buried starts, the second in 9;
  - 3 or more were needed in 2, and 3 episodes have no uncovering label.
- **After uncovering, the operator usually goes straight for the striped ball (21 of 31), but clears 1-3 more balls in 10.**
  - This is the same "is it graspable now?" decision Tian found at the start (`research/smolvla-clear-first.md`).
  - For a single policy that has to decide, those 10 episodes teach the opposite of the other 21.
  - For the per-colour ACT plan (supervisor decides, skill executes), they are harmless: each clearing segment is still a correct "clear one red/blue ball" demo.
- **Pushing the stripe back under is rare:** 1 episode (22). It's not a pattern to curate for.
- The pre-labels put `s_partly` after the first drop almost by rule. Don't use them for this statistic.

### 3. Inconsistent start choices (all labels)

| start | rb_mixed clear first / direct | buried_ball | blue_ball |
|---|---|---|---|
| buried | 78 / 8 | 24 / 1 | 0 / 2 |
| partly | 19 / 16 | 16 / 0 | 1 / 12 |
| visible | 11 / 69 | 3 / 1 | 2 / 83 |

- Partly covered starts are a coin flip in rb_mixed and a fixed rule in the foil datasets: always clear in buried_ball, almost never in blue_ball. That is a scene difference (foil piles), but it also makes "partly" mean different things across datasets.
- For any policy that decides by itself, drop the contradicting cells: buried-direct and visible-clear-first, 19 rb_mixed episodes. Or let the supervisor choose (Tian's fix 1).

### 4. Failures and recoveries
- **Striped misses:**
  - rb_mixed 32/201 episodes (8 with 2 or more), buried_ball 13/45, blue_ball 10/100;
  - striped slips 15; red/blue misses 36; put-backs 7 (4 of them foil moved aside in buried_ball).
- **Every episode ends with the striped ball held and the arm home**, except one: blue_ball has one with no striped grab.
  - So every failure in the data is followed by a recovery: 65 pick segments contain one, 6.6 min from the last failure to home.
- **For ACT:**
  - The recovery tail (from just after the failed close to home) is useful: it starts in a state the policy will reach on the arm.
  - The failed approach before it is a demo of failing. Cut it out by starting the segment after the miss, not by deleting frames inside a segment.
- **For SmolVLA:** use clean segments only. A one-frame policy can't tell "first try" from "retry" frames.

### 5. Recording order and scenario blocks (rb_mixed)

| episodes | buried starts | direct picks | mean length | striped miss | blind approach |
|---|---|---|---|---|---|
| 0-24 | 13 | 8 | 21 s | 4 | 3 |
| 25-49 | 15 | 7 | 24 s | 5 | 10 |
| 50-74 | 21 | 5 | 26 s | 6 | 14 |
| 75-99 | 2 | 20 | 14 s | 1 | 2 |
| 100-124 | 1 | 21 | 14 s | 1 | 8 |
| 125-149 | 11 | 9 | 23 s | 5 | 8 |
| 150-174 | 18 | 3 | 40 s | 8 | 5 |
| 175-200 | 5 | 20 | 14 s | 2 | 4 |

- Choose hold-out and evaluation episodes per scenario, e.g. every 10th episode, not the last N. Tian found the SmolVLA's hold-out had no buried start at all.
- **Frame share:** buried episodes are 2-3 times longer, so they dominate the frames. 76% of rb_mixed frames come from clear-first episodes, against 46% of episodes.
  - With uniform frame sampling, a mixed training set is mostly clearing even when half the episodes are direct picks.

### 6. Foil is not red
- **buried_ball clears foil**, logged as R. It goes into a different hole (out in front: pan about -12, lift +20 to +40) than the red hole (pan -35 to -29).
- **Its 107 clean clearing segments are not red-clearing demos.** Don't mix them into a "clear one red ball" ACT. They fit a "clear one grey ball" skill, as in the original demo idea.

### 7. Timing and positions (small issues)
- **Start delay** before the arm moves: median 0.8 s, max 4.2 s. **Longest pause** away from home: median 0.5 s, max 2.5 s. Hesitation is not a data problem here.
- **The final approach (striped target to grab) is longer in blue_ball** (median 3.0 s) than in rb_mixed (1.8 s): slower, more careful picks.
- **Striped grab positions:**
  - pan -20 to +20 (median -5), lift -45 to +15;
  - rim grabs near pan +20 sit next to the blue-hole pose, so the pick skill has demos close to the drop pose. Not a problem, but worth knowing when reading pan-based skill checks.

## Curation options, with what they keep

| option | rb_mixed | buried_ball | blue_ball |
|---|---|---|---|
| all episodes | 201 eps, 73 min | 45, 25 min | 100, 26 min |
| no failure anywhere | 145, 42 min | 21, 10 min | 87, 21 min |
| no failure, no blind approach (strict VLA, whole episodes) | 103, 29 min | 16, 7 min | 83, 20 min |
| + start choice consistent | 93, 26 min | 14, 7 min | 81, 20 min |
| ACT-tolerant: blind under 1.7 s, at most 1 striped miss, no put-back | 154, 51 min | 33, 19 min | 91, 23 min |

The same data, cut into segments (`segments.csv`, contiguous time ranges):

| segment | rb_mixed clean | buried_ball clean | blue_ball clean |
|---|---|---|---|
| clear one red ball (start: previous drop or home; end: drop + 0.5 s) | 126 (18 min) | 107 foil (14 min) | 1 |
| clear one blue ball | 98 (14 min) | - | - |
| pick striped, view clear | 116 (17 min) | 22 (3 min) | 84 (19 min) |
| pick striped, blind approach | 47 (7 min) | 8 | 4 |
| any segment with a failure | 70 | 27 | 14 |

## Recommended training sets
- **Per-colour ACT** (the plan Tomi chose: one ACT per subtask, the supervisor decides):
  - **clear-red:** rb_mixed `clear_r` clean, 126 segments.
  - **clear-blue:** rb_mixed `clear_b` clean, 98.
  - **pick-striped:** `pick_s` clean from rb_mixed and blue_ball with blind under 1.7 s, 219. Optionally add the recovery tails.
  - Keep `n_action_steps=100` for pick-striped unless the blind ones are removed.
- **SmolVLA** (one policy, prompted per subtask): the same clear segments, plus `pick_s` with `clean=1` and `wrist_blind=0` only (222).
  - If it must decide clear-or-pick itself, also drop the 19 contradicting starts and the 10 "kept clearing after uncovered" episodes. Better: let the supervisor decide.
- **Both:** a hold-out spread over the scenario blocks (section 5).

## Cutting episodes: what to watch
- **Cut only at boundaries the robot will really start from on the arm:** home, just after a drop at a hole, or (for recovery tails) the gripper open over the bowl after a failed close. Never splice two time ranges into one segment: the joint jump would be a teleport in the demo.
- **Every cut segment should start where the supervisor will hand over.** A clear segment starting right after the previous drop matches "park, look, clear the next ball" only if the supervisor does not park in between. If it parks at HOME, start the segments at the moment the arm leaves the hole pose and accept a small mismatch, or add a short HOME→bowl lead-in.
- **The cut points come from labels:** `segments.csv` uses hole time + 0.5 s. The pre-labelled episodes carry about 4% label errors, so spot-check a sample before training on them.
- **LeRobot side:** a new dataset made of time ranges needs our own export (Tian's per-segment dataset builder for the per-colour ACT). `lerobot-edit-dataset` removes whole episodes, so it can only do the episode-level options above.

## Not checked
- Whether a blind demo actually makes the SmolVLA grab the wrong ball on the arm. That needs trials logged with the wrist view.
- Whether the top camera still shows the striped ball during blind approaches. The arm covers the bowl in that view, so it is likely no.

---
Written by Claude (Space0code's agent).
