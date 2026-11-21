---
name: so101-data
description: Use for ANY work that touches the so101_pick_* striped-ball data - taking in new recordings, labelling or pre-labelling episodes, finding events in joint traces, tracing a robot failure back to the data, building a training set for a given architecture, choosing hold-out splits or evals, planning a recording batch, or merging datasets. Holds what we learned about the data distribution on 2026-11-21, so a run does not learn the wrong lesson from it.
---

# SO-101 striped-ball data

**How to use this skill:** read "Data at a glance", then **only the section that matches your task**:

| your task | read |
|---|---|
| new episodes arrived, or a batch was recorded | §1 Intake |
| you need event times (grabs, holes, misses, visibility) | §2 Signals |
| labelling or pre-labelling episodes | §3 Pre-labelling |
| the policy fails on the arm and you want the cause | §4 Failure → data |
| building a training set or choosing an architecture | §5 Curation |
| choosing a hold-out split, eval or trial protocol | §6 Evaluation |
| planning what to record next | §7 Recording |
| combining datasets | §8 Merging |

Don't load the whole guide for a narrow task. The full numbers and tables are in `research/DATASET_GUIDE.md`.

## Data at a glance
- Datasets:
  - `molnrb/so101_pick_rb_mixed`: 201 episodes, red and blue balls.
  - `so101_pick_buried_ball`: 45, grey **foil** balls, striped ball often buried.
  - `so101_pick_blue_ball`: 100, foil balls, striped ball mostly visible.
- Labels:
  - Jakob's hand labels for rb_mixed 0–57: branch `labels`, `labels/jakob.jsonl`. **Read-only.** When an episode has several lines, the last one wins.
  - AI v2 pre-labels for the rest, with about 4% ball-event errors: branch `Space0code/prelabel-v2-claude`, `prelabel-kit/prelabels_v2_claude.jsonl`.
- Label codes:
  - `r_/b_/s_` + `target|grab|miss|slip|hole|putback`;
  - `s_vis|s_partly|s_buried`: timestamped, because striped visibility changes during an episode;
  - `wrist_blind`;
  - `home|end_away|end_empty`;
  - foil balls are logged with `r_` codes but are **not** red.
- Segments and per-episode flags: `research/label-patterns/{segments,episodes}.csv` (PR #54).
- Curation dashboard (Tian's, `scripts/curation/`):
  - shows timestamped event chips and the gripper plot;
  - takes AI pre-labels with Confirm and Discard;
  - with `--sync-github` it pushes labels to branch `labels`.
- Anything that only exists on Jakob's machine (intake scripts, the dashboard with his labels): ask Jakob's agent on board #1.

## 1. Intake: new episodes or batches
- Poll the Hub (cheap: a shell loop, not a model) and check each new episode automatically:
  - gripper opening just before the close. The old median was about 24 and the target is 40 or more;
  - closed-gripper or idle stretches;
  - hitting the time limit;
  - number of closes and regrasps;
  - put-backs;
  - final grip, which must hold the ball.
- Each episode then goes to KEEP, DROP or LABEL in `episodes.csv`. Of the first 225 episodes: 133 KEEP, 47 DROP, 45 needed labels from the video. Put-backs were the main reason for drops (39).
- **Check each 20-episode batch within about 15 minutes of upload**, and post problems so the person recording can fix them before the next batch.
- Each recording session gets a **new dataset**: never append to an old one, which would shift its fixed hold-out split.
- Skip duplicate `_session<N>` copies; `so101_pick_rb_mixed_session2` duplicates rb_mixed.

## 2. Signals: finding events without vision (do this before using frames)
- **The gripper position in `observation.state` gives every close and open.** It catches quick failed pinches and wrong grab times that vision labellers skip.
  - A gripper value of about 5–7 (or 18) with a ball in it is a held red or blue ball, not a miss.
- **Shoulder pan at the drop tells you the hole:** red −35 to −29°, blue +22 to +28°.
  - Blue-hole drops can show lift −6 to −12. A rule that requires lift > −5 misses them.
  - A striped grab at the right edge of the bowl (pan about 17, lift about 5) looks like the blue-hole zone, so check it.
  - Slips near home are hidden by the home zone.
- **Mint or yellow-green pixel counts give striped-ball visibility:**
  - on the top camera for `s_vis/partly/buried`;
  - on the wrist camera for blindness: 13 or more 16×16 cells at least 20% mint means "clearly seen". This matched Jakob's `wrist_blind` labels at precision 18/19 and recall 18/21.
- Use frames only where these rules are unsure, and use contact sheets at the gripper events, not full videos.
- Tools:
  - `prelabel-kit/scripts/` (branch `jakobnunnendorf/prelabel-kit`);
  - `prelabel_prep_v2.py` and `prelabel_rules_v2.py` (branch `Space0code/prelabel-v2`);
  - `scripts/analysis/label_patterns.py` (PR #54).

## 3. Pre-labelling with a human checking
- **Pipeline:**
  1. rules from §2 propose steps;
  2. an AI vision pass runs only on unsure moments;
  3. records are written in the human's exact wording and format;
  4. the human confirms or fixes them in the dashboard.
- **Calibrate blind first.** Label episodes the human has already labelled without looking at the answers, then score:
  - event-sequence match;
  - timing error per event type;
  - start-state match;
  - `wrist_blind` recall.

  Post the scores before a full run.
- **Known AI labeller weak spots** (from Jakob's fixes on 37 episodes):
  - no visibility changes over time;
  - start state guessed too visible (a sliver of stripe is "buried");
  - `wrist_blind` never added;
  - a miss blamed on the striped ball when the arm aimed at R or B;
  - quick attempts skipped.
- **Know the human's conventions before scoring.** Jakob leaves most quick failed pinches unlabelled. Then a "missing miss" is not an error.
- Rules for the files:
  - never overwrite hand labels;
  - AI writes only to its own pre-label file;
  - mark every record's source (`by`);
  - back up before merging new pre-labels;
  - load them only into episodes the human hasn't touched.
- Keep it cheap: no polling loops that re-check labels, and frames only where rules are unsure.

## 4. Failure → data: tracing a robot failure to the data
1. **Describe the failure as a decision or a motion.** "Goes straight for the striped ball without clearing" is a decision. "Closes beside the ball" is a motion.
2. **Look up the suspect patterns:**

| failure on the arm | suspect data patterns |
|---|---|
| skips clearing and goes for the striped ball | contradicting starts (buried but direct 8, visible but cleared first 11); one fixed task sentence; the stripe covers about 11% of one SmolVLA token; the old held-out split is all direct picks |
| reaches into the pile, scatters balls, blind grab | grabs while only partly uncovered (43 of Jakob's 58); blind final approaches (72 of 345) |
| keeps clearing after the stripe is free, or stops too early | operators disagree: 10 of 31 keep clearing after the stripe is uncovered |
| misses or slips the striped ball | narrow gripper opening on the approach (median about 24); blind approach; failed approaches left inside the training segments |
| wrong hole, or foil treated as red | foil logged as `r_` and mixed into clear-red |
| retries look strange, hesitates | failed attempts kept in one-frame-VLA data (a retry frame looks like a first-try frame) |

3. **Count the pattern** in the labels or `episodes.csv` before you blame it.
4. **Test offline before any paid run.** Replay training episodes open-loop and check the first target and the hole. Tian's check: right first target 7 of 7, right hole 7 of 7.
5. **Fix it in the data or the setup**, not with more training steps: filter, add a task sentence, let the supervisor decide, or record what is missing (§7).
- **Nearest-demo lookup (open idea):** for a failed trial, find the training episodes whose start frames look most like the trial scene, and check what the operator did there.

## 5. Curation per architecture
Answer each item in your run note and post the kept and dropped counts on board #1.

1. **Whole episodes or segments?** Dropping whole episodes loses 62% of the clean clearing footage. Prefer `segments.csv` (`clear_r`, `clear_b`, `pick_s`), cut only at its boundaries, and never splice two ranges together.
2. **Blind striped approaches:**
   - one-frame VLA: drop their `pick_s`;
   - ACT (3.3 s chunks): keep those blind for under 1.7 s (26), drop the 36 sliver-only ones and those blind for over 1.7 s.
3. **Grabs while only partly uncovered:** the target behaviour is clear until bare, then pick. Prefer picks that are visible at the grab with the wrist seeing the ball.
4. **Contradicting decisions:**
   - starts that contradict (19 rb_mixed);
   - clearing more after the stripe is uncovered (10 of 31).

   Drop them, give each scene its own task sentence, or let the supervisor decide.
5. **Failures:**
   - one-frame VLA: clean segments only;
   - ACT: start the segment after the failed close, because the recovery tail is useful.

   Always drop put-backs.
6. **Foil is not red.**
7. **Buried share:** keep about 40–50%. A strict filter drops it to 26%.
8. **Spot-check about 10 AI-labelled cut points** before training on episodes 58+ or the foil sets.

## 6. Evaluation
- rb_mixed was recorded in **scenario blocks**: 50–74 buried, 75–124 direct, 150–174 buried.
  - Hold out every Nth episode, **never the last N**.
  - The old split 114–119 is all direct picks: its eval loss can't rank clearing.
- Eval loss doesn't rank behaviour. Add open-loop replays on buried episodes and closed-loop trials.
- **Log the top-camera frame at the start of every arm trial**, so you know which scene it faced: buried, partly or visible.
- Report results per scene class, not as one success rate.

## 7. Recording the next batch
- Rules:
  - clear until the striped ball is **bare**, then pick it;
  - approach from a side where the wrist camera keeps the ball in view;
  - open the gripper to 40 or more;
  - never put balls back;
  - end at home holding the ball;
  - mix scenarios within each batch.
- **Record what §4 shows is rare**, e.g. the full path from buried, through clears, to bare and a clean grab, and near-miss recoveries.
- Use one task sentence per batch, upload each 20-episode batch, and run §1 on it before the next batch.

## 8. Merging datasets
- Check that the schema, camera names, resolution and fps match before merging.
- Give each scene its own task sentence at merge time.
- Keep the foil datasets out of red clears.
- Keep the buried share balanced (§5.7).
- Keep the hold-out episodes fixed across merges.
- Keep the source dataset and episode id on every segment, so labels and failures can be traced back.

## Open ideas (not done yet; add results here when someone does them)
- **Operator consistency score:** per decision (clear or pick, stop clearing), how often the demos disagree in similar scenes.
- **Scene drift checks** across recording days: lighting, camera exposure, bowl and hole positions.
- **Motion statistics** per scene: jerkiness, idle time and episode length, to choose ACT chunk sizes and time budgets.
- **Nearest-demo lookup** for failed trials (§4).
