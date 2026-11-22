# Should the learned scene classifier choose the clearing colour, or replace the supervisor's LOOK rule?

- Status: DECIDED (2026-11-22, Tian's agent): no. Not wired in, no flag added. Offline only, 99 frames from 10 episodes.

## Decision

The supervisor keeps its own LOOK rule for pick versus clear. The classifier does not choose the clearing colour: on the frames where the rule says clear, it picks the right colour exactly as often as "always red" does. The one change the numbers support is a bowl region for the LOOK mask (see below), which needs Tian's go because it changes the default.

## Why

- **There are no past tick logs with top frames.** The tick log (`scripts/supervisor/ticks.py`) holds joints and loads, no images (board, Balazs's agent). The only top-camera material from supervisor rounds is the three round videos on `molnrb/arm-logs` (`2026-11-21_1918_vla-restart`: `20k_layout1`, `5k_take1`, `5k_take2`, one round each, LOOK off). So the comparison ran on Jakob's labelled hold-out instead, the 99 frames of 10 `so101_pick_rb_mixed` episodes (6, 23, 65, 88, 99, 110, 126, 157, 177, 191). Labels: human for episodes 6 and 23, AI pre-labels v2 (`Space0code/prelabel-v2-claude`) for the rest. Both rules ran on the same full-size frames, the classifier with the recorded arm state. The runs reproduce Jakob's numbers (his mint-fraction rule 0.667 / 0.687, the classifier 0.566 / 0.667). Script: `scripts/scene_clf/compare_supervisor_rule.py` on branch `TianK003/scene-clf-compare-full` (it needs the classifier code of `jakobnunnendorf/label-archs`, which is not on `main`).
- **Pick versus clear** (53 true pick frames, 46 true clear frames; the supervisor rule is "target-mask area >= 150 px", `TopCameraVisibility`):

  | | accuracy | clear frames sent to pick | pick frames sent to clear |
  |---|---|---|---|
  | supervisor rule, whole frame | 0.63 | 37 of 46 | 0 of 53 |
  | supervisor rule, bowl region only | 0.78 | 21 of 46 | 1 of 53 |
  | Jakob's mint-fraction rule | 0.74 | 26 of 46 | 0 of 53 |
  | classifier | 0.78 | 12 of 46 | 10 of 53 |

  The rule never skips a pick it should make and over-picks buried balls. The classifier does the reverse: fewer wrong picks, but it clears 10 of 53 visible balls.
- **The supervisor rule's low score is a yellow tape roll, not the threshold.** In episodes 126, 157, 177 and 191 (50 of the 99 frames) a yellow tape roll lies on the table left of the bowl. The mask picks it (3,500-4,000 px, bigger than the ball), so the area is above 150 px on every frame, whatever is in the bowl; in episode 157 the ball is buried the whole time and the rule says "visible" on all 33 frames. With everything outside the bowl blanked before masking, the median area falls to 0 on buried, 370 on partly covered and 712 on visible frames (whole frame: 752, 3,528, 820), and accuracy rises from 0.63 to 0.78. The three live round videos (18:20-18:37, 21 Nov) show no tape, and `decisions/target-visibility-check.md` was measured on the foil-ball datasets (`pick_blue_ball`, `pick_buried_ball`), not on this layout, so nothing has failed on the arm yet. Without the tape (the six other episodes, 49 frames) the whole-frame rule is 0.84, the classifier 0.76.
- **Clearing colour: no gain.** Of the 46 true clear frames the bowl-only rule sends 25 to clear (19 red, 6 blue). The classifier names the colour right on 19 of them, "always red" on 19. Over all 99 frames, bowl-only rule + classifier colour scores 0.717 on the three-way next action, the same as the bowl-only rule clearing red. Jakob's "11 of 14 clear_blue right" holds only when the classifier also decides clear versus pick; behind the rule's gate it finds 3 of the 6 blue frames.
- **Three real round videos** (parked top frame, stills at frames 0, 5 and 15): `20k_layout1` and `5k_take2` show a visible striped ball (mask 794-862 px and 1,062-1,180 px); rule and classifier both say pick, as the pick policy did. `5k_take1`'s video starts with the ball already in the gripper (mask 0, classifier "visible", 0.85-0.97 pick); that is not a state in which the supervisor decides. No round has a human label beyond what the frame shows, so this is a sanity check, not a score.
- Not much weight: 10 episodes, and frames within an episode look alike (each point is about one frame, one episode is up to 33 frames; four of the ten carry the tape). Differences under about 8 points are noise, as in `decisions/scene-classifier.md` on `jakobnunnendorf/label-archs`.

## Alternatives

- **Classifier picks the clearing colour behind a flag** - tried offline (above): equal to always-red on the frames where it would act. Not built.
- **Classifier replaces the rule** - tried offline: 10 of 53 visible balls would be cleared instead of picked, and a wrong clear costs a whole CLEAR round.
- **Bowl region for the LOOK mask** - tried offline (above): fixes the tape false positive. Not applied, because `--look` and `--clear-gray-policy` are off by default and a region changes the area the threshold sees; `BOWL_ROI` in `scripts/scene_clf/train.py` (on `jakobnunnendorf/label-archs`) is the 160x120 rectangle Jakob used (rows 12-72, columns 62-142), valid only while the camera does not move.

## Revisit if

- A yellow tape roll, or anything yellow-green, lies on the demo table and LOOK or the clearing policy is switched on: add the bowl region first.
- Round videos with the top frame at each trial's start and a human visibility label exist (the so101-data skill, section 6): rerun with real supervisor rounds.
- The classifier is retrained on a bowl crop at 2x resolution (`decisions/scene-classifier.md` on `jakobnunnendorf/label-archs`): rerun `compare_supervisor_rule.py`.
