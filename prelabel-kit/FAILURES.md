# Why v1 pre-labels went wrong (rb_mixed 13-49, Jakob's corrections)

How this was checked: Jakob's step labels for rb_mixed 0-49 (`labels/jakob.jsonl`, branch `labels`, 52b8665), the gripper events from `observation.state`, and top and wrist frames at 5 fps around every changed timestamp (`scripts/prelabel_prep_v2.py`, `scripts/prelabel_look.py`). Times are seconds from episode start. The one finding behind three of the five classes: **Jakob judges visibility from the wrist camera, the operator's view. v1 judged it from the top camera.**

## 1. No visibility changes over time (24 missed, 0 found)
- **What Jakob does.** 30 of his 32 mid-episode visibility events come 0.4-2.7 s after a hole drop, miss or slip (median 1.3 s). They mark the moment the arm is back over the bowl and the wrist overview first shows the stripe.
- **Examples.**
  - ep14: R into the hole at 14.09; s_partly at 15.28, when the wrist overview first shows the stripe.
  - ep17: r_hole 9.35; the stripe is plain in the wrist view at 10.4-10.8; s_partly 10.56.
  - ep22 has three changes: s_partly 11.58 after a miss uncovered it, s_buried 16.47 when it was pushed under again, and s_vis 20.16 after a slip.
- **Misleading cue.** The top camera. In ep14 it shows a stripe patch from 13.6 on, before the change. v1 had no step for changes at all; the start state was its only visibility output.
- **Decisive cue.** The first wrist frame after each hole drop or failed attempt in which the stripe can be identified. Compare it with the last state.

## 2. Start state too visible (partly → buried 7, visible → partly 4)
- **Partly → buried** (14, 15, 32, 39, 40, 45, 47). v1 called a small stripe patch on the top start frame "partly". Jakob's partly and buried starts look the same on the top frame: ep6 and ep12 (buried) show patches like ep0 (partly), and ep31 (partly) shows almost nothing. They differ on the **wrist camera during the first approach**, when the arm hovers above the bowl before it goes down:
  - partly starts (0, 19, 26, 31): the stripe is visible in the first wrist overview;
  - buried starts (6, 14, 15): no stripe until the gripper is down among the balls (ep14 6.2 s, ep15 6.8 s).
- **Visible → partly** (23, 30, 33, 36). Most of the ball shows on the top frame, but a red or blue ball overlaps its outline. v1 called that visible. Jakob says visible only when the outline is free (21, 24: ball clear at the edge). The top frame decides this one.
- The yellow-green pixel count on the top start frame overlaps between classes (buried up to 32 px, partly from 0 px), so it is only a hint.

## 3. `wrist_blind` never added (21 missed)
- **What Jakob does.** 19 of his 22 `wrist_blind` events come 0-2.6 s before a grab or miss (median 0.9 s), almost always the striped grab.
- **What it means.** The stripe was in the wrist view during the approach, then drops out because another ball fills the view or sits on top. The gripper then grabs without the camera seeing the striped ball.
  - ep30: from 9.0 s a red ball fills the wrist view and the grab follows at 10.2 (wrist_blind 9.28).
  - ep37: the stripe is gone from 11.2 s (11.61).
  - Counter-example, ep13: the stripe stays in view right up to the grab, so there is no wrist_blind.
- **Why v1 missed it.** It never looked at the wrist frames before a grab; its sheets have one wrist tile 1.5 s before each close.
- **Decisive cue.** Wrist frames every 0.2-0.4 s from about 3 s before each striped grab or miss until the close. wrist_blind goes at the first frame where the stripe has vanished while the arm is still approaching.
- The hue mask is unreliable here: in ep30 the stripe is plain but scores 0 under that lighting.

## 4. The wrong ball on a miss (16, 18, 22, 45, 48; also 32, 40)
- **What v1 did.** It assumed every miss was on the striped ball, because the stripe was "in view".
- **What is in the frames.** In the wrist view the stripe is often centred, while the jaws close on the ball at the **bottom centre of the frame, between the jaw tips**:
  - ep16: at 8.0 s a red ball is between the jaws (r_miss);
  - ep18: a blue ball at 11.4 s (b_miss);
  - ep45: the striped ball at 16.6-16.8 s (s_miss), then R next.
- **Prior.** "The ball grabbed next" is right in 5 of 8 cases (16, 18, 32, 39, 48) and wrong in 40 and 45, where the next grab was a different ball. Use it only as a fallback.
- **Decisive cue.** The colour at the jaw tips in the wrist frames from close onset to close end.

## 5. Missing quick attempts (35, 38, 39, 40, 41, 44)
Two rule bugs and one deliberate v1 rule:
- **Blue-hole drops read as bowl releases.** v1's zone needs lift > -5, but blue-hole releases happen at lift -6 to -12:
  - ep38 17.1 s (pan 27, lift -11);
  - ep39 30.2 s (24, -12);
  - ep41 8.9 s (27, -6).
  So a B grab and its b_hole turned into "unclear grab released in the bowl" and were left out. Pan alone separates them: red hole pan ≤ -25, blue hole pan ≥ 20, bowl -22 to +17.
- **Grab, then release in the bowl within about 1 s.** v1 deleted these as failed pinches (calibrated on ep4 and ep6). Jakob now labels them when the attempt is given up or the ball is lost, as grab + slip or as a miss:
  - ep35: r_grab 9.18, r_slip 9.56;
  - ep38: b_grab 11.52, b_slip 12.01;
  - ep41: r_grab 22.47, r_slip 23.05;
  - ep40: b_miss 7.33;
  - ep45: s_miss 17.27.

  Over his 50 episodes such quick closes are labelled 5 times and left out 8 times. The unlabelled ones are regrasps of the **same** ball right before its real grab (44 at 21.5 and 22.7 s, 25 at 7.7 s, 38 at 23.6 s).
- **Closes high above the bowl** (lift < -50) and **partial pinches** (gripper only to more than 16.5): Jakob labels 1 of 10 (high closes ending empty or in a pinch) and 3 of 30. v1's s_miss in ep17 (to 17) and ep44 (at lift -59) were removed for this reason.

The overall counts, over his 50 episodes:

| close type | labelled | not labelled |
|---|---|---|
| held grab (8-16.5) | 118 | 1 |
| miss (< 8), lift above -50 | 15 | 0 |
| quick grab then open | 5 | 8 |
| partial pinch (> 16.5) | 3 | 27 |

## 6. Retimed grabs
- ep40 s_target: Jakob kept 5.0, then added a second s_target at 7.7 after the B miss.
- ep6 (from the earlier label check): an s_grab sat on a failed close.

Both follow from class 5: once every attempt gets its own target, the retimes go away.

---
Written by Claude (Space0code's agent) for pre-label v2.
