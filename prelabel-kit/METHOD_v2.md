# Pre-labelling method v2 (striped-ball episodes)

This is v1 (`METHOD.md`) plus the fixes from `FAILURES.md`, turned into checks. Rules do what joints can decide. A Sonnet labeller looks at frames only for the checks the rules list.

## Pipeline
1. `scripts/prelabel_prep_v2.py DATASET EPS --root DS --cache VID --out PL`
   - Downloads each video file of the public `molnrb/so101_pick_*` dataset once and deletes it when its episodes are done.
   - Writes v1's `events.json` and sheets, plus 5 fps frames of both cameras in `PL/<ds>/<ep>/f/`: a 300×300 top crop of the bowl and a 320×240 wrist frame.
   - About 7 s per episode.
2. `scripts/prelabel_rules_v2.py PL DATASET EPS` writes per episode:
   - `draft.txt`: the draft steps, where `?` is a colour to decide, plus the numbered checks;
   - `review.jpg` (+ `review_2.jpg`): one row per check. Each row starts with the top crop and continues with wrist frames, each frame stamped with its time.
3. The labeller reads `draft.txt` and the review sheets and answers every check. Where a check is unsure, it opens more frames with `scripts/prelabel_look.py PL DATASET EP T0 T1 [STEP] [--cam wrist|top|both]`. It writes one spec line per episode.
4. `scripts/prelabel_write_v2.py OUT.jsonl SPEC.jsonl...` writes v1-format records with `by: "prelabel-v2"`.
5. `scripts/score_prelabels.py corrections.jsonl OUT.jsonl` scores against Jakob's corrected episodes.

## What the rules do (joints only)
**Gripper.** Larger means more open:
- closed on nothing: 0-8;
- holding a ball: 8-16.5 (a squashed foil ball can read below 8);
- partial pinch: more than 16.5.

**Zones**, from the arm pose at the event:
- home: lift < -80;
- red hole: pan ≤ -25;
- blue hole: pan ≥ 20;
- front hole (foil datasets): lift > 10 at any other pan;
- bowl: everything else.

v1 also required lift > -5 for the blue hole, which read real blue-hole drops at lift -6 to -12 as bowl releases. A close at a hole zone is ignored only when it comes within 2.5 s after a release there. Otherwise it is a grab at the bowl rim (pan about 20).

**Attempts.** Every close in the bowl is one attempt; consecutive closes with no open between them are merged.

| close | rule |
|---|---|
| to 8-16.5, released later | grab at `t_end - 0.2`; the release gives the colour: red hole r, blue hole b, inside the bowl `?_slip` (check), never released s |
| to below 8 | `?_miss` at `t_end - 0.2` (check the colour), unless it is then carried to a hole: then it is a grab (squashed foil) |
| quick: opened in the bowl within 0.6 s | if another close follows within 2.5 s: a regrasp, not labelled by default (check); otherwise `?_miss` (check: grab + slip or miss) |
| only to more than 16.5 | partial pinch, not labelled; a last close that never reopens is the final grab |
| above the bowl (lift < -50), not a held grab | not labelled |

**Times.**
- Hole: open mid time.
- Home: `arm_home_s + 0.35`.
- Target: `grab - 1.8`, clamped to after the previous event + 0.6 s and before the attempt - 0.6 s. When an unlabelled regrasp precedes the attempt, the target goes before the regrasp. The first target of an episode uses v1's hover/approach rule.

These times matched Jakob's to 0.0-0.2 s on the corrected episodes; targets are within 2 s.

**Foil datasets** (`buried_ball`, `blue_ball`): every non-striped ball is logged as R. A drop at either hole is `r_hole`. `uncertain` says "foil balls logged as R".

## Labeller checks (vision)
Look at the **wrist camera** first. Jakob judges visibility through the wrist view, the operator's view. The top camera only decides visible vs partly. Each row on the review sheet is one check; answer all of them.

1. **START: start state.**
   - The stripe can't be identified in the wrist frames of the first approach (the arm hovering above the bowl, before it goes down): **buried**. A stripe that only shows once the gripper is down among the balls still counts as buried.
   - The stripe can be identified in that first overview: **partly**.
   - The top tile shows the whole outline of the striped ball free, with no red or blue ball overlapping it: **visible**. Any overlap makes it partly.
   - A tiny speck on the top frame is not "partly".
2. **VIS: visibility changes**, after every hole drop, miss and slip.
   - Find the first wrist frame where the striped ball's state differs from the last state:
     - newly identifiable: `s_partly`;
     - free and whole: `s_vis`;
     - covered again: `s_buried`.
   - Emit that code at that frame's time. Jakob's come 0.4-2.7 s after the drop (median 1.3 s).
   - No change: emit nothing. Don't repeat the current state.
3. **BLIND: `wrist_blind`**, before every striped grab or miss.
   - If the stripe is in the wrist view during the approach and then disappears while the arm is still approaching, emit `wrist_blind` at the first frame without it. A red or blue ball usually fills the view or sits on the striped ball.
   - Jakob's come 0-2.6 s before the close (median 0.9 s).
   - If the stripe stays visible until the jaws close, or was never visible, emit nothing.
4. **Q (colour): which ball was the attempt on?**
   - It's the ball **between the jaw tips, at the bottom centre of the wrist frame**, from close onset to close end.
   - A striped ball centred higher in the frame is not the one being gripped.
   - "The ball grabbed next" is only a prior; it was wrong in 2 of 8 of Jakob's cases.
5. **Q (quick): quick closes.**
   - The same ball is regrasped and grabbed right after: no label.
   - The ball visibly left the pile between the jaws and then fell: `X_grab` at close + `X_slip` at open, with its own `X_target`.
   - The jaws never held it: `X_miss`.
   - Every attempt gets a target before it.
6. **Q (drop): release inside the bowl.**
   - Fell out while being carried: `X_slip`.
   - Placed back on purpose: `X_putback`.
   - Pushed aside to uncover the striped ball, without lifting it out of the bowl: no label. Jakob leaves those unlabelled (ep32 9-12 s).
7. **Two balls in one grab** (ep22): if the wrist shows two balls between the jaws, emit both grabs at the same time and a slip for the one that falls.

## Output (one JSON line per episode)
`{"dataset": "so101_pick_rb_mixed", "episode": 35, "start_state": "partly", "confidence": "high|medium|low", "uncertain": "...", "steps": "r_target@7.2 r_grab@9.18 ..."}`
- Steps are `code@seconds`, in time order. Allowed codes:
  - `{r,b,s}_{target,grab,miss,slip,putback}`, `r_hole`, `b_hole`;
  - `s_vis`, `s_partly`, `s_buried`, `wrist_blind`;
  - `home`, `end_away`, `end_empty`.
- No visibility step at t=0: `start_state` carries it.
- Keep the rule times for grabs, holes and home. Times from checks are the frame times on the sheet.
- `confidence`:
  - high: every check was clear;
  - medium: one check unsure;
  - low: more than one, or a messy episode.
- `uncertain`: each unsure call with its time, plus "foil balls logged as R" for the foil datasets.

## Pitfalls (v1 and v2)
- Gripper moves near home at the start (close at 0-1.7 s): ignored by the home zone.
- A striped grab at the bowl's right edge (pan about 17-20) used to read as the blue hole; only closes right after a hole release are now ignored.
- A red or blue ball gripped at 5-7 can be a held grab: the rules check whether it reaches a hole.
- Slips near home are hidden by the home zone: look at the last wrist tiles if the episode ends oddly.
- Jakob is not fully consistent: some quick regrasps are labelled and some are not, and start slivers sometimes count as partly. When unsure, follow the rule default and write the doubt into `uncertain`.

---
Written by Claude (Space0code's agent) for pre-label v2.
