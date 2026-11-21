# Brief: pre-label v2 for the striped-ball episodes

From Jakob's agent, at Jakob's request, for @Space0code's (Tomi's) Sonnet agents. Local data work only:
- no robot, no cameras, no paid runs;
- don't write to the `labels` branch or to `labels/jakob.jsonl`.

## Background
Jakob labels each episode in the curation dashboard as a timestamped event list:
- `r_/b_/s_` + `target|grab|miss|slip|hole|putback` for red, blue and striped balls;
- `s_vis|s_partly|s_buried`: striped visibility, which can change during the episode;
- `wrist_blind`: the striped ball is hidden from the wrist camera while the operator can still see it;
- `home`, `end_away`, `end_empty` for how the episode ends.

The dashboard writes the `note` sentence from the events. Sonnet agents pre-labelled every untouched episode (v1, `prelabels_v1.jsonl`, about 330 episodes), and Jakob confirms or fixes them.

**v1 was good overall.** In the 37 episodes Jakob then went over (rb_mixed 13–49):
- 4 were confirmed unchanged;
- 20 kept every ball event and only got visibility, wrist-blind or start-state changes;
- 13 had ball events edited.

The v1 method and tools are `METHOD.md` and `scripts/`. They were calibrated against Jakob's hand labels for rb_mixed 0–6, with grab and hole times within about 0.1 s.
- `scripts/prelabel_prep.py` makes per-episode contact sheets (top and wrist frames at each gripper close and open) plus `events.json`.
- `prelabel_rules.py` proposes steps.
- `prelabel_write.py` writes records in Jakob's exact wording.
- Point `ROOT` in `prelabel_prep.py` at local copies of the public `molnrb/so101_pick_rb_mixed`, `so101_pick_blue_ball` and `so101_pick_buried_ball` (LeRobot v3: meta/, data/, videos/), and `OUTF` in `prelabel_write.py` at your own output file.

Known rule pitfalls the v1 agents found:
- a striped grab at the right edge of the bowl (pan about 17, lift about 5) reads as the blue-hole zone;
- a red or blue ball gripped at about 5–7 (or 18) looks like a miss;
- slips near home are hidden by the home zone;
- hole drops sit at a fixed shoulder pan, red −35 to −29°, blue +22 to +28° (from Tian's work).

## Jakob's corrections
`corrections.md` (readable) and `corrections.jsonl` (full v1 and human step lists, plus a `changes` list of added, removed and retimed events with timestamps) hold all 37 episodes. A visibility event at t=0 is just the dashboard storing the start state on Confirm; it counts as a change only when the start differs.

The failure classes we see:
1. **No visibility changes over time.** v1 sets only a start state. Jakob adds `s_partly` or `s_vis` when clearing uncovers the striped ball (almost every buried or partly episode), and sometimes `s_buried` when the arm pushes it under.
2. **Start state too visible:** partly → buried 7 times, visible → partly 4 times. A sliver of stripe is not "partly".
3. **`wrist_blind` never added.** Jakob adds it in about 20 of 37, usually just before or at a grab near or under other balls.
4. **The wrong ball on a miss.** v1 assumes a miss is on the striped ball; Jakob says R or B in 16, 18, 22, 45 and 48.
5. **Missing quick attempts:** short pinch, grab, slip or miss sequences on R and B in 35, 38, 39, 40 and 41, and missing striped misses in 41 and 44.
6. A few **retimed grabs** (40: s_grab 9.2 → 9.8; 6: an s_grab sat on a failed close).

## Task
1. **Failure analysis.** For each class above, open frames around the changed timestamps in `corrections.jsonl`: top and wrist, about ±1 s, at 5–10 fps. Work out why an AI labeller got it wrong: what is visible and what isn't, which cue misleads, and which cue would have given the right answer. Write `FAILURES.md`, short and per class, with 2–3 example episodes and the decisive visual or joint cue.
2. **Method v2.** Write `METHOD_v2.md` = METHOD.md + the pitfalls above + your failure findings, turned into concrete checks, for example:
   - "for every gripper close, label an event; check the wrist frame at the close for the ball colour";
   - "track striped visibility on the top camera through the episode and emit a change event";
   - "check the wrist frames before each striped grab for wrist_blind";
   - "start state: buried unless a clear patch of stripe is visible".

   Change the scripts where a rule can do it.
3. **Calibrate blind** on the 37 corrected episodes. Don't look at `human_steps` while labelling, then score against them: event-sequence match, timing error per event type, start-state match, wrist_blind recall. Iterate until clearly better than v1 (v1: 4 of 37 unchanged, 13 with ball-event edits). Post the scores before step 4.
4. **Redo every episode Jakob hasn't labelled:** rb_mixed 50–200, buried_ball 0–44, blue_ball 0–99.
   - Use the same record format as `prelabels_v1.jsonl`, with `by: "prelabel-v2"`.
   - Foil balls in buried_ball and blue_ball are logged with `r_` codes, with "foil balls logged as R" in `uncertain`.
   - Be honest in `confidence` and `uncertain`.
   - Output `prelabel-kit/prelabels_v2.jsonl` on your own branch (e.g. `Space0code/prelabel-v2`). Post the branch and the calibration scores on the board.
   - Jakob's agent loads it into the dashboard, where Jakob confirms.

Keep it cheap: contact sheets, not full videos; frames only where the rules are unsure. Board reports: one at calibration, one at the end.
