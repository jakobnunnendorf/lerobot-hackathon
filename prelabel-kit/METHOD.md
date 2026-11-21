# Pre-labelling method (rb_mixed, SO-101 striped-ball task)

Goal: write labels in Jakob's style so he only confirms. Never write to ~/so101/curation/labels.jsonl; output goes to ~/so101/curation/prelabels.jsonl.

## Tools (all in scripts/ of this folder)
- `prelabel_prep.py DATASET EPS` -> prelabel/<dataset>/<ep>/ events.json, sheet.jpg (+sheet_2..), start_bowl.jpg. ~0.7 s/episode.
- `prelabel_rules.py DATASET EPS` -> proposed step list from events.json (rules below). Calibrated on rb_mixed 0-6.
- `prelabel_write.py` -> `write_prelabel(...)`: builds summary/verdict with his exact wording, atomic append, skips existing.
- `batch_rb_13_40.py` is the worked example (one SPEC row per episode: start_state, confidence, uncertain, steps).
- Run: `uv run --no-project --with pandas --with pyarrow --with numpy --with av --with pillow python -I <script>`; `-I` drops the script dir from sys.path, the scripts add it themselves.

## Gripper convention (observed)
Larger = more open. Home/closed on nothing 0-8; approach-open 15-40; holding a ball 10-16 (striped can read 17); partial pinch 17-20 = ball not held.

## Events (events.json)
Gripper events from a 0.3 s window: close/open with t (onset), t_end, g_before/after, arm pose [pan,lift,elbow,wflex,wroll]. arm_leaves_s, arm_home_s (3 big joints within 8 deg of end pose).
Zones by arm pose at the event: red hole = pan < -20 and lift > 0; blue hole = pan > 15 and lift > -5; home = lift < -80; otherwise bowl.

## Rules and timing offsets (calibrated on his episodes 1-6, errors in s)
- Grab = close t_end - 0.2 (median err 0.0, max 0.7). Closes in the bowl with g_after 8-16.5 are grabs; consecutive closes with no open between merge (t_end of the last). Close to <8 on nothing = miss. Close only to 17+ = partial pinch (ignore) unless it is the last event and the ball is held at the end (ep40).
- Hole release = open mid time, in hole zone (err 0.03). Red hole => the ball was R, blue hole => B. This identifies colour without looking; the final ball not released is striped.
- Home = arm_home_s + 0.35 (err 0.04).
- Target: later targets = max(grab - 1.8, previous event + 0.6) (his gaps 1-2.5 s, median abs err ~0.6). First target = mean of (last time the wrist is flat, wflex >= 96, +0.1) and (first approach-open onset - 0.2). First targets are the weak spot (err up to 2.5 s) because he marks it when the arm clearly heads at the ball, which varies.
- A grab followed within 1.3 s by an open inside the bowl, then a new grab = failed pinch; he does not label it (ep4, ep6): keep the first target and the final grab.
- A grab held > ~1.3 s then opened inside the bowl = slip/drop (s_slip) or put-back (X_putback if the ball was R/B and clearly placed back).
- Start state is on top-camera start frame (start_bowl.jpg, bowl close-up): visible = most of the ball shows, partly = some stripe visible, buried = none. A tiny sliver is ambiguous (his ep0 sliver = partly, ep6 sliver = buried): flag it.
- Steps carry no visibility step; start_state carries it and the note starts with "Striped partly covered." etc. (he may add timestamped s_vis/s_partly/s_buried later, ep4 did; not predicted).
- Verdict as in app.js suggest(): drop if any putback, end empty/away, or >= 2 striped misses/slips; keep if s_grab and home; else look.

## Looking at the sheets (required)
1. start_bowl.jpg of all episodes in a montage for the start state.
2. Last wrist tile of all episodes in a montage: the final ball must be striped (all 28 were).
3. Full sheets for every episode whose proposal has "?" balls, misses, slips, pinches, or > 3 s release-grab gaps.
Colour cues: red ball = pink-red crinkled foil, blue = glossy royal blue, striped = green/yellow stripes (may look pale cream/green). Red hole = red tape roll on the right of the bowl (pan ~ -30); blue hole = blue tape roll below the bowl (pan ~ +25).

## Pitfalls
- Gripper moves near home at the episode start (close at 0-1.7 s, g 20 -> 3): ignore (zone home).
- Closes in a hole zone are the gripper closing after release, not grabs.
- The ball identity of an unreleased grab followed by a bowl release is unknown from joints: look at the sheet.
- Episodes may be labelled partially by him while you calibrate (ep4 changed); re-read labels.jsonl before comparing.
- Failed digging attempts (pinches around the striped ball) are left unlabelled and listed in "uncertain" with times.
