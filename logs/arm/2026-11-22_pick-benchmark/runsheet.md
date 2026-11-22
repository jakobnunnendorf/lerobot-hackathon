# PICK benchmark: run sheet for the table (22 Nov)

7 models × 8 layouts = 56 runs. Every run is recorded. Action size 30, cap 90 s, gripper capped at 200.
The window says which run, layout and model is next; this sheet says the same, so you can build ahead.

## 0. Once, before run 1 (about 3 min)

1. **Table:** nothing yellow-green anywhere (tape roll off the table). Lighting as during recording.
2. **Balls:** the original number of red and blue balls in the bowl, plus the one striped ball. Both holes empty.
3. **Wrist camera cable:** it dropped out twice since the gripper swap. Push its plug in firmly at the wrist and give it slack at the elbow.
4. **Reference photo:** one phone photo from the top camera's angle of the empty table and bowl: `ref_top.jpg`.
5. **Arm:** it may still hold the old ABORT pose above the bowl. That is fine: after SPACE it eases to HOME (slowly) before the model starts.
6. **Hand near the keyboard** for the whole session. **ESC** = abort (arm holds), **SPACE** during a run = end it now.

## 1. Every run, the same 6 steps

1. **Build** (or rebuild) the layout for this run from its card below (and from `L<n>.jpg` after its first run).
2. **First run of a layout only:** take the phone photo `L<n>.jpg` from the top camera's angle **before** pressing SPACE.
3. **Press SPACE.** The arm goes HOME, then the model runs until it returns with a ball, 90 s pass, or you press SPACE/ESC.
4. **Script asks "Ball at start?"** (first run of a layout only): **h** hidden, **p** partly, **v** visible, as on the card.
5. **Script asks "Result?":** **s** success, **w** wrong_ball, **e** empty, **d** dropped, **t** still_digging, **n** never_started, **o** other.
6. **Write in `pick_benchmark_fails.csv`** (same run number): one failure code, the number of grasp attempts (how often it closed), and a short note. Then wait about 11 s while the episode saves (the arm holds still).

**Failure codes** (the first thing that went wrong): `ok`, `never_started`, `wrong_ball`, `blind_dive` (reached into the pile with the stripe covered), `stall` (jaws in front of the ball, no motion), `empty_close` (closed beside the ball), `slip` (held, then dropped), `scatter` (pushed balls out, or the stripe deeper), `no_clear` (never moved a covering ball), `stutter` (long pauses), `collision` (hit bowl/table hard), `timeout_other` (describe in note).

**Something went wrong at the table** (camera dropped, layout built wrong, table bumped, someone touched the arm or balls): answer **o**, write `invalid` or `abort` and why in the note. It gets rerun at the end.

**Every 2 layouts** (after L2, L4, L6): tell me, and I check the gripper reading at rest and the slow-tick %; you touch the servos for warmth.

## 2. Layout cards

Bowl positions are seen from above: **left / right** as you face the robot from the front, **centre** = middle of the bowl.

| Layout | Class (key) | Striped ball | How to cover it |
| --- | --- | --- | --- |
| **L1** | hidden (**h**) | left side of the bowl | 1 red + 1 blue on top; **no stripe visible** from above |
| **L2** | partly (**p**) | centre | 1 red ball over about half of it; **some stripe visible** |
| **L3** | hidden (**h**) | right side | 1 red + 1 blue on top; no stripe visible |
| **L4** | visible (**v**) | centre, on top | nothing on it, nothing touching it |
| **L5** | hidden (**h**) | centre | 2 balls on top (1 red, 1 blue); no stripe visible |
| **L6** | partly (**p**) | left side | 1 blue ball over about half of it; some stripe visible |
| **L7** | hidden (**h**) | right side | 2 blue + 1 red on top; no stripe visible |
| **L8** | hidden (**h**) | left side | 2 red + 1 blue on top; no stripe visible |

After every run: take the balls out of the holes, put them back in the bowl, rebuild the same layout from its photo.

## 3. Run order (model per run)

| Layout | Runs | Model order |
| --- | --- | --- |
| L1 hidden | 1–7 | v2, hidden, v3, clean_all, clean_scratch, balazs, vla |
| L2 partly | 8–14 | vla, balazs, clean_scratch, clean_all, v3, hidden, v2 |
| L3 hidden | 15–21 | v2, hidden, v3, clean_all, clean_scratch, balazs, vla |
| L4 visible | 22–28 | vla, balazs, clean_scratch, clean_all, v3, hidden, v2 |
| L5 hidden | 29–35 | v2, hidden, v3, clean_all, clean_scratch, balazs, vla |
| L6 partly | 36–42 | vla, balazs, clean_scratch, clean_all, v3, hidden, v2 |
| L7 hidden | 43–49 | v2, hidden, v3, clean_all, clean_scratch, balazs, vla |
| L8 hidden | 50–56 | vla, balazs, clean_scratch, clean_all, v3, hidden, v2 |

**Models:** `v2` = Tomi v2 @20000 · `hidden` = Tian's hidden-ball fine-tune @002000 · `v3` = Tian's longer v2 @030000 · `clean_all` / `clean_scratch` = Jakob's clean-data models · `balazs` = molnrb/act_pick_rb_mixed @080000 · `vla` = Jakob's SmolVLA @003000 (may move slower or pause: that is `stutter` only if it pauses long).

## 4. Stopping

- Stop only **after a whole layout** (after its 7th run), with **q** at the next "set up" prompt.
- If L4 is not done about 50 min after run 1, stop after L6 (42 runs).
- At the end: tell me. I post the tallies, push the CSVs, `fails.csv` and photos to `molnrb/arm-logs`, and the recording pushes to `molnrb/so101_eval_pick_20261122` (private) when the script exits.
