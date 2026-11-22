# Benchmark: pick policies on the new servo, 22 Nov

From `research/BENCHMARK_DESIGN_TEMPLATE.md`. **This is data collection only; the analysis and decisions come later, in a separate write-up.** Sections 1–9 stay fixed once the first run starts. Changes go into section 11.

| | |
| --- | --- |
| Status | draft, not shared yet |
| Version | v1, 22 Nov 08:00 CET |
| Board posts | earlier plans 7075454867 and 7075469917 (this file replaces them); data: – |
| Code | `main` with PR #69 merged (SmolVLA + `--record`), merge commit 25b666c |

**Open before the first run:**
1. `balazs` training details (section 2): asked on the board, 7075644974.
2. The recording dataset repo (section 7): asked on the board, 7075644974.

---

## 1. What we measure

- **In one sentence:** how often, how fast and in what way each of 7 pick models succeeds or fails on the new servo, on the same 8 layouts, with every run recorded.
- **What the data must allow later** (no analysis is fixed here):
  - success per model and per start class (V / P / H);
  - a layout-by-layout comparison between models;
  - time to success, and stutter (`late_ticks_pct`);
  - how each failure happened: the code plus the recording;
  - each run's similarity to the training demos (so101-data skill §9);
  - VLA vs ACT, under the same conditions.

## 2. Candidates

All ACT models use chunk 100 in training. Every model runs at `n_action_steps` 30 here, the demo setting from #67. All training used `molnrb/so101_pick_rb_mixed` unless noted. Details are from each checkpoint's `train_config.json` on the Hub.

| Short name | Hub repo @ step | Type | Training data | Init from | Training at this checkpoint | Why it is in |
| --- | --- | --- | --- | --- | --- | --- |
| `v2` | `Space0code/act_redblue_from_foil_v2@20000` | ACT | all episodes | `molnrb/act_pick_ball_combined` (the foil-scene model) | 20K steps, batch 32, lr 2e-5 | reference: the current demo model |
| `hidden` | `TianK003/act_hidden_from_v2@002000` | ACT | 40 buried / partly covered episodes | `v2@20000` | +2K steps (of a 3K run), batch 32 | v2 tuned for hidden balls |
| `v3` | `TianK003/act_redblue_from_foil_v3@030000` | ACT | all episodes | `v2@20000` | +10K steps (step numbers continue from 20K), batch 32 | v2 trained longer |
| `clean_all` | `jakobnunnendorf/act_clean_from_all@010000` | ACT | 93 clean episodes (10 more held out) | `brouthen/act_so101_pick_striped_all_pretrain_50k@050000`: 50K steps, batch 8, on `brouthen/so101_pick_striped_all` (314 episodes) | 10K steps, batch 32 | broad pretrain + clean fine-tune |
| `clean_scratch` | `jakobnunnendorf/act_clean_scratch@030000` | ACT | the same 93 clean episodes | none (from scratch) | 30K steps, batch 32 | clean data alone; compare with `clean_all` |
| `balazs` | `molnrb/act_pick_rb_mixed@080000` | ACT | **open item 1** | **open item 1** | 80K | fastest in the 02:15 test (9–15 s on visible balls, old servo) |
| `vla` | `jakobnunnendorf/smolvla_redblue_v2@003000` | SmolVLA | all episodes | `lerobot/smolvla_base` | 3K steps (of a 10K run, only 1K/2K/3K uploaded), batch 64, lr 1e-4; chunk 50 | the only VLA |

- **SmolVLA specifics:**
  - task sentence: "Pick up the striped ball from the bowl of red and blue balls, uncovering it first if it is buried, and return to the start position". It matches the dataset's `meta/tasks.parquet`, and it is the `--task` default.
  - cameras: top/wrist are renamed to camera1/camera2 by the checkpoint's own preprocessor.
- **Checked before the runs:**
  - [x] `v2` and `vla` load and get through `select_action`: offline test, PR #69;
  - [x] `hidden` and `v3` resolve without a token (board 7075472893);
  - `clean_all`, `clean_scratch` and `balazs` aren't load-tested; problems are sorted out at the table.
- **Left out:** `v1` (replaced by `v2`); the old-servo foil models; `brouthen/*` models other than as the init for `clean_all`.
- **Cut order if time runs short:** don't drop models; cut layouts instead (section 9).

## 3. Fixed conditions (the same for every run)

| Condition | Value |
| --- | --- |
| Arm, gripper servo, torque limit | SO-101, new gripper servo (#64), torque cap 200 (#65) |
| Robot laptop, GPU, lerobot version | robot laptop, RTX 4050, lerobot 0.7.0 |
| Control rate, run time cap | 30 Hz, 90 s |
| Chunking at test | `--n-action-steps 30` for every model, including SmolVLA |
| Start pose | HOME (the script parks there after every run) |
| Top camera, wrist camera | not moved since recording; one reference photo at the start, `ref_top.jpg` |
| Lighting | as during recording; one photo at the start, another if it changes |
| Holes | as recorded: red hole pan −35..−29°, blue +22..+28° |
| Balls | the original numbers of red and blue balls, plus one striped ball |
| Table | nothing yellow-green on it, including the tape roll: it fools the visibility check |
| Display / Rerun | off |

**Command** (pre-download all seven checkpoints first, so the first run doesn't stall):
```
bash scripts/eval/eval_pick.sh --runs 8 --seconds 90 --n-action-steps 30 \
  --record ~/so101/evals/rec_20261122 --push-to-hub molnrb/so101_eval_pick_20261122 \
  --policy v2=Space0code/act_redblue_from_foil_v2@20000 \
  --policy hidden=TianK003/act_hidden_from_v2@002000 \
  --policy v3=TianK003/act_redblue_from_foil_v3@030000 \
  --policy clean_all=jakobnunnendorf/act_clean_from_all@010000 \
  --policy clean_scratch=jakobnunnendorf/act_clean_scratch@030000 \
  --policy balazs=molnrb/act_pick_rb_mixed@080000 \
  --policy vla=jakobnunnendorf/smolvla_redblue_v2@003000
```

## 4. Test layouts

- **Classes:**
  - **V** visible: the striped ball is fully seen from the top camera;
  - **P** partly hidden: some stripe visible, but touched or covered;
  - **H** hidden: no stripe visible from the top; it has to be dug out.
- **Layout list and order:** 8 layouts, **H P H V H P H H** (5 hidden, 2 partly, 1 visible).
- **Why this mix:**
  - the earlier new-servo runs already have visible and partly covered layouts (section 13), and hidden is the least measured case;
  - the one visible layout is a sanity check, e.g. for SmolVLA;
  - for comparison, 27 of the 103 clean episodes (26%) start buried.
- **Where the striped ball sits:** a mix of left, centre and right of the bowl (Appendix A).
- **Same layout for every model:**
  - a phone photo from the top camera's angle before the first run of each layout, `L<n>.jpg`;
  - after every run, rebuild the layout from the photo;
  - the script rotates the model order per layout.

## 5. Protocol

- **Design:** every model runs every layout once. 7 models × 8 layouts = **56 runs**.
- **Order:** layouts in the listed order. The script rotates the model order within a layout, so no model always goes first after a reset.
- **Before the first run:**
  1. Pull `main` (PR #69 is merged).
  2. Pre-download all seven checkpoints.
  3. Take the reference photos.
  4. Do one warm-up run (logged as warm-up, not counted).
- **One run:**
  1. Rebuild the layout and check it against `L<n>.jpg`.
  2. The arm is at HOME with the gripper open.
  3. SPACE starts the run. It ends at success, at the 90 s cap or on abort.
  4. Enter the start state (once per layout), the result and the failure code.
  5. The episode saves while the arm holds its pose (about 11 s).
- **Abort:** someone touches the arm or the balls during the run. Write down why, and rerun that run once at the end.
- **Invalid:** a camera timeout, a script crash, a wrongly built layout, or a bumped table. Log it, mark it invalid, keep the recording, and rerun it at the end.
- **Checks every 14 runs (2 layouts):**
  - the gripper reading at rest;
  - servo warmth (touch);
  - a top camera photo compared with `ref_top.jpg`.
- **Stop a model early only for safety:** repeated hard collisions, or a model that never moves at all. Write down why.

## 6. Outcomes and failure codes

- **Success:** the striped ball is held at the start pose before the cap, with a gripper reading of about 6–20.
- **Partial:** written as a note on a failure, e.g. "had it, dropped it on the way back".
- **One code per run.** These are the codes from the earlier plan, plus three more. Pick the first thing that went wrong:

| Code | Meaning |
| --- | --- |
| `ok` | success |
| `never_started` | the arm stays at rest |
| `wrong_ball` | brought back a red or blue ball |
| `blind_dive` | reached into the pile while the stripe was covered; the wrist view is full of other balls |
| `stall` | jaws in front of the ball, no motion |
| `empty_close` | closed beside the ball |
| `slip` | held it, then dropped it |
| `scatter` | pushed balls out of the bowl, or the stripe deeper |
| `no_clear` | never moved a covering ball when the stripe was buried |
| `stutter` | long pauses in the motion (expected risk for SmolVLA on the RTX 4050) |
| `collision` | hit the bowl, the table or itself hard enough to stop |
| `timeout_other` | anything else; describe it in `note` |

- **Second check:** the code given at the table is checked later against the recording, and both codes are kept.

## 7. Data collected

| What | How | Where |
| --- | --- | --- |
| One CSV row per run: run, layout, episode_index, policy, checkpoint, started, n_action_steps, cap_s, seconds, end, gripper_end, left_home, late_ticks_pct, start_state, result | automatic | `~/so101/evals/eval_20261122_*.csv` |
| Every run as a LeRobot dataset episode: top + wrist video, state, action, 30 fps, task sentence | automatic, `--record` | `~/so101/evals/rec_20261122`, then `molnrb/so101_eval_pick_20261122` (open item 2) |
| Per episode: row + policy type + git commit | automatic | `rec_20261122/runs.jsonl` |
| Failure code, number of grasp attempts, note | by hand | `fails.csv`: `run,policy,layout,fail,attempts,note` |
| Layout photos `L1..L8.jpg`, reference photos | phone | uploaded with the logs |
| Checks every 14 runs, anything surprising | by hand | sections 11 and 12 |

- **Upload:**
  - push `eval_*.csv`, `fails.csv` and the photos to `molnrb/arm-logs` under `logs/arm/2026-11-22_pick-benchmark/` after layout 4 and again at the end;
  - the dataset pushes at the end (`--push-to-hub`);
  - keep the local copies until the upload has been checked.
- **Record everything,** including warm-up, aborted and invalid runs. Mark them; don't drop them.
- **Earlier CSVs:** any `eval_20261122_*.csv` on the laptop that the board summaries don't show goes to `molnrb/arm-logs` too.

## 8. How much data, and what it can show

- **Runs:** 8 per model, 5 of them hidden; 56 in total.
- **For orientation:**
  - 4 successes out of 8 has a 95% interval of about 22–78%;
  - 2 out of 5 hidden is about 12–77%.
  - So this gives a first picture per model. It is sharpest on hidden balls when combined with the recordings and failure codes.
- **Extra runs, if time is left:** a second pass over the hidden layouts (L1, L3, L5, L7, L8) for every model.

## 9. Time budget and safety

- **Time per run:** about 50 s on average (successes take about 20–40 s, failures 90 s), plus about 10 s to save and about 45 s to reset. That's about 1.75 min.
- **Total:** 56 × 1.75 min ≈ 98 min, plus about 10 min of setup. That's **about 1 h 50 min**, more than the 1.5 h available.
- **Cut rule:**
  - if layout 4 isn't finished by 50 minutes in, stop after layout 6 (H P H V H P: 3 hidden, 2 partly, 1 visible);
  - that's 42 runs, about 75 min;
  - cut whole layouts so every model keeps the same layouts. Don't cut models.
- **Safety:**
  - only the person at the table starts a run, and agents never start one;
  - torque cap 200;
  - a hand stays near the stop key;
  - stop at a strange sound or a hard collision.

---

## 10. Raw tallies (during the runs, no interpretation)

**Per layout** (`ok` / failure code / abort / invalid):

| Layout | Class | v2 | hidden | v3 | clean_all | clean_scratch | balazs | vla |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | H | | | | | | | |
| L2 | P | | | | | | | |
| L3 | H | | | | | | | |
| L4 | V | | | | | | | |
| L5 | H | | | | | | | |
| L6 | P | | | | | | | |
| L7 | H | | | | | | | |
| L8 | H | | | | | | | |

**Counts per model:**

| Model | V (succ/n) | P | H | Total | Aborts | Invalid |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | | | | | | |
| hidden | | | | | | |
| v3 | | | | | | |
| clean_all | | | | | | |
| clean_scratch | | | | | | |
| balazs | | | | | | |
| vla | | | | | | |

## 11. Deviations from the protocol

| Time | What changed | Why |
| --- | --- | --- |
| | | |

## 12. Notes from the table

Free text: anything seen at the table that the numbers don't show.

## 13. Where the data is, and related data

- **This benchmark:**
  - recording dataset: `molnrb/so101_eval_pick_20261122` (revision: –);
  - logs: `molnrb/arm-logs/logs/arm/2026-11-22_pick-benchmark/`;
  - code: `main` at 25b666c (PR #69), plus the command in section 3;
  - checkpoints: section 2.
- **Earlier runs on the new servo** (same arm and servo, different `n_action_steps`, no recordings):
  - board 7074277776: 8 layouts at n100; `v1@015000` 5/8, `v2@20000` 6/8;
  - board 7074960583: 10 layouts, `v2@20000` at n100 / n50 / n25: 4/10, 6/10, 6/10; hidden 2/5, 3/5, 2/5.
  - board 7075616730: 10 layouts at n50 (6 H, 3 P, 1 V), CSV `eval_20261122_072137.csv`:
    - `hidden@002000` 7/9 (L9 not run), `v2@20000` 6/10, paired 7/9 vs 5/9;
    - hidden layouts: 4/5 vs 3/5 on the same five;
    - mean success time 61.1 s vs 34.2 s; 3 of `hidden`'s successes took over 75 s.
- **Analysis:** later, in a separate write-up (link: –).

---

## Appendix A: layout cards

Positions are the plan. The photo is taken at the table.

| ID | Class | Photo | Striped ball position | Covered or touched by | Notes |
| --- | --- | --- | --- | --- | --- |
| L1 | H | L1.jpg | left | | |
| L2 | P | L2.jpg | centre | | |
| L3 | H | L3.jpg | right | | |
| L4 | V | L4.jpg | centre | free | |
| L5 | H | L5.jpg | centre | | |
| L6 | P | L6.jpg | left | | |
| L7 | H | L7.jpg | right | | |
| L8 | H | L8.jpg | left | | |

## Appendix B: run sheet

The script prints the run number and model. Write one line per run.

| Run | Layout | Model | Result ok/fail/abort/invalid | Failure code | Attempts | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | L1 | | | | | |
