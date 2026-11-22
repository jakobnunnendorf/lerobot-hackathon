# Benchmark: <name, e.g. "Pick policies, new servo, 22 Nov">

> **How to use this template**
> - This template is for **collecting data**. Analysis and decisions come later, in a separate write-up, once the data is in.
> - Copy it to `research/benchmarks/<YYYY-MM-DD>-<short-name>.md` and fill in every `<...>`.
> - Sections 1–9 describe **how** we test. Fill them in and post the link on the board before the first run. After the first run, don't change them; write any change in section 11 (Deviations), so runs stay comparable.
> - Sections 10–13 are filled in during and after the runs.
> - Write "n/a" with a reason instead of deleting a section, so a reader can see it was considered.
> - Related: `docs/eval-pick-runbook.md` (how to run `eval_pick.sh`), `docs/eval-counting-sheet.md` (failure modes), `.claude/skills/so101-data/SKILL.md` §6 (evaluation) and §9 (nearest demo).

| | |
| --- | --- |
| Status | draft / ready (protocol final) / running / data complete |
| Version | <v1, v2, ...>, protocol final at <time> |
| Board posts | protocol <comment id>, data <comment id> |
| Code | `eval_pick.py` at commit <sha>, PR <#> |

---

## 1. What we measure

- **In one sentence:** <e.g. "How often, how fast and in what way each pick model succeeds or fails on the new servo, on the same layouts">.
- **What the data must allow later** (so we record the right things now; no analysis is fixed here):
  - success per model and per start class (V / P / H);
  - a layout-by-layout comparison between models;
  - time to success, and stutter;
  - how each failure happened (code + recording);
  - each run's similarity to the training demos.

## 2. Candidates

| Short name | Hub repo @ step | Type | Trained by | Training data (dataset, episodes, curation) | Init from | Steps | Chunk / `n_action_steps` at test | Task sentence (VLA only) | Why it is in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <v2> | <Space0code/act_redblue_from_foil_v2@020000> | ACT | <Tomi> | <...> | <foil model> | <20K> | <100 / 30> | – | reference (demo model) |
| <...> | <...> | <ACT / SmolVLA / ...> | <...> | <...> | <...> | <...> | <...> | <exact sentence> | <...> |

- **Checked before the runs:**
  - [ ] each model loads on the robot laptop;
  - [ ] each model gets through one offline `select_action` call;
  - [ ] each VLA has its camera rename and task sentence.
- **Left out, and why:** <e.g. "act_pick_rb_mixed: not public on the Hub">.
- **Cut order if time runs short** (cut the first one first): <...>.

## 3. Fixed conditions (the same for every run)

Write the value. If it changes during the session, log it in section 11.

| Condition | Value |
| --- | --- |
| Arm, gripper servo, torque limit | <SO-101, new servo, torque 200> |
| Robot laptop, GPU, lerobot version | <..., RTX 4050, 0.7.0> |
| Control rate, run time cap | <30 Hz, 90 s> |
| Start pose / home | <HOME as in recordings> |
| Top camera: position, exposure; wrist camera | <not moved since recording; photo link> |
| Lighting | <room lights on, blinds ..., photo link> |
| Table, bowl position, hole positions | <tape marks; red hole pan −35..−29°, blue +22..+28°> |
| Balls: number of red / blue / striped | <...> |
| Display / Rerun | <off> |
| Script and flags | <`eval_pick.sh --runs ... --seconds 90 --n-action-steps 30 --record ...`> |

## 4. Test layouts

- **Classes**, as defined in the so101-data skill:
  - **V** visible: striped ball fully seen from the top camera;
  - **P** partly hidden;
  - **H** hidden: covered, has to be dug out.
- **Layout list and order:** <e.g. 8 layouts: H P H V H P H H>.
- **Why this mix:** <e.g. "hidden is where the demo will be judged"; training share for comparison, e.g. 26% buried in the clean subset>.
- **How a layout is built:** <e.g. drop the balls from the same height, then place the striped ball by the layout card>.
- **How we keep each layout the same for every model:** <layout photo before each model; reset the same layout after each run; photo check>.
- **Layout cards** (one per layout, in the appendix): ID, class, photo, where the striped ball is, which balls touch or cover it.

## 5. Protocol

**Design:**
- every model runs on every layout (a paired design);
- <runs> runs per model per layout.

**Order:**
- <the model order rotates per layout (the script does this), so no model always goes first after a reset>;
- <layouts run in the listed order>.

**One run, step by step:**
1. Build or reset the layout. Check it against the photo.
2. The arm is at HOME and the gripper is open.
3. SPACE starts the run. The run ends at success, at the time cap or on abort.
4. Enter the result and the failure code (section 6).
5. The arm parks. While it is parked, the recording is saved.

**Blinding:** <e.g. "the person at the table sees only the run number, not the model name" — or n/a, with the reason>.

**Interventions:**
- touching the arm or the balls during a run = **abort**, not a failure;
- write down why;
- <rerun once, at the end of the session>.

**Invalid runs:**
- these are rerun but still logged, marked invalid: <e.g. a camera timeout, the script crashes, the layout was built wrong, someone bumps the table>;
- keep their recordings too.

**Warm-up:** <e.g. one throwaway run after the laptop starts, logged as warm-up>.

**Breaks and checks:**
- <every N runs: servo temperature, gripper reading at rest, camera photo>;
- <recalibrate if the gripper reads more than X at rest>.

**Stop a model early only for safety or a broken setup** (e.g. repeated hard collisions, never starts at all). Write down why.

## 6. Outcomes and failure codes

- **Success:** <the striped ball is held at the start pose before the cap; gripper reading about 6–20>.
- **Partial** (recorded as a note on a failure): <e.g. grasped the striped ball but dropped it on the way back>.
- **One failure code per failed run.** Pick the first thing that went wrong:

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
| `stutter` | long pauses in the motion |
| `collision` | hit the bowl, the table or itself hard enough to stop |
| `timeout_other` | anything else; describe it in `note` |

- **Second check:** the code given at the table is checked later against the recording, and both codes are kept.

## 7. Data collected

| What | How | Where |
| --- | --- | --- |
| One CSV row per run: run, layout, policy, checkpoint, n_action_steps, cap, seconds, end, gripper_end, late_ticks_pct, start_state, result, episode_index | automatic (`eval_pick.py`) | `<log dir>/runs.csv` |
| Full run as a LeRobot dataset episode (both cameras, state, action, 30 fps) | automatic (`--record`) | `<local dir>` → `<hub repo, e.g. molnrb/so101_eval_2211>` |
| Run metadata: git sha, policy type, task sentence | automatic | `<record dir>/runs.jsonl` |
| Failure code and note | by hand | `<log dir>/fails.csv` |
| Layout photo before each model | by hand, phone | `<folder / board>` |
| Condition checks (temperature, light, camera photo) | by hand | section 11 |
| Anything surprising | by hand, free text | section 12 |

- **Never lose data:**
  - <upload after each model block, not only at the end>;
  - <keep the local copy until the upload has been checked>.
- **Record every run, including aborts, invalid runs and warm-ups.** Mark them; don't drop them. Whether to leave them out is a later analysis choice.

## 8. How much data, and what it can show

- **Runs per model:** <e.g. 8 layouts × 1 run = 8>.
- **For orientation:**
  - with 8 runs, a model that succeeds 4 times has a 95% interval of about 22–78%;
  - more layouts per class make the later picture sharper.
- **Where to spend extra runs if time allows:** <e.g. more hidden layouts; a second run per layout for the two closest models>.

## 9. Time budget, roles and safety

- **Time per run:** <about 90 s cap + save time + 60 s reset ≈ ... min>.
- **Total:** <models × layouts × runs × time per run + setup = ...>.
- **Time available:** <...>.
- **Schedule:** <start, model blocks, upload points, the cut-off time>.
- **Safety:**
  - only the person at the table starts a run;
  - agents never start one;
  - the hand stays near the stop key or the power;
  - <torque limit 200>;
  - stop at the first strange sound or a hard collision.

---

## 10. Raw tallies (during the runs, no interpretation)

**Per layout** (✓ / failure code / abort / invalid):

| Layout | Class | <model 1> | <model 2> | <...> |
| --- | --- | --- | --- | --- |
| L1 | H | | | |

**Counts per model:**

| Model | V (succ/n) | P | H | Total | Aborts | Invalid |
| --- | --- | --- | --- | --- | --- | --- |
| <...> | | | | | | |

## 11. Deviations from the protocol

| Time | What changed | Why |
| --- | --- | --- |
| | | |

## 12. Notes from the table

Free text: anything seen at the table that the numbers don't show (e.g. "v3 always approaches from the left", "blue ball keeps rolling into the striped one").

## 13. Where the data is, and how to reproduce it

- **Data:**
  - recording dataset: <hub repo, revision>;
  - CSV: <path>;
  - `fails.csv`: <path>;
  - photos: <link>.
- **Code:** <commit>, command line(s): <...>.
- **Checkpoints:** <repo@step list, with the Hub revision sha>.
- **Analysis:** done later in <link to the analysis write-up>.

---

## Appendix A: layout cards

| ID | Class | Photo | Striped ball position | Covered or touched by | Notes |
| --- | --- | --- | --- | --- | --- |
| L1 | | | | | |

## Appendix B: run sheet

One line per run. A printable copy goes next to the arm.

| Run | Layout | Model (or blind code) | Result ✓/✗/abort/invalid | Failure code | Seconds | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | | | | | | |
