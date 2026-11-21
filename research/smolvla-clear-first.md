# Why does the SmolVLA go straight for the striped ball instead of clearing the red/blue balls first?

Investigated 2026-11-21 (evening) by the agent of @TianK003, at the request of @jakobnunnendorf's agent (board 7069526645). Offline only: no arm, no paid runs, CPU inference on Tian's laptop. The checkpoint is `jakobnunnendorf/smolvla_redblue_v2` (board 7068370223), files `checkpoints/002000` and `checkpoints/003000` on the Hub's `main` (they are folders, not revisions).

Scripts (all in `scripts/analysis/`):
- `clear_first_stats.py`: scene mix in episodes and frames; phases and clearing orders from Jakob's step labels.
- `start_frames.py`: the top-camera frame the policy sees at episode start (512x512, 8x8 token grid), plus a yellow-pixel count of the striped ball in it.
- `smolvla_first_chunk.py`: open-loop SmolVLA predictions at three moments of a demo (departure, first target, first grab), compared with the demo and with reference poses.

Inputs: `curation/episodes.csv` (branch `jakobnunnendorf/curation-table`), `labels/jakob.jsonl` (branch `labels`, 36 hand-labelled episodes, `rb_mixed` 0-35, read 2026-11-21 ~22:30), `molnrb/so101_pick_rb_mixed` revision `32f1278` (the 120-episode snapshot the checkpoint was trained on).

## Short answer

**Probably not the architecture and not a lack of clearing data. The clearing motions are learnt; the clear-or-pick choice is the weak part.**
- **Measured:** v2 trained on 52% buried-start episodes and 71% buried-start frames.
- **Measured:** given the demo's own frames, the `002000` checkpoint heads for the covering ball and then the colour hole in 7 of 7 buried episodes.
- **Measured:** at the moment the arm leaves its start pose, it leans toward the covering ball only weakly, with less than half the demo's margin.
- **Measured:** the cue it must base that choice on is tiny. At the policy's input the striped ball is at most ~460 px, about a tenth of one of the 64 visual tokens per camera.
- **Measured:** the demos follow that cue inconsistently. 5 of 32 episodes contradict "ball shows → grab directly, else clear", and partly covered starts went direct in 4 of 12.
- **Measured:** the task sentence is the same for all episodes, so it cannot help.
- **Inferred:** in closed loop, the weak early preference plus ambiguous demos let the arm drift to the striped ball whenever some of it shows.
- **The fix with the best expected effect:** let the supervisor make the clear-or-pick choice from a colour check, and prompt the policy per subtask.

## What the checkpoint was trained on (measured, from its `train_config.json` and `config.json`)

- Dataset `molnrb/so101_pick_rb_mixed`, 120 episodes, `eval_split=0.05`. LeRobot 0.7.0 holds out the last `ceil(120 x 0.05) = 6` episodes per task (`datasets/factory.py:229-253`): **episodes 114-119, all `rb_visible` direct picks. Not a single buried episode is held out**, so the eval loss says nothing about clearing, and every buried episode was trained on.
- One task sentence for all 120 episodes (`meta/tasks.parquet` has one row): *"Pick up the striped ball from the bowl of red and blue balls, uncovering it first if it is buried, and return to the start position"*. 29 tokens with the checkpoint's tokenizer, under `tokenizer_max_length=48`, so not truncated.
- `freeze_vision_encoder=true`, `train_expert_only=true`: the whole SmolVLM (vision encoder, connector, language layers) is frozen; only the action expert and the state projection learn (`smolvlm_with_expert.py:153-160`).
- Images: 640x480 → `resize_with_pad` to 512x384, 128 black rows on top (`vla_utils.py:218-242`). SmolVLM2-500M's SigLIP (512 px, patch 16) gives 32x32 patches, and the connector's pixel shuffle (`scale_factor=4`) turns them into **64 tokens per camera, an 8x8 grid; one token covers 64x64 input pixels (80x80 camera pixels)**.
- `n_obs_steps=1`, `chunk_size=n_action_steps=50`: one frame in, 1.67 s of actions out, no state carried between chunks.
- Schedule set up for 10K steps (`scheduler_decay_steps=10000`, warm-up 500), batch 64. 68,657 training frames, so **`002000` has seen 1.9 epochs and `003000` 2.8 epochs**, with the learning rate still close to its 1e-4 peak (cosine over 10K). The run stopped at about 3.6K.

## Hypotheses

### 1. The data mix leans toward direct picks - NOT SUPPORTED for this checkpoint

Measured (`clear_first_stats.py`):

| | buried-start episodes | share of episodes | share of frames |
| --- | --- | --- | --- |
| All 235 KEEP episodes, three datasets (not what v2 trained on) | 86 of 235 | 37% | 55% |
| **v2 training set, `rb_mixed` 0-113, every verdict** | **59 of 114** | **52%** | **71%** |
| v2 held-out, `rb_mixed` 114-119 | 0 of 6 | 0% | 0% |

- Buried episodes are long (27.7 s mean against 12.2 s for `rb_visible`), and LeRobot samples frames uniformly, so **71% of the training frames come from clear-first episodes**.
- Phases in the 36 labelled episodes (by frame time): 25% waiting at the start pose before the first target, **47% clearing** (first target to the first striped target), 16% striped approach and grasp, the rest returning.
- The `foil_*` datasets are not in v2's training data at all; the 149/86 split in the request is about the whole curation table.
- **Warning for the next run:** `rb_mixed` 120-163 are mostly new direct picks. Adding them unbalanced takes buried starts from 52% to about 37% of episodes (estimate: 59 of ~160).

### 2. One task sentence for both behaviours - SUPPORTED as "language cannot help", not as a cause by itself

- Measured: one sentence for every episode. It *does* mention uncovering ("uncovering it first if it is buried"), but because it never changes, it carries no information; the clear-or-pick choice must come from the image alone.
- Inferred: a per-scene sentence would only help if something at inference time knows the scene and picks the sentence. That moves the decision to a supervisor, which is fix 1 below.

### 3. Visibility at the policy's input resolution - SUPPORTED: the cue is small, and the demos do not follow it consistently

Measured (`start_frames.py`, `rb_mixed` 0-31, frame 0 of the top camera as the policy sees it; yellow pixels with PIL hue 25-45, saturation and value > 90, inside the bowl):

| demo at the start | episodes | striped-ball pixels in the 512x512 input, median (range) |
| --- | --- | --- |
| clears first (Jakob: buried) | 16 | 3.5 (0-65) |
| clears first (Jakob: partly covered) | 6 | 89 (0-214) |
| goes direct (Jakob: partly covered) | 4 | 86.5 (1-408) |
| goes direct (Jakob: visible) | 6 | 329 (128-460) |

- **The whole visible striped ball is at most ~460 px, about 11% of one visual token (4,096 px).** Buried starts show 0-65 px. The cue exists (yellow pixel count separates clear-first from direct with AUC 0.85), but it is a sub-token detail for a frozen encoder.
- **The demos contradict the visual rule in 5 of 32 episodes (16%):** e5 and e9 show 212-214 px and still clear first; e11 shows 111 px and clears first; e19 (1 px) and e31 (10 px) go direct. With a 100 px threshold: 8 of 10 direct picks are above it, and 3 of 22 clear-first episodes.
- Partly covered starts are a coin flip in Jakob's labels: **8 of 12 clear first, 4 of 12 go direct** (`rb_mixed` 0-35).
- At the start pose the wrist camera sees the ceiling (e4 frame 0), so the top camera is the only view of the bowl when the policy chooses.
- Not measured: whether the frozen SigLIP features actually encode "striped ball partly visible" (needs a linear probe on the prefix features).

### 4. No memory or progress state - NOT the main cause (inferred)

- Measured: 12 distinct clearing orders among 26 clear-first episodes (RS 7, BS 5, RRS 3, RBRS 2, RBS 2, and seven singletons). Red is cleared first in 19 of 26. 1-6 balls cleared per episode, mean 2.1.
- Inferred: the order follows the scene (which ball lies on top), so the policy does not need memory to clear; the bowl shows the progress. What it does need after each clear is the same visual decision as at the start ("is the striped ball graspable now?"), which is hypothesis 3 again. SmolVLA's flow matching samples one mode instead of averaging modes the way an L1-trained ACT does, so mode averaging is less likely than a wrong mode choice.

### 5. Checkpoint and held-out split - PARTLY SUPPORTED

- Measured: no buried episode is held out (above), so "held-out loss" cannot rank checkpoints for clearing. `002000` = 1.9 epochs with the LR near peak; the run was planned for 10K steps.
- Measured (open loop, `002000`, training episodes): once the gripper is at the bowl, the checkpoint reproduces the clearing demos (7 of 7 head for the covering ball, 7 of 7 carry it to the hole). It has learnt the clearing *motions*, so "undertrained, has not learnt clearing yet" is not supported. The weak spot is the early, uncommitted part of the reach (margin less than half the demo's).
- Not checked: `003000` (downloaded, not run, timebox), and generalisation to unseen buried scenes (none held out).

### 6. The test scene - UNKNOWN

- Nothing on the board says how buried the striped ball was in Balazs's trials. If any of it showed (more than ~100 px at the policy's input, a ball partly visible to a person), going straight for it is what 8 of 11 such training demos do (the direct ones of the 100 px row above), so it would be the expected behaviour, not a failure.
- To settle it: log the top-camera frame at the start of each arm trial (or run `start_frames.py`'s pixel count on it).

## Open-loop check (measured, `smolvla_first_chunk.py`)

**Method.**
- Checkpoint `002000` on CPU (about 40 s per chunk), one flow-matching sample (seed 0) per query. Inputs: the demo's own top and wrist frames and joint state, `rename_map` top→camera1 and wrist→camera2, and the dataset's task sentence.
- Episodes: `rb_mixed` 0-14, the ones with video in the local cache. All are training episodes (see "What could not be checked").
- Each predicted 50-step chunk is compared with the demo's next 50 actions and with reference poses taken from the demo's joint state: `clear` at the first red/blue grab, `striped` at the striped grab, `hole` at the first drop, `home` at the end. Numbers are the closest approach of the chunk to each pose, as the L2 norm over the five arm joints in degrees.
- Queries:
  - **departure:** the arm has just moved 10 degrees from its start pose; the choice is still open.
  - **first target:** Jakob's `*_target` label; the gripper is already over the chosen ball.
  - **first grab:** the gripper has just closed on the first ball.

**Results, `002000`:**

| episode (start) | first target: chunk → clear / → striped pose (demo) | first grab: chunk → hole pose (start distance; demo) |
| --- | --- | --- |
| e4 (buried) | 11.1 / 17.0 (4.3 / 9.7) | 9.3 (113; 3.0) |
| e6 (buried) | 7.8 / 28.2 (1.0 / 24.6) | 18.4 (79; 3.2) |
| e7 (buried) | 3.4 / 44.1 (0.6 / 52.1) | 8.1 (104; 3.0) |
| e8 (buried) | 12.9 / 65.7 (2.3 / 68.5) | 6.2 (108; 2.7) |
| e12 (buried) | 9.0 / 30.0 (2.3 / 32.6) | 4.6 (77; 3.2) |
| e13 (buried) | 16.8 / 25.4 (0.6 / 48.4) | 14.7 (108; 2.8) |
| e14 (buried) | 20.6 / 56.1 (2.6 / 39.2) | 13.3 (107; 7.3) |

- **At the first target, the chunk heads for the covering ball in 7 of 7 buried episodes**, as the demo does. The covering ball's and the striped ball's grasp poses are 9-68 degrees apart (e4 is the only case closer than 25).
- **After grabbing a covering ball, it goes to the colour hole in 7 of 7** (median 9.3 degrees from the drop pose, starting about 107 degrees away).
- Control, the 4 direct-pick episodes (e1, e2, e3, e10): after the striped grab, the chunk goes home in 4 of 4 (5.8-24.8 degrees from the home pose, starting 186-217 degrees away).
- Mean absolute error against the demo's 50 actions: median 6.7 degrees at the first target and 7.4 at the first grab.

**At departure** (the arm 10 degrees off its start pose, 0.4-2.6 s into the episode; the chunk ends 45-100 degrees short of either grasp pose, because 1.67 s does not reach the bowl). The margin is the chunk's distance to the striped pose minus its distance to the clear pose; positive means it is nearer the covering ball:

| episode (start, striped px at start) | chunk margin | demo margin |
| --- | --- | --- |
| e6 (buried, 65) | +7.9 | +10.1 |
| e7 (buried, 2) | +7.3 | +25.1 |
| e8 (buried, 7) | +29.7 | +69.2 |
| e12 (buried, 30) | +7.2 | +17.0 |
| e13 (buried, 0) | +14.8 | +9.9 |
| e14 (buried, 32) | +0.3 | +15.1 |
| e5 (partly, 214, demo clears) | +0.2 | +16.3 |
| e9 (partly, 212, demo clears) | +10.0 | +25.4 |

- The chunk leans toward the covering ball in 8 of 8 clear-first episodes, but **with less than half the demo's margin (median +7.6 against +16.7 degrees)**. Two are ties: e14, and e5, which is one of the episodes where the demo clears although the ball shows 214 px.
- The predicted chunks also travel further than the demo's: 231-252 degrees of joint change against 211-236.
- Inferred: early in the reach the model hedges between the two balls instead of committing. On the arm, where it sees its own motion and not the demonstrator's, a small lean toward the striped ball can grow chunk by chunk. That fits Balazs's observation, but it is not shown here.

## Fixes, ranked

Effects are inferred, not measured; costs are estimates.

1. **Take the clear-or-pick choice out of the policy: the supervisor decides, the policy executes one subtask.** This is the design in `CLAUDE.md` (PICK and CLEAR as separate skills) applied to red/blue.
   - With the arm parked, a colour check on the top camera says "striped ball visible or not". `decisions/target-visibility-check.md` has one, measured on the foil datasets; its threshold has to be re-measured for the red/blue bowl. The 100 px count used here is a first cut, not a tuned threshold.
   - The supervisor then prompts "clear one red ball" / "clear one blue ball" / "pick up the striped ball", parks after each clear and looks again. The frames can be relabelled per subtask from Jakob's labels or the pan rule in `decisions/multistep-voice-policy.md` (branch `TianK003/multistep-research`). No new recording is needed.
   - Expected effect: the policy no longer has to read a sub-token cue that the demos follow only 84% of the time; the stop-clearing decision becomes a rule we can tune and log.
   - Cost: the same groundwork as the multi-step recommendation: a relabel script, one fine-tune of about 40 min (about 2 USD on Modal, by that file's estimate), the supervisor change, and an offline language-swap check. Risk: the model ignores the sentence. The fallback is one ACT per subtask.
2. **Cheaper variant: two task sentences by scene, chosen by the supervisor at the start.** Give `rb_buried` episodes "uncover the striped ball by putting the red and blue balls into their holes, then pick it up" and `rb_visible` episodes "pick up the visible striped ball". `curation/episodes.csv`'s `scene` column already says which, and one sentence per episode is what `lerobot-edit-dataset modify_tasks` does.
   - Expected effect: it fixes the first choice. After each clear, the policy must still judge from the image when to stop, so it removes only part of the problem.
   - Cost: one re-export and one fine-tune. Note that `eval_split` then holds out the last 5% *per sentence*, which finally puts buried episodes in the held-out set.
3. **Make the demos agree with one visual rule.** Drop or re-record the contradicting episodes (in 0-31: e5, e9, e11 clear first although the ball shows 111-214 px; e19 and e31 go direct with 1-10 px). From now on, record by a stated rule, e.g. "grab directly only when the striped ball shows and nothing lies on it". Cost: curation time only. Effect alone: small; it also helps fixes 1 and 2.
4. **Keep the buried share and hold out buried episodes in the next run.** Adding `rb_mixed` 120-163 unbalanced drops buried starts from 52% to about 37% of episodes. Choose the held-out episodes explicitly (`--dataset.episodes`) so that some buried ones are in it; then checkpoints can be ranked on clearing. Cost: none.
5. **Not recommended now:**
   - A progress cue. It needs a model input that SmolVLA does not have (code change plus retraining), and the bowl already shows progress.
   - Simply training longer. See the open-loop result: the checkpoint already reproduces the clearing demos on training frames.

## What could not be checked

- **No held-out buried episode exists** for v2, so every open-loop case is on training data. A good open-loop result shows the model learnt the demos, not that it generalises.
- Open-loop is not closed-loop: the arm on the robot sees its own (imperfect) motion, not the demonstrator's.
- The labels cover `rb_mixed` 0-35 only; Sonnet pre-labels for the rest were not used.
- Only `rb_mixed` 0-31 had top-camera video locally (two ~200 MB files); the pixel counts and contact sheets cover those.
- The arm trials themselves (scene, checkpoint, settings) are not recorded anywhere I could read.
- Time and the CPU (about 40 s per chunk):
  - `003000` was not run.
  - Each query used one noise sample.
  - The partly covered clear-first episodes e0 and e11 were not run at all; e5 and e9 were run only at departure.
- Whether the frozen SigLIP features encode partial visibility (a linear probe) was not tested.
- The pose metric is a closest-approach distance in joint space, not a check of which ball the gripper would actually close on.

Reproduce (local paths in angle brackets):

```
git show origin/jakobnunnendorf/curation-table:curation/episodes.csv > episodes.csv
git show origin/labels:labels/jakob.jsonl > jakob.jsonl
python scripts/analysis/clear_first_stats.py episodes.csv jakob.jsonl
python scripts/analysis/start_frames.py --dataset-root <rb_mixed snapshot 32f1278> --labels jakob.jsonl --episodes 0 1 ... 31 --out start_frames.png
python scripts/analysis/smolvla_first_chunk.py --checkpoint <checkpoints/002000> --dataset-root <rb_mixed snapshot> --labels jakob.jsonl --episodes 4 6 7 8 12 13 14 1 2 3 10 --seeds 1 --queries first_target first_grab --out first_chunk.csv
```

Needs `lerobot[smolvla]==0.7.0` (`transformers` 5.22, `num2words`). Run the scripts from a directory outside the repo, with `tests/` off `sys.path`.
