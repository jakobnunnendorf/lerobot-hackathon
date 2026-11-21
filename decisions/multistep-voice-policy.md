# How does the robot do a spoken multi-step task ("3 blue, 2 red, then the target ball")?

- Status: OPEN (2026-11-21, agent of TianK003, research only; Tomi's request on board 7069530124). Nothing trained or run on the arm. The first answer from Tian or Tomi picks the option.

## Recommendation

**(a): a rule-based planner plus one SmolVLA, fine-tuned once with three subtask sentences and prompted per step.** Gate it with an offline language-swap test before it goes on the arm. If the test shows the model ignores the colour word, switch to **(b)** (one ACT per colour), which reuses all the groundwork. The last step (the striped ball) runs the demo's current best PICK and the existing VERIFY_HOLD -> GIVE / PUT_DOWN chain in both cases.

## Why

- **The data already holds the subtasks, and joint data alone can cut them.** In `molnrb/so101_pick_rb_mixed` (201 episodes, 131,706 frames at 30 fps, one task sentence), Jakob's labels put every drop at a fixed shoulder pan: red hole at -35.4 to -28.7 deg (34 drops), blue hole at +22.3 to +28.2 deg (20 drops), home at -13 to +9 deg. A drop is "gripper was holding (< 16.5), then opens (>= 18) with pan < -25 (red) or > 18 (blue)". That rule gives the same colour sequence as the labels on 34 of 36 labelled episodes (the other two each miss one drop). Over all 201 episodes it finds **144 red and 117 blue drops in 109 episodes**. Counted by this agent from the Hub parquet and `labels/jakob.jsonl` at `e9fee74` (36 labelled episodes, 0-35). DROP rows of `curation/episodes.csv` still need removing.
- **The subtask segments are short but there are many of them.** Labelled segments (previous event to drop): red median 7.0 s (n=34), blue 7.6 s (n=20), striped grab to home 8.6 s (n=36). With the joint-detected counts, that is roughly 17 min of red and 14 min of blue demonstrations (estimate). That is about 2-3 times the usual 50-demo rule per subtask.
- **Language is the only thing that can pick the colour.** The scene looks the same whichever colour is asked for, so per-step prompting needs a language-conditioned policy (a). With ACT, which ignores the task sentence, it needs one policy per colour (b). That reopens the "Revisit if: a skill has to be told what to do" condition of `decisions/skill-policies.md`.
- **(a) is the least new machinery.**
  - SmolVLA reads the task per frame: `item["task"]` comes from each frame's `task_index` (`datasets/dataset_reader.py:529`, LeRobot 0.7.0). Relabelling frames, not cutting episodes, is enough, and no video is touched.
  - `lerobot-edit-dataset modify_tasks` only sets one task per episode (`dataset_tools.py:1576`), so the relabelling is our own script. It rewrites `task_index` in `data/*.parquet` plus `meta/tasks.parquet` and the episodes' `tasks` column, on Modal's volume.
  - The supervisor's `PolicySkill` already passes a `task` string to `predict_action` (`scripts/supervisor/skills.py` on `TianK003/demo-chain`).
  - SmolVLA has already run in the supervisor on the robot laptop (VLA test, `molnrb/arm-logs` 6d68b23).

## Options

| | (a) planner + SmolVLA per step | (b) planner + ACT per colour | (c) pi0.5 / pi0-FAST | (d) end to end |
| --- | --- | --- | --- | --- |
| Data | `rb_mixed` frames relabelled to 3 sentences (red ~144, blue ~117 segments, striped every episode). No new recording | Same segments, but ACT trains on whole episodes, so each segment must become its own episode (new derived dataset per colour) | As (a) | Episodes that follow a spoken count: **0 exist**. Data orders follow the burial, not an instruction; the longest same-colour run is 3 drops, in 1 of 36 labelled episodes |
| Fine-tune from | `jakobnunnendorf/smolvla_redblue_v2` (rev `002000`) or Kamel's `smolvla_so101_pick_rb_mixed_from_blue_10k` `002000`; about 3K steps (Kamel's from-blue run overfit after about 3K) | `Space0code/act_redblue_from_foil` (10 of 13 on the arm, `decisions/act-redblue-init.md` on `Space0code/modal-act-from-foil`), 5-10K steps per colour | `lerobot/pi05_base` | - |
| Modal cost (H100 3.95 USD/h) | One run, about 2 USD (Kamel: SmolVLA 5.1K steps at batch 64 cost about 2.5 USD), about 40 min with start-up. Cap: 5 USD | 2 runs x 10K steps x 0.117 s/step = 2 x 20 min, about 1.5-2 USD each, plus the dataset build on CPU. Cap: 8 USD | LeRobot's pi0.5 reference: 5K steps at batch 32 took 3 h 41 min on 4x H100, about 60 USD at our rate (estimate). Needs 24-40 GB to train | - |
| Inference on the laptop | Measured on the robot laptop (`20k_layout1/ticks.jsonl`, `n_action_steps` 20): median tick 33 ms, but one 0.32-0.39 s stall about every 19 ticks while the chunk is predicted, so about **18 Hz on average**. Peak GPU memory not logged | ACT: about 1 GB in `tests/lerobot_check.py`; inference is negligible, so 30 Hz. Two or three ACTs together not measured | **Does not fit.** pi0.5 needed about 14 GB to build at fp32 on Kamel's 8 GB card (`kb/machines.md`); bf16 is about 7 GB of weights alone. The openpi README says inference needs "> 8 GB". The robot laptop's RTX 4050 has 6 GB (per `scripts/so101/train.sh` and NVIDIA's spec, not checked) | - |
| Code to change | Planner, voice field, drop counter, per-step `task` (below) | Same, plus a segment-dataset builder | Same as (a) | - |
| Main risk | It may ignore the colour word in a fixed scene. LangGap (https://arxiv.org/abs/2603.00592) and LIBERO-Plus (https://arxiv.org/abs/2510.13626) report VLAs largely ignoring the instruction. Its grasps were weak on the arm (VLA test: picks timed out at 40 s, a slip onto a blue ball) | The derived per-segment dataset is untested (below). Each ACT sees only one colour's segments | Cost, memory, and LeRobot 0.7.0's pi05 / pi0_fast do **not** generate subtask text (`supports_text_generation` exists only in eo1, fineart_vla, g05, wall_oss_05, wall_x), so it is only (a) with a bigger model | Cannot count: one frame in, no memory of drops done |
| Ready by Sunday 11:00 | Likely, if Modal is set up tonight: groundwork about 2-3 h of agent work, training about 40 min, offline check about 30 min, then arm rounds in the morning | Possible, but the dataset builder adds 1-2 h and one more thing that can fail | No | No |

## The parts every option shares

- **Planner: rules, no LLM.** Same reasoning as `decisions/voice-commands.md`. The parser turns "pick up 3 blue balls and 2 red balls, then the target ball" into `[(blue, 3), (red, 2), (striped, 1)]`. Number words and digits both count, and an unknown colour or a missing count rejects the command. Parakeet's spelling of numbers is **not measured**; add sentences to `tests/voice_parse.py`.
- **Voice contract: a new message type, old messages unchanged.**
  - The message: `{"plan": [{"target": "blue", "count": 3}, {"target": "red", "count": 2}, {"target": "striped", "count": 1}], "destination": "hand" | "container" | null}`.
  - `link.decode` accepts it as well as `{target, destination}` and `{abort}`.
  - Keyboard: one line such as `p 3b 2r s` in `Commands.listen_keyboard`.
- **"Done" comes from the drop counter.** A policy never reports done (`CLAUDE.md`). A red/blue step ends when that colour's count is reached, with the colour taken from the pan at the moment of the drop (the rule above, measured on teleoperated data, **not on policy runs**).
  - A wrong-colour pick still lands in its own colour's hole (in the SmolVLA pilot it put a mistakenly picked red ball into the red hole, board 7067020779). It is counted for that colour and logged.
  - Each subtask gets a time budget (30 s suggested); after a timeout with nothing held, re-prompt, at most 3 times.
  - The striped step keeps the existing PICK -> VERIFY_HOLD (load >= 200, held-ball area) -> GIVE / PUT_DOWN.
- **Supervisor:** a STEP state between REST and PICK that runs the active subtask's skill until its count is reached. For (a), that is `PolicySkill` with the step's sentence. For (b), it is the colour's ACT. Scenario tests with fakes as for the rest of `machine.py`.
- **Offline test on held-out episodes** (exclude about 10 episodes with labels via `--dataset.episodes`; `eval_split` groups held-out episodes by each episode's first task, `datasets/factory.py:243-253`, which mixes up once frames carry several sentences).
  - L1 per sentence on the held-out segments.
  - **Language swap (gate for (a)):** on held-out frames right after a drop, predict the chunk with the red and with the blue sentence. If the two predicted pan trajectories barely differ, the model ignores the colour word: go to (b).
  - Run it on Modal in the same job; Tian's laptop is short on memory.

## Unverified

- Total segment counts come from the joint rule, checked against 36 hand-labelled episodes only. Jakob's pre-labels for the rest (on his machine) were not used.
- That a mostly-direct-pick SmolVLA learns three sentences in about 3K steps, and that it follows the colour word.
- For (b): a derived dataset whose episodes are segments, done by copying data rows and keeping each frame's original `timestamp` and the source episode's `videos/*/from_timestamp`. Video frames are looked up at `from_timestamp + timestamp` (`dataset_reader.py:437`), so no re-encode should be needed. Not run. The fallback is a re-encode, which is slower.
- GPU memory of SmolVLA plus the PICK ACT loaded together on the robot laptop.
- Whether the 18 Hz of SmolVLA's synchronous inference hurts a 7 s clearing move more than a striped pick.
- Every cost is an estimate from Kamel's measured runs (`decisions/act-redblue-generalisation.md` on `BrouthenKamel/modal-smolvla`). Paid runs are started by Tian only.

## Alternatives

- **(b) one ACT per colour**: considered, not tried. It is the fallback if the language-swap gate fails. ACT has the better arm record so far (from foil 10 of 13), but it needs the untested segment dataset and two runs.
- **(c) pi0.5 / pi0-FAST**: considered, not tried. It does not fit the laptops (above). LeRobot has no subtask prediction for it.
- **(d) one policy on the whole sentence**: considered, not tried. There are no demonstrations of it, and no memory of progress.
- **Retrain the current single-sentence SmolVLA on more data**: does not answer the question. One sentence cannot select a colour.

## Revisit if

- The language-swap gate fails: do (b).
- A laptop with clearly more than 8 GB of GPU memory joins the demo: (c) becomes possible, though still costly.
- Arm rounds show the drop counter miscounting (for example, the policy releasing over the bowl): add the top camera's colour counts as a check.
