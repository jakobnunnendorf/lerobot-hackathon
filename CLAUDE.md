# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team project from the [Bilbao Physical AI Hackathon](https://hackathonbilbao.eus/home-page-eus) (2026-11-20 to 2026-11-22): five people, one LeRobot SO-101 arm (leader + follower), a top and a wrist camera. The hackathon is over; the repo is kept as a clean, working record of the demo. `README.md` says what the robot does; `docs/HISTORY.md` says how the project got there, with every model and dataset.

**The task:** a bowl holds one target ball (yellow and green stripes) among red and blue balls. The arm picks the target wherever it lies, digging it out when it is buried, checks it in the wrist camera, and drops it into an open palm it tracks with the top camera. Older datasets use grey tinfoil balls instead of red and blue; a foil ball is neither red nor blue.

## How the demo is structured

- **One robot program owns the arm.** `scripts/supervisor/run_supervisor.py` (started by `scripts/supervisor/demo_chain.sh`) connects the follower and both cameras once, loads every policy up front and runs a plain 30 Hz loop (`get_observation`, `predict_action`, `send_action`). `lerobot-rollout` and `RolloutController` own one policy and one connection and cannot switch policies, and `lerobot-replay` opens its own connection, so replays (dance) run in the same loop. See `decisions/supervisor.md`.
- **A supervisor decides, skills act.** A state machine runs a round: REST → PICK (learned ACT, with a grasp check on the gripper at each close) → VERIFY_HOLD (gripper reading and a wrong-ball check in the wrist camera) → GIVE (rule-based hand following, `scripts/handover/`) or PUT_DOWN → REST. LOOK and CLEAR_GRAY (a top-camera visibility check before PICK) still exist in `machine.py` but are off in the demo chain. DANCE, BITE and STEP (spoken multi-step plans with per-colour clearing policies) are extra round types.
- **A policy never reports that it is done.** Each skill has a time budget, then the supervisor checks the result (gripper reading, camera).
- **The pick policy is ACT.** pi0.5 ran out of memory on the 8 GB laptop GPU; a SmolVLA fine-tune picked 0 of 4 on the arm. The current model and every alternative are in `docs/HISTORY.md` and `decisions/hidden-ball-picking.md`. `n_action_steps` (how many of the 100 predicted actions run before replanning) is a run-time flag and matters more than retraining did.
- **Other processes never talk to the motors.** The hand tracker (`scripts/handtrack/`, MediaPipe, its own environment because its OpenCV clashes with LeRobot's) and the dashboard (`scripts/dashboard/`) read camera frames from shared memory and talk to the robot program over local sockets: commands TCP 8765, hand positions TCP 8766, status UDP 8767, hand overlay UDP 8771. Nothing they do may slow the 30 Hz loop.

## Working in this repo

- **Never commit to `main`.** Work on a branch named `<github-login>/<topic>` from an up-to-date `main`, open a pull request, and ask the person you work for before merging, unless they said at the start of the task that you may merge it. Never force-push `main`.
- Before merging a change to code or dependencies, check that `uv sync` and the smoke scripts still pass.
- **Never run a command that moves the robot** (`lerobot-teleoperate`, `lerobot-record`, `lerobot-replay`, `lerobot-rollout`, calibration, `demo_chain.sh`, `eval_pick.sh`). Prepare the command and hand it to the person at the table. The same goes for deleting or overwriting datasets, episodes or checkpoints, changing the `lerobot` or `torch` pins, and starting paid cloud jobs.
- Report robot results as raw counts ("7/10"), never as "works" or "mostly works".

## Commands

```bash
uv sync                                  # create .venv from uv.lock
uv run python tests/cuda.py              # torch build, CUDA availability, GPU matmul check
uv run python tests/lerobot_check.py     # builds an ACT policy, one train step + one inference step, prints PASS
```

There is no pytest suite, linter or build step. The files in `tests/` are standalone smoke and scenario scripts; run the one you need with `uv run python tests/<file>.py` (for example `tests/supervisor_scenarios.py` for the state machine, `tests/dashboard_smoke.py` for the dashboard).

**Any work on the striped-ball data** (labelling, finding events, tracing a robot failure to the data, building a training set, splits and evals, merging datasets): load the `so101-data` skill (`.claude/skills/so101-data/SKILL.md`) and read the section for your task. Take the data distribution into account before every training run. The full reference is `research/DATASET_GUIDE.md`.

**Hands-on robot work** (teleop, recording, checking or uploading a dataset, training, running a model on the arm, fixing the arm, cameras or USB): load the `so101-robot` skill (`.claude/skills/so101-robot/SKILL.md`) first and use `scripts/so101/`. Runbooks for people are in `docs/` (`docs/README.md` lists them).

## Things that are easy to get wrong

- **The follower gripper's torque is capped at 200 of 1000. Never raise it above 200** (`Torque_Limit` or `Max_Torque_Limit`, in code, flags or by hand). LeRobot's default 500 stripped the gripper servo's gears on 2026-11-22. Every program that connects the follower calls `scripts/so101/gripper_limit.py` right after `connect()` to write the live `Torque_Limit`; do the same in any new one. LeRobot writes `Max_Torque_Limit` 500 (EEPROM) on every connect and the servo reloads `Torque_Limit` from it at power-up, hence the write after every connect. Details: `kb/gripper-torque-limit.md`.
- **Never reset, disconnect or reconnect the robot while it holds the ball.** `RolloutController.reset()` moves all six joints including the gripper back to the pose at connect time, `disconnect()` switches torque off by default, and `connect()` briefly switches it off as well. Switching skills inside one loop sends nothing, so the arm holds.
- **LeRobot is a pinned dependency (0.7.0), not vendored.** Its source sits in `.venv/Lib/site-packages/lerobot` (Windows) or `.venv/lib/python3.12/site-packages/lerobot`. Config and flag names change between releases, so check them there or in the version-matched docs (https://huggingface.co/docs/lerobot/v0.7.0/en/index), not from memory of another version. In 0.7.0, `lerobot-record` adds a timestamp to the dataset name unless `--dataset.no_stamp=true`, and experiment tracking flags are `--trackio.*` (`research/lerobot-training-feasibility.md`).
- **ACT ignores the task sentence.** `set_task`, `/subtask` and the planner only affect language-conditioned policies (SmolVLA, pi0.5 and similar).
- **Fine-tune, do not resume, when the data changes** (`--policy.path`, `research/continue-training-from-checkpoint.md`). Held-out loss has not predicted arm behaviour for our policies; choose checkpoints on the arm (`kb/act-model-facts.md`).
- **On Windows, LeRobot turns a Hub policy id into `user\repo` and fails**, and checkpoint `last` symlinks need a privilege Windows does not give. Pass a local snapshot path to `--policy.path` there; both work on Linux and WSL. The robot laptop drives the arm from WSL; Windows and WSL keep separate calibration files and dataset caches.
- **torch comes from the CUDA 12.8 wheel index for every platform** (`[tool.uv.sources]` in `pyproject.toml`). `uv.lock` therefore has Linux and Windows torch wheels and no macOS one.
- `pyproject.toml` and `uv.lock` are committed together whenever dependencies change (`uv add <pkg>`). The dashboard, hand tracker, voice, calibration, curation and recolouring tools have their own `requirements.txt` and environments.
- `outputs/`, `data/`, `wandb/` and `*.safetensors` are git-ignored. Datasets and checkpoints go to the Hugging Face Hub, never into git.

## Decisions get documented, code does not

The repo records *why* things are the way they are, so nobody explores the same search space twice. It does not describe *what* the code does: the code says that itself.

- **Read `decisions/` before you explore an approach.** If the question already has a record, start from it. Do not retry an alternative listed there unless its "Revisit if" condition holds, and if you do retry it, add the result to that same file.
- **Write a decision record whenever you choose between alternatives or drop something you tried.** Use `decisions/<question>.md`, one file per question; `decisions/README.md` has the format. Failed attempts are the most valuable part. Put the record in the same pull request as the code or config it explains. A choice with no real alternative needs no record.
- **No documentation layer about the code.** No architecture walkthroughs, module overviews, API docs or READMEs that retell what a script does. Runbooks for the people at the table (which command to run, what to check) belong in `docs/`.
- **Code is clean, concise and self-explanatory.** Use clear names and small functions, not comments.
- **The only comments are about decisions**: why it is done this way, which alternative was tried or rejected, or a constraint the code cannot show (`# no reset() here: it moves the gripper back to its connect pose and drops the ball`). No comments that restate the code, no section banners, no docstrings that repeat the signature, no commented-out code, no change notes. If the reasoning takes more than two lines, write a decision record and point to it (`# see decisions/skill-policies.md`). When you change code, delete or fix comments that no longer hold.

## Where things go

- `scripts/` - our code (robot program, dashboard, perception, data tools, training).
- `tests/` - smoke and scenario scripts (see above).
- `docs/` - runbooks for people (`docs/README.md` is the index) and `docs/HISTORY.md`.
- `decisions/` - one file per question: what was chosen, why, and what else was considered or tried, with results.
- `kb/` - verified facts about our setup (hardware, calibration, model facts, settings that worked or failed). One topic per file. Facts only; choices go in `decisions/` and link to the facts.
- `research/` - external research with source links, and offline analyses of our own data.
- `images/` - pictures used by `README.md`.
- `logs/training/` - loss logs and plots of the main training runs.
- Only `README.md`, `LICENSE` (CC BY-NC 4.0: non-commercial use with attribution) and this file stay at the repo root.
