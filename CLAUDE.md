# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team project for the [Bilbao Physical AI Hackathon](https://hackathonbilbao.eus/home-page-eus): 5 people, about 2 days (started 2026-11-20, deadline Sunday 2026-11-22 11:00), one LeRobot SO-101 arm (leader + follower) and two cameras. Optimise for a working robot demo by the deadline, not for clean abstractions.

**The idea** (chosen by the team on 2026-11-20; the earlier bomb-defusal and cable-cutting ideas are dropped): a bowl holds one target ball (yellow and green stripes) among gray tinfoil balls. If the target ball can be seen, the arm picks it up wherever it lies. If it cannot be seen, the arm picks one gray ball, drops it into the gray hole and looks again. Holding the target ball, it puts it into a human hand and shakes hands if a hand is detected, and puts it down otherwise. Full description, build plan and open questions: `ideas/ball-picking-and-handover.md`.

## How the demo is structured

Read `ideas/ball-picking-and-handover.md` before writing demo code. The parts that span several files:

- **A supervisor decides, skills act.** A decision tree over the two camera feeds picks the next skill: target ball visible well enough to grasp? PICK, otherwise CLEAR GRAY (one gray ball into the gray hole) and look again; holding the ball and a hand is seen? GIVE then SHAKE, otherwise PUT DOWN. It decides only while the arm is parked, because the arm hides the bowl from the camera while it works.
- **Skills are separate and short.** PICK and CLEAR GRAY are learned policies. GIVE, PUT DOWN and SHAKE start as replays of one recorded episode each; GIVE becomes a learned policy if time allows. One dataset per skill, each with its own task sentence, all with the same camera names, resolution and `--robot.id`.
- **Two anchor poses make skills chain.** HOME (above the bowl, gripper open) and CARRY (above the bowl, gripper closed on the target ball). Every demonstration and every replay starts and ends in one of them. CLEAR GRAY goes from HOME to HOME. GIVE and PUT DOWN demonstrations start with the ball already held.
- **A policy never reports that it is done.** Each skill has a time budget, then the supervisor checks the result (gripper reading, camera).
- **The learned skills are ACT policies, one per skill.** Training runs in the cloud; inference has to run on a team laptop, and the only GPU laptop has 8 GB. pi0.5 was tested there on 2026-11-20 and ran out of memory while loading. See `kb/machines.md`.

## Coordination (read before doing any work)

Five people work here in parallel, each with their own agent. The only coordination channel is **GitHub issue #1, the message board**.

- Before starting: `gh issue view 1 --comments` and read the latest comments, so you do not duplicate or collide with someone else's work.
- Post a comment (`gh issue comment 1 --body-file <file>`) when you claim a task, finish one, get blocked, or hand work over. Start it with who you work for and the kind of update, e.g. `**[agent of @TianK003] CLAIM**`, `DONE`, `BLOCKED`, `HANDOFF`, `QUESTION`. Mention a person with `@login` only when they need to act.
- Do not open more issues for coordination; everything goes on #1.
- To be told when someone posts, run `bash scripts/watch_board.sh <state file outside the repo> 43200 60` as a background command. It polls once a minute, skips your own agent's comments, and exits with the new comments as soon as someone else posts; then start it again. Use the long window: a short one makes the script return every few minutes and wakes the session for nothing. The `board-watcher` agent (Haiku, `.claude/agents/board-watcher.md`) wraps the same script and adds a summary. Neither posts anything.
- **Agents never commit to `main`. Work on a branch and open a pull request.**
  - Start every piece of work, docs included, on its own branch named `<github-login>/<topic>`, created from an up-to-date `main`.
  - Commit regularly: whenever something works and before you switch tasks. Push the branch at least once an hour so work never exists only on one laptop.
  - When the work is ready, open a pull request into `main` (`gh pr create`) and post its link on the board. Keep pull requests small: one topic each, opened the same day the work started.
  - **Ask the person you work for before merging.** The one exception: if they told you at the start of the task to merge it yourself, merge without asking again. That permission covers that task only, not the next one.
  - After the feature freeze (Saturday 21:00, see `docs/roadmap.md`) `main` is the demo branch. From then on every merge is asked, whatever was said earlier.
  - Before merging, bring the branch up to date with `main` and check that `uv sync` and the two smoke scripts still pass if code or dependencies changed. Delete the branch after the merge.
  - A change that others depend on (dependencies, shared scripts, anything the demo runs) is also announced on the board before it is merged.
  - Never force-push `main`.
- `docs/roadmap.md` defines the stop points and the human check-ins. At a check-in gate, stop and ask the human instead of continuing on your own.
- **Never run a command that moves the robot** (`lerobot-teleoperate`, `lerobot-record`, `lerobot-replay`, `lerobot-rollout`, calibration). Prepare the command and hand it to the person at the table. The same goes for deleting datasets or checkpoints and for starting paid cloud jobs. The full list is in `docs/roadmap.md`.

## Commands

```bash
uv sync                                  # create .venv from uv.lock
uv run python tests/cuda.py              # torch build, CUDA availability, GPU matmul check
uv run python tests/lerobot_check.py     # builds an ACT policy, one train step + one inference step, prints PASS
```

There is no pytest suite, linter or build step. The files in `tests/` are standalone smoke scripts; run the one you need with `uv run python tests/<file>.py`.

Robot work goes through the LeRobot CLIs, always behind `uv run` (`lerobot-find-port`, `lerobot-calibrate`, `lerobot-teleoperate`, `lerobot-record`, `lerobot-train`). `README.md` has the full SO-101 invocations. Its examples use macOS port names (`/dev/tty.usbmodem...`); on Windows the ports are `COMx`.

## Things that are easy to get wrong

- **LeRobot is a pinned dependency (0.7.0), not vendored.** Its source sits in `.venv/Lib/site-packages/lerobot` (Windows) or `.venv/lib/python3.12/site-packages/lerobot`. Config and flag names change between LeRobot releases, so check them there or in the version-matched docs (https://huggingface.co/docs/lerobot/v0.7.0/en/index) instead of relying on memory of another version.
- **0.7.0 changed the CLI.** According to `research/lerobot-training-feasibility.md` (read from the v0.7.0 docs and source, not yet run on our machines): policies are run on the robot with `lerobot-rollout`, not with `lerobot-record`; `lerobot-record` adds a timestamp to the dataset name unless `--dataset.no_stamp=true`; experiment tracking flags are `--trackio.*`; the two-arm types are `bi_so_follower` and `bi_so_leader`. That file has the full commands. When one is confirmed on hardware, record it in `kb/`.
- **Several skills in one demo need our own control loop.** Read from the v0.7.0 source, not yet run (`research/lerobot-multi-skill-runner.md` has the details and an untested sketch): `lerobot-rollout` and `RolloutController` own exactly one policy and one robot connection and cannot switch policies. The demo therefore connects the robot once, loads all policies up front and runs a plain 30 Hz loop (`get_observation`, `predict_action`, `send_action`). `lerobot-replay` also opens its own connection, so replays inside the demo are done in that same loop.
- **Never reset, disconnect or reconnect the robot while it holds the ball.** `RolloutController.reset()` moves all six joints including the gripper back to the pose at connect time, `disconnect()` switches torque off by default, and `connect()` briefly switches it off as well. Switching skills inside one loop sends nothing, so the arm holds.
- **ACT ignores the task sentence.** `set_task`, `/subtask` and the planner only affect language-conditioned policies (SmolVLA, pi0.5 and similar).
- **Only the control loop talks to the motors.** Vision code runs in another thread or process and reads frames with `cameras[name].read_latest()` or from a shared copy of the observation. Slowing the control loop below the recorded 30 fps hurts the policy.
- **Hand detection (MediaPipe) goes in its own environment and process.** Its OpenCV package clashes with the one LeRobot installs (from web research, not yet tried here).
- **torch comes from the CUDA 12.8 wheel index for every platform** (`[tool.uv.sources]` in `pyproject.toml`). `uv.lock` therefore has only Linux and Windows torch wheels and no macOS one, while `README.md` still describes a macOS/`mps` setup. As of 2026-11-20 this mismatch is open on the board; check there before touching the torch pins.
- `pyproject.toml` and `uv.lock` are committed together whenever dependencies change (`uv add <pkg>`).
- `outputs/`, `data/`, `wandb/` and `*.safetensors` are git-ignored. Datasets and checkpoints go to the Hugging Face Hub, never into git.

## Decisions get documented, code does not

The repo records *why* things are the way they are, so nobody explores the same search space twice. It does not describe *what* the code does: the code says that itself, and at this point agents read the code, not people.

- **Read `decisions/` before you explore an approach.** If the question already has a record, start from it. Do not retry an alternative listed there unless its "Revisit if" condition holds, and if you do retry it, add the result to that same file.
- **Write a decision record whenever you choose between alternatives or drop something you tried.** Use `decisions/<question>.md`, one file per question; `decisions/README.md` has the format. Failed attempts are the most valuable part: record them while the question is still open. Put the record in the same pull request as the code or config it explains. A choice with no real alternative needs no record.
- **No documentation layer about the code.** No architecture walkthroughs, module overviews, API docs or READMEs that retell what a script does; they go stale and agents read the code directly. Runbooks for the people at the table (which command to run, what to check) still belong in `docs/`.
- **Code is clean, concise and self-explanatory.** Use clear names and small functions, not comments.
- **The only comments are about decisions**: why it is done this way, which alternative was tried or rejected, or a constraint the code cannot show (`# no reset() here: it moves the gripper back to its connect pose and drops the ball`). No comments that restate the code, no section banners, no docstrings that repeat the signature, no commented-out code, no change notes. If the reasoning takes more than two lines, write a decision record and point to it (`# see decisions/skill-policies.md`). When you change code, delete or fix comments that no longer hold.

## Where things go

- `scripts/` - our own code (data collection helpers, perception, demo app).
- `tests/` - smoke scripts (see above).
- `decisions/` - one file per question: what was chosen, why, and what else was considered or tried, with results. See the section above.
- `kb/` - knowledge base: verified facts about our setup that the team learned the hard way (port mapping, calibration notes, camera settings, training settings that worked or failed). One topic per file. Facts only: a choice, and the alternatives behind it, goes in `decisions/` and links to the facts it rests on.
- `ideas/` - candidate project ideas, one file per idea, each with a status line.
- `research/` - external research with source links (hackathon rules, the ball-picking idea, LeRobot feasibility and the multi-skill runner). `research/README.md` says which files belong to dropped ideas.
- `docs/` - all other documentation (roadmap, runbooks, demo script, pitch). Only `README.md` and this file stay at the repo root.
