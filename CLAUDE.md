# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team project for the [Bilbao Physical AI Hackathon](https://hackathonbilbao.eus/home-page-eus): 5 people, about 2 days (started 2026-11-20), LeRobot SO-101 arms (leader + follower, possibly a second pair). Optimise for a working robot demo by the deadline, not for clean abstractions.

**The project idea is a PLACEHOLDER.** The current candidate is "bomb defusal with computer vision" (the arm finds and cuts the correct wire on a mock device). It is not finalized; the team is still ideating. Until the message board says the idea is locked, keep code idea-agnostic (teleop, recording, training, perception plumbing) and do not bake "bomb", "wire" or similar into module names, dataset names or task strings. Idea status lives in `ideas/`.

## Coordination (read before doing any work)

Five people work here in parallel, each with their own agent. The only coordination channel is **GitHub issue #1, the message board**.

- Before starting: `gh issue view 1 --comments` and read the latest comments, so you do not duplicate or collide with someone else's work.
- Post a comment (`gh issue comment 1 --body-file <file>`) when you claim a task, finish one, get blocked, or hand work over. Start it with who you work for and the kind of update, e.g. `**[agent of @TianK003] CLAIM**`, `DONE`, `BLOCKED`, `HANDOFF`, `QUESTION`. Mention a person with `@login` only when they need to act.
- Do not open more issues for coordination; everything goes on #1.
- To be told when someone posts, run `bash scripts/watch_board.sh <state file outside the repo> 43200 60` as a background command. It polls once a minute, skips your own agent's comments, and exits with the new comments as soon as someone else posts; then start it again. Use the long window: a short one makes the script return every few minutes and wakes the session for nothing. The `board-watcher` agent (Haiku, `.claude/agents/board-watcher.md`) wraps the same script and adds a summary. Neither posts anything.
- Commit small and often, directly on `main` (the team works trunk-based, no PR flow). `git pull --rebase` before every push, never force-push, and say what you pushed on the board when others depend on it.
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
- **torch comes from the CUDA 12.8 wheel index for every platform** (`[tool.uv.sources]` in `pyproject.toml`). `uv.lock` therefore has only Linux and Windows torch wheels and no macOS one, while `README.md` still describes a macOS/`mps` setup. As of 2026-11-20 this mismatch is open on the board; check there before touching the torch pins.
- `pyproject.toml` and `uv.lock` are committed together whenever dependencies change (`uv add <pkg>`).
- `outputs/`, `data/`, `wandb/` and `*.safetensors` are git-ignored. Datasets and checkpoints go to the Hugging Face Hub, never into git.

## Where things go

- `scripts/` - our own code (data collection helpers, perception, demo app).
- `tests/` - smoke scripts (see above).
- `kb/` - knowledge base: verified facts the team learned the hard way (port mapping, calibration notes, camera settings, what worked and what failed). One topic per file.
- `ideas/` - candidate project ideas, one file per idea, each with a status line.
- `research/` - external research with source links (hackathon rules, domain, LeRobot feasibility).
- `docs/` - all other documentation (roadmap, runbooks, demo script, pitch). Only `README.md` and this file stay at the repo root.
