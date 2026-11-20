# lerobot-hackathon

Our team project for the LeRobot physical-AI hackathon in Bilbao. Hardware: SO-101 follower arm (+ leader arm), optionally a second arm.

**What we are building:** the arm finds the one target ball (yellow and green stripes) among gray tinfoil balls in a bowl and picks it up. If the target ball is hidden, it takes gray balls out one at a time, drops each into the gray hole and looks again. Then it puts the target ball into a person's hand and shakes hands, or puts it down if there is no hand. Details: `ideas/ball-picking-and-handover.md`. Plan and deadlines: `docs/roadmap.md`.

LeRobot is installed as a package (pinned to **0.7.0** in `pyproject.toml`), so nobody needs to clone the LeRobot repo.
Use the matching docs: https://huggingface.co/docs/lerobot/v0.7.0/en/index

## Setup (once per laptop)

You need [uv](https://docs.astral.sh/uv/) and ffmpeg (macOS: `brew install ffmpeg`).

```bash
git clone https://github.com/jakobnunnendorf/lerobot-hackathon.git
cd lerobot-hackathon
uv sync
```

Check it works:

```bash
uv run python -c "import lerobot; print(lerobot.__version__)"
```

Run every LeRobot command with `uv run` in front (or `source .venv/bin/activate` first).

## SO-101 quick commands

Find each arm's USB port (unplug/replug when asked):

```bash
uv run lerobot-find-port
```

Calibrate follower and leader (replace the ports):

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follower
uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodemYYYY --teleop.id=leader
```

Teleoperate (leader moves follower):

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follower \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodemYYYY --teleop.id=leader
```

Record demonstrations (add `--robot.cameras=...` for the wrist camera, see the cameras docs):

```bash
uv run lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follower \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodemYYYY --teleop.id=leader \
  --dataset.repo_id=<hf-user>/<dataset-name> \
  --dataset.single_task="Describe the task in one sentence" \
  --dataset.num_episodes=50 --display_data=true
```

Train a policy (ACT shown; `smolvla` is also installed):

```bash
uv run lerobot-train --dataset.repo_id=<hf-user>/<dataset-name> --policy.type=act --policy.device=mps --output_dir=outputs/act_run1
```

Use `--policy.device=cuda` on an NVIDIA GPU. Hugging Face login for pushing datasets/models: `uv run hf auth login`.

## Adding dependencies

```bash
uv add <package>             # e.g. uv add openai
uv add "lerobot[pi]==0.7.0"  # extra LeRobot policy deps
```

Commit `pyproject.toml` and `uv.lock` so everyone gets the same versions.

## Layout

- `scripts/` - our own scripts (data collection helpers, demo app, ...)
- `tests/` - smoke scripts (`uv run python tests/cuda.py`, `uv run python tests/lerobot_check.py`)
- `kb/` - knowledge base: verified facts we learned (ports, calibration, what worked)
- `ideas/` - candidate project ideas with their status
- `research/` - external research with sources
- `docs/` - all other documentation (roadmap, runbooks, demo script)
- `outputs/`, `data/` - local training runs and data (git-ignored)

## Working together

Coordination between everyone (people and coding agents) happens on one GitHub issue, the message board: https://github.com/jakobnunnendorf/lerobot-hackathon/issues/1. `CLAUDE.md` describes the rules agents follow.
