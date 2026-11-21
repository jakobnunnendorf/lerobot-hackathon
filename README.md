# lerobot-hackathon

Our team project for the LeRobot physical-AI hackathon in Bilbao. Hardware: SO-101 follower arm (+ leader arm), optionally a second arm.

**What we are building** (current plan, as agreed on board issue #1 on 2026-11-21): one dig-and-pick ACT policy (`act_pick_ball_combined`) takes the target ball (yellow and green stripes) from the bowl whether it is visible or buried under gray tinfoil balls, and ends at the folded rest pose holding the ball. Then the arm follows a hand (MediaPipe tracker, camera calibration, inverse kinematics) and drops the ball between two flat palms, or puts it down in a container if there is no hand. Voice picks the destination, and a supervisor chains the steps. The handshake comes last, after everything else works. Colour choice is parked. Details: `ideas/ball-picking-and-handover.md`. Plan and deadlines: `docs/roadmap.md`.

*Changed on 2026-11-21 because of board comments 7061614734 and 7061650334 (molnrb): there is no CLEAR GRAY dataset, and the combined policy digs and picks in one episode. Earlier plan: clear gray balls into a gray hole with a separate skill, hand the ball over with GIVE, then shake hands.*

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

**macOS note:** `uv.lock` pins the CUDA 12.8 torch wheels, so `uv sync` fails on macOS (there is no macOS torch in the lock). Use an existing virtualenv, or work on the robot laptop. Any `--policy.device=mps` line below only applies to such a hand-made Mac environment.

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

## Our datasets

Recorded by the team on the SO-101 (top + wrist camera, 30 fps), all at https://huggingface.co/molnrb:

| Dataset | Episodes | Frames |
|---|---|---|
| [molnrb/so101_pick_ball](https://huggingface.co/datasets/molnrb/so101_pick_ball) | 49 | 16,903 |
| [molnrb/so101_pick_blue_ball](https://huggingface.co/datasets/molnrb/so101_pick_blue_ball) | 100 | 47,401 |
| [molnrb/so101_pick_buried_ball](https://huggingface.co/datasets/molnrb/so101_pick_buried_ball) | 45 | 44,529 |
| [molnrb/so101_pick_ball_combined](https://huggingface.co/datasets/molnrb/so101_pick_ball_combined) | 145 | 91,930 |

Note: the task sentences in these datasets say "blue ball", but the ball is the one with yellow and green stripes (the target ball was blue at first). The dataset names and sentences stay as recorded, so do not rename them. Keep this in mind for language-conditioned policies and model cards.

Browse episodes in the browser: https://huggingface.co/spaces/lerobot/visualize_dataset

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
