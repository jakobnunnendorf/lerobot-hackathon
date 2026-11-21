# DAgger (human corrections) with `lerobot-rollout`: quick reference

- Date: 2026-11-21
- Verified by: read from the LeRobot v0.7.0 source (`research/rl-post-training.md`, full research with sources) and one WSL check by Kamel's agent (board 7062057104). **Not yet run on our arm.** Update this file after the first real try.

## What it is

The trained ACT policy drives the arm. A person pauses it, corrects with the leader arm, and the corrections are recorded as extra episodes. They are merged into the dataset and the policy is fine-tuned. This is interactive imitation learning, not reinforcement learning. Worth it only if the policy lands between about 3/10 and 8/10.

## Command (ports, ids, cameras and task sentence identical to the recording setup)

```bash
lerobot-rollout --strategy.type=dagger --strategy.num_episodes=30 --policy.path=<checkpoint> --robot.type=so101_follower --robot.port=<follower port> --robot.id=mi_follower --robot.cameras="{ <same names, resolution and fps as the training data> }" --teleop.type=so101_leader --teleop.port=<leader port> --teleop.id=<leader id> --dataset.repo_id=<hf user>/rollout_<name> --dataset.single_task="<same task sentence as the pick dataset>" --dataset.fps=30
```

Dataset name must start with `rollout_`; a timestamp is appended unless `--dataset.no_stamp=true`.

## Keys

`Space` pause or resume, `Tab` start or stop a correction (only while paused), `Enter` push to the Hub, `ESC` stop. On pause the leader arm is driven to the follower pose, so there is no jump at takeover. When the policy resumes it plans a fresh chunk.

## WSL

LeRobot 0.7.0 falls back to reading keys from the terminal on Wayland (pynput is not used). Type in the focused `lerobot-rollout` terminal, not a global hotkey, and start it in a visible WSL window with a real TTY. Check without the arm: `python -c "from lerobot.utils.keyboard_input import is_wayland, pynput_can_capture; print(is_wayland(), pynput_can_capture())"`, expected `True False`.

## After the corrections

The recorded data has an extra `intervention` column that training does not read; remove it with `lerobot-edit-dataset ... --operation.type remove_feature --operation.feature_names "['intervention']"` before merging. Then merge and fine-tune from the checkpoint as in `research/continue-training-from-checkpoint.md`.
