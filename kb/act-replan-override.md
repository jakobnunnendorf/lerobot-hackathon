# How often ACT replans can be changed on a trained checkpoint

- Date: 2026-11-21
- Verified by: Kamel's agent, on Kamel's laptop (RTX 4070 Laptop, 8 GB, WSL2), with `outputs/act_pick_blue_ball/checkpoints/020000/pretrained_model` (`molnrb/so101_pick_blue_ball`, chunk 100). Dummy inputs, no robot.

Our ACT checkpoints were trained with `chunk_size=100` and `n_action_steps=100`. With those values the arm plays back 100 actions (3.3 s at 30 fps) before it looks at the cameras again. Both `n_action_steps` and temporal ensembling are only read at inference time, so a saved checkpoint can be run with other values. No retraining is needed.

- `lerobot-rollout` passes `--policy.*` flags on to the saved config (`rollout/configs.py`, `PreTrainedConfig.from_pretrained(policy_path, cli_overrides=...)`). That makes `--policy.n_action_steps=10` work next to `--policy.path=...`. Read from the code; not yet run with `lerobot-rollout` on the robot.
- In our own loop: `PreTrainedConfig.from_pretrained(path, cli_overrides=["--n_action_steps=10"])`, then `ACTPolicy.from_pretrained(path, config=cfg)`. Run and confirmed: the loaded policy reports the new values.
- Temporal ensembling: `--n_action_steps=1 --temporal_ensemble_coeff=0.01`. Any other `n_action_steps` together with ensembling fails validation. With ensembling the model runs on every step.

Measured with the same checkpoint (two 640x480 cameras, CUDA, 60 steps):

| setting | time per `select_action` | peak GPU memory |
|---|---|---|
| `n_action_steps=10` | 0.5 ms median (queued steps); a full model call every 10th step | 272 MB |
| `n_action_steps=1`, ensembling 0.01 | 20.8 ms median, 41 ms max | 272 MB |

At 30 fps a step has 33 ms. On this GPU, ensembling fits within that, though not with much room. On a laptop without a GPU it will not keep up, so measure there before using it. `n_action_steps` between 10 and 20 costs one model call every 10 to 20 steps.

Not measured: how the arm moves with these settings. Smaller `n_action_steps` reacts sooner but can make the motion jerkier when two chunks disagree; ensembling smooths that.
