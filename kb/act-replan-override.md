# How often ACT replans can be changed on a trained checkpoint

- Date: 2026-11-21
- Verified by: Kamel's agent, on Kamel's laptop (RTX 4070 Laptop, 8 GB, WSL2), with `outputs/act_pick_blue_ball/checkpoints/020000/pretrained_model` (`molnrb/so101_pick_blue_ball`, chunk 100). Dummy inputs, no robot.

Our ACT checkpoints were trained with `chunk_size=100` and `n_action_steps=100`. With those values the arm plays back 100 actions (3.3 s at 30 fps) before it looks at the cameras again. Both `n_action_steps` and temporal ensembling are only read at inference time, so a saved checkpoint can be run with other values. No retraining is needed.

- `lerobot-rollout` passes `--policy.*` flags on to the saved config (`rollout/configs.py`, `PreTrainedConfig.from_pretrained(policy_path, cli_overrides=...)`). That makes `--policy.n_action_steps=10` work next to `--policy.path=...`. molnrb's agent confirmed this on the robot laptop: `lerobot-rollout --policy.path=<ckpt> --policy.n_action_steps=20 ...` parses and loads the policy (stopped at robot connect on purpose, board comment 7060970692). The team never drove the arm with `lerobot-rollout`; the override below is what ran.
- In our own loop: `PreTrainedConfig.from_pretrained(path, cli_overrides=["--n_action_steps=10"])`, then `ACTPolicy.from_pretrained(path, config=cfg)`. Run and confirmed: the loaded policy reports the new values. On the arm this is `eval_pick.py`'s per-policy `#n_action_steps` and the supervisor's `--n-action-steps`, run at 25, 30, 50 and 100 on 2026-11-22.
- Temporal ensembling: `--n_action_steps=1 --temporal_ensemble_coeff=0.01`. Any other `n_action_steps` together with ensembling fails validation. With ensembling the model runs on every step.

Measured with the same checkpoint (two 640x480 cameras, CUDA, 60 steps):

| setting | time per `select_action` | peak GPU memory |
|---|---|---|
| `n_action_steps=10` | 0.5 ms median (queued steps); a full model call every 10th step | 272 MB |
| `n_action_steps=1`, ensembling 0.01 | 20.8 ms median, 41 ms max | 272 MB |

The same checkpoint on the **robot/demo laptop (RTX 4050)**, measured by molnrb's agent (130 calls after warm-up, board comment 7060988177):

| setting | median | p95 | max |
|---|---|---|---|
| `n_action_steps=20` | 0.7 ms | 36.5 ms | 60.7 ms |
| `n_action_steps=1`, ensembling 0.01 | 47.4 ms | 59.6 ms | 74.0 ms |

At 30 fps a step has 33 ms. **Ensembling does not fit on the demo laptop**: it would slow the loop to about 20 Hz. It fits on the 4070 only. `n_action_steps` between 10 and 20 works on both: one model call every 10 to 20 steps, and that one step takes 36-60 ms on the 4050.

How the arm moves with these settings was measured on 2026-11-22: 25-50 picked better than 100, and slower (`decisions/hidden-ball-picking.md`). Ensembling was never run on the arm.
