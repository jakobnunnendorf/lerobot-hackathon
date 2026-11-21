# Learned policies for the hand-following handover: ACT and SmolVLA

Researched 2026-11-21 by the agent of @Space0code. Nothing here was run on our machines. Config values were read from the v0.7.0 source (raw files, tag `v0.7.0`), not from memory.

Behaviour to learn: start at HOME holding the ball, hover over one hand, drop the ball when a second hand appears, return to HOME. Context for the VRAM and training-time numbers: `research/lerobot-training-feasibility.md` (section 1), not repeated here.

## TL;DR

1. **ACT is the safe bet**: already trained and run by the team, 2-6 GB, fits the 8 GB laptop. Its weakness is reaction time, and that can be tuned (section 2).
2. **SmolVLA is worth one parallel run, not more.** It takes the task sentence, 450M parameters, so inference should fit 8 GB, but training does not (about 10-16 GB, cloud). Nobody here has run it.
3. Whichever policy: the data decides. Follow the recording checklist (section 4) before the ~20 episodes are finished.

## 1. Settings read from the v0.7.0 source

| Setting | ACT | SmolVLA |
| --- | --- | --- |
| `chunk_size` | 100 | 50 |
| `n_action_steps` | 100 (3.3 s at 30 fps) | 50 (1.7 s) |
| `n_obs_steps` | 1 (anything else raises an error) | 1 |
| Learning rate | 1e-5 (backbone 1e-5) | 1e-4, warmup 1,000, decay 30,000 steps |
| Vision backbone | ResNet-18 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, frozen by default (`freeze_vision_encoder=True`, `train_expert_only=True`) |
| Task sentence | ignored | used |
| Camera names | any | `smolvla_base` expects `camera1..3`; pass `--rename_map` at training and at rollout |
| Images | native size | resized with padding to 512x512; 64 visual tokens per frame (blog) |

Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/configuration_act.py , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/smolvla/configuration_smolvla.py , https://huggingface.co/blog/smolvla

## 2. Making ACT react to a moving hand

`n_action_steps=100` means the arm runs 3.3 s of planned motion without looking again. A hand that moves or a second hand that appears is invisible for that long. Two levers, both in `ACTConfig` and `ACTPolicy` of 0.7.0:

- **Smaller `n_action_steps`** (for example 10-20, about 0.3-0.7 s). The chunk is predicted every time the queue is empty and then cut to `n_action_steps` (`predict_action_chunk(batch)[:, :n_action_steps]`), so this changes only how often the policy re-plans, not the weights. Cost: more inference calls and more jerk at the joins. Confirmed by molnrb's agent on the robot laptop (board, 2026-11-21, parse and load only, arm not moved): `lerobot-rollout --policy.path=<ckpt> --policy.n_action_steps=20` works on a saved checkpoint, and `--policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01` too. Effect on success not yet measured.
- **Temporal ensembling**: `temporal_ensemble_coeff` (the original ACT work used 0.01). It runs inference every step and blends all overlapping predictions with weights `exp(-coeff * i)`. The config raises an error unless `n_action_steps == 1` when it is on. This is the smoothest option but needs one forward pass per control step (30 per second); fine on the 4070, unknown on a CPU.

Reading for this task: start with `n_action_steps` around 10-20 on the checkpoint we get, compare against 100 on the same episodes, and try temporal ensembling only if the motion is too jerky. The second hand arriving matters most: the drop decision is made at re-plan time, so a long chunk delays the drop by up to its length.

Sources: `modeling_act.py` and `configuration_act.py` at https://github.com/huggingface/lerobot/tree/v0.7.0/src/lerobot/policies/act ; ACT paper https://arxiv.org/abs/2304.13705 (temporal ensembling).

## 3. SmolVLA on an 8 GB laptop

- **Inference:** no official VRAM figure and no SO-101 latency benchmark found. The Hugging Face blog and the paper say it runs on consumer GPUs or CPU. One third-party comment claims 0.9 GB and 18 ms per step on a simulated Franka; treat as a rough indicator only. Measure on Kamel's laptop. Sources: https://huggingface.co/blog/smolvla , https://arxiv.org/abs/2506.01844
- **Training:** the LeRobot hardware guide says about 10-16 GB, so it runs on HF Jobs (A10G or L4), not on the laptop. The docs command fine-tunes `lerobot/smolvla_base` for 20k steps at batch 64, roughly 4 h on an A100. The docs also say 25 episodes "was not enough", and one hackathon team's SmolVLA fine-tuned for 2,500 steps did worse than their ACT (`research/ball-picking-and-handover.md`). With about 20 episodes, expect SmolVLA to be the weaker of the two unless the language input helps. That is the thing it can show: one model, different task sentences.
- **Latency:** chunk of 50 steps; the docs recommend RTC (`--inference.type=rtc`) when inference is slow.

## 4. Recording checklist for the ~20 episodes

Before the next episode:

1. **Same start pose, ball held in the gripper, every episode.** The gripper must close on the ball the same way each time; mark the ball position in the gripper if it can be done without moving the arm.
2. **End at HOME**, and hold the last pose for about a second so the policy learns to stop.
3. **Spread the first hand over the whole area where a hand may appear in the demo** (left, centre, right, near, far); no more than 2-3 episodes per spot. Fixed positions teach one fixed motion (the same failure mode as in `research/ball-picking-and-handover.md`).
4. **Second hand at different moments**: early, late, never (a few hover-only episodes where no second hand comes and the ball stays). Without the "never" cases the policy learns that hover always ends in a drop.
5. **The hand that stays is the same demo hand as in the pitch** (palm up, similar height). Vary lighting a little and do not move the top camera between recordings and the demo; a moved camera was the cause of several bad ACT results in the forum thread cited in `research/lerobot-training-feasibility.md`.
6. **One fixed task sentence for the whole dataset** (for example "hand over the ball"). ACT ignores it, SmolVLA uses it; changing the wording between episodes only adds noise.
7. Count about 5 episodes per hand-area cell if 20 is all there is; if more time remains, add episodes before adding policies.

Open from the board that affect this: the three table questions (camera fixed, hand height, WSL and `use_degrees` or normalised joint values).

## 5. What to try, in order

1. ACT on the 20 episodes, both `n_action_steps=100` and a small value, on `molnrb`'s HF job setup or Kamel's 4070 (ACT: 20k steps took 1h40 on the 4070, `kb/machines.md` and the board).
2. SmolVLA fine-tune in the cloud with the same dataset (A10G, `--batch_size=16-32` and `--steps` 5k-10k; the 20k-step docs example is for 50 episodes and an A100), only if the cloud budget allows after ACT runs; check the running total on the board before starting a paid job (a person starts it).
3. Compare on the same table setup: ten tries each, count hover reached, second hand noticed, ball dropped into the hands, back at HOME. Write the numbers into `kb/` and the choice into `decisions/hand-handover-policy.md`.

## 6. Facts from the board (2026-11-21), not verified by this agent

- The robot program runs in WSL2 (Ubuntu 24.04, conda env `lerobot`, lerobot 0.7.0); joints are in degrees (`use_degrees=True`), relative to the `mi_follower` calibration. Cameras `top` and `wrist`, 640x480, 30 fps. Source: comments of the agents of @molnrb and @BrouthenKamel on issue #1.
- SmolVLA fine-tune of `lerobot/smolvla_base` already claimed by Kamel's agent, with cameras renamed `top`->`camera1`, `wrist`->`camera2`.
