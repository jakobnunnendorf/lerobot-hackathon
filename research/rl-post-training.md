# Reinforcement learning after training: what is realistic for our SO-101 demo

Researched 2026-11-21. Question from the team: "Is there a good, promising and at least somewhat realistic way to enhance our robot with reinforcement learning, maybe after training?"

How this was checked:

- **LeRobot facts** were read in the source archive of the `v0.7.0` tag (`https://github.com/huggingface/lerobot/archive/refs/tags/v0.7.0.tar.gz`), in the files named below. Doc pages are the `.mdx` files of that tag, which are the source of `https://huggingface.co/docs/lerobot/v0.7.0/en/<page>`.
- **Papers and blog posts** were read through an automated page reader. Numbers from them should be re-checked in the original before they go on a slide.
- **Nothing here was run.** No command was executed on a robot or a GPU.
- Lines marked **Estimate** or **Inference** are mine, not from a source.

## Short answer

1. Real reinforcement learning (a reward, a critic, exploration on the arm) is **not realistic in the 8-12 hours we have**. The only RL that LeRobot 0.7.0 ships, HIL-SERL, trains a new small policy from scratch, cannot start from or improve our ACT policies, needs Linux-only kinematics, and is documented for 5-10 second tasks.
2. The realistic "learn from its own experience" step is **human corrections on top of the trained ACT policy** (DAgger / RaC). LeRobot 0.7.0 has it built in (`lerobot-rollout --strategy.type=dagger`), it works with ACT and the SO-101 leader arm, and it needs no new hardware. It is interactive imitation learning, not RL, and should be named that way in the pitch.
3. Everything else (residual RL, DSRL, RECAP, RL token, Q-chunking, sim-to-real RL) either does not apply to ACT, has no code that runs on LeRobot datasets, or needs a digital twin of a bowl of tinfoil balls.

## Comparison

| # | Method | Is it RL? | In LeRobot 0.7.0? | Works with our ACT? | Extra needs | Robot hours | Realistic in 8-12 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | HIL-SERL (SAC + interventions) | Yes | Yes (`lerobot.rl`, `hilserl` extra) | No. Separate `gaussian_actor` policy from scratch | URDF, `placo` (Linux/macOS only), reward labels or classifier, bounded workspace, fixed reset | 1-4 h of training after setup, arm blocked | No |
| 2 | Human corrections (DAgger / RaC) | No (interactive imitation) | Yes (`--strategy.type=dagger`) | Yes (docs example is ACT) | Leader arm, keyboard, one operator | about 1 h per round | **Yes** |
| 3 | Success-filtered self-imitation | Borderline (reward = success filter) | Recording yes (`--strategy.type=episodic`), no dedicated tool | Yes | A success label per rollout | 1-2 h for a useful amount | Possible, small expected gain |
| 3b | Reward-weighted BC (RA-BC with SARM) | Borderline | Yes, but only PI0, PI0.5, SmolVLA, FastWAM | No | Reward model training | none | No |
| 4a | Residual RL on a frozen ACT (ResFiT) | Yes | No | In the paper yes | Own actor-learner code, critic, success labels | paper: 15-76 min, after a large engineering effort | No |
| 4b | DSRL (steer diffusion noise) | Yes | No | No (needs diffusion or flow policy) | JAX code tied to openpi | paper: 40-150 episodes | No |
| 4c | RECAP / pi*0.6, RL token | Yes | No | No (VLA-scale) | Not public as trainable code | n/a | No |
| 4d | Q-chunking (QC-FQL), ConRFT, RLPD | Yes | QC-FQL "coming soon"; RLPD-style mixing is inside HIL-SERL | No | Own training stack | n/a | No |
| 4e | Value-guided action selection (V-GPS) | Yes (offline RL critic) | No | No (ACT is deterministic at inference) | Trained value function | none | No |
| 5 | RL in simulation, then sim-to-real | Yes | Not for our task | Separate policy | Digital twin, Linux, Vulkan GPU | little | No for the bowl task |

---

## 1. HIL-SERL as shipped in LeRobot

**What it is.** Soft Actor-Critic on the real robot, seeded with a few demonstrations in an offline buffer, with a human who takes over during exploration; success comes from a learned image classifier or a key press. Actor (robot side) and learner (GPU side) are two processes that talk over gRPC.

**Verified in the v0.7.0 tag**

- Docs: `docs/source/hilserl.mdx`. Needs listed there: "A gamepad (recommended) or keyboard", "A Nvidia GPU", a leader arm (optional), and "A URDF file for the robot for the kinematics package".
- Task advice in the same file: "Avoid extremely long horizon tasks. Focus on tasks that can be completed in 5-10 seconds."
- **It starts from scratch.** The only registered algorithm is `sac` (`src/lerobot/rl/algorithms/sac/configuration_sac.py`, `src/lerobot/rl/algorithms/factory.py`: "Supported names are \"sac\""). `SACAlgorithm.__init__` takes a `GaussianActorPolicy` (`src/lerobot/rl/algorithms/sac/sac_algorithm.py`). The policy type is `gaussian_actor` (`src/lerobot/policies/gaussian_actor/configuration_gaussian_actor.py`): a frozen ResNet-10 image encoder plus small MLP heads that output one action per step. There is no field for loading an ACT checkpoint, and ACT is not an option for `algorithm`. The repo README at the tag lists "HIL-SERL, TDMPC & QC-FQL (coming soon)" under Reinforcement Learning.
- **Action space is end-effector deltas.** `hilserl.mdx`: "HIL-Serl learns actions in the end-effector space of the robot." The example config (`lerobot/config_examples`, `rl/train_config.json` on the Hub) has `action` shape `[3]` (x, y, z) plus 3 discrete gripper actions, 128x128 images, `fps: 10`, `batch_size: 256`, `control_time_s: 30`, and an `inverse_kinematics` block with `urdf_path` and `end_effector_bounds`.
- **Kinematics need `placo`.** `src/lerobot/model/kinematics.py` builds `placo.RobotWrapper(urdf_path)`; the `hilserl` extra in `pyproject.toml` pulls `lerobot[placo-dep]` (`placo>=0.9.6,<0.9.16`), `gym-hil` and `grpcio`. PyPI lists only Linux and macOS wheels for `placo` 0.9.6-0.9.15 and for `pin` 4.1.0 (checked through the PyPI JSON API today). So the actor has to run under WSL2 or Linux.
- **Leader-arm interventions are supported.** `hilserl.mdx`, section "Setting up the SO101 leader": `"control_mode": "leader"`, `"teleop": {"type": "so101_leader", ...}`; `space` takes over and hands back, `s` marks success, `esc` marks failure. A gamepad is recommended but not required.
- **Our existing datasets do not fit.** Demonstrations are recorded again through `python -m lerobot.rl.gym_manipulator --config_path rl/env_config.json` with `mode: "record"`, because the buffer needs rewards, done flags, cropped 128x128 images and end-effector actions (`src/lerobot/rl/buffer.py`, `from_lerobot_dataset`). Our 145 episodes are joint-space, 30 fps, without rewards.
- Commands: `python -m lerobot.rl.learner --config_path rl/train_config.json` and `python -m lerobot.rl.actor --config_path rl/train_config.json`; bounds with `lerobot-find-joint-limits ... --urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf --target_frame_name=gripper`; crops with `python -m lerobot.rl.crop_dataset_roi --repo-id <dataset>`.

**Published evidence**

- Original paper (Franka-class arms, RTX 4090, SpaceMouse): "near-perfect success rates and fast cycle times within just 1 to 2.5 hours of training"; HIL-SERL about 100% on nearly all tasks against 49.7% for HG-DAgger; 20-30 offline demos; the policy is not initialised from an imitation policy, only the ResNet-10 encoder is pretrained. https://arxiv.org/abs/2410.21845
- SO-101, hackathon team (Seeed Studio LeRobot hackathon, Mountain View; write-ups from Feb/Mar 2026): cube pick from scratch on hardware in about 40 minutes with tight workspace bounds, "slightly generalisable" after about 2 hours; a 16-hour simulation run "didn't transfer at all"; fine-tuning that sim checkpoint on hardware took about 4 hours for "some successful picks". They report many small bug fixes in the LeRobot HIL-SERL code before anything trained, training interrupted by camera and robot problems, a resumed run that learned nothing for 2-3 hours, and a gripper that opened when an intervention started. No success rates are given. They used manual key-press rewards, not a classifier. https://indraneelpatil.github.io/blog/2026/hil-serl and https://trunc8.github.io/projects/hil-serl-lerobot
- Third-party study notes on LeRobot (written against v0.5.0): the SAC actor "always begins from random", "you can't load your trained ACT or Diffusion weights into it as a starting point". This matches what I read at v0.7.0. https://sebastiandavidlee.github.io/seb-lab/lerobot-study/rl-on-top.html
- No official LeRobot benchmark of HIL-SERL training time on SO-100/SO-101 was found.

**Verdict: not realistic.** Reasons, in order of weight:

1. It cannot improve the ACT policies. The result would be a second, unrelated policy for one short sub-task (for example "close the gripper on the visible ball in a 10 cm box").
2. Setup on our machines: WSL2 with both serial ports and both cameras passed through USB, the SO-101 URDF, workspace bounds, crops, new demonstrations, a config file of about 200 lines. **Estimate:** 4-8 hours before the first learning curve, on a stack that a full team needed far longer than a hackathon to make work.
3. The bowl has to be reset by hand after every episode, and the task (find, clear, pick) is far beyond 5-10 seconds.
4. The arm is occupied for all of it while the main demo still needs evaluation and rehearsal.

---

## 2. Human corrections without RL (DAgger, HG-DAgger, RaC)

**What it is.** The trained policy runs on the robot; when it is about to fail, a person pauses it, takes over with the leader arm, recovers and finishes the sub-task. The corrections are added to the demonstrations and the policy is fine-tuned; repeat.

**Verified in the v0.7.0 tag**

- It is built in: `docs/source/hil_data_collection.mdx`, `docs/source/inference.mdx` (section "DAgger"), `src/lerobot/rollout/strategies/dagger.py`, `DAggerStrategyConfig` in `src/lerobot/rollout/configs.py`.
- **Works with ACT:** the docs table lists "Standard (default) | (no flag needed) | ACT, Diffusion Policy", and step 1 of the guide trains ACT.
- **Works with the SO-101 leader:** "Compatible teleoperators: ... `so100_leader` / `so101_leader`". On pause the leader is driven to the follower pose (`teleop_smooth_move_to` in `dagger.py`; `send_feedback` exists in `src/lerobot/teleoperators/so_leader/so_leader.py`), so there is no jump at takeover.
- **Flags:** `--strategy.type=dagger`, `--strategy.num_episodes` (number of corrections; falls back to `--dataset.num_episodes`), `--strategy.record_autonomous` (default false: only corrections are recorded, each correction is one episode), `--strategy.upload_every_n_episodes` (5), `--strategy.input_device=keyboard|pedal`, `--strategy.smooth_handover` (true), `--strategy.keyboard.pause_resume|correction|upload`. `--teleop.type` and `--dataset.repo_id` are required.
- **Keys:** `Space` pause/resume, `Tab` start/stop a correction (only while paused), `Enter` push to Hub, `ESC` stop. The keyboard listener uses pynput on Windows (docstring in `dagger.py`); the foot pedal is Linux only.
- When the policy resumes, the inference engine is reset (`dagger.py`, transition to `AUTONOMOUS`), so ACT plans a fresh chunk from the corrected state.
- Dataset rules: the name must start with `rollout_` (`src/lerobot/rollout/context.py`); a timestamp is appended unless `--dataset.no_stamp=true` (`src/lerobot/configs/dataset.py`); `--dataset.fps` must equal `--fps` (default 30).
- The recorded data has an extra boolean column `intervention`. Training does not read it (no reference in `src/lerobot/scripts/lerobot_train.py`, `src/lerobot/datasets` or the ACT policy), and it must be removed before merging.
- Fine-tuning from the existing checkpoint is the documented step 4 (`lerobot-train --policy.path=<checkpoint> --dataset.repo_id=<merged> --steps=20000`).

**Commands (flags verified, not run; ports, ids, camera block and task sentence must be identical to the recording setup)**

```bash
lerobot-rollout \
    --strategy.type=dagger \
    --strategy.num_episodes=30 \
    --policy.path=<pick ACT checkpoint or Hub id> \
    --robot.type=so101_follower --robot.port=COMx --robot.id=<same follower id> \
    --robot.cameras="{ <same names, resolution and fps as the training data> }" \
    --teleop.type=so101_leader --teleop.port=COMy --teleop.id=<leader id> \
    --dataset.repo_id=${HF_USER}/rollout_pick_dagger \
    --dataset.single_task="<same task sentence as the pick dataset>" \
    --dataset.fps=30

lerobot-edit-dataset \
    --repo_id ${HF_USER}/rollout_pick_dagger_YYYYMMDD_HHMMSS \
    --new_repo_id ${HF_USER}/pick_dagger_clean \
    --operation.type remove_feature \
    --operation.feature_names "['intervention']"

lerobot-edit-dataset \
    --new_repo_id ${HF_USER}/pick_plus_dagger \
    --operation.type merge \
    --operation.repo_ids "['${HF_USER}/<pick dataset>', '${HF_USER}/pick_dagger_clean']"

lerobot-train \
    --dataset.repo_id=${HF_USER}/pick_plus_dagger \
    --policy.path=<pick ACT checkpoint> \
    --policy.repo_id=${HF_USER}/act_pick_dagger1 \
    --output_dir=outputs/pick_dagger1 \
    --steps=20000
```

These move the robot or start training, so a person at the table runs them.

**Published evidence**

- RaC (two xArm-7, 368M-parameter flow policy, not a low-cost arm): shirt hanging 78.3% with 5 hours of data, against 75.0% for ALOHA Unleashed with about 89 hours; "an order of magnitude less data"; "consistently outperforms HG-DAgger". Two rules: "Pair each recovery segment with a correction segment" and "After an intervention concludes, we terminate the entire episode". https://arxiv.org/abs/2509.07953
- pi*0.6 / RECAP uses the same deploy, correct, retrain loop (plus a value function) and reports more than doubled throughput and roughly halved failure rate on its hardest tasks. https://arxiv.org/abs/2511.14759
- "So You Think You Can Scale Up Autonomous Robot Data Collection?" (Stanford, CoRL 2024): human demos collected from states where the policy failed beat the same number of random human demos. https://arxiv.org/abs/2411.01813
- TER-DAgger uses ACT as base policy and reports more than 37% average gain over BC and correction baselines (abstract only, robot not confirmed). https://arxiv.org/abs/2603.04038
- SO-101 specifically: public Hub datasets named `rollout_so101_dagger_*` show the workflow is used, but I found **no measured before/after success rate for ACT plus DAgger on SO-100/SO-101/Koch**. One hobbyist project reportedly saw an ACT policy collapse to 0% after a round of bad corrections (search snippet only; the page returned 404 when opened).

**Verdict: realistic, and the only candidate that improves the policies we already have.**

- **Estimate** for one round on the PICK skill: 30 corrections in 45-60 minutes at the arm (one operator, one person resetting balls), 10 minutes for edit and merge, 1-2 hours fine-tuning for 20k steps on the RTX 4070 laptop or on an HF Job, 20 minutes for a 10-trial comparison. About 3 hours wall-clock, 1.5 of them at the arm.
- Use it on one skill at a time (PICK first). `lerobot-rollout` owns one policy and one connection, so this is a data-collection session outside the demo runner.
- Follow RaC: every correction first recovers to a familiar state and then completes the sub-task; end the correction in the anchor pose (HOME or CARRY) the demonstrations end in.
- Risk: corrections that are hesitant or end somewhere else can make the policy worse. Keep the base checkpoint and compare both on the same ball layouts.

---

## 3. Filtered or reward-weighted self-imitation

**What it is.** Run the trained policy, keep only the rollouts that succeeded, add them to the data and retrain. The success filter is the only "reward".

**LeRobot 0.7.0 (verified)**

- Recording policy rollouts as episodes: `lerobot-rollout --strategy.type=episodic --dataset.num_episodes=N --dataset.episode_time_s=T ...`; Left Arrow discards the current episode and re-records it (`src/lerobot/rollout/strategies/episodic.py`), which is a human success filter. Bad episodes can also be removed later with `lerobot-edit-dataset --operation.type delete_episodes`.
- There is no dedicated self-imitation tool. Reward-weighted training exists as RA-BC (`--sample_weighting.type=rabc`, `docs/source/sarm.mdx`, `src/lerobot/utils/sample_weighting.py`), but the docs say "Currently PI0, PI0.5, SmolVLA and FastWAM are supported". The training step calls `policy(batch, reduction="none")` (`src/lerobot/scripts/lerobot_train.py`), and `ACTPolicy.forward(self, batch)` has no `reduction` argument (`src/lerobot/policies/act/modeling_act.py`). **Inference:** RA-BC fails with ACT unless the policy is patched.

**Published evidence**

- Stanford study above (Diffusion Policy, real tasks HangTape and NutInsertion): filtered BC on successful autonomous rollouts "can modestly improve performance (on the order of 10%)": HangTape 44% to 55%, NutInsertion 44% to 58%. A second round made both worse (55% to 50%, 57% to 46%). 320 autonomous rollouts added to few human demos gave 46%, while 50 human demos gave 80%: "additional human data is significantly better than access to even ten times the amount of autonomous data".
- Positive reports exist for VLA-scale systems with extra machinery (pi*0.6 above; REVOLVE reports 48% to 66% over five iterations, https://arxiv.org/abs/2609.14633, abstract-level reading only).

**Verdict: feasible, low value.** Successful rollouts only show the policy what it can already do; they contain no recoveries. An hour at the arm is better spent on corrections (section 2) or on plain extra demonstrations of the failing layouts. If evaluation rollouts are recorded anyway with `--strategy.type=episodic`, adding the successes costs nothing, but do it once, not in rounds.

---

## 4. RL on top of a frozen imitation policy

| Method | Applies to ACT? | Public code on LeRobot datasets? | Cost and evidence |
| --- | --- | --- | --- |
| Residual RL (ResFiT, https://arxiv.org/abs/2509.19301) | Yes: "We use ACT as the base policy for both tasks". A small per-step residual is added to the frozen ACT action and trained with TD3-style off-policy RL from sparse success labels given by the operator | No repository found; the paper links only a project page | Real humanoid: 14% to 64% in 134 rollouts (about 15 min) and 23% to 64% in 343 episodes (about 76 min). Base ACT trained on about 900-1000 demos. Needs an own actor-learner, replay buffer, critic ensemble and image encoder |
| DSRL (https://arxiv.org/abs/2506.15799) | No. It runs RL over the input noise of a diffusion or flow policy. ACT at inference sets its latent to zeros (`modeling_act.py`, "we set the latent to be all zeros"), so there is no noise input to steer | https://github.com/nakamotoo/dsrl_pi0 : JAX, openpi, Libero/Aloha sim and a real Franka. Not LeRobot | Franka pick-and-place 2/10 to 9/10 in about 40 episodes; pi0 toaster 5/20 to 18/20 in 80 episodes. Would need a Diffusion Policy or pi0 base, which we do not have |
| RECAP / pi*0.6 (https://arxiv.org/abs/2511.14759) | No. Advantage-conditioned VLA with a learned value function | No training code found | VLA-scale compute and data |
| RL token (RLT, https://arxiv.org/abs/2604.23073) | No. Needs a VLA trained to emit the token | Not confirmed | "minutes to a few hours of practice", up to 3x faster on the hardest part of the task. The page pi.website/research/rlt returned HTTP 429, so this is from the arXiv abstract |
| ConRFT (https://arxiv.org/abs/2502.05450) | No. Fine-tunes a VLA action head with a consistency policy on HIL-SERL-style infrastructure | Project page says code exists (not opened) | 96.3% average after 45-90 min online, on a Franka-class setup |
| RLPD | It is the "SAC plus prior data" core of HIL-SERL. In LeRobot it is the `online_offline` mixer with `online_ratio=0.5` (`src/lerobot/rl/train_rl.py`) | Only as part of section 1 | See section 1 |
| Q-chunking / QC-FQL (https://arxiv.org/abs/2507.07969) | Not directly: it trains its own flow policy and chunked critic | https://github.com/ColinQiyangLi/qc (simulation benchmarks). LeRobot README: "QC-FQL (coming soon)" | No real-robot result seen in what I read |
| Value-guided selection (V-GPS, https://arxiv.org/abs/2410.13816) | No. It re-ranks several sampled actions with an offline-RL value function and needs a policy that samples stochastically; ACT is deterministic at inference | Not checked | Paper abstract: +82% on real tasks for generalist policies. We have no value function and no reward-labelled data to train one |

**Inference:** a "heuristic critic" version of the last row already exists in our design: the supervisor checks gripper reading and camera after each skill and retries. That is a hand-written check, not RL, and should not be relabelled.

**Verdict: none of these in 8-12 hours.** Residual RL is the only one that fits ACT in principle, and it is a research implementation, not a config flag.

---

## 5. RL in simulation and sim-to-real

**What exists**

- Squint (https://arxiv.org/abs/2602.21203): SO-101, ManiSkill3 digital twin, wrist camera downsampled to 16x16 plus joint state. "We train policies for 15 minutes on a single RTX 3090 GPU"; zero-shot real-world result 73/80 (91.3%) over eight tasks (reach, lift, place, stack with a cube and a can). The paper does not describe how to build the twin for a new task.
- lerobot-sim2real (https://github.com/StoneT2000/lerobot-sim2real): ManiSkill, SO100 cube grasp, RGB policy deployed zero-shot. Pinned to an old LeRobot commit ("latest LeRobot can work but some LeRobot import paths need to be updated"), needs working NVIDIA drivers and Vulkan. The ManiSkill docs report 18/20 for a Koch arm after about an hour on an RTX 4090 (from a search summary, page not opened).
- LeRobot 0.7.0 itself: `docs/source/hilserl_sim.mdx` is a Franka Panda cube pick in MuJoCo (`gym_hil`), not an SO-101. `docs/source/envhub_leisaac.mdx` lists SO-101 tasks (PickOrange, LiftCube, CleanToyTable, FoldCloth) in IsaacLab for imitation learning.
- Counter-evidence from the SO-101 hackathon team in section 1: 16 hours of simulated HIL-SERL training did not transfer to their real arm.

**Verdict: not realistic for our task.** The published successes are single rigid objects on a flat table. We would need a simulated bowl with crumpled tinfoil balls, a matched camera, Linux with Vulkan, and then a second control path in the demo for a policy with a different action space and rate. A cube-lift sim-to-real side demo might be possible in a day for someone who has done it before, but it would not improve the bowl demo.

---

## 6. What hackathon teams actually did when they said "RL"

- **HIL-SERL cube pick on SO-101** (Seeed Studio hackathon, Mountain View): started at the hackathon, the reported results (40 minutes from scratch with narrow bounds, failed sim-to-real) come from follow-up work documented over several experiments. No success rates published. Links in section 1.
- **TD-MPC, offline then online, cube push** (LeRobot hackathon Toulouse, Moss v1 arm): demonstrations recorded with LeRobot, TD-MPC trained offline, then continued online on the arm. From a search summary only; the blog page (https://corentinlger.github.io/posts/lerobot) returned 404 when opened, so the outcome is unknown.
- **DAgger rounds on SO-101** appear on the Hub as datasets and fine-tuned ACT models, without metrics.
- The winners page of the LeRobot Worldwide Hackathon (https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners) loads videos without descriptions, so I could not tell how many winning projects used RL. The winner write-ups I did find (for example https://darthgera123.github.io/blogs/hackathon/robot) used imitation learning with VLAs.

Pattern: projects that finished within a hackathon used imitation learning; RL showed up as a from-scratch toy task (cube pick or push), not as an improvement of a trained policy.

---

## Recommendation for this team

### Ranked

1. **One DAgger round on the PICK skill** (section 2). Try this first, and only this.
2. **Record evaluation rollouts with `--strategy.type=episodic`** and merge the successes once, only if they are recorded anyway. Do not spend arm time on it for its own sake.
3. **Do not start** HIL-SERL, residual RL, or sim-to-real before the deadline.

Cheaper than any of the above and not RL: if ACT commits to a bad chunk, lower `--policy.n_action_steps` (default 100, the full chunk; field in `src/lerobot/policies/act/configuration_act.py`) at rollout so it re-plans more often. Untested here.

### What the DAgger round needs from the humans

- Hardware: follower, leader, both cameras exactly as during recording, a keyboard on the laptop that runs `lerobot-rollout`. No gamepad, no URDF, no WSL.
- People: one operator who recorded the original demonstrations (same style), one person resetting the bowl.
- Arm time: about 20 minutes for a 10-trial baseline of the trained PICK policy, 45-60 minutes for 30 corrections, 20 minutes for the comparison afterwards.
- GPU time: 1-2 hours of fine-tuning (laptop, or an HF Job from the cloud credit; a paid job is started by a person).
- Before starting: write down 10 fixed ball layouts (photos) so that before and after are tested on the same scenes.

### Stop rule

- **Do not start** if the base PICK policy is below about 3/10 (then the problem is data or setup, and corrections will not fix it) or at 9/10 and above (nothing to gain, something to lose).
- **Do not start** if the round cannot be finished, including the comparison, before the feature freeze (Saturday 21:00). After the freeze the demo runs the base checkpoint.
- **One round, 3 hours wall-clock.** If the fine-tuned policy is not at least 2 successes out of 10 better than the base on the same layouts, keep the base checkpoint and stop. No second round.
- If `lerobot-rollout --strategy.type=dagger` does not work on Windows within 30 minutes (keyboard listener, leader handover), stop and record 20 plain extra demonstrations of the failing layouts with `lerobot-record` instead.

### What to say in the pitch

Say what was done, with the numbers:

> "After training, the robot keeps learning from its own mistakes. We let the policy run, a person takes over only at the moment it fails, and the policy is retrained on exactly those corrections. One round of 30 corrections took an hour and moved the pick success from X/10 to Y/10."

If asked about reinforcement learning:

> "This is interactive imitation learning (DAgger / RaC), the human-in-the-loop half of systems like HIL-SERL and pi*0.6. The reward-driven half is the next step, and our supervisor already produces the signal it needs: a success or failure after every skill, from the gripper reading and the camera."

Do not call it reinforcement learning: there is no reward and no value function in what we train, and a judge who knows the field will check. If the round is skipped or fails, the same second quote still holds as a roadmap statement, without the numbers.

---

## Not verified, or could not open

- No command in this file was run. Whether the DAgger keyboard controls and the leader handover behave on our Windows laptop is untested.
- `pi.website/research/rlt` (HTTP 429), `corentinlger.github.io/posts/lerobot` (404) and `tabithako.github.io/projects/policy-training` (404) could not be opened; statements about them come from search summaries or the arXiv abstract.
- The LeRobot Worldwide Hackathon winner list could not be read.
- Paper numbers (RaC, ResFiT, DSRL, Squint, HIL-SERL, the Stanford autonomous-data study, V-GPS, ConRFT) came through an automated reader of the arXiv pages. Re-check before quoting.
- Whether ResFiT, RLT or ConRFT have released code was not confirmed.
- Whether HIL-SERL's learner alone would run on native Windows (only the actor needs `placo`) was not checked; neither was its VRAM use at batch 256.
- That RA-BC fails with ACT is inferred from the function signature, not from a run.
- All time figures marked "Estimate" are mine. The fine-tuning time on the RTX 4070 laptop depends on tonight's measured training speed.
- No before/after success numbers for ACT plus DAgger on an SO-100/SO-101/Koch were found in any source.
