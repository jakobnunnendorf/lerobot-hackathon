# LeRobot 0.7.0 training feasibility: "cut the correct wire" on SO-101 in 2 days

Researched 2026-11-20 against the v0.7.0 docs (`https://huggingface.co/docs/lerobot/v0.7.0/en/...`) and the `huggingface/lerobot` repo at tag `v0.7.0`. Doc pages and source files were downloaded raw from the tag and read directly; community pages and papers were read through an automated summarizer, so numbers from those are marked where they should be re-checked.

## TL;DR

1. Train **ACT** first. It is the only policy that comfortably trains on a laptop GPU (about 2-6 GB VRAM) in an hour or two; SmolVLA needs about 10-16 GB, and pi0/pi0.5/GR00T need 24-40 GB (cloud only).
2. LeRobot 0.7.0 (released 2026-11-05) changed the workflow: evaluation is `lerobot-rollout`, not `lerobot-record --policy.path`; async inference is replaced by `--inference.type=rtc` (local) or `lerobot-policy-server` plus `--inference.type=remote`; W&B flags are now `--trackio.*`.
3. Bimanual SO-101 is supported in software (`bi_so_follower` + `bi_so_leader`), but a learned bimanual policy is not realistic in 2 days: in a July 2026 SO-101 benchmark, ACT with 50 demos scored 2.5% on bimanual tasks.
4. Do not ask one policy to decide the color. Use a hybrid: classical CV (HSV on a fixed overhead camera) picks the slot, then one of three per-slot, color-agnostic ACT skills does the cut.
5. Data budget: 3 slots x 40 episodes = 120 base episodes, plus about 30 DAgger correction episodes on day 2, leaving 20-50 spare for one stretch experiment.
6. Biggest risks: the capped gripper cannot cut real copper wire, a blade near hands during 150+ resets, lighting drift between day 1 and day 2, and Windows USB/camera/video-decoding friction on a 15-day-old release.

---

## 1. Policies in 0.7.0 and what is trainable in hours

### What ships

`src/lerobot/policies/` at v0.7.0 contains: `act`, `diffusion`, `smolvla`, `pi0`, `pi05`, `pi0_fast`, `groot`, `multi_task_dit`, `xvla`, `vqbet`, `tdmpc`, `evo1`, `eo1`, `wall_x`, `wall_oss_05`, `molmoact2`, `vla_jepa`, `lawam`, `lingbot_va`, `lingbot_vla_v2`, `g05`, `fastwam`, `longwam`, `flux3`, `fineart_vla`, `dm05`, `cosmos3`, `gaussian_actor`, plus `rtc` (real-time chunking helper).
Source: https://github.com/huggingface/lerobot/tree/v0.7.0/src/lerobot/policies

The cheat sheet lists the `--policy.type` values `act`, `diffusion`, `multi_task_dit`, `smolvla`, `pi0`, `pi05`, `pi0_fast`, `groot`, `xvla`.
Source: https://huggingface.co/docs/lerobot/v0.7.0/en/cheat-sheet

GR00T is **N1.7 only** in this release (N1.5 checkpoints are rejected). Source: https://huggingface.co/docs/lerobot/v0.7.0/en/groot

### VRAM and training time (official hardware guide)

Source: https://huggingface.co/docs/lerobot/v0.7.0/en/hardware_guide

| Group | Policies | Peak VRAM (batch 8, AdamW) | Starter GPU per docs |
| --- | --- | --- | --- |
| Light BC | `act`, `vqbet`, `tdmpc` | ~2-6 GB | Laptop GPU (RTX 3060), L4, A10G |
| Diffusion | `diffusion`, `multi_task_dit` | ~8-14 GB | RTX 4070+ / L4 / A10G |
| Small VLA | `smolvla` | ~10-16 GB | RTX 4080+ / L4 / A10G |
| Large VLA | `pi0`, `pi0_fast`, `pi05`, `xvla`, `wall_x` | ~24-40 GB | A100 40 GB+ (24 GB tight at batch 1) |
| Multimodal | `groot`, `eo1` | ~24-40 GB | A100 40 GB+ |

Indicative wall-clock for 5 epochs on ~50 episodes (~45k frames, 640x480), same page:

| Setup | Policy | Batch | Wall-clock |
| --- | --- | --- | --- |
| RTX 4090 / 3090 | `act` | 8 | ~30-60 min |
| RTX 4090 / 3090 | `diffusion` | 8 | ~2-4 h |
| L4 / A10G | `act` | 8 | ~1-2 h |
| L4 / A10G | `smolvla` | 4 | ~3-6 h |
| A100 40 GB | `smolvla` | 16 | ~1-2 h |
| A100 40 GB | `pi0` / `pi05` | 4 | ~4-8 h |

The guide says imitation learning "typically converges in 5-10 epochs over the dataset" and gives the formula:

```text
total_frames    = sum of frames over all episodes      # 50 ep x 30 fps x 30 s ~ 45,000
steps_per_epoch = ceil(total_frames / (num_gpus x batch_size))
total_steps     = epochs x steps_per_epoch
```

It also warns: if you shorten training, shorten the LR schedule with `--policy.scheduler_decay_steps` close to `--steps`, and adjust `--save_freq` (default 20,000, so a 20k run saves only one checkpoint).

### Per-policy summary

| Policy | Language-conditioned | Typical episodes (docs) | Training guidance (docs) | Laptop-GPU verdict |
| --- | --- | --- | --- | --- |
| ACT | No. Docs: "`--task` can be skipped for ACT" | "Often achieves high success rates with just 50 demonstrations" | "a few hours for 100k training steps on a single GPU"; batch 8; defaults `chunk_size=100`, `n_action_steps=100`, ResNet-18 | Yes. Primary choice |
| Diffusion | No | No dedicated doc page in 0.7.0 | ~2-4 h on a 4090 for 5 epochs; defaults `horizon=64`, `n_action_steps=32`, `n_obs_steps=2` | Only with 12 GB+ VRAM; otherwise cloud |
| SmolVLA | Yes | "~50 episodes"; docs report 25 episodes "was not enough" | "20k steps will roughly take ~4 hrs on a single A100"; 450M params; defaults `freeze_vision_encoder=True`, `train_expert_only=True` | Tight on 8 GB; use HF Jobs or a 16 GB+ GPU |
| Multitask DiT | Yes (frozen CLIP text encoder) | Not stated | ">30k steps recommended for a single task", batch 192-320 recommended | No (batch requirements) |
| pi0 / pi0.5 | Yes | Not stated | 24-40 GB; pi0.5 reference run 5,000 steps at batch 32 took 3 h 41 min on 4x H100 | Cloud only |
| GR00T N1.7 | Yes | Not stated | Docs example: `--batch_size=64 --steps=20000`, 3B base model | Cloud only |
| X-VLA | Yes | Not stated | Has an `so101_bimanual` action mode (12 real dims padded to 20) | Cloud only |

Sources: https://huggingface.co/docs/lerobot/v0.7.0/en/act , https://huggingface.co/docs/lerobot/v0.7.0/en/smolvla , https://huggingface.co/docs/lerobot/v0.7.0/en/multi_task_dit , https://huggingface.co/docs/lerobot/v0.7.0/en/pi05 , https://huggingface.co/docs/lerobot/v0.7.0/en/groot , https://huggingface.co/docs/lerobot/v0.7.0/en/xvla , and the config files under https://github.com/huggingface/lerobot/tree/v0.7.0/src/lerobot/policies

**Inference rate on SO-101:** the v0.7.0 docs do not publish per-policy inference rates. What the source does show: ACT predicts a 100-step chunk and by default executes all 100 steps (3.3 s at 30 fps) before re-planning, so inference cost is negligible; SmolVLA predicts 50 steps per call and the docs recommend RTC "when inference is slow". `lerobot-rollout` prints the achieved rate itself (see cadence reporting in section 5).

### Independent evidence at 50 demos per task (ArmnetBench v0.1, SO-101)

Paper: https://arxiv.org/abs/2607.24481 . Seven policies, 12 real SO-101 tasks (8 single-arm, 4 bimanual), 50 teleoperated demos per task, 2,518 human-scored rollouts. Numbers below were transcribed by an automated reader from the HTML tables; re-check before citing.

| Policy | Strict success, all 12 tasks | Bimanual tasks |
| --- | --- | --- |
| pi0.5 | 47.6% | 52.1% |
| pi0 | 35.1% | not broken out |
| GR00T N1.7 | 29.4% | not broken out |
| Diffusion | 26.7% | 35.8% |
| ACT | 19.2% | 2.5% |
| MolmoAct 2 | 18.9% | not broken out |
| SmolVLA | 15.0% | not broken out |

ACT per-task results ranged from 0% (block_stack, cable_clip, eye_drops_to_shelf) to 50-63% (ring_insert, cable_unclip, eye_drops_to_basket). `cable_clip` was 0% for all seven policies. The authors say the leaderboard "does not estimate each method's capability ceiling". Reading for this project: 50 demos is not a guarantee for any policy, cable-like precision is hard, and reducing task difficulty (fixed jig, short motion) matters more than policy choice.

A Hugging Face forum thread on ACT with ~50 episodes reports the same spread: original ACT/ALOHA 64-96% across four tasks, one SO-ARM101 user at 60% strict success over 20 trials with 60 demos, and several poor results traced to moved cameras, calibration differences, and a recording loop running below target FPS.
Source: https://discuss.huggingface.co/t/what-is-the-expected-success-rate-for-a-act-trained-model-using-50-episodes/178517

### Exact commands (verified in v0.7.0 docs)

Train ACT (https://huggingface.co/docs/lerobot/v0.7.0/en/il_robots):

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/record-test_YYYYMMDD_HHMMSS \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --trackio.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

Flags confirmed in `src/lerobot/configs/train.py`: `--steps` (default 100,000), `--batch_size` (default 8), `--save_freq` (default 20,000), `--num_workers` (default 4), `--log_freq` (default 200), `--resume`, `--save_checkpoint_to_hub`. `--policy.push_to_hub=false` skips the Hub upload. `--dataset.image_transforms.enable=true` turns on image augmentation (used in the GR00T and EVO1 doc commands).

Resume training:

```bash
lerobot-train \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

Fine-tune SmolVLA (https://huggingface.co/docs/lerobot/v0.7.0/en/smolvla):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --policy.repo_id=${HF_USER}/my_smolvla \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --trackio.enable=true
```

The `--rename_map` is required: `smolvla_base` expects cameras named `camera1`, `camera2`, `camera3`, and the same map must be passed at rollout.

Train on Hugging Face Jobs instead of the laptop (same page as ACT):

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/record-test_YYYYMMDD_HHMMSS \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/my_policy \
  --job.target=a10g-small
```

Flavors per the hardware guide: `t4-small`/`t4-medium` (ACT only), `l4x1`/`l4x4`, `a10g-small/large/largex2/largex4`, `a100-large`. List with `hf jobs hardware`.

Evaluate on the robot. In 0.7.0 `lerobot-record` is "a pure data-collection tool - no policy inference" (docstring of `lerobot_record.py`); use `lerobot-rollout`:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=${HF_USER}/my_policy \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AB90687491 \
  --robot.id=my_follower_arm \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --task="Grab the black cube" \
  --duration=60
```

Record evaluation episodes (mirrors the old `lerobot-record` eval behaviour; dataset name must start with `rollout_`):

```bash
lerobot-rollout \
    --strategy.type=episodic \
    --policy.path=${HF_USER}/my_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=${HF_USER}/rollout_eval_data \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Pick up the red cube"
```

Asynchronous inference. The release notes state `lerobot.async_inference` and the `lerobot[async]` extra were removed (https://github.com/huggingface/lerobot/releases/tag/v0.7.0). Two replacements:

Local RTC, same machine, no server (compatible with pi0, pi0.5, SmolVLA; https://huggingface.co/docs/lerobot/v0.7.0/en/inference):

```bash
lerobot-rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --inference.rtc.max_guidance_weight=10.0 \
    --policy.path=${HF_USER}/pi0_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="Pick up the cube" \
    --duration=60 \
    --device=cuda
```

Remote policy server (needs the `remote` extra on both sides; https://huggingface.co/docs/lerobot/v0.7.0/en/remote_inference). The docs say ACT, SmolVLA, XVLA, EVO1 and LaWAM "have run on real robots through this path", and ACT needs temporal ensembling disabled:

```bash
lerobot-policy-server --config_path=server.yaml

lerobot-rollout \
  --inference.type=remote \
  --inference.endpoint=tcp/127.0.0.1:7447 \
  --inference.deployment=manipulation \
  --inference.mode=chunk \
  --inference.chunk_merge=aligned \
  --inference.semantics=so101-degrees-gripper-percent-v1 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.use_degrees=true \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --fps=30 \
  --task="pick up the cube" \
  --interactive=true
```

`server.yaml` starts from `examples/remote_inference/server.yaml` in the repo. The docs prefix these with `uv run --no-sync`; with a pip install the bare entry points should work (both are registered in `pyproject.toml`).

---

## 2. Bimanual support in 0.7.0

### Type names (verified in source)

- Robot: `bi_so_follower` - `src/lerobot/robots/bi_so_follower/config_bi_so_follower.py`
- Teleoperator: `bi_so_leader` - `src/lerobot/teleoperators/bi_so_leader/config_bi_so_leader.py`

There is no `bi_so100_follower`, `bi_so101_follower` or `bi_so100_leader` in the v0.7.0 tree. Those names appear in older tutorials (for example XLeRobot docs) and in a third-party PyPI package (`lerobot-robot-bi-so101-follower`); do not mix them with the built-in type.

Config fields: `left_arm_config` and `right_arm_config` (each with `port`, `cameras`, `max_relative_target`, `disable_torque_on_disconnect`, `use_degrees`, PID coefficients, `num_read_retries`) and a top-level `cameras` dict. Per-arm camera keys get a `left_` / `right_` prefix; top-level camera keys stay unprefixed. Motor keys become `left_*` and `right_*`, giving a 12-dimensional state and action. Calibration files are looked up as `<robot.id>_left` and `<robot.id>_right`.

There is no bimanual page in the v0.7.0 docs for SO arms. The only official examples are in script docstrings. From `src/lerobot/scripts/lerobot_record.py`:

```bash
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \
  --dataset.num_episodes=25 \
  --dataset.single_task="Grab and handover the red cube to the other arm" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
```

`lerobot_teleoperate.py` has the matching `lerobot-teleoperate` example. `bi_so_leader` is listed as a compatible teleoperator for DAgger in https://huggingface.co/docs/lerobot/v0.7.0/en/hil_data_collection .

### Operating it

- One person operates both leaders, one hand each. Neither the docs nor the Seeed guide describe a two-operator workflow; two people on one episode produce less consistent demonstrations, which hurts imitation learning.
- Calibrate each arm individually, then make sure the files carry the `_left` / `_right` suffix; the Seeed guide notes teleop asks for recalibration otherwise, and that you can copy the JSON files instead.
  Source: https://wiki.seeedstudio.com/lerobot_double_arm_so_arm_training/ (targets an older LeRobot; its `--wandb.*`, `--dataset.vcodec` and `lerobot-record --policy.path` usage is outdated for 0.7.0)
- Hardware load: four serial buses and three or four cameras on one laptop. USB bandwidth is the usual limit; a forum thread cites about 147 Mbps for uncompressed YUYV versus about 9 Mbps for MJPG per 640x480 camera, so set `fourcc: "MJPG"` and use a powered hub.

### Known issues

- Issue #3244 (closed): `--robot.cameras` was not supported for `bi_so_follower` in `lerobot-record`. The top-level `cameras` field exists in the v0.7.0 config, so this appears fixed.
- PR #4449 (open as of the search): depth keys of top-level cameras get wrongly prefixed in bimanual robots. Only matters with depth cameras.
- Remote inference and RTC hold behaviour: the docs say "bimanual wrappers depend on their configuration". In source, `bi_so_follower` declares position-hold support when both arms do.

### How much harder is bimanual?

- ArmnetBench (50 demos per task, bimanual SO-101): ACT 2.5% average over four bimanual tasks (0%, 0%, 0%, 10%); Diffusion 35.8%; pi0.5 52.1%. "Hold the lamp with the left arm and open the door with the right gripper" - structurally the closest to hold-and-cut - scored ACT 0%, Diffusion 73%, pi0.5 23%.
- The policies that did tolerably on bimanual (Diffusion, pi0.5) are the ones that do not fit a typical laptop GPU.
- No source found gives a clean "bimanual needs N times more data" figure.

### Verdict

Recording a bimanual dataset in 2 days is feasible. Getting a working learned bimanual policy in 2 days is not something the evidence supports, especially with ACT on a laptop. If two arms matter for the demo, make the second arm a holder that is not learned: record one "present the device" trajectory and replay it, or hold a fixed pose, while the single-arm policy does the cut (details in the plan below).

---

## 3. Data collection practice

### Official recommendations

- "We suggest recording at least 50 episodes, with 10 episodes per location. Keep the cameras fixed and maintain consistent grasping behavior." Also: "Avoid adding too much variation too quickly, as it may hinder your results." (https://huggingface.co/docs/lerobot/v0.7.0/en/il_robots)
- SmolVLA page: 50 episodes over 5 positions (10 each) worked; "a similar dataset with 25 episodes ... was not enough".
- Cameras (https://huggingface.co/docs/lerobot/v0.7.0/en/cameras): light evenly and avoid direct sunlight and moving shadows ("Lighting that differs between recording and deployment can make a policy fail"); mount rigidly and never move between recording and deployment; object, gripper and target visible in at least one camera; "640x480 at 30 fps is enough for most tasks".
- Rule of thumb from the tutorial: you should be able to do the task yourself looking only at the camera images.
- Dataset blog (https://huggingface.co/blog/lerobot-datasets): preferably two views; keep the leader arm and human limbs out of frame; steady cameras; neutral stable lighting; ~30 fps; task strings of 25-50 characters, specific, never empty or generic; name cameras by location (`top`, `front`, `wrist`).
- Camera names become dataset keys (`observation.images.wrist`), and the policy expects the same names and resolution at deployment.

### Recording command and controls

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
```

(The docs version of this command contains two `--robot.cameras` lines; the first, a single 1920x1080 `front` camera, looks like a leftover. Use one.)

- Defaults (`src/lerobot/configs/dataset.py`): `--dataset.fps=30`, `--dataset.episode_time_s=60`, `--dataset.reset_time_s=60`, `--dataset.num_episodes=50`, `--dataset.push_to_hub=true`, `--dataset.streaming_encoding=false`.
- Keys: Right Arrow or `n` ends the episode early and moves on; Left Arrow or `r` discards and re-records; Escape or `q` stops, encodes and uploads.
- **Timestamped names:** `lerobot-record` appends `_YYYYMMDD_HHMMSS` to `--dataset.repo_id`. Use the full stamped name everywhere afterwards, or pass `--dataset.no_stamp=true`.
- **Resume:** re-run the same command with `--resume=true`, the full stamped repo id, and `--dataset.root="local_path"` (required). `--dataset.num_episodes` is then the number of additional episodes, not the total.
- **Cadence report:** each episode prints the achieved loop rate. Frame timestamps are derived from the frame index, so a session that really ran at 25 Hz still claims 30 fps and the recorded motion is faster than reality. Check this line before training on anything.
- Suggested for this task: `--dataset.episode_time_s=20 --dataset.reset_time_s=10`, ending episodes early with Right Arrow.

### Handling variation for the wire task

- Fix the mock device in a jig (screwed or clamped, with tape outlines). Every degree of freedom removed from the scene is data you do not have to record.
- If positions must vary, follow the 10-episodes-per-position rule and add positions only after the fixed case works.
- Colors: for the hybrid design below, shuffle wire colors across slots every 5 or so episodes so each per-slot skill becomes color-agnostic.
- Lighting: bring your own lamp and shade the scene from windows. Day 1 afternoon and day 2 morning will otherwise look different. Train with `--dataset.image_transforms.enable=true`.
- Demonstration style: one operator, one consistent strategy, no pauses or hesitations, same start pose, stop the episode right after the cut.

### Common failure causes (from the sources above)

Camera bumped between recording and deployment; different `--robot.id` or recalibration between record and rollout; loop running below target FPS; USB bandwidth frame drops; leader arm or hands in frame; inconsistent or hesitant demos; different camera names, resolution or task string at rollout; missing `--rename_map` at SmolVLA rollout; LR schedule not shortened for short VLA runs.

### Dataset tools in 0.7.0

`lerobot-edit-dataset` exists (entry point in `pyproject.toml`; docs at https://huggingface.co/docs/lerobot/v0.7.0/en/using_dataset_tools). Operations: `delete_episodes`, `split`, `merge`, `remove_feature`, `modify_tasks`, `convert_image_to_video`, `reencode_videos`, `info`.

```bash
# Delete bad episodes into a new dataset (keeps the original)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --new_repo_id lerobot/pusht_after_deletion \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"

# Merge datasets (identical features, robot, cameras and fps required)
lerobot-edit-dataset \
    --new_repo_id lerobot/pusht_merged \
    --operation.type merge \
    --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"

# Relabel task strings (in place)
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type modify_tasks \
    --operation.episode_tasks '{"0": "Task A", "1": "Task B", "2": "Task A"}'
```

Add `--push_to_hub true` to upload the result. To train on a subset without editing, `DatasetConfig` has `episodes` and `exclude_episodes` fields (so `--dataset.episodes="[...]"`; field verified, CLI form not seen in the docs).

Push and visualize:

```bash
hf upload ${HF_USER}/record-test_YYYYMMDD_HHMMSS ~/.cache/huggingface/lerobot/${HF_USER}/record-test_YYYYMMDD_HHMMSS --repo-type dataset

lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

Browser visualizer: `https://huggingface.co/spaces/lerobot/visualize_dataset?path=/<repo_id>/episode_<index>` (sign in for private datasets). `lerobot-replay` replays a recorded episode on the robot:

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/record-test_YYYYMMDD_HHMMSS \
    --dataset.episode=0
```

---

## 4. The "which wire?" decision

### a. End-to-end single policy that learns color from pixels (ACT, one task string)

- Mapping: one dataset, `--policy.type=act`, target is always the same color while its slot changes.
- Pros: no extra code; one model.
- Cons: ACT has no language input, so the target color is baked in at training time; you cannot ask for a different wire at demo time. The policy must learn "find green, then go there" from about 40 episodes per slot with nothing forcing it to attend to color. A September 2026 paper studying ACT reports that visually similar distractors cause "incorrect object or destination selection even when the underlying manipulation skill remains intact" (https://arxiv.org/abs/2609.05376 ; abstract only, no numbers retrieved).
- Verdict: plausible but fragile, and a wrong cut is the worst possible demo failure.

### b. Language-conditioned VLA ("cut the green wire")

- Mapping: record one dataset per instruction with different `--dataset.single_task` strings, merge with `lerobot-edit-dataset --operation.type merge`, fine-tune `lerobot/smolvla_base`, pass the string via `--task` at rollout. `--interactive=true` adds `/subtask <text>` to switch instructions without reloading.
- Evidence against, for small data:
  - LIBERO-Plus reports models "tend to ignore language instructions completely" (https://arxiv.org/abs/2510.13626); LangGap, which keeps the layout fixed and varies only the instruction, reports that current VLAs "largely ignore language instructions" (https://arxiv.org/abs/2603.00592). A fixed wire jig where only the instruction differs is exactly that setting.
  - ArmnetBench: SmolVLA 15.0% strict success at 50 demos per task, the lowest of seven policies.
  - A community four-task SO-101 SmolVLA card (930 episodes, 200k steps on an L40S) describes the result as "OK-ish", says SmolVLA "appears capacity-limited", and that "Pi0.5 on the same dataset works really well" (https://huggingface.co/Project-IRA/TPSoSe2026_SmolVLA_LeRobot_SO101_Finetuning_V6_Full). No success rates given.
  - Community multi-task SmolVLA cards use 200-300 episodes per instruction and warn that strings must match training verbatim (https://huggingface.co/Harrysunshine/so101-smolvla-9task ; not validated on hardware).
- No source found shows SmolVLA reliably respecting color instructions with ~50 episodes per instruction.
- Verdict: does not fit the laptop comfortably and the evidence says language following is the weak point. Stretch experiment only.

### c. Hybrid: CV picks the target, a learned skill executes (recommended)

- Mapping: a fixed overhead camera and a fixed jig give each wire slot a known image region. A short OpenCV script reads the mean hue in each region and returns the slot holding the requested color. It then launches the matching per-slot ACT policy with `lerobot-rollout --strategy.type=base --policy.path=<slot_k_policy> --duration=<s>`.
- Each per-slot policy is trained only on "cut the wire in slot k" with shuffled colors, so it has no decision to make.
- Pros: the decision is deterministic, debuggable and testable without the robot; data per skill follows the documented 50-episode recipe; no LeRobot internals to modify; the requested color can change at demo time; a wrong cut can only come from the CV step, which you can verify on screen before the arm moves.
- Cons: N policies to train (same total frames as one policy); scales linearly with slot count; switching policies means reloading a checkpoint (seconds); on Windows the CV script must release the camera before `lerobot-rollout` opens it, or use a separate camera.
- Variants:
  - VLM instead of HSV. 0.7.0 has a built-in external planner: `--planner.model_id`, `--planner.api_base`, and `--planner.instructions='["...","..."]'`, where "the VLM is asked to pick one of these strings". It only steers language-conditioned policies, so it combines with option b, not with per-slot ACT.
  - IK pre-positioning then learned finish. `lerobot.model.kinematics` and `examples/so100_to_so100_EE/` exist, but they depend on `placo`, and hand-eye calibration is a day of work. Not for this timeline.
  - Open-loop replay per slot (`lerobot-replay`). With a rigid jig this may simply work and is the safety-net demo.
- Related precedent: an SO-101 scissors project wrapped its imitation policy in a five-state machine with two YOLO classifiers verifying each step and per-state timeouts, after the robot "became overeager and made repeated cuts" (https://jaredlbailey.github.io/2025/11/01/Imitation-Learning.html).

### d. Visual prompting (marker or overlay on the camera image)

- Mapping: draw a marker on the overhead image at the target wire during both recording and rollout, and train one ACT policy on all slots. In 0.7.0 this needs a custom observation processor step so the overlay lands in the recorded frames (`robot_observation_processor` in `record_loop` and `build_rollout_context`). A zero-code physical version: an LED or pointer next to the target slot, switched by the CV script.
- Evidence: AimBot (overlay cues improve visuomotor policies, https://arxiv.org/abs/2508.08113), ARRO (masking distractors, tested with Diffusion Policy and pi0, https://arxiv.org/abs/2505.08627), Visual Attentive Prompting (training-free highlighting for frozen VLAs, https://arxiv.org/abs/2512.20014), and the ACT distractor paper above lists appearance-based visual prompting among fixes that "substantially improve robustness". None of these was confirmed to report results at the 100-200 episode scale.
- Pros: one policy shares approach-and-cut data across slots. Cons: custom code inside the record and rollout loops on a new release; an overlay bug silently poisons the whole dataset.
- Verdict: good second step if c works early. The physical LED version is the low-risk way to try it.

---

## 5. Other 0.7.0 features useful in a hackathon

- **DAgger-style corrections** (`--strategy.type=dagger`): run the policy, press Space to pause, Tab to take over with the leader, Tab then Space to hand back. The leader arm moves to match the follower on pause, so `so101_leader` works. Corrections-only mode records each correction as an episode. Source: https://huggingface.co/docs/lerobot/v0.7.0/en/hil_data_collection

```bash
lerobot-rollout \
    --strategy.type=dagger \
    --strategy.record_autonomous=true \
    --strategy.num_episodes=50 \
    --policy.path=${HF_USER}/my_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=${HF_USER}/rollout_dagger_data \
    --dataset.single_task="Grasp the block"
```

  Then remove the extra column, merge, and fine-tune:

```bash
lerobot-edit-dataset \
    --repo_id ${HF_USER}/rollout_hil_dataset_YYYYMMDD_HHMMSS \
    --new_repo_id ${HF_USER}/hil_dataset_no_intervention \
    --operation.type remove_feature \
    --operation.feature_names "['intervention']"

lerobot-edit-dataset \
    --new_repo_id ${HF_USER}/demo_plus_hil \
    --operation.type merge \
    --operation.repo_ids "['${HF_USER}/demo-dataset', '${HF_USER}/hil_dataset_no_intervention']"

lerobot-train \
    --dataset.repo_id=${HF_USER}/demo_plus_hil \
    --policy.path=outputs/pretrain/checkpoints/last/pretrained_model \
    --policy.repo_id=${HF_USER}/act_hil_finetune \
    --output_dir=outputs/hil_finetune \
    --steps=20000
```

- **HIL-SERL** (`lerobot[hilserl]`, https://huggingface.co/docs/lerobot/v0.7.0/en/hilserl): real-robot RL with interventions. Needs a URDF, the kinematics package, a reward classifier or manual success labels, ideally a gamepad, and an actor-learner setup. Recommended for 5-10 s tasks. Too much setup for a first 2-day build.
- **RTC and remote inference:** only relevant if you deploy SmolVLA or a pi model. ACT runs synchronously.
- **Interactive sessions** (`--interactive=true`, base and sentry strategies): robot idles until `/start`; `/reset` returns home; `/subtask` changes the instruction. Good for demos because the policy stays loaded between attempts.
- **Programmatic control:** `lerobot.rollout.RolloutController` exposes `start()`, `reset()`, `stop()`, `set_task()` for wrapping a rollout in your own script.
- **Cadence and health logging:** `lerobot-rollout` logs loop Hz, inference latency and GPU memory every 5 s (`--status_interval_s`).
- **Pretrained checkpoints and datasets:** `lerobot/smolvla_base` (pretrained on 487 community datasets "focused on the SO100", ~10M frames, per https://huggingface.co/blog/smolvla), `lerobot/pi0_base`, `lerobot/pi05_base`, `nvidia/GR00T-N1.7-3B`, `lerobot/xvla-base`; datasets `lerobot/svla_so101_pickplace`, `lerobot/svla_so100_pickplace`, and the ArmnetBench SO-101 sets (`armnet/armnetbench_v01_lerobot_so101`, `armnet/armnetbench_v01_lerobot_bimanual_so101`). No pretrained ACT checkpoint transfers to a new scene; ACT trains from scratch.
- **LeLab:** browser GUI for SO-101 calibrate/record/train/rollout (`uv tool install git+https://github.com/huggingface/leLab.git && lelab`). Compatibility with a pinned 0.7.0 environment was not checked.
- **Sim:** LeIsaac (IsaacLab) has SO-101 tasks including bimanual ones, but its documented setup pins `lerobot==0.4.1` and needs Isaac. `lerobot-eval` is for simulation benchmarks. Neither helps a 2-day real-robot build.
- **PEFT/LoRA:** `--peft.method_type=LORA` (or `--policy.use_peft=true`) for SmolVLA and pi models; reduces trainable parameters, not a fix for laptop VRAM by itself.

### Windows-specific notes

- **Ports:** `lerobot-find-port` uses pyserial `list_ports.comports()` on Windows (verified in source), so ports are `COM5`-style: `--robot.port=COM5`.
- **Shell:** doc commands use bash line continuations and `${HF_USER}`. Run them in Git Bash, or in PowerShell replace `\` with a backtick and the variable with a literal username. JSON arguments such as `--rename_map='{"a": "b"}'` are the usual place PowerShell quoting breaks.
- **Cameras:** `camera_opencv.py` sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` on Windows before importing cv2 and applies FOURCC after size and fps on Windows. The camera config has `backend` (`Cv2Backends`: `ANY=0`, `DSHOW=700`, `MSMF=1400`; default `ANY`) and `fourcc` fields. Camera indices can change on re-plug; re-run `lerobot-find-cameras opencv` after any USB change. The Windows Camera app holds cameras exclusively; close it.
- **PyTorch:** the install page says Windows uses the PyPI default wheels and skips the CUDA section. `pyproject.toml` requires `torch>=2.7,<2.16`. Keeping the team's CUDA 12.8 wheels is consistent with that range; confirm with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` after any `pip install` that touches lerobot extras, because pip can silently replace the CUDA build with a CPU one.
- **Video decoding:** `pyproject.toml` pins `torchcodec>=0.7.0,<0.18.0` on win32 and notes it "needs torch>=2.8". TorchCodec needs FFmpeg shared libraries it can load; forum reports of Windows DLL load failures are common (https://discuss.huggingface.co/t/cannot-load-torchcodec/169260). LeRobot falls back automatically: `get_safe_default_video_backend()` returns `pyav` with a warning when TorchCodec cannot load FFmpeg. You can force it with `--dataset.video_backend=pyav` (flag used in the GR00T docs).
- **Video encoding while recording:** default codec is `libsvtav1` (CPU heavy). `--dataset.rgb_encoder.vcodec=auto` probes for hardware encoders; `h264_nvenc` is listed for Linux/Windows with an NVIDIA GPU. Low-resource fallback from the docs: `--dataset.rgb_encoder.vcodec=h264 --dataset.streaming_encoding=false`.
- **Extras:** the extra is named `core_scripts` in `pyproject.toml`; pip normalizes `core-scripts` to the same thing. `feetech` now installs `rustypot` instead of the Feetech SDK. `libero` is Linux-only.
- **WSL:** USB cameras through WSL are a known pain point (forum threads); stay on native Windows.

---

## 6. Practical cutting on an SO-101

### Servo and gripper limits

- The follower uses 6x STS3215 with 1/345 gearing (https://huggingface.co/docs/lerobot/v0.7.0/en/so101).
- Retailer specs: 7.4 V version 16.5 kg.cm at 6 V and 19.5 kg.cm at 7.4 V stall, rated load 5 kg.cm; 12 V version 30 kg.cm stall, rated load 10 kg.cm, stall current 2.7 A (https://www.dfrobot.com/product-2961.html , https://www.dfrobot.com/product-2962.html , https://www.robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215). Sources disagree slightly on which voltage the 19.5 figure belongs to.
- **LeRobot caps the gripper servo.** `so_follower.py` at v0.7.0 writes, for the gripper motor only: `Max_Torque_Limit = 500` ("50% of max torque to avoid burnout"), `Protection_Current = 250` ("50% of max current"), `Overload_Torque = 25` ("25% torque when overloaded"). Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/so_follower/so_follower.py
- My estimate, not a measured value: half of stall torque is about 10 kg.cm (7.4 V) or 15 kg.cm (12 V). With a jaw of roughly 8-10 cm that is on the order of 10-18 N at the tip, briefly, and much less sustained.
- **Payload and repeatability:** TheRobotStudio publishes no rated payload or repeatability. Third-party figures conflict (about 150 g to about 500 g). Assume millimetre-level, not sub-millimetre, positioning from hobby servos with gear backlash and printed links, and measure it: replay one episode ten times with `lerobot-replay` onto a paper target.

### Can it cut?

- Precedent: an SO-101 with ordinary scissors on a custom wrist mount, handles driven by the gripper, cut soft food; the author reports it "couldn't cut harder foods reliably" and switched to softer items (https://jaredlbailey.github.io/2025/11/01/Imitation-Learning.html). A Hackster project ("Cutana") uses SO-101 arms for vegetable chopping; details not verified.
- No project was found cutting electrical wire with an SO-100/SO-101.
- My estimate: shearing 22 AWG copper (0.33 mm2 at roughly 200 MPa shear strength) needs about 65 N at the blade. Flush cutters give perhaps 5-8x leverage, so about 8-13 N at the handle ends, which is right at the capped gripper's limit before insulation and friction. Treat real copper wire as not feasible.

### Safer and more reliable "wires" (engineering suggestions, untested)

| Option | Mechanism | Notes |
| --- | --- | --- |
| Soft solder wire or thin yarn in flush cutters or scissors held by the gripper | Gripper closes the tool | Solder is far softer than copper; closest to a real "snip". Tool needs a spring return and a printed mount |
| Paper or ribbon strips with scissors | Same | Proven tool approach; wide strips are forgiving of position error |
| Magnetic breakaway wires | Arm hooks or pushes the wire until a magnet joint separates | No blade at all. Add a continuity circuit: correct wire lights green, wrong wire triggers the "bomb" |
| Brittle "wires" (dry spaghetti, pencil lead) | Arm pushes sideways until it snaps | No blade, clear visual result, cheap resets |
| Tensioned thread and a blade on the gripper | Arm drags blade across thread | Works only with good tension and a sharp exposed blade |
| Hot wire (nichrome) through nylon thread | Melts | Burn and fume risk at an event; not recommended |

### Blade safety

A learned policy moves unpredictably, and someone reaches into the workspace to reset the scene 150 or more times. If a blade is used: guard it so only a slot is exposed, set `--robot.max_relative_target=<deg>` (field verified in `SOFollowerConfig`) to limit per-step motion, put arm power on a switch within reach, wear eye protection, and never reset while a rollout process is alive. Note `disable_torque_on_disconnect` defaults to true, so the arm drops when a script exits.

---

## Recommended technical plan for 2 days

### Architecture

1. **Scene:** mock device rigidly fixed in a jig with 3 wire slots, own lamp, top camera plus wrist camera at 640x480, 30 fps, MJPG.
2. **Selector:** OpenCV script on the top camera; fixed region per slot; mean hue decides which slot holds the requested color; shows the decision on screen.
3. **Skills:** three per-slot ACT policies, each trained on "cut the wire in slot k" with wire colors shuffled.
4. **Executor:** script launches `lerobot-rollout --strategy.type=base --policy.path=<slot_k> --duration=25`, then re-reads the camera to verify the cut.
5. **Fallback:** one clean recorded episode per slot, replayable with `lerobot-replay`.
6. **Second arm (optional):** not learned. Replay a "present the device" episode with `--robot.disable_torque_on_disconnect=false` so it keeps holding (untested), and record all cutting data with that arm in place.

### Data budget (about 150 of the 200-episode ceiling)

| Block | Episodes | Purpose |
| --- | --- | --- |
| Pipeline smoke test | 10 (throwaway) | Record, train 3-5k steps, rollout once. Proves the whole chain on Windows before real data |
| Slot 1 / 2 / 3 base | 40 + 40 + 40 | Separate datasets, `--dataset.single_task="cut the wire in the left slot"` etc., colors shuffled every 5 episodes |
| DAgger corrections | ~10 per slot = 30 | Day 2, targeted at observed failures |
| Reserve | 20-50 | One stretch experiment (LED visual prompt, or SmolVLA on the merged set via HF Jobs) |

Size check: 40 episodes x ~15 s x 30 fps is about 18k frames per slot. Ten epochs at batch 8 is about 22k steps per policy, about 67k steps for all three. From the docs' anchor (4090: ~28k steps in 30-60 min), expect roughly 1-3 h per policy on a laptop GPU; this is an extrapolation and depends on the actual card. Set `--save_freq=5000` and compare checkpoints on the robot rather than trusting the loss.

### Schedule

- **Day 1 morning:** calibrate, fix cameras and jig, build the cutting tool, write the HSV selector. Run the 10-episode smoke test end to end, including `lerobot-rollout`. Replay an episode ten times to measure repeatability.
- **Day 1 afternoon:** record 120 base episodes with one operator. Review a sample in the visualizer and delete bad ones. Check every cadence line.
- **Day 1 evening and night:** train the three policies (sequentially on the laptop, or in parallel with `--job.target=a10g-small`).
- **Day 2 morning:** 10 rollouts per slot, tally successes. DAgger corrections on the weakest slots, merge, fine-tune 10-20k steps from the existing checkpoint.
- **Day 2 afternoon:** integrate selector and executor, add the verify step, rehearse the demo, freeze by mid-afternoon. Stretch experiment only if all three slots already work.

### Decision points

- Smoke test fails by day 1 lunch: stop adding features and fix the pipeline.
- A slot is below about 6/10 after DAgger: use replay for that slot in the demo.
- All slots at 8/10 or better by day 2 noon: try the LED visual-prompt single policy or the SmolVLA stretch.

---

## Unverified

- **No command in this document was executed.** Flags are quoted from the v0.7.0 docs or read from config dataclasses; behaviour on this Windows machine is untested.
- ArmnetBench per-policy and per-task numbers, and the forum, model-card and blog details, came through an automated page reader. Re-check against the originals before quoting them to judges.
- Per-policy inference rates on SO-101: not published in the v0.7.0 docs.
- Laptop training-time estimates: extrapolated from desktop and cloud figures; the team's GPU model and VRAM are unknown to me.
- `--dataset.episodes` / `--dataset.exclude_episodes` CLI form: fields exist in `DatasetConfig`; exact CLI syntax not seen in docs.
- How to write the camera `backend` value inline (`backend: 700` versus `backend: DSHOW`): field and enum verified, parsing form not.
- Whether `libsvtav1` and `h264_nvenc` are available in the PyAV build installed on Windows.
- Whether `placo` (needed by `kinematics` and `hilserl`) and the `groot` extra install and run on native Windows.
- Whether the installed CUDA 12.8 torch build is 2.8 or newer (required for TorchCodec on Windows per `pyproject.toml`).
- Windows dataset cache path: docs give `~/.cache/huggingface/lerobot/{repo-id}`; the Windows equivalent under the user profile is assumed.
- Second arm holding its pose after `lerobot-replay` exits with `--robot.disable_torque_on_disconnect=false`, and servo heating while holding.
- Whether `bi_so_follower` issue #3244 is fully fixed in 0.7.0 (inferred from the config, not from the issue thread).
- Gripper force, wire shear force, payload and repeatability figures are my estimates or conflicting third-party numbers. No official SO-101 figures exist.
- All "safer wire" options are suggestions; none was found tested on an SO-101.
- LeLab compatibility with a pinned 0.7.0 pip environment.
- No community evidence was found, either way, for SmolVLA following color instructions at ~50 episodes per instruction on SO-101.
