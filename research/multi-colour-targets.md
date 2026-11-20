# Picking a ball of a commanded colour: ways to get from "always the striped ball" to "pick the red ball"

Researched 2026-11-20. LeRobot claims come from reading the `huggingface/lerobot` source and docs at tag `v0.7.0` (commit `410c095`), downloaded with the GitHub CLI. Papers and blog posts were read through an automated page reader unless marked "PDF read directly". Nothing was installed or run; see "Unverified" at the end.

Setting assumed throughout: SO-101, one ACT policy per skill in the team's own 30 Hz loop, top + wrist camera at 640x480, inference on an 8 GB RTX 4070 laptop, a colour detector already running on the top camera, 60-120 striped-ball episodes being recorded tonight.

## TL;DR

1. No option is proven at this data scale. The best-supported direction is to make the target visible to the policy in the image (visual prompt), not to tell it the colour through a number or a sentence: a bounding box given as four numbers "was ignored" by a diffusion policy while the same box as a mask image gave 89% correct picks, and small language models mostly did not follow colour or target changes in the sources I found.
2. Recommended: keep the striped-ball ACT as the demo backbone, and build one colour-agnostic "pick the marked ball" ACT by rewriting the camera images offline (target painted one fixed colour, everything else grey). Tonight's episodes are reusable for this; LeRobot 0.7.0 has no tool that rewrites images, so it needs a small script.
3. Cheap second horse on the same data: a one-hot colour vector stored as `observation.environment_state`. ACT in 0.7.0 turns that feature into its own input token with no library change. Whether ACT then uses it to choose between balls is not shown anywhere.
4. Record raw frames only, label every episode with its target colour, and after the striped set is done add multi-colour scenes (striped, red, blue together) with equal counts and shuffled positions. Never bake an overlay into recordings.
5. Avoid as the main plan: SmolVLA or `multi_task_dit` (weak evidence for colour following at 30-50 demos per colour), pi0.5 on the laptop, coordinates as low-dimensional input, and inference-time recolouring without retraining.
6. Safe fallback for Sunday: one coloured ball in the bowl at a time. It demonstrates voice-to-skill triggering and a colour-independent pick, not selection among colours, and should be presented that way.

---

## 1. One ACT policy per target colour

**What it is.** Train `act_pick_striped`, `act_pick_red`, ... and let the voice command choose the checkpoint. The team's loop already holds several policies, so no new code.

**Data per colour.**
- LeRobot's guidance is "at least 50 episodes, with 10 episodes per location", and the ACT page says it "often achieves high success rates with just 50 demonstrations". Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/il_robots.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/act.mdx
- Community single-object results on SO-100/101 range from 5/5 with 30 fixed-start episodes to 60% with 60 episodes (already collected in `ball-picking-and-handover.md`, section 2). Source: https://discuss.huggingface.co/t/what-is-the-expected-success-rate-for-a-act-trained-model-using-50-episodes/178517
- The other colours must be in the bowl during training. A 2026 study of ACT found that with three colour-matched competitors pick success fell to 39.2% while lift and place stayed above 93%, and on a real UR3e standard ACT scored 0/20 under mixed distractors on both tasks (100 scripted demos per task). A policy that has only ever seen one coloured ball has no reason to ignore a second one. Source: https://arxiv.org/html/2609.05376
- So a realistic figure is 40-60 episodes per colour, recorded with the other colours present. **[estimate, built from the sources above; no source tests coloured balls in a bowl]**

**Does fine-tuning from the striped checkpoint need fewer demos?**
- Mechanics exist: `lerobot-train --policy.path=<checkpoint> --dataset.repo_id=<new dataset>` starts a new run from the checkpoint weights, and when it is not a resume the normalisation statistics are replaced by those of the new dataset. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/train.py#L219-L241 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/scripts/lerobot_train.py#L665-L693
- The checkpoint's input features are kept (`if not cfg.input_features:` only fills them when empty), so the new dataset must have the same cameras and state layout. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/factory.py#L340
- A Seeed guide suggests a lower learning rate for this: "When fine-tuning an existing checkpoint, consider 1e-6 to 3e-6, about 3-10 times lower". It says nothing about episode counts. Source: https://wiki.seeedstudio.com/robotics/hands_on_act_training/
- I found no measurement of ACT fine-tuned from a sibling task against ACT from scratch at several demo counts. **[unverified: "fewer demos after fine-tuning" is plausible because arm motion, bowl and cameras are identical, but nobody has shown it]**
- The one staged-training data point I found points the other way, for a different model: a pi0.5 model trained on dataset 1 and then fine-tuned on the union averaged 54%, against 80% for training on the union directly (10 trials per test). Source: https://justintiensmith.github.io/vla-research/2026/04/24/what-makes-a-good-vla-dataset-lessons-from-real-experiments.html

**How many colours in a day.**
- Recording: one SO-101 blog reports 200 teleoperated episodes in "about 2 hours", so 50 episodes is roughly 30-40 minutes. Source: https://justintiensmith.github.io/vla-research/2026/04/17/200-episodes-later-why-more-data-didnt-fix-our-vlas.html
- Training: about 30-60 minutes for ACT on an RTX 4090/3090 and 1-2 hours on an L4/A10G, for 5 epochs over about 50 episodes. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/hardware_guide.mdx
- Memory is not the limit: ACT is "only ~80M parameters", so several checkpoints fit in 8 GB. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/act.mdx **[not measured on the laptop]**
- With cloud trainings in parallel, recording and on-robot testing are the bottleneck. Two extra colours in a day is realistic, three to four only if nothing else needs attention. **[estimate]**

**Verdict.** Lowest technical risk, highest data cost, and it does not scale: every new colour is a new recording session and a new policy. Tonight's striped data is reusable only for the striped policy, and that policy needs extra episodes with other colours present before it can be trusted in a mixed bowl.

---

## 2. Conditioning ACT without language: a one-hot colour vector

### What the 0.7.0 source does

- ACT's transformer encoder input is `[latent, (robot_state), (env_state), (image_feature_map_pixels)]`. If the config has an environment-state feature, the model creates `encoder_env_state_input_proj = nn.Linear(env_dim, dim_model)` and appends that projection as one extra token. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/modeling_act.py#L349-L370 and https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/modeling_act.py#L469-L477
- A dataset column becomes that feature only by name: `dataset_to_policy_features` types the key `observation.environment_state` as `FeatureType.ENV`; every other non-image key starting with `observation` becomes `STATE`. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/utils/feature_utils.py#L140-L183
- ACT's robot-state token is taken only from the key `observation.state` (`ft_name == OBS_STATE`). A column called `observation.target_colour` would therefore be typed `STATE` and then silently unused. The name must be exactly `observation.environment_state`. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/policies.py#L138-L154
- Normalisation: ACT's mapping lists `VISUAL`, `STATE` and `ACTION` only, and the normaliser falls back to identity for unlisted types and for keys without statistics. A 0/1 vector passes through unchanged. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/configuration_act.py#L88-L94 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/normalize_processor.py#L332-L334
- The variational encoder used during training sees only the robot state and the actions, not the environment state. At inference the latent is zero, so this does not matter for deployment. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/modeling_act.py#L408-L467

### Can it be fed without changing library code? Yes, by reading.

- **Into the dataset, offline.** `add_features(dataset, {"observation.environment_state": (fn, {"dtype": "float32", "shape": [N], "names": None})})` writes a new column into a copy of the dataset. `fn(row, episode_index, frame_index)` may be a callable, so a per-episode colour table is enough. Videos are copied unchanged. The docs say adding features "is only available through the Python API". Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_tools.py#L330-L471 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_tools.py#L1145-L1202 , https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/using_dataset_tools.mdx
- The new column gets no statistics (only statistics of existing keys are copied). That is harmless here because of the identity fallback above. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_tools.py#L1264-L1275
- **At inference, in the team's loop.** `predict_action` converts every NumPy array in the observation dict to a tensor and adds a batch dimension, whatever the key. Adding `frame["observation.environment_state"] = np.array([0, 1, 0], dtype=np.float32)` after `build_dataset_frame(...)` is enough. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/common/control_utils.py#L48-L96 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/utils.py#L111-L155
- The stock `lerobot-rollout` has no place to inject this value; only a custom loop can.
- Limits: the vector length is fixed at training time; a new colour means retraining. A conditioned policy cannot be fine-tuned from tonight's unconditioned checkpoint through the CLI, because the checkpoint's feature list is kept (section 1). Train it from scratch.
- The alternative of appending the one-hot to `observation.state` means rewriting that column and its statistics, and has no advantage.

A contradiction to be aware of: LeRobot issue #4921 (open, filed against 0.4.4) says ACT's `forward()` "never touches" `observation.environment_state`. At v0.7.0 lines 476-477 do read it whenever the feature is configured. Source: https://github.com/huggingface/lerobot/issues/4921 . An older issue shows a user asking for exactly this use ("a task pointer"), answered only with "intended for sim environments". Source: https://github.com/huggingface/lerobot/issues/1580

### Is such an input actually used?

- **For it:** LeX-O (LeRobot Worldwide Hackathon) added a fixed one-hot-style task embedding as an extra token, the same place ACT's environment-state token sits, and trained on 10 demos for each of 9 grid cells (25k steps, 160 minutes on an RTX 5090). Their attention analysis found "High attention to task instructions", "High attention to joint states" and "Almost no attention to vision". No success rate is given. Sources: https://aadarshram.github.io/2025-07-29-Le-X-O , code read directly at https://github.com/aadarshram/lerobot/blob/LeX-O_TicTacToe/src/lerobot/policies/act/modeling_act_lang.py
- **The catch in that result:** in tic-tac-toe each ID means a fixed place, so "ID to memorised trajectory" is enough. Here the commanded ball lies somewhere new every time, so the policy has to combine the ID with the image ("find the ball of colour k"). LeX-O's own finding that the model "ignored the image input, instead memorizing trajectories" is a warning for that case, not support.
- **Against it:** a Stanford CS224R project (UR5, diffusion policy, wrist camera, PDF read directly) fed a tracked bounding box as four low-dimensional numbers on 486 episodes: "Mechanically worked, but the policy mostly ignored the bbox", "likely because they were weak compared with high-dimensional visual features and were corrupted by crop augmentation". The same box as a mask image channel worked (section 3). Source: https://cs224r.stanford.edu/projects/pdfs/Raul%20Garreta%20Tompson%20Joshua%20Alexander%20Bowden%20Swaroop%20Pal%20submission_416215691/cs224r_2026_final_report_point_and_pick.pdf
- A one-hot colour is a cleaner signal than four coordinates (discrete, not disturbed by image augmentation), so that failure does not transfer one to one. **[my reasoning, untested]**
- Research versions of multi-task ACT do not rely on a plain extra token: MT-ACT uses FiLM so that "the policy doesn't get confused about the task when multiple tasks are possible in a scene", reports a 5-10% drop without it, and was trained on 7,500 trajectories. Source: https://arxiv.org/html/2309.01918
- I found no report of ACT choosing between differently coloured objects from a one-hot or task ID. **[unverified]**

**Verdict.** Zero library changes and one extra training run, so it is worth running alongside the visual-prompt policy on the same multi-colour data. Do not make it the only bet. It needs multi-colour scenes for every colour; tonight's single-colour episodes only help the motion.

---

## 3. Visual prompting: mark the commanded ball in the image

**Idea.** The detector marks the commanded ball in the camera images during training and at inference, and the policy learns "pick the marked ball". Colour stops mattering to the policy, so a new colour needs detector tuning but no new demonstrations.

### Evidence

| Source | What was marked, and how | Data | Result |
| --- | --- | --- | --- |
| Point and Pick (Stanford CS224R 2026, PDF read directly) https://cs224r.stanford.edu/projects/pdfs/Raul%20Garreta%20Tompson%20Joshua%20Alexander%20Bowden%20Swaroop%20Pal%20submission_416215691/cs224r_2026_final_report_point_and_pick.pdf | Bounding box as a binary mask, fourth image channel, diffusion policy with ViT-L, wrist camera only | 184 real pile episodes mirrored to 368; earlier stages 75-486 | 89% correct pick on a grid, 76% in piles, 0% wrong-object picks in both; 19% at far-left/right positions outside the training range. Without a prompt the policy "picked arbitrary blocks or oscillated between nearby objects" |
| MOO (Google) https://arxiv.org/html/2303.00905 | OWL-ViT detection reduced to a single pixel at the box centre, concatenated as an image channel | 59,051 episodes | 98% seen, 79% unseen objects (pick); degrades when the mark is off the object |
| ARRO https://arxiv.org/html/2505.08627v3 | Keeps only gripper and task objects, replaces everything else with a black or grid background, in training and at inference; Grounding DINO + SAM 2 + a VLM | 90 teleoperated demos per real task, Diffusion Policy | Vanilla policy "exhibits a substantial drop" with distractors, ARRO "demonstrates strong performance"; the numbers are only in a figure **[not retrieved]** |
| "What Matters, When?" https://arxiv.org/html/2609.05376 | Crops of the target's appearance as a prompt, plus copy-paste distractor augmentation and attention regularisation, ACT | 100 scripted demos per task | Real UR3e under mixed distractors: 13/20 and 12/20, standard ACT 0/20. Swapping the prompt redirected the choice to the distractor in 35-91% of cases |
| AimBot https://arxiv.org/html/2508.08113 | Lines and reticles drawn onto the RGB images, applied offline to the training set ("simply replaces original RGB images with augmented counterparts") and live at inference | 80-150 demos per task, 548 total, pi0 / pi0-FAST / OpenVLA-OFT | 27 to 43, 42 to 47 and 21 to 36 successes out of 50. It marks the gripper's aim, not a target. Deliberately misaligned overlays made things worse than no overlay (85.2 to 77.4 on LIBERO-Long) |
| VCA https://arxiv.org/abs/2602.23583 | Operator click turned into a segmentation mask for the policy | not in the abstract | no numbers in the abstract **[only the abstract read]** |

What this supports: an image-space mark is used by the policy where a low-dimensional one was not, and overlays can be applied offline to an existing dataset (AimBot did exactly that). What it does not show: any of this with ACT on an SO-101 at about 100 demonstrations. The closest ACT result needed more than a mark (augmentation and regularisation too). I found no LeRobot community project that trains on detector overlays. **[unverified at this scale]**

### Which kind of mark

- **Extra mask channel** (the Stanford design) is not possible without changing ACT: the backbone is a stock torchvision ResNet and images are 3-channel. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/modeling_act.py#L333-L343
- **Mask as a third "camera"** is possible without code changes: all `observation.images.*` keys are treated as camera views of the same shape and run through the same backbone with the same positional embedding. It costs one more backbone pass and a new video feature in the dataset. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/configuration_act.py#L33-L39 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/act/modeling_act.py#L479-L495 **[untested idea]**
- **Drawn into the RGB image** needs no feature change at all, and the dataset keeps the same layout as tonight's. Two variants:
  - Ring or crosshair around the target, rest of the image untouched.
  - Canonical target: paint the target's pixels one fixed flat colour and turn everything else grey (a light version of ARRO).
- **Why the ring is risky with tonight's data.** In every episode recorded so far the target is also the only coloured thing, so ring and colour carry the same information. The policy can learn from colour alone and ignore the ring, and would then still go for the striped ball when a red ball is ringed. Removing colour from the image (canonical target) leaves the mark as the only cue. This is my reasoning, supported indirectly by LeX-O's finding that ACT takes the easiest cue and by "Robust Skills, Brittle Grounding" (policies "exploit correlations between instructions and object locations", https://arxiv.org/html/2602.24143). **[inference, not tested]**
- **Cropping** to the target changes the geometry the arm has to learn and loses the bowl rim. I found no support for it on a fixed-camera setup and do not recommend it.

### Can the overlay be applied offline to the episodes already recorded? Yes, with a custom script.

- Format: a v3 dataset stores `videos/` as "MP4 shards per camera; each file typically contains many episodes", with tabular data in Parquet and per-episode offsets in `meta/episodes/`. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/lerobot-dataset-v3.mdx
- No built-in tool rewrites image content. `lerobot-edit-dataset` offers delete, split, merge, remove feature, modify tasks, convert image to video, re-encode videos, recompute stats and info. `add_features` only writes Parquet columns and copies the existing videos. `reencode_dataset` changes codec settings in place and has no per-frame hook. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/using_dataset_tools.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_tools.py#L1205-L1228 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_tools.py#L2103-L2187
- The supported way is to copy the dataset frame by frame with the public writer API: open the source read-only, `LeRobotDataset.create(...)` a new one with the same features, and per episode call `add_frame(...)` with the edited images and the original state, action and task, then `save_episode()`, and `finalize()` at the end. `add_frame` accepts channel-first or channel-last arrays and computes timestamps itself; `finalize()` is mandatory. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py#L500-L569 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/dataset_writer.py#L202-L228 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/feature_utils.py#L235-L257
- Image statistics do not need care: training uses ImageNet statistics for cameras by default. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/factory.py#L215-L221

```python
# UNTESTED SKETCH - written from the v0.7.0 source, never executed.
import numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

src = LeRobotDataset("user/balls_pick_striped")
features = {k: v for k, v in src.meta.features.items() if k not in DEFAULT_FEATURES}
dst = LeRobotDataset.create("user/balls_pick_marked", fps=src.fps, features=features,
                            robot_type=src.meta.robot_type)
current = None
for i in range(len(src)):
    item = src[i]                      # assumed: images as float CHW tensors in [0, 1]
    ep = int(item["episode_index"])
    if current is not None and ep != current:
        dst.save_episode()
    current = ep
    frame = {"task": item["task"]}
    for key in features:
        value = item[key].numpy()
        if key in src.meta.camera_keys:
            rgb = (value.transpose(1, 2, 0) * 255).round().astype(np.uint8)
            value = mark_target(rgb, colour="striped")   # same function as at inference
        frame[key] = value
    dst.add_frame(frame)
dst.save_episode()
dst.finalize()
```

### Where to apply it at inference

- In the team's own loop, on the raw arrays before they become a frame: `obs = robot.get_observation()`, then replace `obs["top"]` and `obs["wrist"]` with `mark_target(...)`, then `build_dataset_frame(...)` and `predict_action(...)` as before. No processor class is needed. Source for the loop: `lerobot-multi-skill-runner.md`, section 1; `predict_action` at https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/common/control_utils.py#L48-L96
- The formal hook, if it should also work in `lerobot-record` or `lerobot-rollout`, is an `ObservationProcessorStep` subclass in the robot observation pipeline; `record_loop` calls `robot_observation_processor(obs)` before building the frame. Sources: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/pipeline.py#L1874-L1900 , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/scripts/lerobot_record.py#L316-L319
- Use one `mark_target` function for both the offline rewrite and the live loop. AimBot's misaligned-overlay result shows what a train/inference mismatch costs.

### How many demos

No source gives a number for this setup. The comparable ones used 90 per task (ARRO), 100 scripted (ACT study) and 184-368 (Stanford). Plan on tonight's 60-120 rewritten episodes plus 60-90 multi-colour episodes, and treat it as an experiment. **[estimate]**

### Failure modes

- **Detector dropouts.** A frame without a mark shows the policy a scene it never trained on. ARRO lists objects "occasionally not tracked reliably, leading to degraded performance" and says its performance "depends heavily on the underlying segmentation model, especially for wrist cameras". Source: https://arxiv.org/html/2505.08627v3
- **The wrist camera.** The current detector runs on the top camera only. Near the grasp the wrist view dominates, and an unmarked wrist image still shows every ball in full colour. Either mark both views or grey out the wrist view completely. The Stanford project marked the wrist view. **[the team's detector on wrist images is untested]**
- **Occlusion by the arm** hides the target in the top view during the approach, exactly when the mark matters (occlusion problems are documented in `ball-picking-and-handover.md`).
- **Foil reflections** of a coloured ball can pass a colour threshold and get painted as a second target (reasoned in `ball-picking-and-handover.md`, section 3). **[unverified]**
- **Offline against live pixels.** The offline rewrite sees video-decoded frames (compressed, chroma-subsampled), the live loop sees raw camera frames. One threshold may not fit both. **[unverified]**
- **Position extrapolation** stays a problem: the Stanford policy picked the right object but only 19% of the time at untrained positions.
- **Latency.** A threshold on two 640x480 images is cheap, a detector on both views is not free; one SO-101 report measured a drop from 9/10 to 4/10 when the loop fell from 30 Hz to about 15 Hz. Source: https://dev.classmethod.jp/en/articles/lerobot-so-arm101-act-training-eval/

---

## 4. Colour remapping at inference, without retraining

**Idea.** Keep tonight's striped-ball policy. At inference, repaint the commanded ball so it looks like the striped ball and grey out the real striped ball.

**Precedent.** Editing observations at run time to fit a frozen policy exists, but not as plain recolouring, and always with heavy generative models:
- ROSO uses Stable Diffusion to make a novel object look like one the policy knows, finishing "up to 57% of tasks otherwise unsuccessful". Source: https://arxiv.org/abs/2311.16680
- Mirage masks out an unseen robot and inpaints the training robot ("cross-painting") in real time. Source: https://arxiv.org/abs/2402.19249
- BYOVLA finds image regions a frozen VLA is sensitive to and edits task-irrelevant ones; distractors otherwise cut success "by up to 40%". Source: https://arxiv.org/abs/2410.01971
- I found no paper or project that recolours a target by hue to match the training target. **[no direct precedent]**

**Why it could work.** The policy most likely keys on "a saturated yellow-green blob among grey things". Rotating the hue of the commanded ball into that range and desaturating the real striped ball reproduces those statistics, costs nothing to train, and can be tried the hour the baseline policy exists. **[my reasoning]**

**Why it may not.**
- Hue rotation gives a plain yellow-green ball, not stripes; whether the policy cares about the pattern is unknown.
- Brightness does not rotate with hue: a dark blue ball becomes olive, not bright yellow.
- Mask edges leave fringes of the original colour; highlights and foil reflections are untouched.
- The risk is lopsided. Whenever the mask on the real striped ball drops out for a few frames, the policy sees its true training target and goes for it, which is the worst demo failure (wrong ball) rather than a miss.
- Both cameras need it, with the same wrist-camera problem as section 3.

**Verdict.** A one-hour experiment once the baseline is trained, not a plan. If the paired-frame check in "Recommended path" looks good it is a free bonus; if not, nothing is lost. The trained version of the same idea (section 3, canonical target) removes the mismatch instead of hoping the policy tolerates it.

---

## 5. Language-conditioned policies that could run on 8 GB or the Mac

### Memory and cost

| Policy | Inference memory found | Training cost | Notes |
| --- | --- | --- | --- |
| SmolVLA (450M) | 2.0 GB and 12.7 Hz on an RTX 4090D (Evo-1 paper, Table 2, https://arxiv.org/html/2511.04555v2); 1,322 MiB device memory on an RTX 3060 and 1,380 MiB RSS with 374 ms mean latency on an Apple M4, both in the vla.cpp runtime, not PyTorch (https://arxiv.org/html/2606.08094) | "20k steps will roughly take ~4 hrs on a single A100"; hardware guide 10-16 GB at batch 8. A community run: 6,000 steps at batch 64 in about 1.5 h on a 16 GB RTX 4090 laptop | Fits the 8 GB laptop by these numbers. **[not measured with LeRobot 0.7.0 on this laptop; no PyTorch-MPS figure found]** Predicts 50 actions per call, so slow inference is tolerable |
| `multi_task_dit` (~450M, CLIP ViT-B/16 image and frozen text encoder) | none stated; docs: "At least an RTX 5070 Ti (or equivalent GPU) is recommended for reasonable speed performance" | batch 64 minimum, 192-320 recommended, ">30k steps recommended for a single task"; hardware guide 8-14 GB at batch 8 | The laptop is below the recommended inference class. No small-data colour evidence found |
| Evo-1 (0.77B, `policy.type=evo1`) | 2.3 GB and 16.4 Hz on an RTX 4090D | not sized in the hardware guide; paper used 100 demos per task | New integration, no SO-101 evidence found **[unverified]** |
| pi0.5 | 5,938 MiB device memory on an RTX 3060 and 5,987 MiB RSS at about 1.2 s per request on an Apple M4, in vla.cpp | 24-40 GB to train | Ran out of memory on the team's laptop while loading. Only reachable through a cloud GPU with `lerobot-policy-server` and `--inference.type=remote`, which the team's own loop does not use today |

Sources for the LeRobot rows: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/smolvla.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/multi_task_dit.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/evo1.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/hardware_guide.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/inference.mdx , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/smolvla/configuration_smolvla.py , https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py . The community run is https://huggingface.co/scalexi/smolvla-raise2026-ripeness-ref

### Do they follow colour instructions with 30-50 demos per colour?

For:
- The SmolVLA paper reports 70% on an SO-100 sorting task ("put the red cube in the right box and the blue cube in the left box", colours flipped across episodes, 50 demos per dataset), against 25% for ACT and 45% for pi0. Without its robot pretraining the same model averaged 40-51.7% across tasks. Source: https://arxiv.org/html/2506.01844v1
- A course project fine-tuned SmolVLA on 90 SO-101 episodes and states "Even when other objects are present the model picks up the red cube based on our instruction". No trial counts, and only red was ever the target, so this does not show switching. Source (PDF read directly): https://yuxng.github.io/Courses/CS6341Fall2025/project_group_11.pdf
- A simulated UR5e model card: 49 episodes, 8/8 correct-colour picks of a red tomato among green ones. Simulation, one target colour, 8 trials. Source: https://huggingface.co/scalexi/smolvla-raise2026-ripeness-ref

Against:
- The closest real test: an SO-101 with four Duplo colours, 50 episodes per colour, distractors varied, SmolVLA, ACT and pi0.5 trained on the same data. pi0.5 was "the only model that consistently conditioned on colour instructions"; overall success was "roughly 30-40%" for all models because grasping failed. The post does not describe SmolVLA's colour behaviour further. Source: https://justintiensmith.github.io/vla-research/2026/04/17/200-episodes-later-why-more-data-didnt-fix-our-vlas.html
- LangGap keeps the scene fixed and changes only the instruction: SmolVLA scored 38.0% on the original tasks and 6.4% on the changed ones, with 0.0% on "change target"; pi0.5 fell from 93.8% to 21.4%. Adding same-scene tasks with different instructions to the training data raised one such task from 3.75% to 90.0%, but the gain faded as more tasks were added. Source: https://arxiv.org/html/2603.00592
- "Robust Skills, Brittle Grounding" (simulation, "grasp the <object>" among five objects): SmolVLA 90% with small position jitter, 37-41% with medium or large jitter, 2% with full randomisation; "in most failures, agents execute smooth and confident grasping motions toward seemingly arbitrary locations". Source: https://arxiv.org/html/2602.24143
- In the Stanford project a text-conditioned pi0.5 baseline got 3% correct picks on the simulated grid against 47% for the box-conditioned diffusion policy, and was competitive only when objects were scattered and easy to describe (83% against 60%). Source: section 3 table.
- ArmnetBench (50 demos per task, real SO-101): SmolVLA 15.0% strict success, lowest of seven policies (numbers in `lerobot-training-feasibility.md`, transcribed by an automated reader). Source: https://arxiv.org/abs/2607.24481

**Verdict.** SmolVLA fits the laptop by published numbers and can be trained in the cloud in a few hours, but the independent evidence says colour or target switching is where small language policies fail, and its plain pick success at 50 demos is below ACT's in the one SO-101 benchmark. The model that does follow colours (pi0.5) does not fit the laptop. A stretch experiment at most, and only useful if the multi-colour data from section 6 exists anyway.

---

## 6. Data design: how to record from now on

**Rules that keep every option open**

1. **Record raw frames. Never record with an overlay.** Offline marking is possible (section 3); un-marking is not, and an overlay bug would poison the whole dataset.
2. **One dataset per (target colour, scene type), merged later.** For example `balls_pick_striped_solo`, `balls_pick_striped_mixed`, `balls_pick_red_mixed`. Merging and per-episode selection are supported (`lerobot-edit-dataset --operation.type merge`, `--dataset.episodes=[...]`). Source: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/using_dataset_tools.mdx . Merging requires identical features, so keep camera names, resolution, fps and robot id fixed.
3. **One fixed sentence per colour**, differing only in the colour word: "Pick the striped ball from the bowl", "Pick the red ball from the bowl". ACT ignores it; `--dataset.single_task` stores it on every frame, which is the only per-episode colour label the stock recorder writes. Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/scripts/lerobot_record.py#L372 . Community multi-task SmolVLA cards warn that strings must match training verbatim (https://huggingface.co/Harrysunshine/so101-smolvla-9task ; not validated on hardware).
4. **Finish the striped baseline set first, unchanged** (striped ball plus foil). The docs warn "Avoid adding too much variation too quickly". Source: https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/il_robots.mdx
5. **Then record mixed scenes**: striped, red and blue balls in the bowl together with the foil, one of them the target.
   - Distractor colours present: yes. A policy that never saw a second coloured ball has no reason to ignore it (ACT distractor study, section 1), and conditioned policies can only learn to choose if there is a choice.
   - Equal counts per colour, 20-30 each to start. The blog's 25-per-colour, position-varied dataset beat a 100-episode single-colour one on everything except the constrained test, and the author concludes "dataset structure matters more than dataset size" (pi0.5). Source: https://justintiensmith.github.io/vla-research/2026/04/24/what-makes-a-good-vla-dataset-lessons-from-real-experiments.html
   - Shuffle which colour lies where, so colour is never predictable from position ("exploit correlations between instructions and object locations", https://arxiv.org/html/2602.24143).
   - Record contrast pairs: same arrangement, different target. This is the data fix LangGap found effective (same scene, different instruction).
   - Consistent grasp style across all colours. The four-colour blog found "more data did not help. Consistency did" after mixing horizontal and top-down grasps.
6. **How many colours.** Three (striped, red, blue). Their hues are far apart for both the detector and the augmentation (below). Avoid a plain yellow or plain green ball next to a yellow-and-green striped one, and avoid orange next to red. **[my recommendation]**

**Which option uses which data**

| Data | Per-colour ACT | One-hot ACT | Marked-ball ACT | SmolVLA | One ball at a time |
| --- | --- | --- | --- | --- | --- |
| Striped + foil (tonight) | striped policy only | motion only | yes, after rewrite | "striped" sentence only | yes |
| One other colour + foil | that colour's policy, but untested against other colours | motion only | yes, after rewrite | weak (no choice to learn) | yes, this is its training set |
| Mixed scenes, labelled target | yes | required | yes, after rewrite; teaches ignoring unmarked balls | required | no, unless rewritten |

**Colour jitter and learning to tell colours apart**

- LeRobot's image augmentation is off by default (`enable: bool = False`). When enabled it samples up to 3 of 6 transforms per frame: brightness (0.8, 1.2), contrast (0.8, 1.2), saturation (0.5, 1.5), hue (-0.05, 0.05), sharpness (0.5, 1.5), affine (rotation +-5 degrees, translation 0.05). Source: https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/transforms/transforms.py#L1443-L1511
- These are torchvision `ColorJitter` factors. A hue factor of +-0.5 is a "complete reversal of hue channel", so +-0.05 is one tenth of that, about +-18 degrees on the colour wheel. Saturation is multiplied by 0.5 to 1.5. Sources: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_hue.html , https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.ColorJitter.html **[the conversion to degrees is my arithmetic]**
- Consequences:
  - Red (about 0 degrees), yellow-green (60-120) and blue (240) stay apart under +-18 degrees. Neighbouring colours (red and orange, yellow and lime) can overlap, which turns the augmentation into label noise for any colour-dependent policy. **[arithmetic, not tested]**
  - Half saturation still leaves a coloured ball clearly different from foil.
  - RoCoDA states the rule directly: colour jitter "cannot be used when colors are relevant to the task", and its authors dropped it for their colour-dependent stacking task. Source: https://arxiv.org/html/2411.16959
  - For per-colour, one-hot and language policies: keep brightness and contrast for lighting robustness, drop hue, or leave augmentation off and control the lighting instead.
  - For the marked-ball policy the frames are grey plus one painted colour, so hue and saturation jitter only vary the paint. The affine transform moves image and mark together, so they stay aligned; that is an advantage over coordinates as numbers, which the Stanford project found "corrupted by crop augmentation".
  - The default jitter does not make a striped-only policy colour-blind: red is about 60 degrees from yellow, far outside +-18. Do not expect tonight's policy to pick a red ball because augmentation was on. **[arithmetic, not tested]**

---

## 7. Plain engineering alternative: only one pickable ball at a time

**What it is.** The bowl holds foil balls and exactly one coloured ball per round. The voice command names the colour; the state machine checks with the detector that a ball of that colour is visible and then runs the single pick policy.

**What it needs.**
- A pick policy that grasps whichever single coloured ball is there. Tonight's striped-only policy is not known to do that (section 6, last point). The honest version adds 15-20 episodes each of a red and a blue ball alone with the foil to the same dataset and trains one ACT on "pick the coloured ball". **[the count is an estimate]**
- A weaker variant with no new data at all: every round uses the striped ball, and the voice command only starts the round.

**What it demonstrates.** Speech to intent to skill, a perception gate that refuses when the named colour is absent ("I can't see a red ball"), and a learned pick that does not depend on one particular ball colour.

**What it does not demonstrate.** Choosing between coloured balls. With one candidate in the bowl the command carries no information for the arm; the experimenter has made the choice by loading the bowl. A judge who puts two colours in will see the arm take either. Say so in the pitch: "target selection is done by the detector and the scene today; the multi-colour policy is the next step", and show the multi-colour policy separately if it works.

**A stronger hybrid if one extra colour is ready.** Two per-colour policies (section 1) with both balls in the bowl do demonstrate selection, at the cost of 40-60 mixed-scene episodes per colour.

---

## Comparison

| Option | Effort | New data | Risk | Reuses tonight's data |
| --- | --- | --- | --- | --- |
| 1. One ACT per colour | Low: no new code, one training run per colour | 40-60 mixed-scene episodes per colour, and the same again for striped if it must work in a mixed bowl | Low per policy; does not scale; wrong-colour picks if the other colours were absent in training | Yes, for the striped policy only |
| 2. One-hot as `observation.environment_state` | Low to medium: `add_features` script, one line in the loop, train from scratch | 20-30 mixed-scene episodes per colour, all colours | Medium to high: no evidence that ACT binds the vector to the image when positions vary; may memorise trajectories | Partly: adds motion data, teaches nothing about choosing |
| 3. Marked-ball ACT (canonical target, offline rewrite) | Medium: mask function for both cameras, rewrite script, live hook, one training run | 0 to start; 60-90 mixed-scene episodes recommended | Medium: depends on mask quality in both views; no result at this scale; best external evidence of all options | Yes, fully, after the rewrite |
| 4. Recolour at inference, no retraining | Low: mask function and a hue shift | none | High: no precedent for plain recolouring; a mask dropout sends the arm to the real striped ball | Yes (uses the baseline policy as is) |
| 5a. SmolVLA | Medium: cloud fine-tune (hours), rename map, new policy type in the loop | 30-50 per colour in mixed scenes, verbatim sentences | High: independent evidence against colour switching at this scale; lower pick success than ACT at 50 demos | Partly: striped episodes with the striped sentence |
| 5b. `multi_task_dit` | Medium to high: batch 64+ training, >30k steps | as 5a | High: laptop below the recommended inference GPU, no small-data evidence | Partly |
| 7. One coloured ball at a time | Very low | 0, or 15-20 per extra colour alone | Very low technically; the claim has to be stated honestly | Yes |

---

## Recommended path

Priority order stays: the striped-ball pick and the other skills first. Everything below is a stretch goal that must not touch the baseline dataset or checkpoint.

**First step, tonight, under two hours, no robot time and no training: a mask audit.**
1. Write `mark_target(rgb, colour)` with OpenCV: threshold the target colour inside the bowl region, keep the largest roughly round blob, paint its pixels one fixed colour, convert everything else to grey. For the striped ball the threshold is the union of its yellow and green ranges.
2. Run it over 5-10 of tonight's episodes for both cameras and write a side-by-side preview video (original next to marked).
3. Count, per camera, the share of frames with a clean single blob while the ball is in view, and note where it breaks: under the gripper, on foil reflections, at the bowl rim.
4. Decision: if the top view holds almost always and the wrist view holds through the approach, continue with the marked-ball policy. If the wrist view is hopeless, plan to grey out the wrist image without a mark and accept the weaker cue near the grasp. If the top view is unreliable too, drop options 3 and 4 and go to option 7 or option 1 with one extra colour.

**Then, in this order.**
1. Tonight, while recording: follow section 6 (raw frames, fixed sentence, striped-only baseline set finished first).
2. Saturday morning: record the mixed scenes, 20-30 episodes per target colour with striped, red and blue all in the bowl, as contrast pairs. About an hour by the 200-episodes-in-2-hours figure.
3. Rewrite baseline plus mixed data with `mark_target` and train a marked-ball ACT in the cloud (30-60 minutes on a 4090-class GPU per the hardware guide). In parallel, add the one-hot column to the unmarked copy of the same data and train the one-hot ACT. Both runs are cheap, and they fail for different reasons.
4. Evaluate both on the robot, 10 trials per colour with all three colours in the bowl. Score wrong-colour picks separately from missed grasps; only the first number says whether conditioning works.
5. Once the striped baseline is trained, spend at most an hour on the option-4 check: capture a frame with the striped ball at a spot, and a frame with a red ball at the same spot and the arm in the same pose; recolour the red one; compare the two predicted action chunks. Close agreement means inference-time recolouring may work as a bonus. This check is my suggestion, not a published method.
6. Decide Saturday evening. If one conditioned policy picks the right colour reliably, demo it with a mixed bowl. If not, demo option 7 with the honest description from section 7 and show the multi-colour attempt as work in progress.

**Avoid.**
- SmolVLA or `multi_task_dit` as the route to colour commands before Sunday.
- pi0.5 on the laptop.
- Coordinates or bounding boxes as low-dimensional inputs.
- A ring overlay trained only on tonight's single-colour data.
- Recording with overlays baked in.
- Hue jitter on colour-dependent policies when colours are close.
- Relying on fine-tuning from the striped checkpoint to cut the demo count per colour.

---

## Unverified

Nothing in this file was executed. LeRobot was not installed, no dataset was opened, no policy was trained or run, and no camera or robot was used. All LeRobot statements come from reading the v0.7.0 source and docs; all external numbers come from the linked pages, most of them read through an automated summariser.

- The code sketch in section 3 and the `add_features` / inference recipe in section 2 are untested. In particular: what `LeRobotDataset.__getitem__` returns for images in 0.7.0, whether `add_features` writes a vector-valued column that loads back as a float32 tensor of shape `(N,)`, whether a feature without statistics causes trouble in training or in `merge`, and whether video decoding works on the Windows laptop.
- That ACT uses a one-hot `observation.environment_state` to choose between balls whose positions vary: no source. LeX-O used fixed positions per ID and reports no success rate.
- That a marked-ball ACT works with roughly 100-200 episodes on an SO-101: no source. The supporting papers used other policies, robots and data sizes. ARRO's distractor numbers are only in a figure and were not retrieved; VCA was read as an abstract only.
- The argument that a ring would be ignored when the target is also the only coloured object, and that a greyed image forces use of the mark, is reasoning, not a result.
- Mask quality on the wrist camera, under the gripper, with foil reflections, and the difference between thresholds on decoded video and on live frames: not measured.
- That fine-tuning ACT from the striped checkpoint reduces the demos needed for another colour: no source found.
- Whether tonight's striped-only policy would pick a different single coloured ball: unknown.
- Episode counts per colour (20-30, 40-60, 15-20), the number of colours achievable in a day, and the rewrite and training times are estimates.
- SmolVLA inference memory was measured by others on other hardware and partly in another runtime (vla.cpp); it was not measured in LeRobot 0.7.0 on the RTX 4070 laptop, and no PyTorch-MPS figure for the M3 Pro was found. `multi_task_dit` inference memory is not stated anywhere I looked. Evo-1 in LeRobot was not examined beyond its doc page.
- The four-colour SO-101 blog gives no per-model numbers and does not say explicitly how SmolVLA behaved on colour; "did not follow consistently" is read from "pi0.5 was the only model that consistently conditioned on colour instructions". How ACT was given the colour there is not stated.
- ArmnetBench figures are carried over from `lerobot-training-feasibility.md`, where they are already marked as transcribed by an automated reader.
- The hue-to-degrees conversion and the claim that red, yellow-green and blue stay separable under the default jitter are arithmetic. The exact command-line syntax for removing only the hue transform from `--dataset.image_transforms.tfs` was not checked.
- Issue #4921 says ACT never reads the environment state (filed against 0.4.4); the v0.7.0 source reads it. I trust the source, but this was not confirmed by running the model.
- Remote inference for pi0.5 from a cloud GPU or on the Mac was not explored beyond noting that the backend exists.
- The paired-frame check for option 4 and the mask audit are my own suggestions.
- Speech recognition and mapping a sentence to a colour were not part of this research.
