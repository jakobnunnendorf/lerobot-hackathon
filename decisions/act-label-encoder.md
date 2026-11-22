# Does an ACT image encoder pretrained on our labels beat ImageNet and the all-data start?

- Status: OPEN (2026-11-22). Init folder built and checked; no training run yet.

## Decision

Add a third start to the clean-episode comparison: `act_clean_labelenc` = the same ACT as `act_clean_scratch` except that its ResNet18 is pretrained on our labels (`scripts/label_encoder/`). Nothing else differs, so a gap to `act_clean_scratch` is the encoder's.

## Why

- The failure is grabbing the striped ball while it is only partly uncovered, or skipping the clearing. ImageNet features were never asked to tell buried from partly from visible; the labels say what matters.
- Not known yet whether it helps: this was only checked for loading, not for robot behaviour.

## The encoder

- ResNet18 (ImageNet start) on top and wrist frames, 160x120, 3 fps, the 191 non-hold-out episodes (25.5k frames), 4 epochs on MPS (about 6 minutes). Batch norm stays in eval mode, as in ACT, whose backbone uses frozen batch norm.
- Heads: striped visibility (top frames), target ball of the running or next segment (red / blue / striped / none), wrist blind (wrist frames), phase (approach / grab / carry / home, from event times: grab for 0.7 s after `*_grab`, carry until hole or home).
- Hold-out (10 episodes, 994 frames, never trained on). Majority-class accuracy in brackets.

| head | accuracy | majority |
|---|---|---|
| visibility, top | 0.610 | 0.443 |
| target, top | 0.648 | 0.475 |
| target, wrist | 0.789 | 0.475 |
| phase, top | 0.726 | 0.441 |
| phase, wrist | 0.833 | 0.441 |
| wrist blind, wrist | 0.984 | 0.981 |

- Wrist blind is almost never seen as positive (6 positives in the hold-out, 1 found): the number is not evidence of a skill. Visibility confusion (buried, partly, visible): `[[81,89,5],[17,90,53],[9,21,132]]`, the same buried/partly weakness as `decisions/scene-classifier.md`.
- Resolution caveat: pretrained at 160x120, ACT sees 480x640. The convolutions carry over, but the cues were learned on 4x smaller objects.

## The init folder

- `outputs/act_labelenc_init/` (207 MB, git-ignored) has the layout of `brouthen/act_so101_pick_striped_all_pretrain_50k` step 050000: same `config.json` (copied byte for byte, device cuda), processors and `train_config.json`.
- `make_act_init.py` builds a fresh `ACTPolicy` from that config and replaces **only `model.backbone.*`**; the transformer, VAE and projections are re-initialised (seed 0). Verified: backbone tensors equal the pretrained ones, a transformer tensor differs from Kamel's.
- The processors in the folder carry Kamel's dataset stats; `lerobot-train` replaces them with the dataset's own stats when `--policy.path` is given without resume (`lerobot_train.py`, `processor_kwargs["dataset_stats"]`).
- Checked on CPU: `ACTPolicy.from_pretrained(folder)` (the abstract `PreTrainedPolicy` cannot be instantiated) loads, and `forward` on a real batch of 2 from episode 0 of the local dataset returns a loss (96.7, a fresh model with kl_weight 10; not meaningful by itself).

## Commands Jakob runs

```bash
uv run modal volume put lerobot-smolvla outputs/act_labelenc_init /init/act_labelenc_init
uv run modal volume ls lerobot-smolvla /init/act_labelenc_init          # 7 files, model.safetensors 207 MB
uv run modal run scripts/modal_train.py::prepare --recipe act_clean_labelenc
uv run modal run scripts/modal_train.py::train --recipe act_clean_labelenc --gpu L40S --smoke
uv run modal run --detach scripts/modal_train.py::train --recipe act_clean_labelenc --gpu L40S --steps 30000 --max-minutes 150
uv run modal volume get lerobot-smolvla outputs/act_clean_labelenc_L40S_b32_30k outputs/
```

30k steps at about 0.26 s/step is about 130 minutes, so the 150-minute cap leaves a little slack. The default save frequency is 5000 steps, so partial checkpoints exist if it is cut. `modal volume put` and the recipe are untested here (no cloud run was started).

## How to compare

- Same clean episodes (93 train, the 10 hold-out episodes above), same augmentation, seed, lr 2e-5 and batch 32 as `act_clean_scratch` and `act_clean_from_all` (`decisions/act-clean.md`). Run all three to the same step count.
- Compare the eval loss logged every 5000 steps (`--eval_steps=5000`) in Trackio project `hackathon`, then the real test: the same rollouts on the robot for buried and partly-covered starts, counting grabs of a partly covered striped ball and skipped clearings.
- Reading: labelenc below scratch = the encoder helps; labelenc about equal to scratch = no gain from labels at this size; from_all below both = the all-data pretraining of the transformer matters more than the encoder.

## Alternatives

- **Keep Kamel's transformer and swap only the backbone**: not done; mixing a transformer tuned to ImageNet features with a new encoder confounds the test. Re-initialising was easy, so the fair version was built. If labelenc loses to scratch, try this next.
- **Fine-tune only the last block**: not tried, whole backbone was trained since cheap.

## Revisit if

- Labels grow, or the encoder is re-pretrained at 480x640 (a re-extraction at full size).
