# Our ACT checkpoints: size, settings, what the network sees

- Date: 2026-11-22
- Verified by: agent of TianK003. Configs and safetensors headers read from the Hub; parameter count reproduced by building `ACTConfig` in a local LeRobot 0.7.0 install. Nothing here was measured on the arm.

## Size

- Every chunk-100 checkpoint is the stock LeRobot 0.7.0 ACT: **51,597,190 learned parameters**, 71,424 buffers, float32, `model.safetensors` 206,699,768 bytes. This covers `Space0code/act_redblue_from_foil` (v1), `act_redblue_from_foil_v2`, `act_redblue_v2`, `jakobnunnendorf/act_clean_from_all`, `brouthen/act_so101_pick_rb_mixed_b32_38k`, `TianK003/act_clear_red` and the base `molnrb/act_pick_ball_combined`.
- Split: ResNet-18 11,166,912; transformer encoder (4 layers) 17,332,736; decoder (1 layer) 5,385,856; VAE encoder 17,332,736; projections and position tables 335,360; VAE projections 40,512; action head 3,078.
- The VAE encoder runs only in training (zero latent at inference), so **34.2M parameters are used on the arm**.
- `brouthen/act_so101_pick_rb_mixed_v2_chunk50_20k` is the only different shape: chunk 50, 51,571,590 parameters.
- `molnrb/act_pick_rb_mixed` needs a login; not read.

## Settings that differ from the LeRobot defaults

- v1 and v2 are identical apart from the data: `config.json` is byte-identical. Both: lr and backbone lr 2e-5 (default 1e-5), batch 32 (default 8), bf16, augmentation on (brightness, contrast, sharpness, affine; hue and saturation weight 0), `eval_split` 0.05, seed 1000, start weights `molnrb/act_pick_ball_combined`.
- Kamel's chunk-50 model: chunk 50, `n_action_steps` 25, dropout 0.2, stronger augmentation, EMA 0.999, from scratch on the 80-episode data.
- Everything else is default everywhere: resnet18 ImageNet weights, `dim_model` 512, 8 heads, feed-forward 3200, 4 encoder and 1 decoder layer, latent 32, `n_obs_steps` 1, `kl_weight` 10, no temporal ensembling.

## What the network sees

- Images enter at 480x640. ResNet-18 has stride 32, so each camera gives a 15x20 grid: 300 tokens per camera, one token per 32x32 px.
- The striped ball in the top camera is 30 to 38 px across (715 to 1,107 px area, live demo frames): **about one token**. In the wrist camera it covers 5 to 10 tokens over the bowl and 20 to 25 at the grasp (rescaled frames, approximate).
- `replace_final_stride_with_dilation` cannot be used: torchvision raises `NotImplementedError` for resnet18.

## Data seen

| model | train episodes / frames | passes over the data |
|---|---|---|
| v1 at 15K | 76 / 53,968 | 8.9 |
| v2 at 20K | 190 / 127,384 | 5.0 |
| `act_clean_from_all` at 10K | 88 / 43,901 (LeRobot held out 5 more than Jakob's list) | 7.3 |
| `b32_38k` at 30K | 76 / 53,968 | 17.8 |

## Loss records

- v1: train L1 0.190, 0.148, 0.126, 0.114, 0.104 at 5K to 25K; held-out loss 0.337, 0.331, 0.338, 0.340, 0.338. Train falls, held-out is flat from 5K.
- v2: held-out 0.337, 0.342, 0.338, 0.335 at 5K to 20K.
- Clearing policies: held-out lowest at 1K (0.307), rising to 0.343 at 8K.
- Train and held-out loss are not the same quantity: training uses a latent computed from the true actions plus the KL term, with dropout and augmentation; evaluation uses a zero latent and neither.
- Held-out loss has not predicted arm behaviour for these policies (Tomi's model card); choose checkpoints on the arm.
