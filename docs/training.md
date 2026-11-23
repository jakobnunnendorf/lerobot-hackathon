# Training

All our demo policies are ACT (LeRobot 0.7.0), one checkpoint per skill, run at 30 fps on the robot laptop's RTX 4050 (8 GB). Larger policies do not fit there (`kb/machines.md`). Which models and datasets we tried, and how they did: `docs/HISTORY.md`.

Training runs in the cloud (Modal) or on a local GPU. Checkpoints go to the Hugging Face Hub, never into git.

## On Modal

`scripts/modal_train.py` trains under a cost cap: prepare the data on a CPU, benchmark GPUs with a short run, then train. **Every `bench`, `train` and `resume` call starts a paid GPU job**; the script prints the worst-case cost first.

The Modal client is not a repo dependency. Install it and log in once (`pip install modal`, `modal setup`), or run it through uv:

```bash
uv run --with modal modal run scripts/modal_train.py::prepare --recipe act_redblue
uv run --with modal modal run scripts/modal_train.py::bench --recipe act_redblue --gpus L40S,A100-80GB --budget-usd 10
uv run --with modal modal run --detach scripts/modal_train.py::train --recipe act_redblue --gpu L40S --steps 20000 --max-minutes 170
uv run --with modal modal run --detach scripts/modal_train.py::resume --run-name <run name> --steps <new total> --max-minutes 60
uv run --with modal modal volume get lerobot-smolvla outputs/<run name> outputs/
```

- `prepare` downloads the recipe's dataset and base models onto the Modal volume `lerobot-smolvla`.
- `bench` runs 400 steps per GPU and prints, for the budget, how many steps fit and the exact `train` command.
- `train` names the run `<recipe>_<gpu>_b<batch>_<steps/1000>k` unless you give `--run-name`. Checkpoints every 5000 steps (`--save-freq`), logs to Trackio.
- `resume` continues a run that stopped (same data); `--steps` is the new total.

Recipes (`RECIPES` in the script):

| Recipe | Dataset | Start |
| --- | --- | --- |
| `act_redblue` | `molnrb/so101_pick_rb_mixed` | from scratch, batch 32, lr 2e-5, light colour augmentation (hue and saturation off) |
| `act_redblue_from_foil` | `molnrb/so101_pick_rb_mixed` | as `act_redblue`, from the grey-foil policy `molnrb/act_pick_ball_combined` (`decisions/act-redblue-init.md`) |
| `act_redblue_v2` | `molnrb/so101_pick_rb_mixed` | from scratch, chunk 50, dropout 0.2, EMA, heavier augmentation (`decisions/act-redblue-generalisation.md`) |
| `act_pretrain_all` | `brouthen/so101_pick_striped_all` (all pick data merged) | LeRobot ACT defaults |
| `act_finetune_rb` | `molnrb/so101_pick_rb_mixed` | from the `act_pretrain_all` run at 50000 |
| `smolvla_combined` | `molnrb/so101_pick_ball_combined` | `lerobot/smolvla_base` |
| `smolvla_redblue_from_combined` | `molnrb/so101_pick_rb_mixed` | from a `smolvla_combined` run on the volume (`smolvla_pick_ball_h100_23k`) |

`prepare_merged` builds `brouthen/so101_pick_striped_all` (task sentences rewritten for the striped ball, then merged) and needs a Modal secret named `huggingface` with a Hub token.

Our final pick model, `TianK003/act_hidden_from_v2`, was fine-tuned from `Space0code/act_redblue_from_foil_v2@20000` with `scripts/modal_v2_finetune.py` (`run_b`; `run_a` trained `TianK003/act_redblue_from_foil_v3`). It starts from v2's checkpoint on the Hub, uses v1's recipe and pushes every checkpoint to the Hub:

```bash
uv run --with modal modal run --detach scripts/modal_v2_finetune.py::run_b
```

**Publish checkpoints while a run trains:** `python scripts/publish_checkpoints.py --run <run name> --repo-name <hub repo name> --dataset <dataset>` polls the Modal volume and uploads each finished checkpoint as `<step>/` with a model card (held-out loss per checkpoint). It prefers the EMA weights when a run saved them. It needs the `modal` CLI on the PATH and `HF_TOKEN` in a `.env` file at the repo root.

## On a local GPU

Inside WSL on the robot laptop, or any machine with LeRobot and CUDA:

```bash
bash scripts/so101/train.sh <dataset name or user/dataset> [steps=20000] [batch=8]
```

Output: `~/so101/outputs/<dataset>_<time>/checkpoints/` (`SO101_OUTPUTS` changes the folder). These are LeRobot's ACT defaults; 49 episodes trained in about 2 h on the RTX 4050.

Try a local checkpoint on the arm with `bash scripts/so101/run_policy.sh <.../pretrained_model> [seconds=40]` (one run, the arm goes limp at the end: hold it) or compare it properly with `docs/evaluation.md`.

## New data: fine-tune, do not resume

When the data changes, start from a checkpoint with `--policy.path=<checkpoint>/pretrained_model`, not `--resume`. Fine-tuning loads only the weights and takes the new dataset's normalisation stats; resume keeps the old stats on purpose, so they go stale. `--steps` then counts additional steps. Details and pitfalls: `research/continue-training-from-checkpoint.md`.

- Record new episodes with the same camera names, resolution and fps, then merge them with the old data (`lerobot-edit-dataset --operation.type merge`). `--dataset.repo_id` takes one dataset, not a list.
- Give the fine-tune a new `--policy.repo_id` and `--output_dir`. Otherwise it pushes to the old model's repo, or stops because the folder exists.
- Re-pass the augmentation, `eval_split` and batch size: fine-tuning reads the dataset settings from the command line. Pass the augmentation as the full `--dataset.image_transforms.tfs='<JSON>'`; single-field forms fail.
- The held-out split is the last episodes of each task sentence, so new episodes under the old sentence fill it. Use a small `--dataset.eval_split` or a distinct task sentence (ACT ignores the text).
- Smoke-test with `--steps=2` first.

## Run-time settings, no retraining

`n_action_steps` and temporal ensembling are read at inference time, so you can change them on any checkpoint (`kb/act-replan-override.md`):

- demo program: `--n-action-steps 30`, the default of `demo_chain.sh` (`docs/demo.md`);
- eval: `--n-action-steps 30` (`docs/evaluation.md`);
- `lerobot-rollout`: `--policy.n_action_steps=30` next to `--policy.path`.

Temporal ensembling (`n_action_steps` 1) takes about 47 ms per step on the RTX 4050, too slow for 30 fps.
