#!/usr/bin/env bash
# Usage: train.sh <dataset_name|user/dataset> [steps=20000] [batch=8]
# Output: outputs/<dataset>_<timestamp>/checkpoints/last/pretrained_model (relative to $SO101_OUTPUTS, default ~/so101/outputs)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET=${1:?dataset name}
[[ "$DATASET" == */* ]] || DATASET=$SO101_HF_USER/$DATASET
STEPS=${2:-20000}
BATCH=${3:-8}
RUN=${DATASET#*/}_$(date +%Y%m%d_%H%M%S)
OUT=${SO101_OUTPUTS:-$HOME/so101/outputs}/$RUN

# These settings trained the first pick policy (49 episodes) to a first-try success on 2026-11-20: ~2 h on an RTX 4050, 3.7 GB.
lerobot-train \
  --dataset.repo_id="$DATASET" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=$(( STEPS < 5000 ? STEPS : 5000 )) \
  --log_freq=50 \
  --num_workers=4 \
  --job_name="$RUN" \
  --output_dir="$OUT"
