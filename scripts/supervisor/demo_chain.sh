#!/usr/bin/env bash
# Usage (WSL on the robot laptop): bash scripts/supervisor/demo_chain.sh [--step 015000] [more run_supervisor.py flags]
# The demo chain voice -> PICK (learned) -> GIVE (hand following) -> REST on one connection. Runbook: docs/demo-chain-test.md.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
if [ -f "$HOME/.so101.env" ]; then
  set -a
  source "$HOME/.so101.env"
  set +a
fi
if [ -n "${SO101_ACTIVATE:-}" ]; then
  eval "$SO101_ACTIVATE"
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

step=015000
if [ "${1:-}" = "--step" ]; then
  step=$2
  shift 2
fi
if [ -z "${PUT_DOWN_POSE:-}" ]; then
  echo "Set PUT_DOWN_POSE in ~/.so101.env first: 5 joints over the bowl (docs/demo-chain-test.md, step 2)."
  exit 1
fi

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
policy="${PICK_POLICY:-Space0code/act_redblue_from_foil}@$step"
checkpoint=$(python -c "import sys; sys.path.insert(0, 'scripts/eval'); from eval_pick import resolve_checkpoint; print(resolve_checkpoint(sys.argv[1])[0])" "$policy") || exit 1
echo "PICK policy $policy -> $checkpoint"
mkdir -p "$HOME/so101/demo"

# --rest is eval_pick's HOME (median first frame of the training episodes, as P runs it) with the gripper at 30, not
# 17.8: the supervisor refuses to start with the gripper in the holding band (6-20), so 17.8 would block every restart.
# --max-step-deg 15 is eval_pick's max_relative_target, not 3: see decisions/supervisor.md (training speeds).
# --min-held-target-area 120: on the arm 3 of 6 holds were red/blue (striped 0 px) and the one striped hold read 274 px
# (board 7065772934). Append --min-held-target-area 0 to only log again.
# A later --max-step-deg on the command line wins.
# shellcheck disable=SC2086
python scripts/supervisor/run_supervisor.py --robot \
  --port "${SO101_FOLLOWER_PORT:-/dev/so101_follower}" --robot-id "${SO101_FOLLOWER_ID:-mi_follower}" \
  --top-camera "${SO101_CAM_TOP:-/dev/v4l/by-id/usb-BSW-V06-200512_Integrated_Camera-video-index0}" \
  --wrist-camera "${SO101_CAM_WRIST:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}" \
  --pick-policy "$checkpoint" --pick-timeout 90 --max-step-deg 15 --min-held-target-area 120 \
  --rest -5.6 -102.8 96.6 -100.1 -5.2 30 \
  --put-down-pose $PUT_DOWN_POSE \
  --give-mode follow --voice --voice-host 0.0.0.0 --rounds 10 \
  --log "$HOME/so101/demo/chain_$(date +%Y%m%d_%H%M%S).jsonl" --tick-log \
  "$@"
