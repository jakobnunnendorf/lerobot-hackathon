#!/usr/bin/env bash
# Written by Claude
# Usage (inside WSL on the robot laptop): bash scripts/eval/eval_pick.sh [--step 015000] [--runs 10] [--seconds 90] ...
# A wrapper so PowerShell's Start-Process needs no quotes: it loads ~/.so101.env (ports, ids, cameras) and the env.
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
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
python scripts/eval/eval_pick.py "$@"
echo -e "\n\033[1m  Finished. Press ENTER to close this window.\033[0m"
read -r
