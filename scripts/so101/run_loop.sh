#!/usr/bin/env bash
# Usage: run_loop.sh <model_dir/pretrained_model> [seconds=40]
# Repeats run_policy.sh for counting successes; every run is logged to ~/so101/logs/.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${1:?path to .../pretrained_model}
mkdir -p ~/so101/logs
n=1
while true; do
  LOG=~/so101/logs/run_$(date +%Y%m%d_%H%M%S).log
  bash "$HERE/run_policy.sh" "$MODEL" "${2:-40}" 2>&1 | tee "$LOG"
  echo -e "\n\033[1m  Run $n done (log: $LOG). Any key = NEXT run, q = quit.\033[0m"
  IFS= read -rsn1 k; [ "$k" = q ] && break
  n=$((n + 1))
done
