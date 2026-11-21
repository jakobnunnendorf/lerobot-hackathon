#!/usr/bin/env bash
# Usage: run_policy.sh <model_dir/pretrained_model> [seconds=40] ["task sentence"]
# The arm moves by itself after SPACE. Ctrl+C stops it early and the arm goes limp.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODEL=${1:?path to .../pretrained_model}
DUR=${2:-40}
TASK=${3:-run}
[ -f "$MODEL/model.safetensors" ] || { echo "No model.safetensors in: $MODEL"; exit 1; }

clear
banner "97;41" "ROBOT WILL MOVE BY ITSELF"
echo -e "${B}  Model: $MODEL${X}"
echo -e "${B}  Runs $DUR s, then stops and goes limp - hold the arm at the end.${X}"
echo -e "${B}  Before: follower in REST POSE, scene ready, hands out of reach, someone at the power plug.${X}"
echo -e "${B}  Stop early: click here, Ctrl+C (hold the arm!).${X}"
echo ""
banner "30;43" "CLICK IN THIS TERMINAL, then press SPACE to START (Ctrl+C = cancel)"
say "Press space to start the robot"
while true; do IFS= read -rsn1 k; [ "$k" = " " ] && break; done
say "Robot starting"
echo -e "\n  ${B}Starting in 3 s...${X}"
sleep 3

# max_relative_target=15 caps each step's joint jump, so a bad prediction cannot slam the arm.
lerobot-rollout \
  --strategy.type=base \
  --policy.path="$MODEL" \
  "${FOLLOWER_ARGS[@]}" \
  --robot.max_relative_target=15 \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --duration="$DUR" \
  --display_data=true

say "Robot stopped"
banner "30;42" "ROBOT RUN FINISHED - the arm is limp now."
