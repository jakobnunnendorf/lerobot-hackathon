#!/usr/bin/env bash
# Usage: record.sh <dataset_name> <num_episodes> "<task sentence>" ["<label shown on screen>"]
# Running again with the same dataset name ADDS episodes. Keys: SPACE start/finish (kept), LEFT throw away, ESC end.
# Needs the LeRobot patch once per install: python scripts/so101/patch_record.py
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NAME=${1:?dataset name}
NUM=${2:?number of episodes}
TASK=${3:?task sentence}
LABEL=${4:-}
REPO=$SO101_HF_USER/$NAME
ROOT=$LEROBOT_HOME/$REPO

RESUME=false; HAVE=0
if [ -f "$ROOT/meta/info.json" ]; then
  RESUME=true
  HAVE=$(python -c "import json;print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])")
fi

clear
banner "30;46" "SO-101 RECORDING SESSION"
[ -n "$LABEL" ] && banner "97;45" ">>> $LABEL <<<"
echo -e "${B}  Dataset: $REPO   (already has $HAVE episodes)${X}"
echo -e "${B}  This session: $NUM episodes.   Task: $TASK${X}"
echo ""
echo -e "${B}  1. CLICK INSIDE THIS TERMINAL so it gets your key presses.${X}"
echo -e "${B}  2. Wait for the yellow WAITING box (about 15 s).${X}"
echo -e "${B}  3. SPACE = start / finish an episode.  LEFT = throw it away.  ESC = end.${X}"
echo ""

export SO101_WAIT_START=1 SO101_BANNERS=1
NOISE="Svt\[|SVT \[|asm level|^-{20,}|moov atom|Map: |^[ {]+'|py:[0-9]+ \{'|Streaming encoding|Text-to-speech|Cadence|ticks|loop-body|^ +(observe|process_obs|teleop|send|record|telemetry) |pacing headroom|effective cadence|Control loop is running slower|tqdm.py|warnings.warn"

# reset_time_s=3600: the pause between episodes ends on SPACE, not on a timer.
lerobot-record "${FOLLOWER_ARGS[@]}" "${LEADER_ARGS[@]}" \
  --robot.cameras="$CAMERAS" \
  --display_data=true \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.no_stamp=true \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM" \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=3600 \
  --dataset.fps=30 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --resume=$RESUME 2>&1 | grep --line-buffered -vE "$NOISE"
