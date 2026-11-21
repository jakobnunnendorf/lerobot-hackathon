#!/usr/bin/env bash
SO101_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SO101_DIR/../.." && pwd)"

if [ -f "$HOME/.so101.env" ]; then
  source "$HOME/.so101.env"
else
  source "$SO101_DIR/so101.env.example"
fi

if [ -n "${SO101_ACTIVATE:-}" ]; then
  eval "$SO101_ACTIVATE"
elif [ -f "$REPO_DIR/.venv/bin/activate" ]; then
  source "$REPO_DIR/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1 SVT_LOG=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1
LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

# Every dataset so far was recorded with exactly these camera names and settings; policies expect them.
CAMERAS="{ top: {type: opencv, index_or_path: $SO101_CAM_TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200}, wrist: {type: opencv, index_or_path: $SO101_CAM_WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200} }"
FOLLOWER_ARGS=(--robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID")
LEADER_ARGS=(--teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID")

B='\033[1m'; X='\033[0m'
banner() { echo -e "\033[$1m${B}$(printf '%-72s' "   $2")${X}"; }
say() { espeak-ng -s 160 "$1" >/dev/null 2>&1 & }
