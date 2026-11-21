#!/usr/bin/env bash
# Usage: teleop.sh [--no-cams]     Leader drives follower; with cameras the Rerun window shows both feeds. Nothing is recorded.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

clear
banner "30;42" "TELEOPERATION - nothing is recorded"
echo -e "${B}  Move the LEADER, the FOLLOWER copies it.${X}"
echo -e "${B}  TO STOP: hold the follower (it goes limp), click here, press Ctrl+C.${X}"
echo ""

CAM_ARGS=()
[ "${1:-}" != "--no-cams" ] && CAM_ARGS=(--robot.cameras="$CAMERAS" --display_data=true)
NOISE="^[ {]+'|py:[0-9]+ \{'|Control loop is running slower|Teleop loop time|^-{5,}|^NAME|\.pos +\||libEGL|MESA|egui_wgpu|re_renderer|winit|re_grpc_server|telemetry|Cadence|ticks|loop-body|^ +(observe|teleop|send|telemetry|process) |pacing headroom|effective cadence|^\s*$"

lerobot-teleoperate "${FOLLOWER_ARGS[@]}" "${LEADER_ARGS[@]}" "${CAM_ARGS[@]}" --fps=30 2>&1 \
  | grep --line-buffered -vE "$NOISE" \
  | sed -u 's/.*SOFollower connected.*/\n  >>> ALL CONNECTED - move the leader now. Ctrl+C to stop. <<<\n/'
echo -e "\n${B}  Teleoperation stopped (the follower is limp).${X}"
