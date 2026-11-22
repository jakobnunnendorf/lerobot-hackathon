#!/usr/bin/env bash
# Usage: collect.sh <dataset name from datasets.json>
# Records until ESC, then checks and pushes the new episodes. After a crash it offers a restart.
# Result line for the agent: ~/so101/logs/collect_last.txt
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NAME=${1:?dataset name from scripts/so101/datasets.json}
entry() { python -c "import json,sys;print(json.load(open('$SO101_DIR/datasets.json'))['$NAME'].get('$1') or '')"; }
TASK=$(entry task) || { echo "$NAME is not in datasets.json"; exit 1; }
LABEL=$(entry label)
VISIBILITY=$(entry visibility)
REPO=$SO101_HF_USER/$NAME
ROOT=$LEROBOT_HOME/$REPO
mkdir -p "$HOME/so101/logs"
LOG=$HOME/so101/logs/collect_${NAME}_$(date +%Y%m%d_%H%M%S).log
touch "$LOG"
RESULT=$HOME/so101/logs/collect_last.txt

episodes() { [ -f "$ROOT/meta/info.json" ] && python -c "import json;print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])" || echo 0; }
finish() { echo "$(date +%H:%M) $NAME $*" | tee "$RESULT"; }

echo "$(date +%H:%M) $NAME RECORDING (log $LOG)" > "$RESULT"
START=$(episodes)
CRASHES=0
while true; do
  pkill -f "rerun --port" && sleep 1
  MARK=$(($(wc -l < "$LOG") + 1))
  # not "| tee": the Rerun viewer lerobot spawns inherits the pipe and keeps it open, so the script never got past here
  bash "$SO101_DIR/record.sh" "$NAME" 1000 "$TASK" "$LABEL" > >(tee -a "$LOG") 2>&1
  sleep 1
  tail -n +"$MARK" "$LOG" | grep -q Traceback || break
  CRASHES=$((CRASHES + 1))
  echo ""
  banner "97;41" "RECORDER CRASHED - episodes saved before the crash are kept"
  tail -n +"$MARK" "$LOG" | grep -E '^[A-Za-z]*(Error|Exception):' | tail -1
  echo -e "${B}  R + Enter = re-attach USB and record again.   Enter = stop and push.${X}"
  read -r ANSWER
  [[ "$ANSWER" =~ ^[rR] ]] || break
  bash "$SO101_DIR/attach_usb.sh"
done

END=$(episodes)
NEW=$((END - START))
if [ "$NEW" -le 0 ]; then
  finish "DONE new=0 total=$END crashes=$CRASHES - nothing to push"
  exit 0
fi

banner "30;46" "CHECKING $NEW NEW EPISODES"
FLAGGED=$(python "$SO101_DIR/check_dataset.py" "$REPO" --from "$START" 2>&1 | tee -a "$LOG" | grep 'flagged episodes' | cut -d: -f2-)

case "$VISIBILITY" in
  public) PRIVATE=() ;;
  private) PRIVATE=(--private) ;;
  *) finish "NOT PUSHED new=$NEW (ep $START-$((END - 1))) total=$END flagged:$FLAGGED - set visibility in datasets.json"; exit 1 ;;
esac

banner "30;46" "PUSHING $REPO ($VISIBILITY)"
HUB=$(python "$SO101_DIR/upload_dataset.py" "$REPO" "${PRIVATE[@]}" 2>&1 | tee -a "$LOG" | grep 'ON HUB')
if [ -n "$HUB" ]; then
  finish "PUSHED new=$NEW (ep $START-$((END - 1))) total=$END crashes=$CRASHES flagged:$FLAGGED | $HUB"
else
  finish "PUSH FAILED new=$NEW total=$END - see $LOG"
fi
banner "30;42" "ALL DONE - you can close this window"
