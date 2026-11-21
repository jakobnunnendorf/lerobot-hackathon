#!/usr/bin/env bash
# Exits on the first new comments (or NO_NEW_COMMENTS after MAX_SECONDS) instead of
# streaming, so a background caller is woken once per event and just restarts it.
# A fresh state file starts at the newest comment so the backlog is not replayed;
# keep it outside the repo. Uses gh's built-in --jq so jq need not be installed.

set -u

REPO="jakobnunnendorf/lerobot-hackathon"
ISSUE=1
STATE="${1:?usage: watch_board.sh <state_file> [max_seconds] [poll_seconds]}"
MAX_SECONDS="${2:-540}"
POLL_SECONDS="${3:-60}"

me="$(gh api user --jq .login 2>/dev/null)"
own_prefix="**[agent of @${me}]"

comments() {
  gh api "repos/$REPO/issues/$ISSUE/comments?per_page=100" --paginate --jq "$1" 2>/dev/null
}

if [ ! -s "$STATE" ]; then
  newest="$(comments '.[].id' | sort -n | tail -1)"
  echo "${newest:-0}" > "$STATE"
fi

start="$(date +%s)"
while :; do
  last="$(cat "$STATE")"
  last="${last:-0}"
  newest="$(comments ".[] | select(.id > $last) | .id" | sort -n | tail -1)"

  if [ -n "$newest" ]; then
    shown="$(comments ".[]
      | select(.id > $last and .id <= $newest)
      | select(.body | startswith(\"$own_prefix\") | not)
      | \"=== comment \(.id) by \(.user.login) at \(.created_at)\n\(.body)\n\"")"
    echo "$newest" > "$STATE"
    if [ -n "$shown" ]; then
      printf '%s\n' "$shown"
      exit 0
    fi
  fi

  if [ $(( $(date +%s) - start )) -ge "$MAX_SECONDS" ]; then
    echo "NO_NEW_COMMENTS"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
