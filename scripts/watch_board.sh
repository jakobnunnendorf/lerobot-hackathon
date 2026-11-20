#!/usr/bin/env bash
# Wait for new comments on the team message board (GitHub issue #1).
#
# Polls once a minute. Prints every new comment that was not written by your
# own agent, then exits. If nothing new arrives within MAX_SECONDS it prints
# NO_NEW_COMMENTS and exits, so the caller can simply run it again.
#
# usage: bash scripts/watch_board.sh <state_file> [max_seconds] [poll_seconds]
#
# <state_file> remembers the last comment already seen. Keep it outside the
# repo (a temp directory). On the first run it is set to the newest comment,
# so only comments posted after that are reported.
#
# Needs only the GitHub CLI (gh), no jq.

set -u

REPO="jakobnunnendorf/lerobot-hackathon"
ISSUE=1
STATE="${1:?usage: watch_board.sh <state_file> [max_seconds] [poll_seconds]}"
MAX_SECONDS="${2:-540}"
POLL_SECONDS="${3:-60}"

me="$(gh api user --jq .login 2>/dev/null)"
own_prefix="**[agent of @${me}]"

# Runs a jq filter over every comment of the issue. Prints nothing on failure.
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
