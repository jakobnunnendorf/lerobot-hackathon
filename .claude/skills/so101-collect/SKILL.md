---
name: so101-collect
description: Use when the person wants to record (collect) SO-101 teleop episodes and push them to the Hub - "record", "collect data", "continue recording", "more episodes of X". One command records until ESC, checks the new episodes and pushes them. For crashes it cannot handle, or other robot work, use the so101-robot skill.
model: sonnet
---

# Collect SO-101 episodes

`scripts/so101/collect.sh <name>` does the whole session in its own window: records until the person presses ESC, checks the new episodes, pushes them to the Hub, and writes one result line to `~/so101/logs/collect_last.txt`. After a crash it asks the person R+Enter (re-attach USB, record again) or Enter (stop and push). Do not hand-build `lerobot-record` commands. The person lets the agent open this window on this laptop.

## 1. Pick the dataset

Read `scripts/so101/datasets.json`. Each entry is a Hub dataset `molnrb/<name>` with its task sentence, on-screen label and visibility.

- **Same skill and same setup as an entry → append** to that name. "Continue recording", "more episodes", "the ball pickups" mean the entry used last; if unsure which, ask.
- **Different task, objects, start/end pose or camera setup → new dataset.** Add an entry: name `so101_<verb>_<thing>`, a task sentence that says what the arm does and where it ends, a short label. Ask the person once public or private and store it as `visibility`. Never edit the task of an existing entry: the next session would add a second task to that dataset.

## 2. Start it

From PowerShell (no double quotes inside the `bash -lc` string besides the outer pair):

```powershell
Start-Process -FilePath wsl.exe -ArgumentList '-e','bash','-lc','"bash /mnt/c/Users/molib/Documents/elrobot/lerobot-hackathon/scripts/so101/collect.sh <name>"'
```

Tell the person, in one short message: click into the window, wait for the yellow WAITING box, SPACE start/finish episode (kept), LEFT throw it away, ESC (in the pause between episodes) end the session; the push runs by itself; close the window at "ALL DONE".

Then stop. Do not poll or sleep-wait.

## 3. When the person says it is done (or asks)

Read only the result line: `wsl -e bash -lc 'cat ~/so101/logs/collect_last.txt'`.

| Line starts with | Meaning / do |
| --- | --- |
| `RECORDING` | Still running, or the window was closed before the push. Ask which. If closed, check and push by hand: `check_dataset.py molnrb/<name> --from <first new episode>`, then `upload_dataset.py molnrb/<name>` (so101-robot skill) |
| `PUSHED` | Report new episodes, total, flagged episodes and the Hub link. Flagged episodes: ask before deleting any |
| `DONE new=0` | Nothing was recorded; nothing pushed |
| `NOT PUSHED` | Visibility missing in `datasets.json`: ask, set it, then `upload_dataset.py` |
| `PUSH FAILED` | Read the last 20 lines of the log named in the line |

To share it, post the Hub link on the board (issue #1) only if the person asks.

## Known

- The window also shows crashes; the person handles them with R+Enter. Repeated crashes (several an hour) are usbip dropouts: ask them to reseat the arm and camera USB cables and hub.
- If the window never shows the WAITING box within a minute, close it and start again.
- A separate dataset of only some episodes (for example one session): `lerobot-edit-dataset --repo_id molnrb/<name> --root ~/.cache/huggingface/lerobot/molnrb/<name> --new_repo_id molnrb/<new> --new_root ~/.cache/huggingface/lerobot/molnrb/<new> --operation.type delete_episodes --operation.episode_indices="[0,1,...]"` (the episodes to leave out), then `upload_dataset.py molnrb/<new>`. Only while no recording runs.
