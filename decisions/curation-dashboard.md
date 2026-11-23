# How does the team pick the episodes for the next fine-tune?

- Status: DECIDED (2026-11-21). Asked for by Jakob; on `main` (#49), tested by `tests/curation_smoke.py` against `molnrb/so101_pick_rb_mixed`.

## Decision

A separate read-only page, `scripts/curation/server.py` on port 8090, over the LeRobot v3 datasets in the local caches.
- **People** browse, inspect and label every episode.
- **Labels** are stored in an append-only `labels.jsonl`.
- **A mix** is saved as a spec plus a script of `lerobot-edit-dataset` commands that a person runs.

## Why

- **Jakob's SmolVLA trained on 76 of about 324 episodes.** Two datasets say "blue" in their task sentence but show the striped ball, and put-backs and idle stretches need a human look. An agent's automatic table (Jakob's `curation/episodes.csv`) is the starting layer, and people override it per episode.
- **It reads the dataset files directly** (parquet, json, mp4) instead of `LeRobotDataset`, so the server needs no torch and runs in any small env. 265 episodes load in about 1 s.
- **Video: the browser plays the dataset's own mp4,** served with HTTP ranges and seeked to the episode's `from_timestamp`. That is instant and costs the server nothing.
  - If the browser reports no AV1 support or fails to decode, the page falls back to JPEG frames decoded by the server with PyAV, which is slower but works anywhere.
- **Labels are append-only JSONL with "last line wins".** Nothing in a dataset changes, several people's files can be concatenated, and the history of who changed what stays.
- **The export is a script, not an action.** The merge writes several GB and deletes nothing, but it should still be a person's deliberate step, and the Hub push even more so. `delete_episodes` renumbers the kept episodes 0..n-1, and `merge` appends the sources in the order given. The script computes each episode's new index from both rules so that `modify_tasks` sets the right sentence.
- **The merge runs with `--operation.concatenate_videos false`.** Tried on 2026-11-21 with LeRobot 0.7.0, running a generated script end to end in a scratch `HF_LEROBOT_HOME`: `so101_pick_ball` plus `so101_give_hand`, 4 episodes excluded, person and per-class tasks.
  - **The default merge failed** with "non monotonically increasing dts". `delete_episodes` re-encodes the video files it changes at time base 1/90000, but stream-copies the others at 1/15360, and the stream-copy concat cannot join the two.
  - **Without concatenation** (one output file per source file), the merge produced 95 episodes. State, action and task matched the spec's mapping for every episode. The first top frame was pixel-identical to its source episode in the 6 sampled.
  - That run also confirmed both renumbering rules above.

## Alternatives

- **A tab in the recording dashboard (`scripts/dashboard/`):** rejected. That server runs on the robot laptop during recording, and video decoding for inspection would compete with the recording's encoder. A separate server also cannot disturb a session.
- **Editing task sentences inside the source datasets:** rejected. The sources stay untouched, and the mix gets its sentences at merge time.
- **Only the server decoding frames (no `<video>`):** kept only as the fallback. It costs about 50-100 ms per frame and cannot play smoothly.

## Revisit if

- Labels from several laptops need merging often. A shared labels file on one server is simpler than concatenating.
- LeRobot changes how `delete_episodes` renumbers episodes or how `merge` orders them (`lerobot/datasets/dataset_tools.py`). The script's indices depend on both.
