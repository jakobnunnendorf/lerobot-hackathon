# How is a recording session run, checked and pushed?

- Status: DECIDED (2026-11-21)

## Decision

One script, `scripts/so101/collect.sh <name>`, records until ESC, checks the new episodes, pushes them and writes one result line to `~/so101/logs/collect_last.txt`. The `so101-collect` skill picks the dataset from `scripts/so101/datasets.json`, starts the script and later reads only that line. Read retries on both arms are raised from lerobot's 2 to 10.

## Why

- The robot-laptop agent ran recording by hand (launch, read logs, restart after crashes, check, upload) and used many tokens per session. With the script the agent needs one launch and one short read, so a cheap model (Sonnet, medium effort) can run it.
- `datasets.json` keeps the task sentence per dataset. Appending with a different sentence adds a second task to the dataset, which a policy trained on it then sees.
- On 2026-11-21 (molnrb, `so101_pick_rb_mixed` ep 120-163) the recorder crashed four times in about an hour: leader Timeout, follower Timeout, follower motor 1 "Parsing error", and once the top camera froze. All 12 motors were healthy right after each crash, and the kernel log showed usbip URBs being unlinked. A single dropped packet ended the session because lerobot gives up after 2 retries. After raising them to 10, the next session (ep 156-163, 8 episodes) ended cleanly with no arm crash. The camera freeze is not covered by this.

## Alternatives

- **Recording from Tian's dashboard** - considered, not built: the dashboard only sends commands to a running supervisor and by design does not start robot programs (`decisions/dashboard.md`).
- **Auto-restart after a crash without asking** - rejected: a crash mid-episode can leave the arm in any pose, so the person decides (R + Enter).

## Revisit if

- Arm read errors still end sessions with 10 retries, or the camera freezes keep happening (then the USB setup, not the retries, needs fixing).
