# How does the team test and show the demo chain?

- Date: 2026-11-21
- Status: decided by Tian; built on `TianK003/dashboard`; tested with `scripts/dashboard/fake_robot.py`, not yet on the arm.

## Decision

A browser page, `scripts/dashboard/server.py`, runs in WSL on the robot laptop and is opened at `http://localhost:8080`. It starts chains (tiles, chat, push-to-talk), shows both cameras while a round runs, shows the supervisor's state and live values, stores the tester's result per round, and writes a board summary. Why: Balazs tested from three terminals plus a Windows voice program, posted results by hand, and his agent was running out of tokens.

- **The page only sends commands to a running supervisor**, the same `{target, destination}` JSON on 8765 as voice. ABORT sends `{"abort": true}`, the same as `a` + Enter. It cannot start the robot program, which a person starts.
- **One exception: the page starts a data collection session** (Tian, 2026-11-21: "through the dashboard so it's visual and easier to do").
  - A person at the table clicks Start after a confirm.
  - The follower only mirrors the leader arm; no policy drives it.
  - Start is refused while a robot program sends status.
- **The session is `lerobot-record` itself**, run by `scripts/collect/record_session.py` with `record.sh`'s exact arguments (PR #6). Four module-level names of `lerobot_record` are replaced, plus `LeRobotDataset.save_episode`:
  - the page's buttons set the same `events` as the keys, through `apply_recording_control`;
  - the spoken lines give the phase;
  - `get_observation` also feeds the frame blocks;
  - `save_episode` counts saved episodes.

  So the datasets and molnrb's installed patches (SPACE, wait before the first episode, banners and voice) stay as they are. The terminal keys keep working when the session has a TTY.
  - **A session uploads its dataset when it ends** (default on, 20 episodes per session; Tian and Jakob, 2026-11-21, board 7067885958), so the training that follows the recording sees each batch (`decisions/continual-training.md`). It uses the same `push_to_hub` as PR #6's `upload_dataset.py`, in the session's own process, after `lerobot-record` returns.
    - **The next session waits for the upload**, because both read and write the same files; a copy for the upload would cost about 2 GB per session.
    - **A failed upload leaves the dataset on the laptop**, and the page offers a retry (`--upload-only`).
    - Rejected: `--dataset.push_to_hub=true` inside `lerobot-record`. A failed push there gives no retry without recording again.
  - Rejected: a record loop of our own. It would drift from `record.sh`'s datasets, which every policy was trained on.
  - Rejected: typing keys into a terminal `lerobot-record`. Its keyboard needs a TTY, and a pty bridge is fragile.
- **No commands while a round runs, or within 3 s of the last one** (ABORT always goes through). The supervisor reads the destination again at VERIFY_HOLD, so a command mid-round would change that round.
- **The mic is in the browser; speech is recognised in the server**, with the same Parakeet model, phrase table and padding as `listen.py` (`decisions/voice-commands.md`). So there is no Python install on Windows; `getUserMedia` works on localhost without HTTPS. On Tian's laptop, two recorded commands were recognised in 0.3 s each.
- **Frames come from shared memory and live values over UDP.** The supervisor writes the top block (as for the tracker) and a wrist block, both every second observe, about 0.3 ms per write. A thread sends the newest values (`scripts/supervisor/status.py`) at 5 Hz, fire and forget. The page needs no work in the 30 Hz loop, and a slow or closed page cannot stall it. History comes from the supervisor's JSONL log, which the server tails.
- **Every round is filmed by the dashboard, not the supervisor** (Jakob, board 7066848083). It writes `<log stem>_roundNN.mp4` next to the log: top | wrist, 15 fps, with a time bar under the frames. A `.frames.csv` gives each frame's capture time on the log's UTC clock, converted from the writers' `perf_counter_ns` stamps (one monotonic clock per machine). Jakob lined up phone footage of exp1 with the log by hand to within 1-3 s; on the fake robot, the file's first frame is 0.05-0.15 s after `round_start`.
  - Codec `mp4v`: the pip OpenCV in the `dashboard` env could not open an `avc1` or `H264` writer (checked 2026-11-21). VLC plays it; browsers may not.
  - The per-tick numbers come from the supervisor's `--tick-log` (`scripts/supervisor/ticks.py`). That is a JSON line per tick, with no image work, so the loop stays at 30 Hz.
- **Red and blue tiles are greyed out.** The only training data is "Pick up the striped ball from the bowl of red and blue balls" (`molnrb/so101_pick_rb_mixed`), so there is no red or blue pick skill. The tiles turn on when the supervisor reports a `pick_colour` skill.

## Alternatives

- **Running inference on Tian's laptop** (to spare Balazs's tokens), rejected. It has no NVIDIA GPU, and a full ACT chunk of Tomi's 15K checkpoint took a median 563 ms on its CPU. With 100-action chunks, the arm would stall about 0.5 s every 3.3 s. Moving the arm would also mean copying the calibration and a different camera stack. Instead, Tian's agent reaches the robot laptop over Tailscale SSH for everything that does not move the arm.
- **Encoding video or computing status inside the supervisor**, rejected: it would put work and network writes into the control loop.
- **Keeping the Windows voice listener**, kept only as a fallback (`listen.py`). The browser mic removes the Windows install from the runbook.

## Revisit if

- The page or the extra frame block slows the loop: the live panel shows the loop rate, which should be about 30 Hz.
- A red or blue pick skill exists.
