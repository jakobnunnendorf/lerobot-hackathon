# Recording episodes from the dashboard

Same recording as `scripts/so101/record.sh` (PR #6): `lerobot-record` with the same arguments, so the datasets match the ones every policy was trained on. The page starts it, shows both cameras and the phase in large type, and its buttons replace the keys.

## Once

- The robot laptop already has what this needs: `~/.so101.env`, the patched LeRobot in conda env `lerobot`, and conda env `dashboard` with `scripts/dashboard/requirements.txt`.
- Update the checkout: `git fetch origin && git checkout --detach origin/main`.

## Each session

1. Stop the robot program (`demo_chain.sh` / `run_supervisor.py`) if it runs: the session is refused while it does, because both need the arm and the cameras.
2. Start the dashboard in its own terminal, if it does not run yet: `conda activate dashboard && python scripts/dashboard/server.py --no-speech`. Open http://localhost:8080 and click **Collect data** at the top right.
3. **Form:**
   - **Dataset:** pick an existing one to add episodes, or **New dataset…** and a name.
   - **Task sentence:** an existing dataset needs the sentence it was recorded with. The page remembers the last one per dataset.
   - **Episodes:** how many in this session, 20 by default (one batch for the training that follows the recording, `docs/continual-training.md`).
   - **Episode limit:** an episode ends by itself after this, 120 s as in `record.sh`.
   - **Upload to the Hugging Face Hub when the session ends:** ticked by default. Needs `hf auth login` (or `huggingface-cli login`) done once in the `lerobot` env.
4. **Start session**, confirm. After about 15 s the banner turns yellow (**WAITING**) and the follower mirrors the leader.
5. Per episode:
   - **Space** or **▶ Start episode**: red RECORDING, with a timer.
   - **Space** or **■ Finish episode**: yellow PAUSED. The episode is **not saved yet**.
   - **Space** or **▶ Keep it · next episode** saves it and starts the next one.
   - **←** or **✗ Throw away**, while recording or paused, drops it; it is recorded again.
   - The last episode of the session has no pause: it is saved as soon as it finishes.
6. **Esc** or **End session** ends early. It keeps the episode that is running or paused, so throw it away first if it was bad. The banner turns green: **SESSION ENDED**, with the dataset's new total.
7. If the page shows **Force stop**, the session did not end within 30 s. Force stop works like Ctrl+C, and LeRobot still saves what it has.
8. **Upload:** if ticked and the session saved episodes, the banner shows **UPLOADING**, then **UPLOADED** with the episode count on the Hub.
   - Only new and changed files are sent.
   - A new session cannot start until it ends, because both use the same files.
   - **UPLOAD FAILED** shows the error and a **Retry upload** button. The dataset stays on the laptop either way.
   - Unticked: **⇪ Upload to the Hub** after the session does the same.

The keys act only when the cursor is not in a text field. Click an empty part of the page first.

## What lands where

- Dataset: `~/.cache/huggingface/lerobot/molnrb/<name>/`, and on the Hub as `molnrb/<name>` after the upload.
- `~/so101/collect/collect_<name>_<time>.jsonl`: session log, with each episode started, saved and thrown away, and the upload.
- `~/so101/collect/collect_<name>_<time>.out`: the recorder's full terminal output, for errors.
- `~/so101/collect/upload_<name>_<time>.jsonl` / `.out`: the same for a retried upload.

## Without the arm

`python scripts/dashboard/server.py --no-speech --fake-collect` starts `scripts/dashboard/fake_recorder.py` instead: the same page and phases, a test pattern for the cameras, no arm, no dataset.
