# Recording episodes

Teleoperate the follower with the leader arm and record LeRobot datasets. Two ways, the same recording underneath (`lerobot-record` with the same arguments), so the datasets match the ones every policy was trained on:

- **From the dashboard**: buttons instead of keys, both cameras and the phase in large type.
- **From a terminal**: `scripts/so101/collect.sh` or `record.sh`.

Every dataset uses camera names `top` and `wrist`, 640x480, 30 fps, MJPG, and robot id `mi_follower`. A policy trained on them expects exactly these. Datasets go to `~/.cache/huggingface/lerobot/<SO101_HF_USER>/<name>/` and to the Hub as `<SO101_HF_USER>/<name>`.

Before the first time: `docs/setup.md`, including `patch_record.py` and `hf auth login` in the `lerobot` env.

## Good episodes

- Every episode starts and ends at the folded rest pose. Policies learn to return there.
- Keep the scene as the policy will see it: same lamp, top camera not moved, exposure locked (`python scripts/preflight.py --lock-exposure`).
- Close the leader gripper only until the ball holds. Squeezing overheats the follower's gripper motor; pause every 10 or so episodes.
- An existing dataset needs the exact task sentence it was recorded with. The sentences we used are in `scripts/so101/datasets.json`.

## From the dashboard

1. Stop the robot program (`demo_chain.sh`) if it runs: both need the arm and the cameras, so the session is refused while it runs.
2. Start the dashboard (`dashboard` env), without the speech model: `python scripts/dashboard/server.py --no-speech`. Open http://localhost:8080 and click **Collect data** at the top right.
3. Fill in the form:
   - **Dataset**: an existing one to add episodes, or **New dataset…** and a name.
   - **Task sentence**: the page remembers the last one per dataset.
   - **Episodes** in this session (default 20) and the **episode limit** (default 120 s; an episode ends by itself after it).
   - **Upload to the Hugging Face Hub when the session ends** (ticked by default).
4. **Start session**. After about 15 s the banner turns yellow (**WAITING**) and the follower mirrors the leader.
5. Per episode:
   - **Space** or **▶ Start episode**: red RECORDING with a timer.
   - **Space** or **■ Finish episode**: yellow PAUSED. The episode is **not saved yet**.
   - **Space** or **▶ Keep it · next episode**: saves it and starts the next one.
   - **←** or **✗ Throw away**, while recording or paused: drops it, and it is recorded again.
   - The last episode of the session has no pause: it is saved as soon as it finishes.
6. **Esc** or **End session** ends early and keeps the episode that is running or paused, so throw a bad one away first. The banner turns green, **SESSION ENDED**, with the new total.
7. If **Force stop** appears, the session did not end within 30 s. It works like Ctrl+C; LeRobot still saves what it has.
8. **Upload**: the banner shows **UPLOADING**, then **UPLOADED**. Only new and changed files are sent. **UPLOAD FAILED** shows the error and a **Retry upload** button; the dataset stays on the laptop either way. Unticked, **⇪ Upload to the Hub** does the same after the session.

The keys act only when the cursor is not in a text field. Click an empty part of the page first.

Logs in `~/so101/collect/`: `collect_<name>_<time>.jsonl` (episodes started, saved, thrown away, the upload) and `.out` (the recorder's full output, for errors).

Try the page without the arm: `python scripts/dashboard/server.py --no-speech --fake-collect` (test pattern, no arm, no dataset).

## From a terminal

These scripts need a terminal you can type in (SPACE, LEFT, ESC). Run them inside WSL from the repo root.

**Record, check and push in one go**, for a dataset listed in `scripts/so101/datasets.json` (task sentence, on-screen label, public or private):

```bash
bash scripts/so101/collect.sh so101_pick_rb_mixed
```

It records until ESC, checks the new episodes with `check_dataset.py`, and pushes them. If the recorder crashes, it keeps the saved episodes and offers to re-attach the USB and go on. The result line is in `~/so101/logs/collect_last.txt`. Add a new dataset to `datasets.json` first.

**Record only:**

```bash
bash scripts/so101/record.sh <dataset name> <episodes> "<task sentence>" ["<on-screen label>"]
```

The same name again **adds** episodes. Keys: SPACE start / finish (kept), LEFT throw away, ESC save and end. LEFT during recording followed by ESC can still save that episode: check afterwards.

**Teleop without recording:** `bash scripts/so101/teleop.sh` (`--no-cams` for arms only).

**Stopping:** Ctrl+C does nothing in a recording terminal, because the recorder reads keys raw. Press ESC. If it hangs, `kill -INT` the `lerobot-record` process, and check the dataset afterwards.

## After every session

1. Read the end of the terminal output: a clean "SESSION ENDED", or a traceback.
2. Check the new episodes and look at the contact sheet:

   ```bash
   python scripts/so101/check_dataset.py <user>/<dataset> --from <first new episode> --sheet /tmp/sheet.jpg
   ```

   It flags episodes that end with an empty gripper and episodes where the follower gripper did not follow the leader.
3. Delete bad episodes. It keeps the old version as `<repo>_old`:

   ```bash
   lerobot-edit-dataset --repo_id=<repo> --new_repo_id=<repo> --operation.type=delete_episodes --operation.episode_indices="[i,j]"
   ```

4. Upload, if `collect.sh` or the dashboard did not:

   ```bash
   python scripts/so101/upload_dataset.py <user>/<dataset> [--private]
   ```

Browse episodes on the Hub with https://huggingface.co/spaces/lerobot/visualize_dataset, or label and mix them with `docs/curation-dashboard.md`.

To combine datasets for training, they need identical fps, robot type and features, camera names included: `lerobot-edit-dataset --operation.type merge`. See `docs/training.md`.
