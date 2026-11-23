# Curation dashboard: browse, label and mix episodes

Look at every episode of our datasets, set a verdict and a scene class, and build the episode mix for the next fine-tune. It reads the datasets, never changes them, and pushes nothing. Why it works this way: `decisions/curation-dashboard.md`.

Run it on a laptop other than the robot laptop while recording runs: decoding video competes with the recording's encoder.

## Start

1. **Once, get the datasets** into the Hugging Face cache, in any env with LeRobot 0.7.0:

   ```bash
   python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; [D(r) for r in ['molnrb/so101_pick_rb_mixed', 'molnrb/so101_pick_blue_ball', 'molnrb/so101_pick_buried_ball']]"
   ```

   Optional: a per-episode table to start from (keep / drop / look for 265 episodes), on the branch `jakobnunnendorf/curation-table`:

   ```bash
   mkdir -p ~/so101/curation
   git fetch origin jakobnunnendorf/curation-table
   git show origin/jakobnunnendorf/curation-table:curation/episodes.csv > ~/so101/curation/episodes.csv
   ```

2. **Server:** `pip install -r scripts/curation/requirements.txt`, then:

   ```bash
   python scripts/curation/server.py --datasets molnrb/so101_pick_rb_mixed molnrb/so101_pick_blue_ball molnrb/so101_pick_buried_ball \
       --table ~/so101/curation/episodes.csv
   ```

   - Without `--datasets` it loads every v3 dataset in the local caches.
   - Do not run it with `python -I`: it imports `mixes.py` from its own folder.

3. **Open** http://localhost:8090. Others on the same network use `http://<laptop ip>:8090`.

## Browse, inspect, label

- **Type your name first** (top right). Every label records who set it.
- **The table** has one row per episode; click a column to sort, filter above it.
  - Values from `--table` show as **auto**; your labels override them and show as **person**.
  - Seconds, widest gripper opening, opening at the first close, final grip and number of closes are computed from the dataset.
- **Inspect:** the selected episode plays top | wrist with the gripper and joint traces underneath, close and open events marked. If the browser cannot play the video (AV1), the server decodes frames instead; **server frames** forces that. There is no gripper load trace: the datasets record positions only.
- **Keys** (when no text field has focus):

  | Key | Action |
  | --- | --- |
  | `j` / `k` | Next / previous row |
  | Space | Play / pause |
  | ← / → | 1 s back / forward (Shift: 1 frame) |
  | Home | Back to the episode start |
  | `1` / `2` / `3` | Verdict keep / drop / look |
  | `4`-`7` | Scene class `rb_buried`, `rb_visible`, `foil_buried`, `foil_visible` |

- **Labels** are appended to `~/so101/curation/labels.jsonl` (`--labels`), one line per change; the last line per dataset and episode wins. To combine several people's labels, concatenate their files, or have everyone use one server.

## Mix builder (`/mix.html`)

- **Choose** the sources, a filter (default: verdict keep), episodes to include or exclude by hand, and one task sentence per scene class.
- **Live counts:** episodes and minutes per class and dataset, direct picks against clear-first, histograms of the gripper openings, the task sentences, and warnings when the sources' features differ (`merge` needs them identical).
- **Save mix** writes `~/so101/curation/mixes/<name>.json` (the spec, counts, each episode's new index and task) and `<name>.sh`.
- **You run the script, not the page.** It holds the `lerobot-edit-dataset` commands: `delete_episodes` per source, `merge` into the output repo id, `modify_tasks` with the final task per episode. It writes local datasets only and stops if an output folder already exists. Pushing the mix to the Hub is a separate step.

## Check it

`python tests/curation_smoke.py [repo_id]` starts the server on port 18090 against a downloaded dataset (default `molnrb/so101_pick_rb_mixed`) and checks the episode list, table layer, traces, labels, video, frames, and the mix preview and export.
