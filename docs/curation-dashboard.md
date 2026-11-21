# Curation dashboard: browse, label and mix episodes

Use it to look at every episode of the team's datasets, set a verdict and scene class, and build the episode mix for the next fine-tune. It reads the datasets and never changes them, and it pushes nothing.

## Start

Run it **on a laptop other than the robot laptop** while recording runs, because decoding video competes with the recording's encoder.

1. **Once:** get the datasets and the table.
   - Datasets: in an env with LeRobot 0.7.0, `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; [D(r) for r in ['molnrb/so101_pick_rb_mixed', 'molnrb/so101_pick_blue_ball', 'molnrb/so101_pick_buried_ball']]"`. This downloads data and videos into the Hugging Face cache.
   - Jakob's table: `git fetch origin jakobnunnendorf/curation-table && git show origin/jakobnunnendorf/curation-table:curation/episodes.csv > ~/so101/curation/episodes.csv`.
2. **Server:** `pip install -r scripts/curation/requirements.txt` (fastapi, uvicorn, pandas, pyarrow, av), then:
   ```
   python scripts/curation/server.py --datasets molnrb/so101_pick_rb_mixed molnrb/so101_pick_blue_ball molnrb/so101_pick_buried_ball --table ~/so101/curation/episodes.csv
   ```
   - Without `--datasets`, it loads every v3 dataset it finds in the local caches.
   - Don't run it with `python -I`; it imports `mixes.py` from its own folder.
3. **Open** http://localhost:8090. Others on the same network can use http://<laptop ip>:8090.

## Browse, inspect, label (main page)

- **Type your name first** (top right). Every label records who set it.
- **The table** has one row per episode, with columns sortable by click and filters above.
  - The values from Jakob's table appear as **auto**. Your labels override them and appear as **person**.
  - Seconds, widest opening, opening at the first close, final grip and closes are computed from the dataset itself.
- **Inspect:** the selected episode plays top | wrist with the gripper and joint traces underneath. The traces share the time cursor; close and open events are marked.
  - The page plays the dataset's mp4 directly. If the browser cannot play it (AV1), it uses frames decoded by the server; the **server frames** box forces that.
  - There is no gripper load trace, because the datasets record positions only.
- **Keys** (when no text field has focus):

  | Key | Action |
  | --- | --- |
  | `j` / `k` | Next / previous row |
  | Space | Play / pause |
  | ← / → | 1 s back / forward (with Shift: 1 frame) |
  | Home | Back to the episode start |
  | `1` / `2` / `3` | Verdict keep / drop / look |
  | `4`-`7` | Scene class `rb_buried`, `rb_visible`, `foil_buried`, `foil_visible` |

  Task sentence and note are typed into their fields.
- **Labels** are appended to `~/so101/curation/labels.jsonl` (`--labels`), one line per change; the last line per dataset + episode wins.
  - To combine several people's labels, concatenate their files, or point everyone at one server.

## Mix builder (`/mix.html`)

- **Choose:** the sources, a filter (default: verdict keep), episodes to include or exclude by hand, and one task sentence per scene class.
- **Live counts:**
  - episodes and minutes per class and per dataset;
  - direct picks against clear-first;
  - histograms of the widest gripper opening and of the opening at the first close;
  - the task sentences;
  - warnings when the sources' features differ, because `merge` needs them identical.
- **Save mix** writes `~/so101/curation/mixes/<name>.json` (the spec, counts, and each episode's new index and task) and `<name>.sh`.
- **The script is run by a person, not the page.** It holds the `lerobot-edit-dataset` commands:
  - `delete_episodes` per source;
  - `merge` into the output repo id;
  - `modify_tasks` with the final task per episode.
  - It writes local datasets only and stops if one of its output folders already exists. Pushing the mix to the Hub is a separate, deliberate step.

## Check it

`python tests/curation_smoke.py [repo_id]` starts the server on port 18090 against a downloaded dataset (default `molnrb/so101_pick_rb_mixed`) and checks:
- the episode list and the table layer;
- traces, labels, video ranges and frames;
- the mix preview and export.
