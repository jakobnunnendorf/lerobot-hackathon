# Shared curation labels

Branch `labels` of this repo. Each labeller owns one file, `labels/<name>.jsonl`, so nothing conflicts.
Also here: `labels/agent_table.csv` (the agents' per-episode table), `labels/claims/<name>.json` (who claimed which episodes) and `labels/presence/<name>.json` (who has what open right now).

## Run it (about 5 minutes)

```bash
git clone https://github.com/jakobnunnendorf/lerobot-hackathon && cd lerobot-hackathon   # use your checkout if you have one
git checkout jakobnunnendorf/curation-steps   # the branch with scripts/curation (ask Jakob if it is not pushed yet)

# 1. datasets (symlinked into ~/so101/curation/ds, which labels depend on)
uv run --no-project --with huggingface_hub python scripts/curation/fetch_datasets.py

# 2. the agents' table
mkdir -p ~/so101/curation
gh api -H "Accept: application/vnd.github.raw" "repos/jakobnunnendorf/lerobot-hackathon/contents/labels/agent_table.csv?ref=labels" > ~/so101/curation/episodes.csv

# 3. the dashboard (GH_TOKEN needs push access to this repo)
export GH_TOKEN="$(gh auth token)"
D=~/so101/curation/ds
uv run --no-project --with 'fastapi>=0.115' --with 'uvicorn>=0.30' --with 'pandas>=2.0' --with 'pyarrow>=15' --with 'numpy>=1.26' --with 'av>=12' --with 'pillow>=10' \
  python scripts/curation/server.py --datasets $D/so101_pick_rb_mixed $D/so101_pick_blue_ball $D/so101_pick_buried_ball \
  --table ~/so101/curation/episodes.csv --host 127.0.0.1 --port 8090 --sync-github jakobnunnendorf/lerobot-hackathon
```

Open http://127.0.0.1:8090, type your name top left (it names your file), and label. Labels sync every 15 s.
Tick episodes and press "Claim selected" so nobody double-labels; the "claimed" column and its filter (mine / unclaimed / claimed by others) show the split.
A blue dot means someone else opened that episode in the last 3 minutes.
