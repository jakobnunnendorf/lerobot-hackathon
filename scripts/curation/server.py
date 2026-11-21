import argparse
import json
import os
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from mixes import build_mix, write_mix

GRIPPER = 5
# A gripper move counts as a close or an open only past this many units; smaller wiggles are servo noise.
ZIGZAG_STEP = 6.0
TIME_LIMIT_S = 120
SCENE_CLASSES = ["rb_buried", "rb_visible", "foil_buried", "foil_visible"]
VERDICTS = ["keep", "drop", "look"]
LABEL_FIELDS = ["verdict", "scene_class", "task", "note"]
TABLE_ALIASES = {
    "dataset": ["dataset", "repo_id", "dataset_repo_id"],
    "episode": ["episode_index", "episode", "ep"],
    "verdict": ["verdict"],
    "scene_class": ["scene_class", "scene", "class"],
    "task": ["task", "task_sentence"],
    "note": ["note", "reason", "verdict_reason"],
    "clears": ["clears", "n_clears", "clear_count", "num_clears", "balls_cleared"],
    "regrasps": ["regrasps", "regrasp_count", "n_regrasps", "regrasp"],
    "label_source": ["label_source"],
    "confidence": ["confidence"],
}


def lerobot_home():
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def hub_caches():
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return [lerobot_home() / "hub", Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser()]


def snapshot_root(repo_dir):
    snapshots = repo_dir / "snapshots"
    for ref in ("v3.0", "main"):
        ref_file = repo_dir / "refs" / ref
        if ref_file.exists() and (snapshots / ref_file.read_text().strip() / "meta/info.json").exists():
            return snapshots / ref_file.read_text().strip()
    candidates = [s for s in snapshots.glob("*") if (s / "meta/info.json").exists()]
    return max(candidates, key=lambda s: s.stat().st_mtime) if candidates else None


def resolve_dataset(spec):
    path = Path(spec).expanduser()
    if (path / "meta/info.json").exists():
        return f"{path.parent.name}/{path.name}", path
    local = lerobot_home() / spec
    if (local / "meta/info.json").exists():
        return spec, local
    for cache in hub_caches():
        root = snapshot_root(cache / f"datasets--{spec.replace('/', '--')}")
        if root:
            return spec, root
    raise SystemExit(f"dataset not found: {spec} (looked in {lerobot_home()} and {[str(c) for c in hub_caches()]})")


def discover_datasets():
    found = {}
    home = lerobot_home()
    for info in sorted(home.glob("*/*/meta/info.json")):
        found.setdefault(f"{info.parents[2].name}/{info.parents[1].name}", info.parents[1])
    for cache in hub_caches():
        for repo_dir in sorted(cache.glob("datasets--*--*")):
            root = snapshot_root(repo_dir)
            if root:
                found.setdefault(repo_dir.name.removeprefix("datasets--").replace("--", "/", 1), root)
    return list(found.items())


def read_parquets(folder, columns=None):
    files = sorted(folder.glob("*/*.parquet"))
    if not files:
        return None
    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in files], ignore_index=True)


def list_column(table, name):
    column = table.column(name).combine_chunks()
    return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), -1).astype(np.float32)


def gripper_events(gripper, step=ZIGZAG_STEP):
    events, trend, hi, lo = [], 0, 0, 0
    for i in range(1, len(gripper)):
        if gripper[i] > gripper[hi]:
            hi = i
        if gripper[i] < gripper[lo]:
            lo = i
        if trend >= 0 and gripper[i] <= gripper[hi] - step:
            events.append({"kind": "close", "frame": hi, "value": round(float(gripper[hi]), 1)})
            trend, lo = -1, i
        elif trend <= 0 and gripper[i] >= gripper[lo] + step:
            events.append({"kind": "open", "frame": lo, "value": round(float(gripper[lo]), 1)})
            trend, hi = 1, i
    return events


class Dataset:
    def __init__(self, index, repo_id, root):
        self.index, self.repo_id, self.root = index, repo_id, Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = self.info["fps"]
        self.cameras = [k for k, f in self.info["features"].items() if f["dtype"] == "video"]
        self.joint_names = self.info["features"]["observation.state"].get("names") or []
        self.episodes = read_parquets(self.root / "meta/episodes").sort_values("episode_index").set_index("episode_index", drop=False)
        self.frames = self.load_frames()
        self.video_files = {self.video_path(cam, ep) for cam in self.cameras for ep in self.episodes.index}

    def load_frames(self):
        files = sorted((self.root / "data").glob("*/*.parquet"))
        if not files:
            return None
        tables = [pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "action"]) for f in files]
        episode = np.concatenate([t.column("episode_index").to_numpy() for t in tables])
        frame = np.concatenate([t.column("frame_index").to_numpy() for t in tables])
        state = np.concatenate([list_column(t, "observation.state") for t in tables])
        action = np.concatenate([list_column(t, "action") for t in tables])
        order = np.lexsort((frame, episode))
        episode, state, action = episode[order], state[order], action[order]
        starts = np.searchsorted(episode, self.episodes.index.to_numpy())
        ends = np.searchsorted(episode, self.episodes.index.to_numpy(), side="right")
        return {ep: (state[s:e], action[s:e]) for ep, s, e in zip(self.episodes.index, starts, ends)}

    def video_path(self, camera, episode):
        e = self.episodes.loc[episode]
        return self.info["video_path"].format(
            video_key=camera,
            chunk_index=int(e[f"videos/{camera}/chunk_index"]),
            file_index=int(e[f"videos/{camera}/file_index"]),
        )

    def videos(self, episode):
        e = self.episodes.loc[episode]
        return {
            cam: {
                "url": f"/video/{self.index}/{self.video_path(cam, episode)}",
                "present": (self.root / self.video_path(cam, episode)).exists(),
                "from": float(e[f"videos/{cam}/from_timestamp"]),
                "to": float(e[f"videos/{cam}/to_timestamp"]),
            }
            for cam in self.cameras
        }

    def metrics(self, episode):
        length = int(self.episodes.loc[episode, "length"])
        row = {"seconds": round(length / self.fps, 1), "hit_limit": length >= (TIME_LIMIT_S - 1) * self.fps}
        if not self.frames or episode not in self.frames or not len(self.frames[episode][0]):
            return row
        gripper = self.frames[episode][0][:, GRIPPER]
        closes = [e for e in gripper_events(gripper) if e["kind"] == "close"]
        return row | {
            "open_max": round(float(gripper.max()), 1),
            "open_at_close": closes[0]["value"] if closes else None,
            "final_grip": round(float(gripper[-1]), 1),
            "closes": len(closes),
        }

    def traces(self, episode):
        if not self.frames or episode not in self.frames:
            raise HTTPException(404, f"no frame data for {self.repo_id} episode {episode}")
        state, action = self.frames[episode]
        return {
            "fps": self.fps,
            "names": self.joint_names,
            "state": np.round(state.T, 2).tolist(),
            "action": np.round(action.T, 2).tolist(),
            "events": gripper_events(state[:, GRIPPER]),
        }

    def original_task(self, episode):
        tasks = self.episodes.loc[episode, "tasks"]
        return str(tasks[0]) if len(tasks) else ""

    def summary(self):
        features = self.info["features"]
        return {
            "index": self.index,
            "repo_id": self.repo_id,
            "root": str(self.root),
            "fps": self.fps,
            "robot_type": self.info.get("robot_type"),
            "total_episodes": self.info["total_episodes"],
            "loaded_episodes": len(self.episodes),
            "has_frames": self.frames is not None,
            "cameras": {c: {"shape": features[c]["shape"], "codec": features[c].get("info", {}).get("video.codec")} for c in self.cameras},
            "features": {k: {"dtype": f["dtype"], "shape": f["shape"], "names": f.get("names")} for k, f in features.items()},
        }


def first_present(row, names):
    for name in names:
        if name in row and pd.notna(row[name]) and row[name] != "":
            return row[name]
    return None


def load_table(paths, datasets):
    by_name = {d.repo_id: d.repo_id for d in datasets} | {d.repo_id.split("/")[-1]: d.repo_id for d in datasets}
    table = {}
    for path in paths:
        path = Path(path).expanduser()
        frame = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
        for record in frame.to_dict("records"):
            dataset, episode = first_present(record, TABLE_ALIASES["dataset"]), first_present(record, TABLE_ALIASES["episode"])
            if dataset is None or episode is None:
                continue
            repo_id = by_name.get(str(dataset), by_name.get(str(dataset).split("/")[-1], str(dataset)))
            table[(repo_id, int(episode))] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}
    return table


class Labels:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.latest, self.mtime = {}, None

    def refresh(self):
        if not self.path.exists():
            return
        mtime = self.path.stat().st_mtime_ns
        if mtime == self.mtime:
            return
        latest = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[(record["dataset"], int(record["episode"]))] = record
        self.latest, self.mtime = latest, mtime

    def append(self, record):
        record = {**record, "time": datetime.now().astimezone().isoformat(timespec="seconds")}
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.refresh()
        return record


class FrameDecoder:
    def __init__(self):
        self.lock = threading.Lock()

    def jpeg(self, path, seconds):
        with self.lock, av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(max(0.0, seconds - 0.5) / stream.time_base), stream=stream)
            chosen = None
            for frame in container.decode(stream):
                chosen = frame
                if frame.time >= seconds - 0.5 / float(stream.average_rate or 30):
                    break
            if chosen is None:
                raise HTTPException(404, "no frame at that time")
            buffer = BytesIO()
            chosen.to_image().save(buffer, format="JPEG", quality=80)
            return buffer.getvalue()


class Catalog:
    def __init__(self, datasets, table_paths, labels_path, mixes_dir):
        self.datasets = datasets
        self.table = load_table(table_paths, datasets) if table_paths else {}
        self.labels = Labels(labels_path)
        self.mixes_dir = Path(mixes_dir).expanduser()
        self.metrics = {(d.index, ep): d.metrics(ep) for d in datasets for ep in d.episodes.index}

    def row(self, dataset, episode):
        auto = self.table.get((dataset.repo_id, episode))
        person = self.labels.latest.get((dataset.repo_id, episode))
        row = {
            "id": f"{dataset.index}:{episode}",
            "ds": dataset.index,
            "dataset": dataset.repo_id,
            "episode": int(episode),
            **self.metrics[(dataset.index, episode)],
            "clears": first_present(auto, TABLE_ALIASES["clears"]) if auto else None,
            "regrasps": first_present(auto, TABLE_ALIASES["regrasps"]) if auto else None,
            "auto_source": first_present(auto, TABLE_ALIASES["label_source"]) if auto else None,
            "confidence": first_present(auto, TABLE_ALIASES["confidence"]) if auto else None,
            "orig_task": dataset.original_task(episode),
            "auto": auto,
            "person": {k: person.get(k) for k in LABEL_FIELDS + ["by", "time"]} if person else None,
        }
        for field in LABEL_FIELDS:
            person_value = person.get(field) if person else None
            auto_value = first_present(auto, TABLE_ALIASES[field]) if auto else None
            row[field] = person_value if person_value not in (None, "") else auto_value
        row["verdict"] = row["verdict"].lower() if isinstance(row["verdict"], str) else row["verdict"]
        row["source"] = "person" if person else ("auto" if auto else "")
        return row

    def rows(self):
        self.labels.refresh()
        return [self.row(d, ep) for d in self.datasets for ep in d.episodes.index]

    def dataset(self, index):
        if not 0 <= index < len(self.datasets):
            raise HTTPException(404, "unknown dataset")
        return self.datasets[index]


def create_app(catalog):
    app = FastAPI()
    decoder = FrameDecoder()

    @app.get("/api/datasets")
    def datasets():
        return {
            "datasets": [d.summary() for d in catalog.datasets],
            "scene_classes": SCENE_CLASSES,
            "verdicts": VERDICTS,
            "labels_path": str(catalog.labels.path),
            "table_rows": len(catalog.table),
        }

    @app.get("/api/episodes")
    def episodes():
        return catalog.rows()

    @app.get("/api/episode/{ds}/{episode}")
    def episode(ds: int, episode: int):
        dataset = catalog.dataset(ds)
        if episode not in dataset.episodes.index:
            raise HTTPException(404, "unknown episode")
        catalog.labels.refresh()
        return catalog.row(dataset, episode) | {"videos": dataset.videos(episode), "fps": dataset.fps}

    @app.get("/api/traces/{ds}/{episode}")
    def traces(ds: int, episode: int):
        return catalog.dataset(ds).traces(episode)

    @app.get("/api/frame/{ds}/{episode}/{camera}")
    def frame(ds: int, episode: int, camera: str, t: float = 0.0):
        dataset = catalog.dataset(ds)
        if camera not in dataset.cameras or episode not in dataset.episodes.index:
            raise HTTPException(404, "unknown camera or episode")
        video = dataset.videos(episode)[camera]
        path = dataset.root / dataset.video_path(camera, episode)
        if not path.exists():
            raise HTTPException(404, f"video not downloaded: {path}")
        seconds = min(video["from"] + max(t, 0.0), video["to"] - 1 / dataset.fps)
        return Response(decoder.jpeg(path, seconds), media_type="image/jpeg")

    @app.get("/video/{ds}/{relative:path}")
    def video(ds: int, relative: str):
        dataset = catalog.dataset(ds)
        if relative not in dataset.video_files:
            raise HTTPException(404, "unknown video file")
        path = dataset.root / relative
        if not path.exists():
            raise HTTPException(404, f"video not downloaded: {path}")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/label")
    def label(body: dict):
        dataset = next((d for d in catalog.datasets if d.repo_id == body.get("dataset")), None)
        if dataset is None or int(body.get("episode", -1)) not in dataset.episodes.index:
            raise HTTPException(400, "unknown dataset or episode")
        if body.get("verdict") not in VERDICTS + [None, ""] or body.get("scene_class") not in SCENE_CLASSES + [None, ""]:
            raise HTTPException(400, "bad verdict or scene_class")
        record = {"dataset": dataset.repo_id, "episode": int(body["episode"])}
        record |= {k: body.get(k) or None for k in LABEL_FIELDS} | {"by": body.get("by") or None}
        catalog.labels.append(record)
        return catalog.row(dataset, record["episode"])

    @app.post("/api/mix/preview")
    def mix_preview(spec: dict):
        return build_mix(catalog, spec)

    @app.post("/api/mix/save")
    def mix_save(spec: dict):
        return write_mix(catalog, spec)

    @app.get("/api/mixes")
    def mixes():
        return [json.loads(p.read_text(encoding="utf-8")) | {"file": str(p)} for p in sorted(catalog.mixes_dir.glob("*.json"))]

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="repo ids or dataset folders; default: every v3 dataset in the local caches")
    parser.add_argument("--table", nargs="*", default=[], help="per-episode table(s), CSV or JSONL, keyed by dataset + episode_index")
    parser.add_argument("--labels", default="~/so101/curation/labels.jsonl")
    parser.add_argument("--mixes", default="~/so101/curation/mixes")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    specs = [resolve_dataset(s) for s in args.datasets] if args.datasets else discover_datasets()
    datasets = []
    for repo_id, root in specs:
        try:
            datasets.append(Dataset(len(datasets), repo_id, root))
        except Exception as error:
            print(f"skipped {repo_id} at {root}: {error!r}")
            continue
        d = datasets[-1]
        print(f"{repo_id}: {len(d.episodes)} episodes, frames {'yes' if d.frames else 'MISSING'}, root {root}")
    catalog = Catalog(datasets, args.table, args.labels, args.mixes)
    print(f"labels: {catalog.labels.path}  table rows: {len(catalog.table)}  mixes: {catalog.mixes_dir}")
    uvicorn.run(create_app(catalog), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
