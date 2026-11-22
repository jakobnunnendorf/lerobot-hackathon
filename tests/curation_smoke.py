# scripts/curation/server.py against a real downloaded dataset (default molnrb/so101_pick_rb_mixed, fetched with
# LeRobotDataset or snapshot_download): episode list, traces, table layer, labels, video and frames, mix preview and
# export. Needs fastapi, uvicorn, pandas, pyarrow and av.
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = "http://127.0.0.1:18090"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "molnrb/so101_pick_rb_mixed"
TOP = "observation.images.top"
failures = []


def check(name: str, ok: bool, detail="") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def call(path: str, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(SERVER + path, data=data,
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    return status, raw


def get_json(path: str, body=None):
    status, raw = call(path, body)
    return status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw


def wait_ready(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            if call("/api/datasets")[0] == 200:
                return True
        except OSError:
            pass
        time.sleep(1)
    return False


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        work = Path(work)
        table = work / "episodes.csv"
        table.write_text("dataset,episode_index,verdict,scene,reason,label_source\n"
                         f"{DATASET},0,DROP,rb_visible,smoke reason,drop_smoke\n"
                         f"{DATASET.split('/')[-1]},1,KEEP,rb_buried,,auto\n", encoding="utf-8")
        server = subprocess.Popen([sys.executable, str(ROOT / "scripts/curation/server.py"), "--host", "127.0.0.1",
                                   "--port", "18090", "--datasets", DATASET, "--table", str(table), "--labels",
                                   str(work / "labels.jsonl"), "--mixes", str(work / "mixes")], cwd=ROOT)
        try:
            check("server starts and loads the dataset", wait_ready(120))
            run_checks(work)
        finally:
            server.terminate()
            server.wait(timeout=10)
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


def run_checks(work: Path) -> None:
    _, info = get_json("/api/datasets")
    dataset = info["datasets"][0]
    check("dataset summary: all episodes and the frame data loaded",
          dataset["repo_id"] == DATASET and dataset["loaded_episodes"] == dataset["total_episodes"]
          and dataset["has_frames"], str({k: dataset[k] for k in ("repo_id", "total_episodes", "loaded_episodes")}))

    _, rows = get_json("/api/episodes")
    by_episode = {r["episode"]: r for r in rows}
    check("one row per episode", len(rows) == dataset["total_episodes"], f"{len(rows)} rows")
    first, second = by_episode[0], by_episode[1]
    check("the table is the 'auto' layer, matched by repo id or by bare dataset name",
          first["verdict"] == "drop" and first["scene_class"] == "rb_visible" and first["note"] == "smoke reason"
          and first["source"] == "auto" and second["verdict"] == "keep" and second["scene_class"] == "rb_buried",
          str({k: first[k] for k in ("verdict", "scene_class", "note", "source")}))

    _, traces = get_json("/api/traces/0/0")
    frames = round(first["seconds"] * dataset["fps"])
    check("traces: 6 joints, one value per frame, close/open events",
          len(traces["state"]) == 6 and abs(len(traces["state"][0]) - frames) <= 3
          and len(traces["action"][0]) == len(traces["state"][0]) and isinstance(traces["events"], list),
          f"{len(traces['state'][0])} values for {first['seconds']} s")

    status, row = get_json("/api/label", {"dataset": DATASET, "episode": 0, "verdict": "keep",
                                          "scene_class": "rb_buried", "note": "smoke", "by": "smoke"})
    lines = (work / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    _, again = get_json("/api/episodes")
    reloaded = next(r for r in again if r["episode"] == 0)
    bad, _ = call("/api/label", {"dataset": DATASET, "episode": 0, "verdict": "maybe"})
    check("a label is appended to labels.jsonl, overrides the table and survives a reload; a bad verdict is refused",
          status == 200 and row["verdict"] == "keep" and row["source"] == "person" and len(lines) == 1
          and json.loads(lines[0])["verdict"] == "keep" and reloaded["verdict"] == "keep" and bad == 400,
          f"{status} {lines} bad={bad}")

    played = None
    for episode in sorted(by_episode)[:40]:
        _, detail = get_json(f"/api/episode/0/{episode}")
        if detail["videos"][TOP]["present"]:
            played = (episode, detail["videos"][TOP])
            break
    if played is None:
        check("video: a downloaded video to play (fetch at least one file under videos/)", False)
        return
    episode, video = played
    status, chunk = call(video["url"], headers={"Range": "bytes=0-1023"})
    frame_status, jpeg = call(f"/api/frame/0/{episode}/{TOP}?t=0.5")
    check("video: the mp4 is served in ranges for the <video> element, and a JPEG frame at a time",
          status == 206 and len(chunk) == 1024 and frame_status == 200 and jpeg[:2] == b"\xff\xd8",
          f"episode {episode}: range {status}, frame {frame_status}")

    keep = sum(1 for r in again if r["verdict"] == "keep")
    spec = {"name": "smoke_mix", "sources": [DATASET], "filter": {"verdict": ["keep"]},
            "task_by_class": {"rb_buried": "Pick up the striped ball"}, "output_repo_id": "local/smoke_mix"}
    _, preview = get_json("/api/mix/preview", spec)
    check("mix preview counts the episodes the filter keeps", preview["episodes"] == keep and keep > 0,
          f"{preview['episodes']} vs {keep} keep")
    status, saved = get_json("/api/mix/save", spec)
    mix = json.loads(Path(saved["json"]).read_text(encoding="utf-8")) if status == 200 else {}
    script = Path(saved["sh"]).read_text(encoding="utf-8") if status == 200 else ""
    new_indices = [e["new_index"] for e in mix.get("episodes", [])]
    check("mix export: spec JSON with renumbered episodes, and a script of lerobot-edit-dataset commands it does not run",
          status == 200 and sorted(new_indices) == list(range(keep))
          and "--operation.type delete_episodes" in script and "--operation.type merge" in script
          and "--operation.type modify_tasks" in script and not (work / "lerobot").exists(),
          f"{status} {len(new_indices)} episodes")


if __name__ == "__main__":
    main()
