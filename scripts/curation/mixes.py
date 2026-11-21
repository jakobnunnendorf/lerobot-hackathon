import json
import re
import shlex

from fastapi import HTTPException

OPEN_BINS = list(range(0, 75, 5))
# modify_tasks takes the whole mapping on the command line; Windows caps a command line at 32k characters.
TASKS_PER_CALL = 60
RANGE_FIELDS = ["seconds", "open_max", "open_at_close", "final_grip", "closes"]


def episode_key(row):
    return f"{row['dataset']}:{row['episode']}"


def matches(row, filt):
    verdicts = filt.get("verdict") or []
    if verdicts and (row["verdict"] or "none") not in verdicts:
        return False
    classes = filt.get("scene_class") or []
    if classes and (row["scene_class"] or "none") not in classes:
        return False
    if filt.get("exclude_hit_limit") and row["hit_limit"]:
        return False
    for field, bounds in (filt.get("ranges") or {}).items():
        low, high = (list(bounds) + [None, None])[:2]
        value = row.get(field)
        if value is None and (low is not None or high is not None):
            return False
        if (low is not None and value < low) or (high is not None and value > high):
            return False
    return True


def select(rows, spec):
    sources = set(spec.get("sources") or [])
    include, exclude = set(spec.get("include") or []), set(spec.get("exclude") or [])
    filt = spec.get("filter") or {"verdict": ["keep"]}
    return [
        r for r in rows
        if episode_key(r) in include or (r["dataset"] in sources and episode_key(r) not in exclude and matches(r, filt))
    ]


def pick_style(row):
    clears = row.get("clears")
    if clears is not None and str(clears).strip() != "":
        try:
            return "direct" if float(clears) == 0 else "clear_first"
        except ValueError:
            pass
    if row["scene_class"]:
        return "direct" if row["scene_class"].endswith("visible") else "clear_first"
    return "unknown"


def final_task(row, task_by_class):
    person_task = (row.get("person") or {}).get("task")
    return person_task or task_by_class.get(row["scene_class"] or "") or row["task"] or row["orig_task"]


def histogram(values):
    counts = [0] * len(OPEN_BINS)
    for v in values:
        if v is not None:
            counts[min(int(v // 5), len(OPEN_BINS) - 1)] += 1
    return {"bins": OPEN_BINS, "counts": counts}


def tally(selected, key, fps_by_dataset):
    out = {}
    for r in selected:
        bucket = out.setdefault(key(r), {"episodes": 0, "minutes": 0.0})
        bucket["episodes"] += 1
        bucket["minutes"] += r["seconds"] / 60
    return {k: {"episodes": v["episodes"], "minutes": round(v["minutes"], 1)} for k, v in sorted(out.items())}


def feature_warnings(datasets):
    if len(datasets) < 2:
        return []
    first, warnings = datasets[0].summary(), []
    for d in datasets[1:]:
        other = d.summary()
        for field in ("fps", "robot_type"):
            if other[field] != first[field]:
                warnings.append(f"{other['repo_id']}: {field} {other[field]} differs from {first['repo_id']} ({first[field]})")
        if set(other["features"]) != set(first["features"]):
            warnings.append(f"{other['repo_id']}: features {sorted(set(other['features']) ^ set(first['features']))} not in both")
        for name in set(other["features"]) & set(first["features"]):
            if other["features"][name] != first["features"][name]:
                warnings.append(f"{other['repo_id']}: {name} {other['features'][name]} differs from {first['features'][name]}")
        for cam in set(other["cameras"]) & set(first["cameras"]):
            if other["cameras"][cam]["codec"] != first["cameras"][cam]["codec"]:
                warnings.append(f"{other['repo_id']}: {cam} codec {other['cameras'][cam]['codec']} vs {first['cameras'][cam]['codec']} (merge allows it, re-check playback)")
    return warnings


def build_mix(catalog, spec):
    rows = catalog.rows()
    selected = select(rows, spec)
    task_by_class = spec.get("task_by_class") or {}
    used = [d for d in catalog.datasets if any(r["dataset"] == d.repo_id for r in selected)]
    tasks = {}
    for r in selected:
        task = final_task(r, task_by_class)
        tasks[task] = tasks.get(task, 0) + 1
    styles = {s: sum(1 for r in selected if pick_style(r) == s) for s in ("direct", "clear_first", "unknown")}
    return {
        "episodes": len(selected),
        "minutes": round(sum(r["seconds"] for r in selected) / 60, 1),
        "by_class": tally(selected, lambda r: r["scene_class"] or "none", None),
        "by_dataset": tally(selected, lambda r: r["dataset"], None),
        "pick_style": styles,
        "open_max_hist": histogram(r.get("open_max") for r in selected),
        "open_at_close_hist": histogram(r.get("open_at_close") for r in selected),
        "tasks": tasks,
        "warnings": feature_warnings(used),
        "selected": [episode_key(r) for r in selected],
    }


def export_plan(catalog, selected, spec):
    """Per source: which episodes to delete, and where each kept episode lands after delete + merge."""
    output = spec["output_repo_id"]
    task_by_class = spec.get("task_by_class") or {}
    plan, offset = [], 0
    for d in catalog.datasets:
        kept = sorted(r["episode"] for r in selected if r["dataset"] == d.repo_id)
        if not kept:
            continue
        # delete_episodes in LeRobot 0.7.0 renumbers the kept episodes 0..n-1 in their old order (dataset_tools.py),
        # and merge appends each dataset after the previous ones in the order of --operation.repo_ids.
        deleted = sorted(set(int(e) for e in d.episodes.index) - set(kept))
        by_episode = {r["episode"]: r for r in selected if r["dataset"] == d.repo_id}
        plan.append({
            "dataset": d,
            "delete": deleted,
            "part": f"{output}_part{len(plan)}" if deleted else d.repo_id,
            "new_index": {ep: offset + i for i, ep in enumerate(kept)},
            "tasks": {offset + i: final_task(by_episode[ep], task_by_class) for i, ep in enumerate(kept)},
        })
        offset += len(kept)
    return plan


def shell_script(catalog, selected, spec):
    output = spec["output_repo_id"]
    plan = export_plan(catalog, selected, spec)
    folder_sources = any(p["dataset"].root.parent.parent.name != "snapshots" and not p["dataset"].root.as_posix().endswith(p["dataset"].repo_id) for p in plan)
    new_dirs = [p["part"] for p in plan if p["delete"]] + [output]
    lines = [
        "#!/usr/bin/env bash",
        f"# Mix '{spec['name']}': {len(selected)} episodes from {len(plan)} dataset(s) into the local dataset {output}.",
        "# Written by scripts/curation/server.py. It only writes under $HF_LEROBOT_HOME and pushes nothing to the Hub.",
        "set -euo pipefail",
        'LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"',
        f"for d in {' '.join(shlex.quote(d) for d in new_dirs)}; do",
        '  if [ -e "$LEROBOT_HOME/$d" ]; then echo "$LEROBOT_HOME/$d exists; pick another output_repo_id or move it away" >&2; exit 1; fi',
        "done",
        "",
    ]
    for p in plan:
        if p["delete"]:
            root = [f"--root {shlex.quote(str(p['dataset'].root))}"] if folder_sources else []
            lines.append(" ".join(["lerobot-edit-dataset", f"--repo_id {shlex.quote(p['dataset'].repo_id)}", *root,
                                   f"--new_repo_id {shlex.quote(p['part'])}", "--operation.type delete_episodes",
                                   f"--operation.episode_indices {shlex.quote(json.dumps(p['delete']))}"]))
    merge = ["lerobot-edit-dataset", f"--new_repo_id {shlex.quote(output)}", "--operation.type merge",
             f"--operation.repo_ids {shlex.quote(str([p['part'] for p in plan]))}"]
    if folder_sources:
        roots = ['$LEROBOT_HOME/' + p["part"] if p["delete"] else str(p["dataset"].root) for p in plan]
        merge.append(f"--operation.roots \"{str(roots)}\"")
    lines += ["", " ".join(merge), ""]
    tasks = {str(k): v for p in plan for k, v in p["tasks"].items()}
    items = list(tasks.items())
    for start in range(0, len(items), TASKS_PER_CALL):
        chunk = dict(items[start:start + TASKS_PER_CALL])
        lines.append(" ".join(["lerobot-edit-dataset", f"--repo_id {shlex.quote(output)}", "--operation.type modify_tasks",
                               f"--operation.episode_tasks {shlex.quote(json.dumps(chunk, ensure_ascii=False))}"]))
    lines += ["", f'echo "done: $LEROBOT_HOME/{output}"', ""]
    return "\n".join(lines), plan


def write_mix(catalog, spec):
    name = spec.get("name") or ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise HTTPException(400, "name must be letters, digits, _ . -")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", spec.get("output_repo_id") or ""):
        raise HTTPException(400, "output_repo_id must look like user/name")
    counts = build_mix(catalog, spec)
    if not counts["episodes"]:
        raise HTTPException(400, "the mix selects no episodes")
    rows = {episode_key(r): r for r in catalog.rows()}
    selected = [rows[k] for k in counts["selected"]]
    script, plan = shell_script(catalog, selected, spec)
    catalog.mixes_dir.mkdir(parents=True, exist_ok=True)
    mix = {k: spec.get(k) for k in ("name", "sources", "filter", "include", "exclude", "task_by_class", "output_repo_id")}
    mix["counts"] = {k: v for k, v in counts.items() if k != "selected"}
    mix["episodes"] = [
        {"dataset": p["dataset"].repo_id, "episode": ep, "new_index": new, "task": p["tasks"][new]}
        for p in plan for ep, new in p["new_index"].items()
    ]
    json_path, sh_path = catalog.mixes_dir / f"{name}.json", catalog.mixes_dir / f"{name}.sh"
    json_path.write_text(json.dumps(mix, indent=1, ensure_ascii=False), encoding="utf-8")
    sh_path.write_bytes(script.encode("utf-8"))
    return {"json": str(json_path), "sh": str(sh_path), "script": script, "counts": mix["counts"]}

