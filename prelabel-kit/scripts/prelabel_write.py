"""Append pre-labels to ~/so101/curation/prelabels.jsonl (one atomic line per episode, skip episodes already present).
Wording copied from static/app.js (WORD map, summarize(), suggest()). Library: write_prelabel(...)."""
import json, os, datetime, fcntl
OUTF = os.path.expanduser("~/so101/curation/prelabels.jsonl")
NAME = {"r": "R", "b": "B", "s": "striped"}
WORD = {"home": "Return home", "end_away": "Ends away from home", "end_empty": "Ends empty", "wrist_blind": "Wrist blind to striped",
        "s_vis": "Striped visible", "s_partly": "Striped partly covered", "s_buried": "Striped buried",
        "r_hole": "R → red hole", "b_hole": "B → blue hole"}
for o, n in NAME.items():
    WORD[o + "_target"] = f"Target {n}"; WORD[o + "_grab"] = f"Grabbed {n}"; WORD[o + "_miss"] = f"Missed {n}"
    WORD[o + "_slip"] = f"Slip/drop {n}"; WORD[o + "_putback"] = f"{n[0].upper() + n[1:]} put back"
START = {"visible": "Striped visible", "partly": "Striped partly covered", "buried": "Striped buried"}

def summarize(codes, start_state):
    parts = [START[start_state]]
    i = 0
    while i < len(codes):
        n = 1
        while i + n < len(codes) and codes[i + n] == codes[i]: n += 1
        parts.append(WORD[codes[i]] + (f" ×{n}" if n > 1 else "")); i += n
    return ". ".join(parts) + "."

def verdict(codes):
    c = codes; bad = c.count("s_miss") + c.count("s_slip")
    drop = any(x.endswith("_putback") for x in c) or "end_empty" in c or "end_away" in c or bad >= 2
    return "drop" if drop else ("keep" if "s_grab" in c and "home" in c else "look")

def existing():
    if not os.path.exists(OUTF): return set()
    s = set()
    for l in open(OUTF):
        try: r = json.loads(l); s.add((r["dataset"], r["episode"]))
        except Exception: pass
    return s

def write_prelabel(dataset, episode, start_state, steps, confidence, uncertain, verdict_override=None):
    """steps: [(code, t_seconds)] in time order, no visibility step (start_state carries it, as in his own records)."""
    steps = sorted(steps, key=lambda x: x[1]); codes = [c for c, _ in steps]
    summ = summarize(codes, start_state)
    rec = {"dataset": dataset, "episode": episode,
           "steps": [{"t": round(float(t), 2), "code": c, "label": WORD[c]} for c, t in steps],
           "start_state": start_state, "summary": summ, "note": summ,
           "verdict_suggestion": verdict_override or verdict(codes), "confidence": confidence, "uncertain": uncertain,
           "by": "prelabel-sonnet", "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with open(OUTF, "a+") as f:                      # lock, re-check, one single write() call = atomic append
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        for l in f:
            try:
                r = json.loads(l)
                if (r["dataset"], r["episode"]) == (dataset, episode): return False
            except Exception: pass
        f.seek(0, 2); f.write(line); f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    return True
