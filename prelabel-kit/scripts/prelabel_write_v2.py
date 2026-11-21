# Written by Claude (pre-label v2)
"""Turn labeller spec lines into pre-label records in the v1 format, with by "prelabel-v2".
usage: prelabel_write_v2.py OUT.jsonl SPEC.jsonl [SPEC.jsonl ...]
spec line: {"dataset", "episode", "start_state", "confidence", "uncertain", "steps": "code@t code@t ..."}
Episodes already in OUT are skipped; an invalid spec line is reported and skipped."""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prelabel_write import WORD, START, summarize, verdict

BY = "prelabel-v2"


def record(spec):
    steps = []
    for tok in spec["steps"].split():
        code, t = tok.rsplit("@", 1)
        if code not in WORD: raise ValueError(f"unknown code {code}")
        steps.append((code, round(float(t), 2)))
    steps.sort(key=lambda x: x[1])
    if spec["start_state"] not in START: raise ValueError(f"bad start_state {spec['start_state']}")
    codes = [c for c, _ in steps]
    summ = summarize(codes, spec["start_state"])
    return {"dataset": spec["dataset"], "episode": int(spec["episode"]),
            "steps": [{"t": t, "code": c, "label": WORD[c]} for c, t in steps],
            "start_state": spec["start_state"], "summary": summ, "note": summ,
            "verdict_suggestion": spec.get("verdict") or verdict(codes), "confidence": spec["confidence"],
            "uncertain": spec.get("uncertain") or "none", "by": BY,
            "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


def main():
    out = sys.argv[1]
    have = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); have.add((r["dataset"], r["episode"]))
    n = bad = 0
    with open(out, "a") as f:
        for path in sys.argv[2:]:
            for l in open(path):
                if not l.strip(): continue
                try:
                    spec = json.loads(l); spec["dataset"] = "ds/" + spec["dataset"].replace("ds/", "")
                    if (spec["dataset"], int(spec["episode"])) in have: continue
                    rec = record(spec)
                except Exception as e:
                    print("skipped:", path, l[:80], e); bad += 1; continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); have.add((rec["dataset"], rec["episode"])); n += 1
    print("written", n, "bad", bad)


if __name__ == "__main__": main()
