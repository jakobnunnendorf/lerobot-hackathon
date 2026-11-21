# Written by Claude (pre-label v2)
"""Score pre-labels against Jakob's labels for the corrected episodes.
usage: score_prelabels.py corrections.jsonl [PRED.jsonl]   (no PRED: scores v1 from corrections.jsonl itself)"""
import json, sys, collections

VIS = {"s_vis", "s_partly", "s_buried"}
TOL = {"target": 2.0, "wrist_blind": 1.5, "vis": 2.5}
DEFAULT_TOL = 0.7


def tol(code):
    if code in VIS: return TOL["vis"]
    if code.endswith("_target"): return TOL["target"]
    return TOL.get(code, DEFAULT_TOL)


def match(pred, human):
    """greedy one-to-one match of equal codes within tolerance, closest pairs first"""
    pairs = sorted((abs(p[1] - h[1]), i, j) for i, p in enumerate(pred) for j, h in enumerate(human)
                   if p[0] == h[0] and abs(p[1] - h[1]) <= tol(p[0]))
    used_p, used_h, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_p or j in used_h: continue
        used_p.add(i); used_h.add(j); out.append((pred[i], human[j]))
    return out, [p for i, p in enumerate(pred) if i not in used_p], [h for j, h in enumerate(human) if j not in used_h]


def split(steps):
    s = [(x["code"], float(x["t"])) for x in steps if not (x["code"] in VIS and float(x["t"]) < 0.5)]
    return ([x for x in s if x[0] not in VIS and x[0] != "wrist_blind"], [x for x in s if x[0] in VIS], [x for x in s if x[0] == "wrist_blind"])


def score(corr, pred):
    n = collections.Counter(); dt = collections.defaultdict(list); rows = []
    for c in corr:
        key = (c["dataset"], c["episode"])
        if key not in pred: continue
        p = pred[key]
        pb, pv, pw = split(p["steps"]); hb, hv, hw = split(c["human_steps"])
        mb, xb, mhb = match(pb, hb); mv, xv, mhv = match(pv, hv); mw, xw, mhw = match(pw, hw)
        for a, b in mb: dt[a[0].split("_", 1)[-1]].append(abs(a[1] - b[1]))
        seq_ok = [x[0] for x in sorted(pb, key=lambda x: x[1])] == [x[0] for x in sorted(hb, key=lambda x: x[1])]
        start_ok = p["start_state"] == c["human_start"]
        ball_ok = not xb and not mhb
        n["eps"] += 1; n["start_ok"] += start_ok; n["ball_ok"] += ball_ok; n["seq_ok"] += seq_ok
        n["unchanged"] += start_ok and ball_ok and not xv and not mhv and not xw and not mhw
        n["ball_h"] += len(hb); n["ball_m"] += len(mb); n["ball_x"] += len(xb)
        n["vis_h"] += len(hv); n["vis_m"] += len(mv); n["vis_x"] += len(xv)
        n["wb_h"] += len(hw); n["wb_m"] += len(mw); n["wb_x"] += len(xw)
        rows.append((c["episode"], start_ok, [f"{a}@{t}" for a, t in mhb], [f"{a}@{t}" for a, t in xb],
                     [f"{a}@{t}" for a, t in mhv], [f"{a}@{t}" for a, t in xv], [f"{a}@{t}" for a, t in mhw], [f"{a}@{t}" for a, t in xw]))
    return n, dt, rows


def report(name, n, dt, rows, verbose):
    e = n["eps"]
    print(f"== {name}: {e} episodes")
    print(f"  unchanged (all events, start, vis, wrist_blind): {n['unchanged']}/{e}")
    print(f"  ball events need no edit: {n['ball_ok']}/{e} (ball-event sequence identical: {n['seq_ok']}/{e})")
    print(f"  ball events: recall {n['ball_m']}/{n['ball_h']}, extra {n['ball_x']}")
    print(f"  start state: {n['start_ok']}/{e}")
    print(f"  visibility changes: recall {n['vis_m']}/{n['vis_h']}, extra {n['vis_x']}")
    print(f"  wrist_blind: recall {n['wb_m']}/{n['wb_h']}, extra {n['wb_x']}")
    print("  timing |error| median/max s: " + ", ".join(f"{k} {sorted(v)[len(v) // 2]:.2f}/{max(v):.2f}" for k, v in sorted(dt.items())))
    if verbose:
        for ep, st, mh, x, mhv, xv, mhw, xw in rows:
            if mh or x or mhv or xv or mhw or xw or not st:
                print(f"  ep {ep}: start {'ok' if st else 'WRONG'} | missed {mh} extra {x} | vis missed {mhv} extra {xv} | wb missed {mhw} extra {xw}")


def main():
    corr = [json.loads(l) for l in open(sys.argv[1])]
    v1 = {(c["dataset"], c["episode"]): dict(steps=c["prelabel_steps"], start_state=c["prelabel_start"]) for c in corr}
    report("v1", *score(corr, v1), verbose=len(sys.argv) < 3)
    if len(sys.argv) > 2:
        pred = {}
        for l in open(sys.argv[2]):
            r = json.loads(l); pred[(r["dataset"], r["episode"])] = r
        report(sys.argv[2].rsplit("/", 1)[-1], *score(corr, pred), verbose=True)


if __name__ == "__main__": main()
