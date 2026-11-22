# Written by Claude (Space0code's agent)
"""Per-episode and per-segment table of the striped-ball pick episodes: wrist blindness before the striped grab,
clearing, failures and timing. Writes episodes.csv and segments.csv. Findings: research/label-patterns.md.
usage: label_patterns.py --pl PL --labels jakob.jsonl --prelabels prelabels_v2_claude.jsonl --out DIR
PL is the frame folder written by prelabel-kit/scripts/prelabel_prep_v2.py (branch Space0code/prelabel-v2):
PL/<dataset>/<ep>/events.json and f/wrist_*.jpg, f/top_*.jpg at 5 fps."""
import argparse, csv, glob, json, os
from multiprocessing import Pool
import numpy as np
from PIL import Image

# a solid stripe patch: 13+ cells of 16x16 px that are 20% mint/yellow-green; slivers stay below it.
# Against Jakob's wrist_blind on rb_mixed 0-57: precision 18/19, recall 18/21 at a 0.6 s blind threshold.
SEEN_CELLS = 13
BLIND_S = 0.6
FAIL = ("miss", "slip", "putback")


def wrist_cells(path):
    hsv = np.asarray(Image.open(path).convert("HSV")).astype(int)
    m = (hsv[..., 0] >= 40) & (hsv[..., 0] <= 124) & (hsv[..., 1] > 50) & (hsv[..., 2] > 70)
    return int((m.reshape(15, 16, 20, 16).mean(axis=(1, 3)) >= 0.2).sum())


def wrist_series(d):
    ps = sorted(glob.glob(d + "/f/wrist_*.jpg"))
    return d, ([float(os.path.basename(p)[6:-4]) for p in ps], [wrist_cells(p) for p in ps])


def load_labels(jakob, pre):
    lab = {}
    for path, src in ((pre, "pre"), (jakob, "jakob")):
        for l in open(path):
            r = json.loads(l); lab[(r["dataset"].replace("ds/", ""), r["episode"])] = dict(r, src=src)
    return lab


def steps(r):
    return sorted(((s["code"], float(s["t"])) for s in r["steps"]), key=lambda x: x[1])


def longest_idle(traj):
    best = run = 0.0
    for a, b in zip(traj, traj[1:]):
        if b[2] > -80 and max(abs(x - y) for x, y in zip(a[1:6], b[1:6])) < 1.5: run += 0.5; best = max(best, run)
        else: run = 0.0
    return best


def episode_row(ds, ep, r, ev, T, W):
    st = steps(r); codes = [c for c, _ in st]
    sg = [t for c, t in st if c == "s_grab"]; g = sg[-1] if sg else None
    s_tg = [t for c, t in st if c == "s_target"]
    tg = max([t for t in s_tg if g and t < g] or [g - 3 if g else 0])
    seen = [t for t, c in zip(T, W) if g and tg - 1.0 <= t <= g - 0.2 and c >= SEEN_CELLS]
    blind_s = round(g - seen[-1], 1) if seen else None
    never = int(g is not None and not seen)
    holes = [c[0].upper() for c, _ in st if c in ("r_hole", "b_hole")]
    n = lambda *k: sum(codes.count(x) for x in k)
    gev = [e for e in ev["gripper_events"] if e["kind"] == "close" and g and abs(e["t_end"] - 0.2 - g) < 0.8]
    return dict(dataset=ds, episode=ep, label_source=r["src"], start_state=r.get("start_state"), length_s=ev["length_s"],
                start_delay_s=ev["arm_leaves_s"], clears="".join(holes), direct=int(not holes),
                s_miss=n("s_miss"), s_slip=n("s_slip"), rb_miss=n("r_miss", "b_miss"), rb_slip=n("r_slip", "b_slip"),
                putback=sum(c.endswith("putback") for c in codes), s_grab_t=g, striped_approach_s=round(g - tg, 1) if g else None,
                wrist_blind_s=blind_s, wrist_never_clear=never, wrist_blind=int(bool(never or (blind_s or 0) >= BLIND_S)),
                grab_pan=gev[-1]["arm"][0] if gev else None, grab_lift=gev[-1]["arm"][1] if gev else None,
                longest_idle_s=longest_idle(ev["traj_half_second"]))


def segment_rows(row, r):
    """contiguous frame ranges: one per cleared ball (anchor .. drop + 0.5 s) and the striped pick (last anchor .. home)"""
    st = steps(r)
    home = next((t for c, t in st if c == "home"), row["length_s"])
    out, anchor, cur = [], 0.0, []
    for c, t in st:
        if c[:2] not in ("r_", "b_", "s_") or c in ("s_vis", "s_partly", "s_buried"): continue
        cur.append(c)
        if c in ("r_hole", "b_hole"):
            out.append(dict(dataset=row["dataset"], episode=row["episode"], kind="clear_" + c[0], t0=anchor, t1=round(t + 0.5, 2),
                            clean=int(not any(x.split("_")[1] in FAIL for x in cur)), wrist_blind=""))
            anchor, cur = round(t + 0.5, 2), []
    out.append(dict(dataset=row["dataset"], episode=row["episode"], kind="pick_s", t0=anchor, t1=home,
                    clean=int(not any(x.split("_")[1] in FAIL for x in cur)), wrist_blind=row["wrist_blind"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("pl", "labels", "prelabels", "out"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    lab = load_labels(a.labels, a.prelabels)
    dirs = [f"{a.pl}/{ds}/{ep}" for ds, ep in sorted(lab)]
    with Pool(4) as p: ser = dict(p.map(wrist_series, dirs))
    eps, segs = [], []
    for (ds, ep), r in sorted(lab.items()):
        d = f"{a.pl}/{ds}/{ep}"
        row = episode_row(ds, ep, r, json.load(open(d + "/events.json")), *ser[d])
        eps.append(row); segs += segment_rows(row, r)
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("episodes.csv", eps), ("segments.csv", segs)):
        with open(f"{a.out}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(len(eps), "episodes,", len(segs), "segments ->", a.out)


if __name__ == "__main__": main()
