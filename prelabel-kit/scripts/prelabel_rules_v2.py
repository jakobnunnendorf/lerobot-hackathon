# Written by Claude (pre-label v2)
"""v2 rules: a draft step list with explicit vision checks, plus one review sheet per episode with the frames for those checks.
usage: prelabel_rules_v2.py PL_DIR DATASET EPS   -> PL_DIR/<ds>/<ep>/draft.txt and review.jpg (+review_2.jpg ...)
Why each rule exists: FAILURES.md. The labeller answers the checks on review.jpg and writes the final steps."""
import sys, os, json, glob
from PIL import Image, ImageDraw, ImageFont


def parse(s):
    r = []
    for p in s.split(","):
        a, _, b = p.partition("-")
        r += range(int(a), int(b or a) + 1)
    return r

GRAB_OFF, HOME_OFF, TARGET_LEAD = 0.2, 0.35, 1.8
QUICK_OPEN_S = 0.6
HIGH_LIFT = -50


def zone(arm):
    pan, lift = arm[0], arm[1]
    if lift < -80: return "home"
    # pan alone: blue-hole drops happen at lift -6..-12, which v1's "lift > -5" called bowl (FAILURES.md class 5)
    if pan <= -25: return "red_hole"
    if pan >= 20: return "blue_hole"
    # buried_ball drops foil far out in front (pan about -12, lift +20..+40); the bowl itself ends near lift -10
    if lift > 10: return "front_hole"
    return "bowl"


def attempts(ev):
    """bowl closes that Jakob labels, each with how it ended: held (released later), quick (opened within 0.6 s in the bowl)"""
    E = []
    for e in ev["gripper_events"]:
        if zone(e["arm"]) == "home": continue
        # a close at a hole right after a release there is the gripper shutting; elsewhere (bowl rim at pan ~20) it is a grab
        if e["kind"] == "close" and zone(e["arm"]) != "bowl" and E and E[-1]["kind"] == "open" and zone(E[-1]["arm"]) != "bowl" and e["t"] - E[-1]["t_end"] < 2.5: continue
        if E and E[-1]["kind"] == "close" and e["kind"] == "close":
            E[-1] = dict(E[-1], t_end=e["t_end"], g_after=e["g_after"])
        else: E.append(dict(e))
    out, skipped = [], []
    for k, e in enumerate(E):
        if e["kind"] != "close": continue
        nxt = E[k + 1] if k + 1 < len(E) else None
        last = all(x["kind"] != "close" for x in E[k + 1:])
        quick = nxt is not None and nxt["kind"] == "open" and zone(nxt["arm"]) == "bowl" and nxt["t"] - e["t_end"] < QUICK_OPEN_S
        a = dict(t=e["t"], t_end=e["t_end"], g=e["g_after"], lift=e["arm"][1], pan=e["arm"][0], quick=quick,
                 open_mid=nxt["mid_t"] if nxt is not None and nxt["kind"] == "open" else None)
        if e["g_after"] > 16.5 and last and nxt is None and e["g_after"] <= 21:
            a["g"] = 16.5; skipped.append(f"last close {e['t']}-{e['t_end']} only to {e['g_after']} but never reopened: taken as the final grab"); out.append(a); continue
        if e["g_after"] > 16.5: a["skip"] = True; out.append(a); skipped.append(f"partial pinch {e['t']}-{e['t_end']} (gripper only to {e['g_after']}): not labelled"); continue
        if e["arm"][1] < HIGH_LIFT and not (8 <= e["g_after"] and not quick):
            a["skip"] = True; out.append(a); skipped.append(f"close high above the bowl at {e['t']} (lift {e['arm'][1]}): not labelled"); continue
        out.append(a)
    return E, out, skipped


def releases(E):
    return [e for e in E if e["kind"] == "open" and zone(e["arm"]) != "home"]


def draft(ev):
    foil = ev["dataset"] != "so101_pick_rb_mixed"
    E, A_all, skipped = attempts(ev)
    A = [a for a in A_all if not a.get("skip")]
    skip_t = []
    R = releases(E)
    wf = ev.get("wflex_10hz", [])
    steps, checks, prev_t = [], [], 0.0
    first_target = None
    for n, a in enumerate(A):
        t_att = round(a["t_end"] - GRAB_OFF, 2)
        tg = t_att - TARGET_LEAD
        if not steps:
            idx = [i for i, v in enumerate(wf) if v >= 96 and i / 10 < t_att]
            h = idx[-1] / 10 + 0.1 if idx else None
            op = [x for x in R if x["t"] < a["t"] and zone(x["arm"]) == "bowl" and x["g_after"] >= 12]
            c = []
            if h is not None and t_att - 5.3 < h < t_att - 0.8: c.append(h)
            if op and t_att - 6 < op[0]["t"] - 0.2 < t_att - 0.8: c.append(op[0]["t"] - 0.2)
            if c: tg = sum(c) / len(c)
            first_target = tg
        # Jakob puts the target before the first try at a ball, also before a regrasp he leaves unlabelled (ep38, 44, 49);
        # not before partial pinches or high closes (ep17, 33, 47)
        early = [t for t in skip_t if prev_t < t < a["t"]]
        if early: tg = min(tg, early[0] - GRAB_OFF - TARGET_LEAD)
        tg = round(min(max(tg, prev_t + 0.6), t_att - 0.6), 2)
        nxt_att = A[n + 1] if n + 1 < len(A) else None
        rel = next((r for r in R if r["t"] >= a["t_end"] - 0.3 and (nxt_att is None or r["t"] < nxt_att["t"])), None)
        carried = rel is not None and zone(rel["arm"]) != "bowl"
        # a squashed foil ball (or a ball held at the jaw tips) closes the gripper below 8 but still reaches the hole
        if a["g"] < 8 and not carried:
            steps += [["?_target", tg], ["?_miss", t_att]]
            checks.append(dict(kind="colour", t0=a["t"], t1=a["t_end"], at=t_att, ask=f"miss at {t_att}: which ball is between the jaw tips (bottom centre of the wrist view)? r/b/s"))
            prev_t = t_att
        elif a["quick"] and not carried:
            if nxt_att and nxt_att["t"] - a["open_mid"] < 2.5 and nxt_att["g"] >= 8:
                checks.append(dict(kind="quick", t0=a["t"], t1=a["open_mid"] + 0.4, at=t_att,
                                   ask=f"quick close {a['t']}-{a['t_end']} (gripper to {a['g']}) opened at {a['open_mid']}, next close {nxt_att['t']}: "
                                       f"same ball regrasped -> no label (default); different ball or ball lifted and lost -> X_grab@{t_att}+X_slip@{a['open_mid']} or X_miss@{t_att}"))
                skipped.append(f"quick regrasp {a['t']}-{a['open_mid']} before the next close: not labelled unless the check says otherwise")
                skip_t.append(a["t_end"]); skip_t.sort()
                continue
            steps += [["?_target", tg], ["?_miss", t_att]]
            checks.append(dict(kind="quick", t0=a["t"], t1=a["open_mid"] + 0.4, at=t_att,
                               ask=f"quick close {a['t']}-{a['t_end']} (gripper to {a['g']}) opened at {a['open_mid']}: ball lifted then lost -> X_grab@{t_att}+X_slip@{a['open_mid']}, "
                                   f"never held -> X_miss@{t_att} (default); which ball X at the jaw tips?"))
            prev_t = t_att
        else:
            steps += [["?_target", tg], ["?_grab", t_att]]
            prev_t = t_att
            if rel is None:
                steps[-2][0], steps[-1][0] = "s_target", "s_grab"
            else:
                z = zone(rel["arm"])
                if foil and z != "bowl": c, code = "r", "r_hole"
                elif z == "red_hole": c, code = "r", "r_hole"
                elif z == "blue_hole": c, code = "b", "b_hole"
                elif z == "front_hole":
                    c, code = "?", "?_hole"
                    checks.append(dict(kind="drop", t0=rel["t"] - 0.4, t1=rel["t_end"] + 0.4, at=rel["mid_t"], ask=f"release out of the bowl at {rel['mid_t']}: which hole (r/b)?"))
                else:
                    c, code = "?", "?_slip"
                    checks.append(dict(kind="drop", t0=rel["t"] - 0.4, t1=rel["t_end"] + 0.4, at=rel["mid_t"],
                                       ask=f"ball released inside the bowl at {rel['mid_t']}: which ball (r/b/s), and slip/drop or deliberate put back?"))
                steps[-2][0] = f"{c}_target"; steps[-1][0] = f"{c}_grab"
                steps.append([code, rel["mid_t"]]); prev_t = rel["mid_t"]
    home = round(ev["arm_home_s"] + HOME_OFF, 2)
    steps.append(["home", home])
    for c in checks:
        if c["kind"] in ("colour", "quick"):
            nxt = next((k[0] for k, t in steps if t > c["at"] + 0.3 and k.endswith("_grab") and k[0] != "?"), None)
            if nxt: c["ask"] += f" (prior only: the next grab is {nxt}; wrong in 2 of 8 of Jakob's cases)"
    return steps, checks, skipped, first_target


def windows(ev, steps, first_target):
    L = ev["length_s"]
    w = []
    ft = first_target if first_target is not None else 3.0
    w.append(dict(kind="start", times=[0.4] + [round(ft - 1.6 + 0.4 * k, 1) for k in range(6)], ask="start state: top tile 1 = bowl at rest; wrist tiles = first approach. Stripe identifiable in the first wrist overview? none -> buried; yes -> partly; top shows the whole outline free -> visible"))
    for c, t in steps:
        if c and c.split("_")[-1] in ("hole", "miss", "slip"):
            w.append(dict(kind="vis", times=[round(t + 0.2 + 0.4 * k, 1) for k in range(7) if t + 0.2 + 0.4 * k < L], ask=f"after {c}@{t}: first wrist frame where striped visibility changed vs before (uncovered -> s_partly / s_vis, covered again -> s_buried)?"))
        if c in ("s_grab", "s_miss", "?_miss", "?_grab"):
            w.append(dict(kind="blind", times=[round(t - 3.0 + 0.4 * k, 1) for k in range(9) if t - 3.0 + 0.4 * k > 0.2], ask=f"before {c}@{t}: stripe in the wrist view, then gone while still approaching? -> wrist_blind at the first frame without it"))
    return w


def sheet(pl, ds, ep, rows, out):
    d = f"{pl}/{ds}/{ep}/f"
    have = {cam: {float(os.path.basename(p)[len(cam) + 1:-4]): p for p in glob.glob(f"{d}/{cam}_*.jpg")} for cam in ("top", "wrist")}
    keys = sorted(have["wrist"])
    snap = lambda t: min(keys, key=lambda k: abs(k - t))
    TW, TH, TT, LBL = 176, 132, 132, 22
    font = ImageFont.load_default(size=14)
    files, per = [], 7
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        width = max(len(r["times"]) for r in chunk) * (TW + 4) + TT + 4
        im = Image.new("RGB", (width, len(chunk) * (TH + LBL + 4)), (20, 20, 20)); dr = ImageDraw.Draw(im)
        for i, r in enumerate(chunk):
            y = i * (TH + LBL + 4)
            dr.text((4, y + 3), f"{r['id']}: {r['title']}", fill=(120, 220, 255), font=font)
            y += LBL
            top_t = r.get("top_t", r["times"][0])
            im.paste(Image.open(have["top"][snap(top_t)]).resize((TT, TT)), (0, y))
            dr.text((3, y + 2), f"top {snap(top_t):.1f}", fill=(255, 255, 0), font=font)
            for k, t in enumerate(r["times"]):
                x = TT + 4 + k * (TW + 4)
                im.paste(Image.open(have["wrist"][snap(t)]).resize((TW, TH)), (x, y))
                dr.rectangle([x, y, x + 44, y + 16], fill=(0, 0, 0)); dr.text((x + 2, y + 1), f"{snap(t):.1f}", fill=(255, 255, 0), font=font)
        fn = out + ("" if s == 0 else f"_{s // per + 1}") + ".jpg"
        im.save(fn, quality=80); files.append(os.path.basename(fn))
    return files


def run(pl, ds, ep):
    ev = json.load(open(f"{pl}/{ds}/{ep}/events.json"))
    steps, checks, skipped, ft = draft(ev)
    rows = []
    for n, w in enumerate(windows(ev, steps, ft)):
        rows.append(dict(id=f"{w['kind'].upper()}{n}", title=w["ask"][:150], ask=w["ask"], times=w["times"], top_t=w["times"][0] if w["kind"] != "start" else 0.4))
    for n, c in enumerate(checks):
        ts = [round(c["t0"] - 0.4 + 0.2 * k, 1) for k in range(int((c["t1"] - c["t0"] + 0.8) / 0.2) + 1)][:9]
        rows.append(dict(id=f"Q{n}", title=c["ask"][:150], ask=c["ask"], times=ts, top_t=c["at"]))
    files = sheet(pl, ds, ep, rows, f"{pl}/{ds}/{ep}/review")
    lines = [f"{ds} ep {ep}  length {ev['length_s']} s  review sheets: {', '.join(files)}",
             "draft steps (rules; ? = colour to decide on the sheet):",
             "  " + " ".join(f"{c}@{t}" for c, t in steps),
             "checks (row id on the sheet: question):"] + [f"  {r['id']}: {r['ask']}" for r in rows]
    if skipped: lines += ["not labelled by rule (Jakob skips these):"] + [f"  {s}" for s in skipped]
    open(f"{pl}/{ds}/{ep}/draft.txt", "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    pl, ds = sys.argv[1], sys.argv[2].replace("ds/", "")
    for ep in parse(sys.argv[3]):
        print("\n".join(run(pl, ds, ep))); print()
