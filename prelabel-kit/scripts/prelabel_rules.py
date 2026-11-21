"""Propose a step list from events.json (rules from METHOD.md). Ball identity of the non-final grabs comes from the
pan angle at release (red hole pan<-20, blue hole pan>15); putbacks / the final ball must be checked on the sheet.
usage: prelabel_rules.py DATASET EP[,A-B] -> prints JSON proposals"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prelabel_prep import parse, OUT
GRAB_OFF, HOME_OFF, TARGET_LEAD = 0.2, 0.35, 1.8   # grab = close t_end - 0.2 ; home = arm_home + 0.35 ; target = grab - 1.8

def zone(arm):
    pan, lift = arm[0], arm[1]
    if pan < -20 and lift > 0: return "red_hole"
    if pan > 15 and lift > -5: return "blue_hole"
    if lift < -80: return "home"
    return "bowl"

def propose(ev):
    E = [e for e in ev["gripper_events"] if zone(e["arm"]) != "home" and not (e["kind"] == "close" and zone(e["arm"]) in ("red_hole", "blue_hole"))]
    M = []
    for e in E:                                   # merge consecutive closes
        if M and M[-1]["kind"] == "close" and e["kind"] == "close":
            M[-1]["t_end"] = e["t_end"]; M[-1]["g_after"] = e["g_after"]
        else: M.append(dict(e))
    wf = ev.get("wflex_10hz", [])
    def hover_end(before):   # last time the wrist is still flat (>=96) before `before`; the descent to the ball starts there
        idx = [i for i, v in enumerate(wf) if v >= 96 and i / 10 < before]
        return idx[-1] / 10 + 0.1 if idx else None
    steps, notes, holding, prev_t = [], [], False, 0.0
    grabs = []
    for e in M:
        z = zone(e["arm"])
        if e["kind"] == "close" and z == "bowl" and 8 <= e["g_after"] <= 16.5:
            g = round(e["t_end"] - GRAB_OFF, 2)
            tg = g - TARGET_LEAD
            if not steps:                       # first target of the episode: mean of wrist-flex descent and last approach-open onset
                h = hover_end(g)
                op = [x for x in E if x["kind"] == "open" and x["t"] < e["t"] and zone(x["arm"]) == "bowl" and x["g_after"] >= 12]
                c = []
                if h is not None and g - 5.3 < h < g - 0.8: c.append(h)
                if op and g - 6 < op[0]["t"] - 0.2 < g - 0.8: c.append(op[0]["t"] - 0.2)
                if c: tg = sum(c) / len(c)
            tg = round(min(max(tg, prev_t + 0.6), g - 0.8), 2)
            steps += [["target", tg], ["grab", g]]; holding = True; prev_t = g
        elif e["kind"] == "close" and z == "bowl" and e["g_after"] < 8:
            g = round(e["t_end"] - GRAB_OFF, 2)
            tg = round(min(max(g - TARGET_LEAD, prev_t + 0.6), g - 0.8), 2)
            steps += [["target", tg], ["miss", g]]; prev_t = g; holding = False
            notes.append(f"close to {e['g_after']} on nothing at {e['t']}: MISS (which ball? check sheet)")
        elif e["kind"] == "close" and z == "bowl": notes.append(f"close only to {e['g_after']} at {e['t']}: partial pinch, ignored")
        elif e["kind"] == "open" and holding:
            t = e["mid_t"]
            steps.append([{"red_hole": "r_hole", "blue_hole": "b_hole"}.get(z, "putback_or_drop"), t]); holding = False; prev_t = t
    # failed pinch: grab immediately (<1 s) followed by a release inside the bowl, then a new target+grab = one attempt.
    # Keep the first target and the final grab; he does not label the pinch (ep4, ep6).
    i = 1
    while i <= len(steps) - 3:
        if steps[i][0] == "putback_or_drop" and steps[i - 1][0] == "grab" and steps[i][1] - steps[i - 1][1] < 1.3 \
           and steps[i + 1][0] == "target" and steps[i + 2][0] == "grab":
            notes.append(f"failed pinch at {steps[i-1][1]}-{steps[i][1]} ignored; ball may have slipped (check sheet)")
            del steps[i - 1:i + 2]  # drop first grab, release, second target
        else: i += 1
    home = round(ev["arm_home_s"] + HOME_OFF, 2)
    steps.append(["home", home])
    if holding: notes.append("last grab not released -> striped?")
    return steps, notes

def codes(steps):
    """name the balls: grab followed by r_hole/b_hole -> that colour; unreleased last grab -> striped; others '?' (check sheet)"""
    out = []; i = 0
    while i < len(steps):
        k, t = steps[i]
        if k == "target":
            j = next(n for n in range(i + 1, len(steps)) if steps[n][0] in ("grab", "miss"))
            kind = steps[j][0]
            nxt = steps[j + 1][0] if j + 1 < len(steps) else None
            o = {"r_hole": "r", "b_hole": "b", "home": "s"}.get(nxt, "?") if kind == "grab" else "?"
            out += [[o + "_target", t], [o + "_" + kind, steps[j][1]]]; i = j + 1
        elif k == "putback_or_drop": out.append(["?_slip_or_putback", t]); i += 1
        else: out.append([k, t]); i += 1
    return out

if __name__ == "__main__":
    ds = sys.argv[1].replace("ds/", "")
    for ep in parse(sys.argv[2]):
        ev = json.load(open(f"{OUT}/{ds}/{ep}/events.json"))
        s, n = propose(ev); print(ep, json.dumps(codes(s)), n)
