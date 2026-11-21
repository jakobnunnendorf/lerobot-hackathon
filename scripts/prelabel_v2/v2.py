# Written by Claude: pre-label v2 patcher. v1 steps + gripper-close patches + stripe-pixel visibility/wrist_blind rules.
import json, glob, numpy as np, pandas as pd
P = dict(open_level=20.0, vis_after_hole=0.3, vis_min_after=1.0, buried_max=30, partly_max=150, vis_thr=40, vis_sustain=3, wb_thr=3000, wb_before_grab=0.9, retime_tol=0.35, grab_after_close=0.15,
         add_attempts=True, attempt_lvl=17.5)
WORD = {"home": "Return home", "wrist_blind": "Wrist blind to striped", "s_vis": "Striped visible", "s_partly": "Striped partly covered", "s_buried": "Striped buried",
        "r_hole": "R → red hole", "b_hole": "B → blue hole"}
for o, n in {"r": "R", "b": "B", "s": "striped"}.items():
    WORD[o + "_target"] = f"Target {n}"; WORD[o + "_grab"] = f"Grabbed {n}"; WORD[o + "_miss"] = f"Missed {n}"; WORD[o + "_slip"] = f"Slip/drop {n}"; WORD[o + "_putback"] = f"{n[0].upper() + n[1:]} put back"
def load_state(root='ds'):
    d = pd.concat([pd.read_parquet(f, columns=['episode_index', 'frame_index', 'timestamp', 'observation.state']) for f in sorted(glob.glob(f'{root}/data/chunk-000/*.parquet'))])
    return d
def gripper(d, e):
    x = d[d.episode_index == e].sort_values('frame_index')
    t = x['timestamp'].values - x['timestamp'].values[0]; g = np.array([s[5] for s in x['observation.state']]); arm = np.array([s[:5] for s in x['observation.state']])
    return t, g, arm
def closes(t, g, arm):
    gs = np.convolve(g, np.ones(5) / 5, mode='same'); res = []
    for k in range(3, len(gs) - 3):
        if gs[k - 1] >= 17 > gs[k] and (not res or t[k] - res[-1]['t'] > 1.0):
            res.append(dict(t=round(float(t[k]), 2), pre=float(g[max(0, k - 20):k].max()), lvl=float(np.median(g[k + 3:k + 12])), arm=arm[k]))
    return res
def start_state(top, v1start=None):
    """hybrid: keep v1's start state unless the stripe pixels clearly disagree (none visible -> buried; plenty visible -> not buried)"""
    v = [r[1] for r in top if 0.2 <= r[0] <= 0.8]; m = float(np.median(v)) if v else 0
    if v1start is None: return ('buried' if m < P['buried_max'] else 'partly' if m < P['partly_max'] else 'visible'), m
    if v1start != 'buried' and m < P['buried_max']: return 'buried', m
    if v1start == 'buried' and m > P['partly_max']: return 'partly', m
    return v1start, m
def vis_events(top, start, steps):
    """buried/partly start: s_partly when the stripe is seen on the top camera after a clear (hole release); Jakob puts it 1-2.5 s after the release"""
    holes = sorted(t for c, t in steps if c.endswith('_hole'))
    if start == 'visible' or not holes: return []
    px = [(r[0], r[1]) for r in top]; thr = P['vis_thr']
    for h in holes:
        for i in range(len(px) - 1):
            if px[i][0] >= h + P['vis_after_hole'] and px[i][1] >= thr and px[i + 1][1] >= thr:
                if start == 'buried' or px[i][0] > h + 0.5:
                    return [('s_partly', round(max(px[i][0], h + P['vis_min_after']), 2))]
    return []
def wrist_blind(wrist, steps):
    sg = [t for c, t in steps if c == 's_grab']
    if not sg: return []
    g = sg[-1]; w = [r[1] for r in wrist if g - 2.0 <= r[0] <= g - 0.2]
    return [('wrist_blind', round(g - P['wb_before_grab'], 2))] if w and min(w) < P['wb_thr'] else []
def build(v1, F, state, e, v1start=None, use=('start', 'vis', 'wb', 'retime', 'misscolour')):
    steps = [(s['code'], s['t']) for s in v1 if s['code'] not in ('s_vis', 's_partly', 's_buried')]
    top, wrist = F[str(e)]['top'], F[str(e)]['wrist']
    start, m = start_state(top, v1start)
    t, g, arm = gripper(state, e); cl = closes(t, g, arm)
    if 'retime' in use:
        out = []
        for c, tt in steps:
            if c.endswith('_grab'):
                gi = int(np.argmin(abs(t - tt)))
                later = [x for x in cl if 0.2 < x['t'] - tt < 1.2]
                if g[gi] > P['open_level'] and later:     # grab labelled while the jaws are still open: it is the next close
                    tt = round(later[0]['t'] + P['grab_after_close'], 2)
            out.append((c, tt))
        steps = out
    if 'misscolour' in use:
        steps = sorted(steps, key=lambda x: x[1])
        for i, (c, tt) in enumerate(steps):
            if c != 's_miss': continue
            nxt = next(((c2, t2) for c2, t2 in steps[i + 1:] if c2.endswith(('_target', '_grab')) and t2 - tt < 2.5), None)
            if nxt and nxt[0][0] in 'rb':
                col = nxt[0][0]; steps[i] = (col + '_miss', tt)
                for j in range(i - 1, -1, -1):
                    if steps[j][0] == 's_target' and tt - steps[j][1] < 3.5: steps[j] = (col + '_target', steps[j][1]); break
                    if steps[j][0].endswith(('_grab', '_hole', '_miss')): break
    if 'attempts' in use:
        holes = [tt for c, tt in steps if c.endswith('_hole')]; labelled = [tt for c, tt in steps if c.endswith(('_grab', '_miss', '_slip'))]
        for c in cl:
            if c['t'] < 2 or c['pre'] < 22 or c['lvl'] > P['attempt_lvl']: continue
            if any(abs(c['t'] - x) <= 0.9 for x in labelled) or any(0 <= c['t'] - h <= 3.5 for h in holes): continue
            steps.append(('?_attempt', c['t']))
    if 'vis' in use: steps += vis_events(top, start, steps)
    if 'wb' in use: steps += wrist_blind(wrist, steps)
    return sorted(steps, key=lambda x: x[1]), start, m
