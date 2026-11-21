# Written by Claude (scratch): score proposed step lists against Jakob's final labels
VIS = ('s_vis', 's_partly', 's_buried')
def split(steps):
    ball = [(s['code'], s['t']) for s in steps if s['code'] not in VIS + ('wrist_blind',) and s['t'] is not None]
    vis = [(s['code'], s['t']) for s in steps if s['code'] in VIS and s['t'] > 0.05]
    wb = [(s['code'], s['t']) for s in steps if s['code'] == 'wrist_blind']
    return sorted(ball, key=lambda x: x[1]), sorted(vis, key=lambda x: x[1]), wb
def match(pred, true, tol):
    used = set(); hit = 0; errs = []
    for c, t in pred:
        best = None
        for i, (c2, t2) in enumerate(true):
            if i in used or c2 != c or abs(t2 - t) > tol: continue
            if best is None or abs(t2 - t) < abs(true[best][1] - t): best = i
        if best is not None: used.add(best); hit += 1; errs.append(abs(true[best][1] - t))
    return hit, len(pred), len(true), errs
def score(pred_steps, pred_start, gt):
    pb, pv, pw = split(pred_steps); tb, tv, tw = split(gt['steps'])
    r = dict(start=pred_start == gt['start'], start2=(pred_start == 'buried') == (gt['start'] == 'buried'))
    r['ball'] = match(pb, tb, 0.6); r['vis'] = match(pv, tv, 2.0); r['wb'] = match(pw, tw, 1.5)
    r['ball_exact'] = r['ball'][0] == r['ball'][1] == r['ball'][2]
    return r
def summarise(results):
    n = len(results); out = {}
    out['n'] = n; out['start_exact'] = sum(r['start'] for r in results) / n; out['start_buried_vs_not'] = sum(r['start2'] for r in results) / n
    for k in ('ball', 'vis', 'wb'):
        h = sum(r[k][0] for r in results); p = sum(r[k][1] for r in results); t = sum(r[k][2] for r in results)
        out[k + '_prec'] = round(h / p, 3) if p else None; out[k + '_rec'] = round(h / t, 3) if t else None
    out['ball_episodes_exact'] = sum(r['ball_exact'] for r in results)
    return out
