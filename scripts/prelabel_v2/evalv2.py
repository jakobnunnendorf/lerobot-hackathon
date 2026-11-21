# Written by Claude
import json, sys
from gt import load
from score import score, summarise
import v2
from drafts import draft_steps
from clears import add_missing_clears
gt = load(); F = json.load(open('f_0_49.json')); state = v2.load_state()
def run(use, eps=None):
    res = []
    for e in sorted(gt):
        g = gt[e]
        if g['v1'] is None: continue
        base = [dict(g['v1'][i]) for i in range(len(g['v1']))]
        if 'clears' in use:
            d = draft_steps('so101_pick_rb_mixed', e)
            if d: base = [dict(code=c, t=t) for c, t in add_missing_clears([(x['code'], x['t']) for x in base], d)]
        steps, start, m = v2.build(base, F, state, e, g['v1start'], use)
        steps = [(c, t) for c, t in steps if c != '?_attempt']
        res.append(score([dict(code=c, t=t, label='') for c, t in steps], start if 'start' in use else g['v1start'], g))
    return summarise(res)
if __name__ == '__main__':
    for use in [(), ('start',), ('retime',), ('vis',), ('wb',), ('misscolour',), ('start', 'vis', 'wb', 'retime', 'misscolour'), ('start', 'vis', 'wb', 'retime', 'misscolour', 'clears')]:
        print(use, run(use))
