# Written by Claude (scratch): ground truth from Jakob's final labels (corrections for 13-49, jakob.jsonl for 0-12)
import json
def load():
    gt = {}
    for l in open('kit/corrections.jsonl'):
        r = json.loads(l); gt[r['episode']] = dict(start=r['human_start'], steps=r['human_steps'], v1=r['prelabel_steps'], v1start=r['prelabel_start'])
    last = {}
    for l in open('jakob.jsonl'):
        r = json.loads(l); last[r['episode']] = r
    for e in range(0, 13):
        if e in last: gt[e] = dict(start=last[e].get('start_state'), steps=last[e].get('steps'), v1=None, v1start=None)
    return gt
