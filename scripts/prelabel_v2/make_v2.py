# Written by Claude: build prelabel-v2 records = v1 steps patched by v2 rules (see v2.py), same record format as prelabels_v1.jsonl
import sys, json, glob, datetime, pandas as pd
import v2
NAME = {"r": "R", "b": "B", "s": "striped"}
START = {"visible": "Striped visible", "partly": "Striped partly covered", "buried": "Striped buried"}
def summarize(codes, start):
    parts = [START[start]]; i = 0
    while i < len(codes):
        n = 1
        while i + n < len(codes) and codes[i + n] == codes[i]: n += 1
        parts.append(v2.WORD[codes[i]] + (f" ×{n}" if n > 1 else "")); i += n
    return ". ".join(parts) + "."
def verdict(c):
    bad = c.count("s_miss") + c.count("s_slip")
    drop = any(x.endswith("_putback") for x in c) or "end_empty" in c or "end_away" in c or bad >= 2
    return "drop" if drop else ("keep" if "s_grab" in c and "home" in c else "look")
def main(dataset, root, feats_json, v1_path, lo, hi, out):
    F = json.load(open(feats_json)); state = v2.load_state(root)
    foil = dataset != 'so101_pick_rb_mixed'
    use = ('start', 'retime', 'misscolour', 'attempts') if foil else ('start', 'vis', 'wb', 'retime', 'misscolour', 'attempts')
    n = 0
    with open(out, 'a') as fo:
        for l in open(v1_path):
            r = json.loads(l)
            if r['dataset'] != 'ds/' + dataset or not lo <= r['episode'] <= hi or str(r['episode']) not in F or 'top' not in F[str(r['episode'])] or 'wrist' not in F[str(r['episode'])]: continue
            steps, start, m = v2.build(r['steps'], F, state, r['episode'], r['start_state'], use)
            att = [t for c, t in steps if c == '?_attempt']; steps = [(c, t) for c, t in steps if c != '?_attempt']
            codes = [c for c, _ in steps]
            summ = summarize(codes, start)
            notes = [r['uncertain']] if r.get('uncertain') and r['uncertain'] != 'none' else []
            if att: notes.append('unlabelled gripper closes (jaws open before, closed below 17.5, not a hole return) at ' + ', '.join(f'{t:.1f}s' for t in att) + ': not added, Jakob leaves most failed pinches unlabelled')
            if start != r['start_state']: notes.append(f"v2 start state {start} (v1 {r['start_state']}): stripe pixels at start = {int(m)}")
            if foil: notes.append("v2 rules for visibility changes and wrist_blind were calibrated on rb_mixed only and are not applied here; foil balls logged as R")
            rec = dict(dataset=r['dataset'], episode=r['episode'], steps=[dict(t=round(float(t), 2), code=c, label=v2.WORD[c]) for c, t in steps], start_state=start,
                       summary=summ, note=summ, verdict_suggestion=verdict(codes), confidence=r['confidence'], uncertain='; '.join(notes) or 'none',
                       by='prelabel-v2-claude', time=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n'); n += 1
    print(dataset, 'wrote', n)
if __name__ == '__main__': main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
