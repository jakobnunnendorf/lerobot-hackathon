# Written by Claude: read the rule drafts of the other v2 session (pl/<dataset>/<ep>/draft.txt)
import re, os
PL = os.environ['PRELABEL_PL']  # output folder of scripts/prelabel_rules_v2.py on branch Space0code/prelabel-v2
def draft_steps(dataset, ep):
    p = f'{PL}/{dataset}/{ep}/draft.txt'
    if not os.path.exists(p): return None
    lines = open(p).read().split('\n'); i = next(k for k, l in enumerate(lines) if l.startswith('draft steps'))
    return [(m.group(1), float(m.group(2))) for m in re.finditer(r'(\S+?)@([\d.]+)', lines[i + 1])]
