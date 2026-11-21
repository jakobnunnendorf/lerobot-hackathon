# Written by Claude: add clears (target, grab, hole) the draft found but v1 dropped (blue-hole drops at lift -6..-12 read as bowl releases in v1)
def add_missing_clears(steps, draft):
    out = list(steps)
    have = [t for c, t in steps if c.endswith('_hole')]
    for k, (c, t) in enumerate(draft):
        if c in ('r_hole', 'b_hole') and not any(abs(t - h) < 1.5 for h in have):
            col = c[0]; grab = next(((c2, t2) for c2, t2 in reversed(draft[:k]) if c2 == col + '_grab'), None)
            targ = next(((c2, t2) for c2, t2 in reversed(draft[:k]) if c2 == col + '_target'), None)
            if grab and targ and t - grab[1] < 6 and not any(abs(grab[1] - t2) < 1.0 for c2, t2 in steps if c2.endswith('_grab')):
                out += [targ, grab, (c, t)]
    return sorted(out, key=lambda x: x[1])
