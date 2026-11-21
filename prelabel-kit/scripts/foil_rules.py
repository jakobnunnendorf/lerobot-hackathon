import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prelabel_prep import parse, OUT
import prelabel_rules as PR
def dist(a,b): return max(abs(x-y) for x,y in zip(a[:4],b[:4]))
def propose(ev):
    lv, hm = ev["arm_leaves_s"], ev["arm_home_s"]
    E=[e for e in ev["gripper_events"] if lv+0.2 < e["t"] < hm+0.3]
    wf=ev.get("wflex_10hz",[])
    def hover_end(before):
        idx=[i for i,v in enumerate(wf) if v>=96 and i/10<before]
        return idx[-1]/10+0.1 if idx else None
    steps,notes,holding,prev_t=[],[],False,0.0
    mpose=None; mt=-9; gpose=None; relpose=None; relt=-9; first_open=[x for x in E if x["kind"]=="open"]
    def mk(e,kind):
        g=round(e["t_end"]-0.2,2); tg=g-1.8
        if not steps and kind=="grab":
            h=hover_end(g); op=[x for x in E if x["kind"]=="open" and x["t"]<e["t"] and x["g_after"]>=12]; c=[]
            if h is not None and g-5.3<h<g-0.8: c.append(h)
            if op and g-6<op[0]["t"]-0.2<g-0.8: c.append(op[0]["t"]-0.2)
            if c: tg=sum(c)/len(c)
        return g,tg
    for e in E:
        if e["kind"]=="close":
            if holding:
                if e["g_after"]<8:   # grab turned into closed-on-nothing
                    g=round(e["t_end"]-0.2,2); steps[-1]=["miss",g]; holding=False; notes.append(f"close to {e['g_after']} on nothing at {e['t']}: MISS")
                elif 8<=e["g_after"]<=16.5: steps[-1]=["grab",round(e["t_end"]-0.2,2)]
                continue
            if relpose is not None and e["t"]-relt<3.5 and dist(e["arm"],relpose)<20:
                notes.append(f"close at {e['t']} after hole release: ignored"); continue
            if 8<=e["g_after"]<=16.5:
                g,tg=mk(e,"grab"); tg=round(min(max(tg,prev_t+0.6),g-0.8),2)
                steps+=[["target",tg],["grab",g]]; holding=True; prev_t=g; gpose=e["arm"]
            elif e["g_after"]<8:
                g,tg=mk(e,"miss"); tg=round(min(max(tg,prev_t+0.6),g-0.8),2)
                steps+=[["target",tg],["miss",g]]; prev_t=g; mpose=e["arm"]; mt=e["t_end"]
                notes.append(f"close to {e['g_after']} on nothing at {e['t']}: MISS")
            else: notes.append(f"close only to {e['g_after']} at {e['t']}: partial pinch")
        else:
            if (not holding) and steps and steps[-1][0]=="miss" and 1.5<e["t"]-mt<9 and dist(e["arm"],mpose)>=25:
                steps[-1][0]="grab"; steps.append(["hole",e["mid_t"]]); prev_t=e["mid_t"]; relpose=e["arm"]; relt=e["t_end"]; notes.append(f"'miss' at {mt} was a squeezed foil carried to hole")
            elif holding:
                t=e["mid_t"]; far=dist(e["arm"],gpose)>=25
                steps.append(["hole" if far else "putback_or_drop",t]); holding=False; prev_t=t; relpose=e["arm"] if far else None; relt=e["t_end"]
    i=1
    while i<=len(steps)-3:
        if steps[i][0]=="putback_or_drop" and steps[i-1][0]=="grab" and steps[i][1]-steps[i-1][1]<1.3 and steps[i+1][0]=="target" and steps[i+2][0]=="grab":
            notes.append(f"failed pinch {steps[i-1][1]}-{steps[i][1]} ignored"); del steps[i-1:i+2]
        else: i+=1
    steps.append(["home",round(hm+0.35,2)])
    if holding: notes.append("last grab not released -> striped")
    return steps,notes
def fmt(steps):
    # letter for each target/grab/miss: grab followed by hole -> r; last grab (home next) -> s; miss -> letter of the next grab; slip -> putback of r (foil aside) unless s
    items=[]  # (kind,t,letter)
    i=0
    while i<len(steps):
        k,t=steps[i]
        if k=="target":
            j=next(n for n in range(i+1,len(steps)) if steps[n][0] in("grab","miss")); kind=steps[j][0]
            nxt=steps[j+1][0] if j+1<len(steps) else None
            L="?"
            if kind=="grab": L={"hole":"r","home":"s","putback_or_drop":"r"}.get(nxt,"?")
            items+=[["target",t,L],[kind,steps[j][1],L]]; i=j+1
        elif k=="hole": items.append(["hole",t,"r"]); i+=1
        elif k=="putback_or_drop": items.append(["putback",t,"r"]); i+=1
        else: items.append([k,t,""]); i+=1
    # resolve misses
    for n,it in enumerate(items):
        if it[0]=="miss" and it[2]=="?":
            nxt=next((x for x in items[n+1:] if x[0]=="grab"),None); L=nxt[2] if nxt else "s"
            it[2]=L
            if n>0 and items[n-1][0]=="target": items[n-1][2]=L
    out=[]
    for k,t,L in items:
        out.append(f"{k}@{t}" if k=="home" else (f"{L}_{k}@{t}"))
    return " ".join(out)
if __name__=="__main__":
    ds=sys.argv[1]
    for ep in parse(sys.argv[2]):
        ev=json.load(open(f"{OUT}/{ds}/{ep}/events.json")); s,n=propose(ev); print(ep,f"len{ev['length_s']}",fmt(s),n)
