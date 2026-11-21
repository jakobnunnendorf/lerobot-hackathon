import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_buried_blue import load, P
from prelabel_write import write_prelabel
DS="ds/so101_pick_blue_ball"
prop=load("pl2/prop_blue.txt")
PARTLY={15,44,49,63,64,67,75,86,87,89,92,95,98}; BURIED={58,81}
SPECIAL={
33:("r_target@10.53 r_miss@12.33 r_target@16.43 r_grab@18.23 r_hole@25.46 home@26.98","drop"),
49:("r_target@2.96 r_grab@7.57 r_putback@9.37 s_target@15.2 s_grab@17.0 home@22.25","look"),
81:("r_target@10.7 r_grab@12.53 r_putback@16.27 s_target@16.53 s_grab@17.33 home@19.92","look"),
55:("s_target@11.0 s_miss@12.8 "+" ".join(prop[55][0].split()[3:]),None),
57:("s_target@8.9 s_miss@10.57 "+" ".join(prop[57][0].split()[3:]),None),
67:("s_target@5.4 s_miss@7.2 "+" ".join(prop[67][0].split()[3:]),None),
}
n=0
for ep in range(100):
    steps,notes=prop[ep]; vo=None; un=[]
    if ep in SPECIAL: steps,vo=SPECIAL[ep]
    elif any(x.startswith("r_") for x in steps.split()):
        steps=" ".join(x for x in steps.split() if not x.startswith("r_"))
        un.append("startup gripper twitches/early closes before the arm reached the bowl (R events) removed")
    st="buried" if ep in BURIED else ("partly" if ep in PARTLY else "visible")
    if ep in SPECIAL and ep in(33,49,81): un.append("foil balls logged as R (r_ codes); r_putback = foil moved aside inside the bowl, not a real put-back")
    for x in notes:
        if "MISS" in x: un.append(x.replace("MISS","labelled miss on striped (foil around it)"))
    if ep==33: un.append("human hand in the scene at 26-30 s; episode ends empty with striped left in the bowl")
    if ep in (55,57,67): un.append("early grab+release under 1 s on striped logged as Missed striped; foil not involved")
    if ep in PARTLY|BURIED: un.append("start state judged from the top-camera tile, striped sits at the bowl edge/partly hidden, could be called visible" if ep in PARTLY else "start state: striped not seen on top camera")
    miss=any("_miss" in x for x in steps.split())
    conf="low" if ep==33 else ("medium" if (miss or ep in PARTLY|BURIED|set(SPECIAL)) else "high")
    if write_prelabel(DS,ep,st,P(steps),conf,"; ".join(un) or "none",vo): n+=1
print(n)
