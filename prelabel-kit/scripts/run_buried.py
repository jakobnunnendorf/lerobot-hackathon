import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_buried_blue import run
SL="sliver of striped visible, could be buried"
S={0:("partly",""),1:("partly",SL),2:("partly",""),3:("partly",""),4:("partly",""),5:("partly",""),6:("buried","none visible"),7:("buried","none visible"),
8:("partly",SL),9:("partly",SL),10:("partly",SL),11:("partly",SL),12:("partly",SL),13:("buried","none visible"),14:("buried","none visible"),15:("partly",SL),
16:("partly",SL),17:("partly",SL),18:("partly",SL),19:("partly",""),20:("partly",SL),21:("partly",SL),22:("buried","none visible"),23:("buried","none visible"),
24:("partly",SL),25:("buried","none visible"),26:("buried","none visible"),27:("visible","most of striped shows, right side"),28:("visible","striped at bowl edge, clearly visible"),29:("partly",""),30:("partly","half visible at left"),31:("partly",""),
32:("partly",""),33:("visible",""),34:("visible",""),35:("buried","none visible"),36:("partly",SL),37:("buried","none visible"),38:("partly",SL),39:("partly",""),
40:("buried","none visible"),41:("buried","none visible"),42:("partly",SL),43:("buried","none visible"),44:("partly",SL)}
OVR={
28:"r_target@5.77 r_grab@7.57 r_hole@11.21 r_target@11.81 r_grab@13.2 r_putback@16.75 s_target@17.35 s_grab@19.0 home@21.08",
43:"r_target@3.1 r_grab@4.5 r_hole@6.75 r_target@7.7 r_grab@9.5 r_hole@11.27 r_target@12.67 r_grab@14.47 r_hole@16.3 r_target@17.37 r_grab@19.17 r_hole@21.05 s_target@26.13 s_grab@27.93 s_slip@29.4 s_target@29.65 s_grab@30.17 home@32.78",
37:"r_target@7.4 r_grab@9.4 r_hole@11.56 r_target@13.1 r_grab@14.9 r_hole@17.56 s_target@27.1 s_miss@28.9 s_target@29.5 s_grab@31.1 s_slip@32.47 s_target@32.7 s_grab@33.47 home@40.08",
17:"r_target@7.4 r_grab@9.47 r_hole@11.12 r_target@12.1 r_grab@13.9 r_hole@15.44 r_target@16.04 r_grab@17.7 r_hole@19.37 s_target@19.9 s_miss@21.07 s_target@27.57 s_miss@29.37 s_target@29.97 s_grab@31.3 home@34.05",
}
CONF={7:"low",16:"low",5:"medium",9:"medium",24:"low",32:"medium",41:"medium",28:"medium",43:"medium",37:"medium",17:"medium",1:"medium"}
EXTRA={7:"26-31 s: grab+release at 28.5-29 and a miss at 29.8 unchecked on frames; striped misses at 37-41 s assumed",
16:"15-25 s: early putback/hole sequence not checked on frames (a putback at 19.1 may be a hole drop; 20.2/20.7 pinch+release unclear)",
24:"26-30 s: pinch at 27.4-27.7 not labelled; foil grab 29.6 -> hole 32.6 not frame-checked",
28:"foil at 13.2-16.8 moved aside inside the bowl (striped then reachable), not into a hole",
43:"striped grabbed at 28.7, dropped at 29.4, regrabbed 30.2 (slip)",37:"striped grabbed 31.1-32.5 then dropped (slip), regrabbed 33.5",
1:"foil grabbed at 24.7, put aside inside the bowl at 26.1 (not into the hole)"}
print(run("ds/so101_pick_buried_ball","pl2/prop_buried.txt",S,OVR,CONF,EXTRA))
