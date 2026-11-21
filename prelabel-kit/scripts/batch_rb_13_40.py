import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prelabel_write import write_prelabel
DS = "ds/so101_pick_rb_mixed"
def P(s):  # "r_target@4.67 ..." -> [(code,t)]
    return [(a.split("@")[0], float(a.split("@")[1])) for a in s.split()]
SL = "start state judged on the top camera only; a sliver of green/yellow is called partly"
SPEC = {
 13: ("buried", "high", "", "r_target@4.67 r_grab@6.6 r_hole@7.98 b_target@8.63 b_grab@10.43 b_hole@12.28 r_target@15.23 r_grab@17.03 r_hole@19.5 s_target@22.13 s_grab@23.93 home@25.62"),
 14: ("partly", "medium", "start state: only a sliver of striped visible, could be buried; target times +-1 s", "r_target@5.52 r_grab@7.67 r_hole@9.77 r_target@10.9 r_grab@12.7 r_hole@14.09 s_target@15.77 s_grab@17.57 home@19.22"),
 15: ("partly", "medium", "start state: only a sliver of striped visible, could be buried; target times +-1 s", "b_target@6.13 b_grab@7.87 b_hole@9.52 r_target@10.47 r_grab@12.27 r_hole@13.6 s_target@17.53 s_grab@19.33 home@21.48"),
 16: ("buried", "medium", "closing on striped at 7.2-8.0 closed on nothing (gripper 5.6): labelled Missed striped, could instead be a pinch he would skip; first R target time guessed", "s_target@6.2 s_miss@8.0 r_target@8.6 r_grab@9.57 r_hole@11.09 s_target@13.53 s_grab@15.33 home@16.88"),
 17: ("buried", "medium", "striped closed at 16.6-17.3 only to gripper 17 and was not held (dropped beside fingers): labelled Missed striped, could be Slip/drop", "r_target@5.04 r_grab@7.47 r_hole@9.35 r_target@11.63 r_grab@13.43 r_hole@14.82 s_target@15.4 s_miss@17.1 s_target@18.8 s_grab@20.23 home@21.68"),
 18: ("buried", "medium", "miss at 11.3 (gripper closed to 1.9 on nothing while striped in view) labelled Missed striped; striped start state hard to see on top camera", "r_target@3.17 r_grab@5.97 r_hole@7.56 s_target@9.47 s_miss@11.27 b_target@12.6 b_grab@14.4 b_hole@15.96 r_target@16.9 r_grab@18.7 r_hole@19.91 r_target@21.43 r_grab@23.23 r_hole@24.52 s_target@26.6 s_grab@30.2 home@32.08"),
 19: ("partly", "medium", "start state: sliver at right edge of bowl; first target time uncertain (arm hovers long before the grab)", "s_target@4.0 s_grab@7.7 home@9.15"),
 20: ("buried", "high", "", "r_target@4.9 r_grab@7.87 r_hole@9.1 b_target@10.2 b_grab@12.0 b_hole@13.63 b_target@15.5 b_grab@17.3 b_hole@18.97 s_target@24.03 s_grab@25.83 home@27.68"),
 21: ("visible", "medium", "gripper closed to 2.1 on nothing at 8-9.7 (striped rolled away): Missed striped; closed only to 19 at 6.1-6.8 (partial pinch) not labelled; first target time guessed", "s_target@4.3 s_miss@9.47 s_target@11.4 s_grab@13.2 home@14.78"),
 22: ("buried", "medium", "drop of the striped ball at 19.7 after a lift (Slip/drop) and a miss at 9.9-11.2 (closed to 0.9 on nothing, striped in view); early partial pinch at 2.7-3.3 not labelled", "r_target@2.44 r_grab@5.27 r_hole@6.67 s_target@9.23 s_miss@11.03 s_target@16.27 s_grab@18.07 s_slip@19.72 s_target@20.3 s_grab@21.6 home@23.32"),
 23: ("visible", "high", "", "s_target@5.04 s_grab@7.83 home@9.95"),
 24: ("visible", "high", "", "s_target@4.21 s_grab@8.47 home@10.28"),
 25: ("buried", "medium", "first R grab was a regrasp (pinch at 7.0-7.7, re-open, grab at 8.7), labelled as one grab; striped target time guessed (arm hovers 16-25 s)", "r_target@3.06 r_grab@8.7 r_hole@11.56 r_target@12.67 r_grab@14.47 r_hole@16.11 s_target@23.5 s_grab@25.3 home@27.88"),
 26: ("partly", "medium", "start state: sliver visible, could be buried", "b_target@4.0 b_grab@6.67 b_hole@8.88 s_target@13.17 s_grab@14.97 home@17.05"),
 27: ("buried", "high", "", "b_target@2.65 b_grab@6.7 b_hole@8.58 s_target@11.2 s_grab@13.0 home@14.62"),
 28: ("partly", "medium", "striped grabbed at 28.2, lifted and dropped back at 30.4 (Slip/drop), regrabbed at 34.2; pinch at 26.6-27.7 not labelled", "r_target@5.57 r_grab@9.87 r_hole@11.52 b_target@17.07 b_grab@18.87 b_hole@21.29 s_target@24.83 s_grab@28.2 s_slip@30.41 s_target@31.0 s_grab@34.2 home@35.68"),
 29: ("buried", "high", "", "r_target@3.5 r_grab@5.97 r_hole@7.44 b_target@9.0 b_grab@10.8 b_hole@12.27 s_target@16.07 s_grab@17.87 home@19.75"),
 30: ("visible", "high", "", "s_target@5.3 s_grab@10.2 home@12.22"),
 31: ("partly", "medium", "start state: sliver visible, could be buried", "s_target@4.98 s_grab@8.63 home@10.58"),
 32: ("partly", "low", "9-16 s: arm repeatedly pinches covering balls next to striped (grab at 9.2 and 12.9 released inside the bowl, ball colour unclear); not labelled. B identified from hole position and wrist frame at 17.5", "b_target@12.85 b_grab@16.73 b_hole@20.46 s_target@24.8 s_grab@26.6 home@28.38"),
 33: ("visible", "medium", "closed only to 19.7 at 8.7-9.4 (partial pinch, maybe an unlabelled miss on striped); striped target time guessed", "r_target@2.06 r_grab@4.27 r_hole@5.56 s_target@9.7 s_grab@11.5 home@13.02"),
 34: ("buried", "low", "miss at 22.5-23.3 (gripper closed to 2.2 on nothing) assumed to be on striped, not checked on frames; early pinch at 11.9-12.6 not labelled", "b_target@10.63 b_grab@13.67 b_hole@15.64 s_target@21.3 s_miss@23.1 s_target@23.7 s_grab@25.5 home@27.22"),
 35: ("partly", "low", "6-11 s: pinch at 9.0-9.5 and a close on nothing at 10.0 around the striped ball (colour of ball between fingers unclear), not labelled; R identified from hole and wrist frame at 13.1", "r_target@11.5 r_grab@12.3 r_hole@14.1 s_target@17.8 s_grab@19.6 home@22.68"),
 36: ("visible", "medium", "close at 9.2-10.1 only to 8.3 over a red ball sitting on the striped one, labelled Missed striped; first target time guessed", "s_target@7.2 s_miss@9.9 s_target@11.0 s_grab@13.03 home@14.85"),
 37: ("buried", "high", "R carried slowly (grab 4.9, hole 9.0)", "r_target@2.52 r_grab@4.87 r_hole@9.0 s_target@10.6 s_grab@12.4 home@14.15"),
 38: ("partly", "low", "start state: tiny sliver; 10-17 s: miss (closed to 6.2) and a grab at 15.0 dropped inside the bowl at 17.1, ball colour unclear, not labelled (maybe a put-back); striped target 21.6 and grab 26.1 -- a pinch at 23.4-23.9 not labelled", "b_target@3.2 b_grab@6.33 b_hole@7.85 s_target@21.6 s_grab@26.1 home@28.18"),
 39: ("partly", "low", "26-39 s: three grab+release attempts in the bowl (grabs 28.3, 31.5, 36.4, releases 30.2, 33.2, 38.3; striped visible and not lifted, probably covering R/B balls pushed aside or striped slipping) not labelled; check the sheets; start state: sliver", "r_target@7.67 r_grab@12.63 r_hole@17.45 s_target@39.5 s_grab@41.3 home@43.48"),
 40: ("partly", "medium", "blue ball pinched at 7.4 and released at 8.0 in the bowl (not labelled); striped grab at 9.2 closed only to gripper 17.5 but the ball is held at the end; target time guessed", "s_target@5.0 s_grab@9.2 home@11.12"),
}
done = skipped = 0
for ep, (st, conf, unc, steps) in sorted(SPEC.items()):
    ok = write_prelabel(DS, ep, st, P(steps), conf, unc or "none")
    done += ok; skipped += (not ok)
print("written", done, "skipped (already present)", skipped)
