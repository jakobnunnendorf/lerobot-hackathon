"""After the two-palm drop: the gripper opens, `--celebrate N` close-open cycles with the arm still, then a plain joint
move to `--rest` that never brings the tip below the hover height. Tian's behaviour (TianK003/hand-tracking-test,
3bb344d, board 7064748304). Simulated arm, scripted follower decisions; no hardware."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handover"))
import handover_loop  # noqa: E402
from following import Decision, FollowSettings, Mode  # noqa: E402
from kinematics import forward_kinematics, inverse_kinematics  # noqa: E402

REST = np.array([-3.69, -103.43, 97.01, -100.26, 1.45])
FPS = 30


class ScriptedFollower:
    """FOLLOW for a while, then DROP for open_time_s, then DONE for ever (as Follower does after a drop)."""

    def __init__(self, follow_s=0.3, open_s=1.0):
        self.follow_s, self.open_s, self.start = follow_s, open_s, None

    def step(self, hands, tip, now):
        self.start = now if self.start is None else self.start
        t = now - self.start
        if t < self.follow_s:
            return Decision(Mode.FOLLOW, tip.copy())
        return Decision(Mode.DROP if t < self.follow_s + self.open_s else Mode.DONE)


class NoHands:
    def newest_hands(self):
        return []


def run_once(rest, cycles, duration_s=8.0):
    hover = handover_loop.DRY_RUN_START_DEG.copy()
    arm = handover_loop.SimArm(hover, 13.0)
    rows = []
    handover_loop.run(arm, NoHands(), ScriptedFollower(), hover, 13.0, 60.0, handover_loop.Limits(max_step_deg=6.0),
                      FPS, duration_s, lambda t, d, c, g, tip, goal: rows.append((t, d.mode, c.copy(), g)), None, rest, cycles)
    return arm, rows


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
        if not ok:
            failures.append(name)

    # run() sleeps to the tick, so these runs take their duration in wall time (about 10 s).
    arm, rows = run_once(REST, 3)
    done = [r for r in rows if r[1] is Mode.DONE]
    first_done = done[0][0]
    celebrating = [r for r in done if r[0] - first_done < 6 * handover_loop.CELEBRATE_PHASE_S]
    grips = [round(r[3]) for r in celebrating]
    flips = sum(1 for a, b in zip(grips, grips[1:]) if a != b)
    check("gripper open at the end of DROP", abs(rows[[r[1] for r in rows].index(Mode.DONE) - 1][3] - 60.0) < 0.5)
    check("3 close-open cycles after the drop (5 gripper flips)", flips == 5 and grips[0] == 5 and grips[-1] == 60, f"flips {flips}")
    check("arm still while celebrating", all(np.allclose(r[2], celebrating[0][2]) for r in celebrating))
    check("ends in the rest pose", np.allclose(arm.joints, REST, atol=1e-6), str(np.round(arm.joints, 2).tolist()))
    check("gripper stays open after celebrating", abs(arm.gripper - 60.0) < 1e-6)
    steps = [np.abs(b[2] - a[2]).max() for a, b in zip(rows, rows[1:])]
    check("joint speed limit holds on the way to rest", max(steps) <= 6.0 + 1e-9, f"max {max(steps):.2f} deg/tick")

    arm, _ = run_once(None, 0, duration_s=2.0)
    check("without --rest and --celebrate: arm stays at the hover pose", np.allclose(arm.joints, handover_loop.DRY_RUN_START_DEG))

    # Tian's safety check: from every hover point in the hand zone, the plain joint move to rest never dips below the
    # hover height (and never comes near the table).
    hover = FollowSettings.hover_height
    lowest = np.inf
    for radius in (0.15, 0.20, 0.25, 0.30, 0.35):
        for deg in (-45, -20, 0, 20, 45):
            joints = inverse_kinematics(np.array([radius * np.cos(np.radians(deg)), radius * np.sin(np.radians(deg)), hover]),
                                        np.zeros(5))
            if joints is None:
                continue
            path = [forward_kinematics(joints + f * (REST - joints))[2] for f in np.linspace(0, 1, 101)]
            lowest = min(lowest, min(path))
    check(f"joint move hover ({hover} m) -> rest stays at or above hover height", lowest >= hover - 0.005, f"lowest {lowest:.3f} m")

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
