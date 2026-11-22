import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# tests/cuda.py would shadow the `cuda` package, so this script's own folder leaves sys.path (as supervisor_scenarios.py).
sys.path[:] = [str(SCRIPTS / "supervisor"), str(SCRIPTS / "handover")] + [
    p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import kinematics  # noqa: E402
from bite import BiteSettings, BiteSkill  # noqa: E402
from skills import action_of, pose_of  # noqa: E402

FPS = 30
# Balazs's folded rest pose (board 7064702491) with the gripper at 30.
FOLDED_REST = np.array([-3.69, -103.43, 97.01, -100.26, 1.45, 30.0])
READY = np.concatenate([kinematics.lerobot_degrees(kinematics.solve(np.array([0.28, 0.0, 0.15]), 0, 0, margin_deg=3)),
                        [60.0]])
MOUTH = kinematics.forward_kinematics(READY[:5])

failures: list[str] = []
all_tips: list[np.ndarray] = []
step_problems: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


# The tracker sends a new message about every 2 ticks (15 fps of MediaPipe against the 30 Hz loop); between two, the
# receiver hands out the newest one again. tips_at(message number, skill) scripts each new message.
class ScriptedReceiver:
    def __init__(self, tips_at, every: int = 2):
        self.tips_at, self.every, self.tick, self.skill = tips_at, every, 0, None
        self.message, self.tips = -1, []

    def newest_pointing(self) -> tuple[int, list[np.ndarray]]:
        if self.tick // self.every != self.message:
            self.message = self.tick // self.every
            self.tips = [np.array(tip, dtype=float) for tip in self.tips_at(self.message, self.skill)]
        return self.message, list(self.tips)

    def newest_pointing_tips(self) -> list[np.ndarray]:
        return self.newest_pointing()[1]


# One MediaPipe detection, handed out again for its 0.5 s of freshness, then messages without a pointing hand.
class OneFalseDetection:
    def __init__(self, finger):
        self.finger, self.tick, self.skill = np.array(finger, dtype=float), 0, None

    def newest_pointing(self) -> tuple[int, list[np.ndarray]]:
        return (0, [self.finger.copy()]) if self.tick < 15 else (self.tick, [])

    def newest_pointing_tips(self) -> list[np.ndarray]:
        return self.newest_pointing()[1]


def held(*tips):
    return lambda message, skill: list(tips)


class Trace:
    def __init__(self):
        self.phases, self.commanded, self.tips = [], [], []

    def entered(self, phase: str) -> int:
        return sum(1 for previous, current in zip(["", *self.phases], self.phases) if current == phase != previous)

    def first(self, phase: str) -> int:
        return self.phases.index(phase) if phase in self.phases else -1


def run(tips_at, settings: BiteSettings, start=READY, seconds: float = 20.0, ready=READY, receiver=None):
    receiver = receiver or ScriptedReceiver(tips_at)
    skill = BiteSkill(kinematics, receiver, settings, ready)
    receiver.skill = skill
    last_sent = action_of(start)
    obs, now, trace = dict(last_sent), 0.0, Trace()
    skill.start(obs, last_sent, now)
    previous = pose_of(last_sent)
    for tick in range(int(seconds * FPS)):
        receiver.tick = tick
        action = skill.step(obs, now)
        if action is not None:
            last_sent, obs = action, dict(action)
        sent = pose_of(last_sent)
        phase = str(skill.phase)
        limit = settings.lunge_step_deg if phase == "LUNGE" else settings.max_step_deg
        step = np.abs(sent[:5] - previous[:5]).max()
        if step > limit + 1e-6:
            step_problems.append(f"{phase} step {step:.2f} deg > {limit}")
        previous = sent
        tip = kinematics.forward_kinematics(sent[:5])
        trace.phases.append(phase)
        trace.commanded.append(sent)
        trace.tips.append(tip)
        all_tips.append(tip)
        if skill.done(obs, now):
            break
        now += 1 / FPS
    trace.commanded, trace.tips = np.array(trace.commanded), np.array(trace.tips)
    return skill, trace


def radial(point: np.ndarray, distance: float) -> np.ndarray:
    radius = np.hypot(point[0], point[1])
    return np.array([*(point[:2] * (radius + distance) / radius), point[2]])


def aim_bite() -> None:
    finger = np.array([0.30, 0.05, 0.15])
    skill, trace = run(held(finger), BiteSettings(max_bites=1), start=FOLDED_REST)
    aim, lunge = radial(finger, -0.06), radial(finger, 0.01)
    near_aim = np.linalg.norm(trace.tips - aim, axis=1)
    near_lunge = np.linalg.norm(trace.tips - lunge, axis=1)
    reached_aim = int(np.argmax(near_aim < 0.01)) if (near_aim < 0.01).any() else -1
    reached_lunge = int(np.argmax(near_lunge < 0.015)) if (near_lunge < 0.015).any() else -1
    grippers = trace.commanded[:, 5]
    closed_at = int(np.argmax(grippers <= 5.0)) if (grippers <= 5.0).any() else -1
    check("aim: from the folded rest it goes to the mouth pose, aims to the standoff point, then lunges past the finger",
          0 <= reached_aim < reached_lunge and trace.first("READY") < trace.first("WATCH") < trace.first("AIM")
          < trace.first("LUNGE"),
          f"aim {near_aim.min():.4f} m at {reached_aim}, lunge {near_lunge.min():.4f} m at {reached_lunge}")
    check("aim: the gripper closes to 5 after the lunge, opens to 60 again, result 'bit' after max_bites=1",
          closed_at > reached_lunge and (grippers[closed_at:] == 60.0).any() and grippers[-1] == 60.0
          and skill.result() == "bit" and np.allclose(trace.commanded[-1, :5], READY[:5]),
          f"closed at {closed_at}, result {skill.result()}")


def outside_workspace() -> None:
    pan = np.radians(80)
    cases = {"below the floor": (0.30, 0.05, 0.08), "too far": (0.45, 0.0, 0.15),
             "80 deg off the mouth pose": (0.30 * np.cos(pan), 0.30 * np.sin(pan), 0.15)}
    for name, finger in cases.items():
        skill, trace = run(held(finger), BiteSettings(window_s=2.0))
        still = np.allclose(trace.commanded[:, :5], READY[:5])
        check(f"outside the workspace ({name}): the arm never leaves the mouth pose, result 'no_finger'",
              still and skill.result() == "no_finger" and "AIM" not in trace.phases and "LUNGE" not in trace.phases,
              f"phases {sorted(set(trace.phases))}")


def lost_before_freeze() -> None:
    finger = np.array([0.30, 0.15, 0.15])
    skill, trace = run(held(finger), BiteSettings(max_bites=1))
    near_lunge = np.linalg.norm(trace.tips - radial(finger, 0.01), axis=1).min()
    check("a finger 15 cm from the jaws that stays in view: turned to, aimed at and bitten",
          skill.result() == "bit" and near_lunge < 0.015 and abs(trace.commanded[:, 0] - READY[0]).max() > 15,
          f"lunge point {near_lunge:.4f} m, pan turned {abs(trace.commanded[:, 0] - READY[0]).max():.1f} deg")
    aimed = {"yes": False}

    def tips_at(tick, skill):
        aimed["yes"] |= str(skill.phase) == "AIM"
        return [] if aimed["yes"] else [finger]

    skill, trace = run(tips_at, BiteSettings(window_s=4.0))
    after_aim = trace.phases[trace.first("AIM"):] if "AIM" in trace.phases else []
    back = "READY" in after_aim and "WATCH" in after_aim[after_aim.index("READY"):]
    check("lost for over 1 s before the freeze (finger 15 cm from the jaws): back to READY, then WATCH, no lunge",
          back and "LUNGE" not in trace.phases and skill.result() == "no_finger",
          f"result {skill.result()}, phases after AIM {list(dict.fromkeys(after_aim))}")


def lost_after_freeze() -> None:
    finger = np.array([0.30, 0.05, 0.15])
    aimed = {"yes": False}

    def tips_at(tick, skill):
        aimed["yes"] |= str(skill.phase) == "AIM"
        return [] if aimed["yes"] else [finger]

    skill, trace = run(tips_at, BiteSettings(max_bites=1))
    near_lunge = np.linalg.norm(trace.tips - radial(finger, 0.01), axis=1).min()
    check("lost after the freeze (the arm hides the finger): the lunge still happens, result 'bit'",
          "LUNGE" in trace.phases and "SNAP" in trace.phases and skill.result() == "bit" and near_lunge < 0.015,
          f"lunge point {near_lunge:.4f} m")


def seen_too_rarely() -> None:
    finger = (0.30, 0.05, 0.15)
    skill, trace = run(lambda message, skill: [finger] if message % 8 < 3 else [], BiteSettings(window_s=2.0))
    check("a finger in only 3 of 8 tracker messages: no aim, result 'no_finger'",
          "AIM" not in trace.phases and skill.result() == "no_finger" and np.allclose(trace.commanded[:, :5], READY[:5]))


def one_false_detection() -> None:
    receiver = OneFalseDetection((0.30, 0.05, 0.15))
    skill, trace = run(None, BiteSettings(window_s=2.0), receiver=receiver)
    check("one pointing detection handed out again for 15 ticks counts once: no aim, result 'no_finger'",
          "AIM" not in trace.phases and skill.result() == "no_finger" and np.allclose(trace.commanded[:, :5], READY[:5]),
          f"phases {list(dict.fromkeys(trace.phases))}")
    skill, trace = run(held((0.30, 0.05, 0.15)), BiteSettings(max_bites=1), receiver=ScriptedReceiver(held((0.30, 0.05, 0.15)), every=2))
    check("a finger in every tracker message at the tracker's rate (a new message every 2 ticks) is still bitten",
          skill.result() == "bit" and "AIM" in trace.phases)


# Folded rest of the demo chain on the robot laptop; from it the tip passes 35 cm up, where pitch-0 IK has no solution
# 32-35 cm out, so the Cartesian part of the way to such a mouth pose stalled until the bite timeout.
DEMO_REST = np.array([-5.6, -102.8, 96.6, -100.1, -5.2, 30.0])


def far_mouth_from_demo_rest() -> None:
    for tip in ((0.0, -0.35, 0.15), (0.0, 0.35, 0.15), (0.25, -0.25, 0.15)):
        ready = np.concatenate([kinematics.lerobot_degrees(kinematics.solve(np.array(tip), 0, 0, margin_deg=3)), [60.0]])
        skill, trace = run(held(), BiteSettings(window_s=0.5), start=DEMO_REST, ready=ready, seconds=15)
        check(f"from the demo rest pose to a mouth pose at {tip}: arrives (READY -> WATCH), no stall mid-air",
              "WATCH" in trace.phases and skill.done({}, 0.0) and np.allclose(trace.commanded[-1, :5], ready[:5]),
              f"{len(trace.phases) / FPS:.1f} s, last tip {np.round(trace.tips[-1], 3).tolist()}")


def three_bites() -> None:
    skill, trace = run(held((0.30, 0.05, 0.15)), BiteSettings(max_bites=3))
    closes = int(np.sum((trace.commanded[1:, 5] <= 5.0) & (trace.commanded[:-1, 5] > 5.0)))
    check("max_bites=3 with the finger held: exactly 3 snaps, then done",
          trace.entered("SNAP") == 3 and closes == 3 and skill.done({}, 0.0) and skill.result() == "bit",
          f"{trace.entered('SNAP')} snaps, {closes} closes, {len(trace.phases) / FPS:.1f} s")


def zone_mode() -> None:
    settings = BiteSettings(mode="zone", max_bites=1, window_s=3.0)
    ahead = radial(MOUTH, 0.04)
    skill, trace = run(held(ahead), settings)
    check("zone: a fingertip 4 cm ahead of the jaws -> a bite without any pan change, no AIM",
          skill.result() == "bit" and "AIM" not in trace.phases and trace.entered("SNAP") == 1
          and np.allclose(trace.commanded[:, 0], READY[0], atol=0.01),
          f"pan range {trace.commanded[:, 0].min():.2f}..{trace.commanded[:, 0].max():.2f}")
    side = MOUTH + np.array([0.0, 0.20, 0.0])
    skill, trace = run(held(side), settings)
    check("zone: a fingertip 20 cm to the side -> no motion, 'no_finger'",
          skill.result() == "no_finger" and np.allclose(trace.commanded[:, :5], READY[:5]))


def ready_pose_outside_refused() -> None:
    low = np.concatenate([kinematics.lerobot_degrees(kinematics.solve(np.array([0.28, 0.0, 0.10]), 0, 0, margin_deg=3)),
                          [60.0]])
    try:
        BiteSkill(kinematics, ScriptedReceiver(held()), BiteSettings(), low)
        refused = False
    except ValueError:
        refused = True
    check("a mouth pose below the minimum bite height is refused when the skill is built", refused)


def garbage_tips_ignored() -> None:
    skill, trace = run(held((np.nan, 0.0, 0.15), (0.30, 0.05)), BiteSettings(window_s=1.0))
    check("non-finite or short fingertips are ignored", skill.result() == "no_finger"
          and np.allclose(trace.commanded[:, :5], READY[:5]))


def main() -> None:
    aim_bite()
    outside_workspace()
    lost_before_freeze()
    lost_after_freeze()
    seen_too_rarely()
    one_false_detection()
    far_mouth_from_demo_rest()
    three_bites()
    zone_mode()
    ready_pose_outside_refused()
    garbage_tips_ignored()
    floor = BiteSettings().min_height_m - 0.005
    lowest = min(tip[2] for tip in all_tips)
    check("no commanded tip below the minimum bite height (-5 mm) over all runs", lowest >= floor, f"lowest {lowest:.4f} m")
    check("no joint step above the phase's limit over all runs", not step_problems, "; ".join(step_problems[:3]))
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
