import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# tests/cuda.py would shadow the `cuda` package torch imports, so this folder leaves sys.path.
sys.path[:] = [str(ROOT / "scripts" / "vision")] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from colour_detector import ColourDetector, calibrate, hue_chroma_value, load_detector
from supervisor_adapter import ColourVisibility, colour_of

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def foil_scene(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((480, 640, 3), 40, np.uint8)
    for _ in range(18):
        x, y = int(rng.integers(300, 480)), int(rng.integers(130, 240))
        cv2.circle(frame, (x, y), 17, (175, 178, 180), cv2.FILLED)
        cv2.circle(frame, (x - 5, y - 5), 5, (255, 255, 255), cv2.FILLED)
    return frame


def ball(frame: np.ndarray, colour_rgb: tuple[int, int, int], centre: tuple[int, int], radius: int = 17, stripes: bool = False) -> None:
    cv2.circle(frame, centre, radius, colour_rgb, cv2.FILLED)
    cv2.circle(frame, (centre[0] - 5, centre[1] - 5), 4, (255, 255, 255), cv2.FILLED)
    if stripes:
        for dx in (-8, 8):
            cv2.line(frame, (centre[0] + dx, centre[1] - radius + 4), (centre[0] + dx, centre[1] + radius - 4), (235, 225, 170), 2)


detector = ColourDetector()
CENTRE = (390, 190)

scene = foil_scene()
ball(scene, (150, 190, 40), CENTRE, stripes=True)
hit = detector.detect(scene)["striped"]
check("synthetic striped ball found at its centre", hit.present and np.hypot(hit.centre[0] - CENTRE[0], hit.centre[1] - CENTRE[1]) < 6, str(hit))

for name, rgb in (("red", (200, 30, 30)), ("blue", (30, 60, 200))):
    scene = foil_scene(1)
    ball(scene, rgb, CENTRE)
    found = detector.detect(scene)
    others = [n for n in ("striped", "red", "blue") if n != name and found[n].present]
    check(f"synthetic {name} ball found, no other ball colour", found[name].present and not others, f"{found[name]} others={others}")

scene = foil_scene(2)
check("foil only: no coloured ball", not any(detector.detect(scene)[n].present for n in ("striped", "red", "blue")))

scene = foil_scene(3)
for x, y, tint in ((330, 200, (150, 165, 200)), (420, 160, (200, 165, 150)), (370, 230, (165, 200, 170))):
    cv2.circle(scene, (x, y), 4, tint, cv2.FILLED)
check("tinted reflections on foil (4 px, low chroma) are no ball", not any(detector.detect(scene)[n].present for n in ("striped", "red", "blue")))

scene = foil_scene(4)
cv2.line(scene, (260, 100), (520, 140), (200, 40, 30), 4)
check("thin red cable is no red ball", not detector.detect(scene)["red"].present)

scene = foil_scene(5)
ball(scene, (200, 30, 30), CENTRE)
ball(scene, (30, 60, 200), (330, 200))
scene[330:470, 60:230] = (30, 60, 200)
scene[330:470, 260:430] = (200, 30, 30)
found = detector.detect(scene)
check("red and blue ball and both containers in one frame",
      all(found[n].present for n in ("red", "blue", "red_container", "blue_container")),
      str({n: found[n].area for n in ("red", "blue", "red_container", "blue_container")}))
check("a container is not mistaken for a ball", found["red"].area < 2200 and found["blue"].area < 2200)

manifest = json.loads((ROOT / "tests" / "fixtures" / "colour" / "manifest.json").read_text())
for item in manifest:
    rgb = cv2.cvtColor(cv2.imread(str(ROOT / "tests" / "fixtures" / "colour" / item["file"])), cv2.COLOR_BGR2RGB)
    found = ColourDetector(roi=None).detect(rgb)
    wrong = {n: want for n, want in item["expect"].items() if found[n].present != want}
    check(f"real crop {item['file']} ({item['dataset']} ep {item['episode']} t={item['seconds_into_episode']}s)", not wrong, str(wrong))

crop = lambda name: cv2.cvtColor(cv2.imread(str(ROOT / "tests" / "fixtures" / "colour" / name)), cv2.COLOR_BGR2RGB)
seen = ColourDetector(roi=None).detect(crop("bowl_e48_t00.0.png"))["striped"].centre
box = (int(seen[0]) - 25, int(seen[1]) - 25, int(seen[0]) + 25, int(seen[1]) + 25)
spec = calibrate([crop("bowl_e48_t00.0.png")], [box], [crop("bowl_e48_t14.0.png")], "striped")
check("calibrate from one real crop finds the striped ball again",
      ColourDetector((spec,), roi=None).detect(crop("bowl_e96_t00.0.png"))["striped"].present and
      not ColourDetector((spec,), roi=None).detect(crop("bowl_e48_t14.0.png"))["striped"].present)
red_scene = foil_scene(6)
ball(red_scene, (190, 25, 35), CENTRE)
red_spec = calibrate([red_scene], [(365, 165, 415, 215)], [foil_scene(7)], "red")
check("calibrate wraps around hue 0 for red", len(red_spec.ranges) in (1, 2) and
      ColourDetector((red_spec,), roi=None).detect(red_scene)["red"].present)
check("config round trip", load_detector(Path("/nonexistent")).names == detector.names)

visibility = ColourVisibility(detector)
check("voice words map to colours", colour_of("Striped") == "striped" and colour_of("red") == "red" and colour_of("purple") is None)
scene = foil_scene(8)
ball(scene, (30, 60, 200), CENTRE)
visibility.set_target("blue")
look = [scene] * 5
check("adapter: blue asked, blue in the bowl", visibility.in_bowl(look, 150))
visibility.set_target("red")
check("adapter: red asked, no red ball: refuse before clearing anything", not visibility.in_bowl(look, 150))
check("adapter: unknown colour keeps the old target", not visibility.set_target("purple") and visibility.target == "red")


def supervisor_look() -> None:
    sup = ROOT / "scripts" / "supervisor"
    if not (sup / "machine.py").exists():
        print("SKIP  supervisor LOOK step: scripts/supervisor is not on this branch (PR #17)")
        return
    sys.path.insert(0, str(sup))
    from fakes import build, SCENARIOS  # noqa: F401
    from machine import Supervisor, TransitionLog
    from skills import pose_of

    for target, buried_expect in (("blue", ["REST", "LOOK", "PICK", "VERIFY_HOLD", "PUT_DOWN", "REST", "DONE"]),):
        scenario = next(s for s in SCENARIOS if s.name == "pick_then_give")
        scenario.settings = {"look": True}
        scenario.hand = False
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        frame = foil_scene(9)
        ball(frame, (30, 60, 200), CENTRE)
        arm.observe = (lambda inner: lambda: {**inner(), "top": frame})(arm.observe)
        visibility.set_target(target)
        sensors.visible_area = visibility
        log = TransitionLog(None, echo=False)
        result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(1)
        states = [s.value for s in result.states]
        check("supervisor LOOK sees the asked colour and goes to PICK", states == buried_expect, str(states))
        visibility.set_target("red")
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        arm.observe = (lambda inner: lambda: {**inner(), "top": frame})(arm.observe)
        sensors.visible_area = visibility
        settings.clear_gray = True
        result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(2)
        check("supervisor LOOK with an absent colour goes to CLEAR_GRAY (why in_bowl must refuse first)",
              "CLEAR_GRAY" in [s.value for s in result.states])


supervisor_look()
print("\nPASS" if not failures else f"\nFAIL: {failures}")
sys.exit(1 if failures else 0)
