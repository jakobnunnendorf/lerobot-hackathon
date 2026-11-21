import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

SUPERVISOR = Path(__file__).resolve().parents[1] / "scripts" / "supervisor"
# tests/cuda.py would shadow the `cuda` package torch imports, so this script's own folder leaves sys.path.
sys.path[:] = [str(SUPERVISOR)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from commands import Commands
from fakes import SCENARIOS, FakeArm, Scenario, World, build, pick_effect, release_effect, scripted
from machine import RealClock, Sensors, Settings, Skills, Supervisor, TransitionLog
from skills import KEYS, LoadedPolicy, PolicySkill, capped, give_sequence, import_from, pose_of

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def fake_scenarios(log_path: str) -> None:
    log = TransitionLog(log_path, echo=False)
    for number, scenario in enumerate(SCENARIOS, 1):
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(number)
        states = [s.value for s in result.states]
        problems = []
        if states != scenario.expected:
            problems.append(f"states {states}")
        if result.outcome != scenario.outcome:
            problems.append(f"outcome {result.outcome}")
        if world.violations:
            problems.append(world.violations[0])
        if arm.sends_after_abort:
            problems.append(f"{arm.sends_after_abort} actions sent after abort")
        if scenario.outcome in ("given", "put_down", "gave_up", "wrong_ball") and not np.allclose(pose_of(arm.last_sent), pose_of(FakeArm(World(), abort).last_sent)):
            problems.append("did not end at the rest pose")
        check(f"scenario {scenario.name}", not problems, "; ".join(problems))
    with open(log_path) as lines:
        events = [json.loads(line) for line in lines]
    ends = [e for e in events if e["event"] == "round_end"]
    check("log has one round_end per scenario with an outcome",
          len(ends) == len(SCENARIOS) and all(e["outcome"] for e in ends))


def policy_runner(checkpoint_cameras=("top", "wrist"), rename=None, n_action_steps=None) -> None:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    height, width = 96, 128
    hw_obs = {**{key: float for key in KEYS}, "top": (height, width, 3), "wrist": (height, width, 3)}
    features = {**hw_to_dataset_features({key: float for key in KEYS}, ACTION),
                **hw_to_dataset_features(hw_obs, OBS_STR)}
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
    config = make_policy_config(
        "act", device="cpu", pretrained_backbone_weights=None,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                        **{f"observation.images.{cam}": image for cam in checkpoint_cameras}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))})
    stats = {"observation.state": {"mean": torch.zeros(6), "std": torch.ones(6)},
             "action": {"mean": torch.zeros(6), "std": torch.ones(6)},
             **{f"observation.images.{cam}": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}
                for cam in checkpoint_cameras}}
    with tempfile.TemporaryDirectory() as checkpoint:
        get_policy_class("act")(config).save_pretrained(checkpoint)
        pre, post = make_pre_post_processors(config, dataset_stats=stats)
        pre.save_pretrained(checkpoint)
        post.save_pretrained(checkpoint)

        loaded = LoadedPolicy(checkpoint, device="cpu", n_action_steps=n_action_steps, camera_rename=rename)
        skill = PolicySkill("pick", loaded, features, "so101_follower")
        obs = {**{key: 0.0 for key in KEYS},
               "top": np.zeros((height, width, 3), np.uint8), "wrist": np.zeros((height, width, 3), np.uint8)}
        skill.start(obs, {key: 0.0 for key in KEYS}, 0.0)
        actions = [skill.step(obs, i / 30) for i in range(3)]
    ok = all(set(a) == set(KEYS) and all(np.isfinite(list(a.values()))) for a in actions)
    ok &= n_action_steps is None or loaded.policy.config.n_action_steps == n_action_steps
    check(f"policy runner: checkpoint cameras {checkpoint_cameras}, rename {rename}, n_action_steps {n_action_steps}"
          " -> 6 named joint goals", ok,
          f"first action {np.round(list(actions[0].values()), 3).tolist()}")


def voice_container() -> None:
    link = import_from(SUPERVISOR.parent / "voice", "link")
    commands, port = Commands(), 18765
    commands.listen_voice("127.0.0.1", port)
    sent = link.send({"target": "striped", "destination": "container", "transcript": "hey bot, put it in the container",
                      "source": "voice"}, port=port)
    deadline = time.monotonic() + 3
    while commands.destination() is None and time.monotonic() < deadline:
        commands.poll_voice()
        time.sleep(1 / 30)
    scenario = Scenario("voice_link_container", ["REST", "PICK", "VERIFY_HOLD", "PUT_DOWN", "REST", "DONE"], "put_down")
    arm, skills, sensors, settings, abort, clock, _world = build(scenario)
    sensors.destination = commands.destination
    result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), TransitionLog(None, echo=False), abort,
                        clock, on_tick=commands.poll_voice).run_round(1)
    states = [s.value for s in result.states]
    check("voice: link.send -> CommandReceiver -> 'container' with a hand present -> PUT_DOWN",
          sent and states == scenario.expected, f"sent={sent} states={states}")
    commands.voice.close()


def round_hygiene(log_path: str) -> None:
    from multiprocessing import shared_memory

    from run_supervisor import unlink_stale_block

    # The destination is for one round: round 2 must not reuse round 1's "container".
    commands = Commands()
    commands.set_target("striped")
    commands.set_destination("container")
    log = TransitionLog(log_path, echo=False)
    scenario = Scenario("hygiene", [], "")
    results = []
    for number in (1, 2):
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        sensors.destination, sensors.target, sensors.finish_round = commands.destination, commands.target, commands.clear
        results.append(Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(number))
        if number == 1:
            check("destination and target are cleared after a round", commands.destination() is None and commands.target() is None)
    states = [[s.value for s in r.states] for r in results]
    check("round 1 'container' -> PUT_DOWN, round 2 without a command (hand present) -> GIVE",
          states[0][3] == "PUT_DOWN" and states[1][3] == "GIVE", str(states))
    # a refused or aborted round clears it as well
    commands.set_destination("hand")
    arm, skills, sensors, settings, abort, clock, world = build(Scenario("refused", [], "", target="red"))
    sensors.destination, sensors.finish_round = commands.destination, commands.clear
    Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(3)
    check("a refused round clears the destination too", commands.destination() is None)
    # colours: the container of the ball's colour, the generic pose for striped
    for name in ("red_target_container", "striped_container_uses_generic_pose"):
        scenario = next(s for s in SCENARIOS if s.name == name)
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(9)
    with open(log_path) as lines:
        puts = [e for e in map(json.loads, lines) if e.get("source") == "PUT_DOWN" and e["event"] == "transition"]
    check("PUT_DOWN uses the red container for a red ball and the default pose for the striped ball",
          [e["container"] for e in puts][-2:] == ["red", "default"], str([e["container"] for e in puts]))
    # keys and voice commands accept the three targets
    commands.poll_voice()
    for target in ("striped", "red", "blue"):
        commands.set_target(target)
        check(f"target {target} accepted", commands.target() == target)
    commands.set_target("green")
    check("unknown target rejected", commands.target() == "blue")
    # a block left by a crashed run is removed, a missing one is fine
    name = f"stale_frames_{int(time.time())}"
    stale = shared_memory.SharedMemory(name=name, create=True, size=64)
    stale.close()
    unlink_stale_block(name)
    unlink_stale_block(name)
    try:
        shared_memory.SharedMemory(name=name)
        gone = False
    except FileNotFoundError:
        gone = True
    check("a stale shared-memory block is unlinked on start", gone)


def held_ball_check(log_path: str) -> None:
    def run(name: str) -> list[dict]:
        scenario = next(s for s in SCENARIOS if s.name == name)
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        log = TransitionLog(log_path, echo=False)
        Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(1)
        log.file.close()
        with open(log_path) as lines:
            events = [json.loads(line) for line in lines]
        open(log_path, "w").close()
        return events

    def verify(events):
        return [e for e in events if e["event"] == "transition" and e["source"] == "VERIFY_HOLD"]

    events = run("held_check_target_is_given")
    checks = verify(events)
    check("held check: striped target in the wrist frame -> handed over, area and threshold logged",
          [(e["target"], e["held_check"], e["held_area_px"], e["held_threshold_px"]) for e in checks]
          == [("GIVE", "target", 900, 300)], str(checks))
    events = run("held_check_red_ball_every_pick")
    checks = verify(events)
    gives = [e for e in events if e["event"] == "transition" and e["target"] == "GIVE"]
    end = next(e for e in events if e["event"] == "round_end")
    check("held check: red ball held -> never handed over, put down 3 times, outcome wrong_ball",
          not gives and [e["held_check"] for e in checks] == ["wrong"] * 3 and end["wrong_balls"] == 3
          and end["outcome"] == "wrong_ball", f"{[e['held_check'] for e in checks]} {end}")
    events = run("held_check_red_ball_sorted_into_red_container")
    puts = [e["container"] for e in events if e["event"] == "transition" and e["source"] == "PUT_DOWN"]
    check("held check: a wrong red ball goes to the red container when the colour sensor says red",
          puts == ["red"], str(puts))
    events = run("held_check_red_ball_put_down_then_retry")
    puts = [e["container"] for e in events if e["event"] == "transition" and e["source"] == "PUT_DOWN"]
    check("held check: without a colour sensor the wrong ball goes to the generic put-down pose",
          puts == ["default"], str(puts))
    checks = verify(run("held_check_detector_unavailable_accepts"))
    check("held check: detector unavailable -> accepted as before (gripper band), logged with the error",
          [(e["target"], e["held_check"]) for e in checks] == [("GIVE", "unavailable")]
          and "no wrist frame" in checks[0]["held_check_error"], str(checks))
    checks = verify(run("held_check_unmeasured_threshold_logs_only"))
    check("held check: no threshold yet -> the area is logged, the ball is accepted",
          [(e["target"], e["held_check"], e["held_area_px"]) for e in checks] == [("GIVE", "unmeasured", 0)], str(checks))
    checks = verify(run("held_check_blue_untrusted_by_default_logs_only"))
    check("held check: blue is not trusted on the wrist camera by default -> logged as untrusted, not judged",
          [(e["held_check"], e["held_ball"], e["held_area_px"]) for e in checks] == [("untrusted", "blue", 0)], str(checks))
    checks = verify(run("held_check_red_trusted_striped_held_in_red_round_is_wrong"))
    check("held check: red trusted (--held-check-colours striped red) -> a striped ball in a red round is wrong",
          [e["held_check"] for e in checks] == ["wrong", "target"], str(checks))
    checks = verify(run("held_check_red_per_colour_threshold_above_area_is_wrong"))
    check("held check: per-colour threshold (red=2000) judges a red round, every ball's area is logged",
          [(e["held_check"], e["held_threshold_px"], e["held_areas_px"]) for e in checks]
          == [("wrong", 2000, {"striped": 0, "red": 900, "blue": 0})], str(checks))
    checks = verify(run("pick_then_give"))
    check("held check: no hook -> held_check off, flow unchanged", [e["held_check"] for e in checks] == ["off"], str(checks))


def vision_adapter(vision_dir: Path) -> None:
    import cv2
    from run_supervisor import held_ball_area

    area = held_ball_area(vision_dir)
    # A 640x480 wrist frame: one ball held in the centre, larger than any ball in the top camera's bowl, and a
    # bluish room around it (the wrist camera faces the ceiling at rest), outside the held-ball region.
    frames = {}
    for name, rgb in (("striped", (200, 200, 40)), ("red", (200, 20, 20)), ("blue", (20, 40, 200))):
        frame = np.full((480, 640, 3), (40, 70, 190), np.uint8)
        # the default region is centred at (0.52, 0.72) of the frame: (333, 346) px, radii 128 x 120 px
        cv2.ellipse(frame, (333, 346), (140, 132), 0, 0, 360, (230, 230, 230), cv2.FILLED)
        cv2.circle(frame, (333, 346), 80, rgb, cv2.FILLED)
        frames[name] = frame
    areas = {(held, wanted): area(frames[held], wanted) for held in frames for wanted in frames}
    ok = all((areas[held, wanted] > 10_000) == (held == wanted) for held, wanted in areas)
    check("held check wired to #24's ColourVisibility: each held ball found only as itself, the blue room outside the "
          "held-ball region ignored", ok, str({f"{h}->{w}": a for (h, w), a in areas.items()}))
    whole_frame = held_ball_area(vision_dir, (0.5, 0.5, 2.0, 2.0))
    blue_room = whole_frame(frames["striped"], "blue")
    check("held check: without the region the blue room counts as a blue ball (why the region exists)",
          blue_room > 100_000, f"blue area on the whole frame with the striped ball held: {blue_room}")


def robot_flags() -> None:
    from run_supervisor import WRIST_HELD_ROI, camera_configs, parse_args, settings_from

    def parsed(*flags):
        saved = sys.argv
        sys.argv = ["run_supervisor.py", "--fake", "all", "--top-camera", "0", "--wrist-camera", "/dev/video2", *flags]
        try:
            return parse_args()
        finally:
            sys.argv = saved

    args = parsed()
    settings, cameras = settings_from(args), camera_configs(args)
    check("robot defaults: striped judged only, red candidate 2000 kept, PR #30 region, both cameras MJPG",
          settings.held_check_colours == ("striped",) and settings.min_held_area_px == {"red": 2000}
          and settings.min_held_target_area_px is None and tuple(args.held_check_roi) == WRIST_HELD_ROI == (0.52, 0.72, 0.2, 0.25)
          and [c.fourcc for c in cameras.values()] == ["MJPG", "MJPG"] and cameras["wrist"].index_or_path == "/dev/video2",
          f"{settings.held_check_colours} {settings.min_held_area_px} {[c.fourcc for c in cameras.values()]}")
    args = parsed("--held-check-colours", "striped", "red", "--min-held-target-area", "120",
                  "--min-held-area", "red=1500", "--camera-fourcc", "none")
    settings, cameras = settings_from(args), camera_configs(args)
    check("one flag adds red; --min-held-area red=1500 overrides the candidate; --camera-fourcc none",
          settings.held_check_colours == ("striped", "red") and settings.min_held_area_px == {"red": 1500}
          and settings.min_held_target_area_px == 120 and [c.fourcc for c in cameras.values()] == [None, None],
          f"{settings.held_check_colours} {settings.min_held_area_px} {[c.fourcc for c in cameras.values()]}")


def speed_cap() -> None:
    last = {key: 0.0 for key in KEYS}
    wanted = {**last, "shoulder_pan.pos": 10.0, "elbow_flex.pos": -5.0, "gripper.pos": 40.0}
    sent = capped(wanted, last, 1.0)
    check("speed cap: largest arm joint step 1.0 deg, direction kept, gripper not capped",
          sent["shoulder_pan.pos"] == 1.0 and sent["elbow_flex.pos"] == -0.5 and sent["gripper.pos"] == 40.0
          and capped(wanted, last, None) == wanted, str(sent))


def send_hands(port: int, stop: threading.Event) -> None:
    hands = [{"x": 0.20, "y": 0.03, "z": 0.13, "palm_up": True}, {"x": 0.20, "y": 0.09, "z": 0.13, "palm_up": True}]
    with socket.create_connection(("127.0.0.1", port)) as sock:
        while not stop.is_set():
            sock.sendall((json.dumps({"t": time.time(), "hands": hands}) + "\n").encode())
            time.sleep(1 / 30)


class TipRecorder(FakeArm):
    def __init__(self, world, abort, kinematics):
        super().__init__(world, abort)
        self.kinematics, self.tips = kinematics, []

    def send(self, action):
        if self.world.holding and action["gripper.pos"] > 30:
            self.world.holding = False
        self.tips.append(self.kinematics.forward_kinematics(pose_of(action)[:5]))
        self.last_sent = dict(action)


def give_integration(handover_dir: Path, log_path: str) -> None:
    hand_socket = import_from(handover_dir, "hand_socket")
    kinematics = import_from(handover_dir, "kinematics")
    port = 18766
    receiver = hand_socket.HandReceiver("127.0.0.1", port)
    stop = threading.Event()
    threading.Thread(target=send_hands, args=(port, stop), daemon=True).start()

    abort, world = threading.Event(), World(hand=True)
    arm = TipRecorder(world, abort, kinematics)
    settings = Settings()
    skills = Skills(pick=scripted("pick", world, ["ok"], pick_effect(world), ticks=10),
                    put_down=scripted("put_down", world, ["ok"], release_effect(world, "ok")),
                    give=lambda: give_sequence(arm, handover_dir=handover_dir, receiver=receiver, abort=abort,
                                                timeout_s=settings.give_timeout_s, gripper_open=60.0))
    sensors = Sensors(palms_seen=lambda: len(receiver.newest_hands()) >= 2)
    rest = pose_of(arm.last_sent)
    started = time.monotonic()
    result = Supervisor(arm, skills, sensors, settings, rest, TransitionLog(log_path, echo=False), abort,
                        RealClock(settings.fps)).run_round(99)
    stop.set()
    tips = np.array(arm.tips)
    radius = np.hypot(tips[:, 0], tips[:, 1])
    check("GIVE via the #13 loop on a sim arm, folded rest pose", result.outcome == "given"
          and np.allclose(pose_of(arm.last_sent), rest),
          f"{' -> '.join(s.value for s in result.states)}; {time.monotonic() - started:.1f} s; "
          f"tip z {tips[:, 2].min():.3f}-{tips[:, 2].max():.3f} m, radius {radius.min():.3f}-{radius.max():.3f} m, "
          f"rest tip {np.round(kinematics.forward_kinematics(rest[:5]), 3).tolist()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--handover-dir", default=str(SUPERVISOR.parent / "handover"),
                   help="directory with #13's handover_loop.py; that test is skipped if it is missing")
    p.add_argument("--skip-policy", action="store_true")
    p.add_argument("--vision-dir", default=str(SUPERVISOR.parent / "vision"),
                   help="directory with #24's colour_detector.py and supervisor_adapter.py; that test is skipped if missing")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as work:
        fake_scenarios(f"{work}/fake.jsonl")
        voice_container()
        round_hygiene(f"{work}/hygiene.jsonl")
        held_ball_check(f"{work}/held.jsonl")
        robot_flags()
        speed_cap()
        vision_dir = Path(args.vision_dir)
        if (vision_dir / "supervisor_adapter.py").exists():
            vision_adapter(vision_dir)
        else:
            print(f"SKIP  held check with #24's detector: no supervisor_adapter.py in {vision_dir}")
        if not args.skip_policy:
            policy_runner()
            policy_runner(("camera1", "camera2"), {"top": "camera1", "wrist": "camera2"}, n_action_steps=20)
        handover_dir = Path(args.handover_dir)
        if (handover_dir / "handover_loop.py").exists():
            give_integration(handover_dir, f"{work}/give.jsonl")
        else:
            print(f"SKIP  GIVE integration: no handover_loop.py in {handover_dir}")
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
