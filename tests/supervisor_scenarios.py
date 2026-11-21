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
from skills import KEYS, LoadedPolicy, PolicySkill, give_sequence, import_from, pose_of

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
        if scenario.outcome in ("given", "put_down", "gave_up") and not np.allclose(pose_of(arm.last_sent), pose_of(FakeArm(World(), abort).last_sent)):
            problems.append("did not end at the rest pose")
        check(f"scenario {scenario.name}", not problems, "; ".join(problems))
    with open(log_path) as lines:
        events = [json.loads(line) for line in lines]
    ends = [e for e in events if e["event"] == "round_end"]
    check("log has one round_end per scenario with an outcome",
          len(ends) == len(SCENARIOS) and all(e["outcome"] for e in ends))


def policy_runner() -> None:
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
                        "observation.images.top": image, "observation.images.wrist": image},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))})
    stats = {"observation.state": {"mean": torch.zeros(6), "std": torch.ones(6)},
             "action": {"mean": torch.zeros(6), "std": torch.ones(6)},
             **{f"observation.images.{cam}": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}
                for cam in ("top", "wrist")}}
    with tempfile.TemporaryDirectory() as checkpoint:
        get_policy_class("act")(config).save_pretrained(checkpoint)
        pre, post = make_pre_post_processors(config, dataset_stats=stats)
        pre.save_pretrained(checkpoint)
        post.save_pretrained(checkpoint)

        loaded = LoadedPolicy(checkpoint, device="cpu")
        skill = PolicySkill("pick", loaded, features, "so101_follower")
        obs = {**{key: 0.0 for key in KEYS},
               "top": np.zeros((height, width, 3), np.uint8), "wrist": np.zeros((height, width, 3), np.uint8)}
        skill.start(obs, {key: 0.0 for key in KEYS}, 0.0)
        actions = [skill.step(obs, i / 30) for i in range(3)]
    ok = all(set(a) == set(KEYS) and all(np.isfinite(list(a.values()))) for a in actions)
    check("policy runner: checkpoint dir -> LoadedPolicy -> PolicySkill.step gives 6 named joint goals", ok,
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
    sensors = Sensors(hands_present=lambda: bool(receiver.newest_hands()))
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
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as work:
        fake_scenarios(f"{work}/fake.jsonl")
        voice_container()
        if not args.skip_policy:
            policy_runner()
        handover_dir = Path(args.handover_dir)
        if (handover_dir / "handover_loop.py").exists():
            give_integration(handover_dir, f"{work}/give.jsonl")
        else:
            print(f"SKIP  GIVE integration: no handover_loop.py in {handover_dir}")
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
