import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SUPERVISOR = Path(__file__).resolve().parents[1] / "scripts" / "supervisor"
# tests/cuda.py would shadow the `cuda` package torch imports, so this script's own folder leaves sys.path.
sys.path[:] = [str(SUPERVISOR)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from commands import Commands
from grasp_check import requery
from fakes import SCENARIOS, FakeArm, FakeGraspPick, Scenario, World, build, pick_effect, release_effect, scripted
from machine import RealClock, Sensors, Settings, Skills, Supervisor, TransitionLog
from skills import KEYS, GiveSettings, LoadedPolicy, PolicySkill, Settled, capped, give_sequence, import_from, pose_of

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
        queued = len(loaded.policy._action_queue)
        requery(skill)
        cleared = len(loaded.policy._action_queue)
    check(f"grasp check re-query empties ACT's queued chunk (n_action_steps {n_action_steps})",
          queued == loaded.policy.config.n_action_steps - 3 and cleared == 0, f"queued {queued} -> {cleared}")
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
    checks = verify(run("held_check_deep_grasp_unseen_is_kept"))
    check("held check: every ball 0 px (deep grasp, ball above the region) -> unseen, kept and handed over, not wrong",
          [(e["target"], e["held_check"], e["held_areas_px"]) for e in checks]
          == [("GIVE", "unseen", {"striped": 0, "red": 0, "blue": 0})], str(checks))
    events = run("held_check_red_ball_every_pick")
    check("held check: a red ball over the floor is still wrong (unseen needs every ball below the floor)",
          [e["held_check"] for e in verify(events)] == ["wrong"] * 3, str(verify(events)))


def wrist_frames(work: str) -> None:
    from run_supervisor import wrist_frame_saver

    def run(save) -> list[dict]:
        scenario = next(s for s in SCENARIOS if s.name == "held_check_red_ball_put_down_then_retry")
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        sensors.save_wrist_frame = save
        log = TransitionLog(f"{work}/frames.jsonl", echo=False)
        result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(3)
        log.file.close()
        with open(f"{work}/frames.jsonl") as lines:
            events = [json.loads(line) for line in lines]
        open(f"{work}/frames.jsonl", "w").close()
        return result.outcome, [e for e in events if e["event"] == "transition" and e["source"] == "VERIFY_HOLD"]

    save = wrist_frame_saver(f"{work}/chain_test.jsonl")
    outcome, checks = run(save)
    saved = [Path(e.get("wrist_frame", "")) for e in checks]
    check("wrist frame: one JPEG per VERIFY_HOLD check next to the log, named by round and pick attempt",
          outcome == "given" and len(saved) == 2 and all(p.is_file() and p.read_bytes()[:2] == b"\xff\xd8" for p in saved)
          and [p.name.startswith(f"chain_test_verify_hold_r3_p{n}_") for n, p in enumerate(saved, 1)] == [True, True],
          f"{outcome} {saved}")

    def broken(frame, name):
        raise OSError("disk full")
    outcome, checks = run(broken)
    check("wrist frame: a failed save is logged, the round carries on",
          outcome == "given" and all("disk full" in e["wrist_frame_error"] for e in checks), f"{outcome} {checks}")
    check("wrist frame: no --log -> nothing saved", wrist_frame_saver(None) is None)


def free_names() -> None:
    """Names a run_supervisor.py function reads but nothing binds: a NameError on the arm only (board 7065772934).

    The fake runs never reach run_connected, so this reads the source: every name loaded in a top-level function
    must be bound in it (arguments, assignments, nested defs, lambdas, comprehensions), at module level or a builtin.
    """
    import ast
    import builtins

    tree = ast.parse((SUPERVISOR / "run_supervisor.py").read_text())
    module_names = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            module_names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    problems = {}
    for function in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        bound, loaded = set(), set()
        for n in ast.walk(function):
            if isinstance(n, ast.Name):
                (loaded if isinstance(n.ctx, ast.Load) else bound).add(n.id)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
        free = loaded - bound - module_names
        if free:
            problems[function.name] = sorted(free)
    check("run_supervisor.py: no function reads a name it does not define or get passed (run_connected's NameError)",
          not problems and "run_connected" in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}, str(problems))


def grasp_check_log(log_path: str) -> None:
    def run(name: str):
        scenario = next(s for s in SCENARIOS if s.name == name)
        arm, skills, sensors, settings, abort, clock, world = build(scenario)
        log = TransitionLog(log_path, echo=False)
        Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(1)
        log.file.close()
        events = [json.loads(line) for line in open(log_path)]
        open(log_path, "w").close()
        return [e for e in events if e["event"] == "grasp_check"], world, events

    checks, world, _ = run("grasp_check_ball_continues")
    check("grasp check (a): ball held -> one 'held' check logged (reading, goal, error), policy re-queried, carried on",
          [c["verdict"] for c in checks] == ["held"] and checks[0]["reading"] == 13.0 and checks[0]["goal"] == 1.0
          and checks[0]["error"] == 12.0 and world.queries == 1, str(checks))
    checks, world, events = run("grasp_check_nothing_requery_then_ball")
    picks = [e["reason"] for e in events if e["event"] == "transition" and e["source"] == "PICK"]
    check("grasp check (b): closed on nothing -> miss (reading 0.5, not in band), opened and backed off, PICK again at once",
          [(c["verdict"], c["in_band"], c.get("back_off")) for c in checks] == [("miss", False, True), ("held", True, None)]
          and checks[0]["pick_attempt"] == 1 and checks[1]["pick_attempt"] == 2
          and "grasp check miss, re-query" in picks[0], f"{checks} {picks}")
    checks, _, _ = run("grasp_check_tips_slip_in_settle_is_miss")
    check("grasp check (c): tips grasp in the band at first, slips during the settle -> miss",
          checks[0]["verdict"] == "miss" and checks[0]["first_reading"] == 13.0 and checks[0]["reading"] == 0.5, str(checks[0]))
    checks, _, _ = run("grasp_check_slip_during_lift_is_miss")
    check("grasp check: held at the check, slips during the lift -> watch miss, opened in place",
          [(c["phase"], c["verdict"]) for c in checks] == [("settle", "held"), ("watch", "miss"), ("settle", "held")]
          and checks[1]["back_off"] is False, str(checks))
    checks, _, events = run("grasp_check_misses_hit_pick_limit")
    end = next(e for e in events if e["event"] == "round_end")
    check("grasp check (d): 3 misses -> pick limit, VERIFY_HOLD not holding, gave_up at rest",
          [c["verdict"] for c in checks] == ["miss"] * 3 and end["outcome"] == "gave_up" and end["pick_attempts"] == 3, str(end))
    checks, _, _ = run("grasp_check_too_high_miss_before_reading")
    check("grasp check: close with the tip above --grasp-max-tip-height -> miss before reading the gripper",
          [(c["phase"], c.get("reason")) for c in checks] == [("trigger", "too_high")] * 3, str(checks[0]))
    checks, world, events = run("grasp_check_lock_blocks_reopen")
    verdicts = [c["verdict"] for c in checks if "verdict" in c]
    blocked = [c for c in checks if c["phase"] == "release_blocked"]
    pick_end = next(e for e in events if e["event"] == "transition" and e["source"] == "PICK")
    check("grasp check lock (r1 pattern): held, policy re-opens to 30 for 1.3 s and closes -> one 'held', "
          "release_blocked 2 lines (once a second, fake clock), ball kept, PICK ends ok, no miss",
          verdicts == ["held"] and len(blocked) == 2 and all(b["command"] == 30.0 and b["sent"] == 5.0 for b in blocked)
          and checks[0]["lock_goal"] == 5.0
          and pick_end["target"] == "VERIFY_HOLD"
          and pick_end["skill"] == "ok" and not world.violations, f"{checks} {pick_end}")
    from grasp_check import GraspChecked, GraspCheckSettings
    pick = FakeGraspPick(World())
    lock = GraspChecked(pick, GraspCheckSettings(), 30).lock_goal_for
    check("grasp check lock goal = held reading - 4, at least close goal 1, at most hold low - 1 (5: an empty jaw "
          "there still reads below 6, so the slip watch is unchanged)",
          (lock(11.0), lock(8.33), lock(4.0), round(GraspChecked(pick, GraspCheckSettings(lock_margin=2.0), 30)
                                                      .lock_goal_for(6.5), 2)) == (5.0, 4.33, 1.0, 4.5),
          str((lock(11.0), lock(8.33), lock(4.0))))
    checks, _, events = run("grasp_check_unlocked_reopen_rearms")
    check("grasp check unlocked: re-open logged as release, re-armed, the empty close is a fresh check (miss), "
          "re-pick holds",
          [(c["phase"], c.get("verdict")) for c in checks][:3] == [("settle", "held"), ("release", None), ("settle", "miss")]
          and checks[0]["lock_goal"] is None
          and checks[2]["pick_attempt"] == 1 and checks[-1]["verdict"] == "held" and checks[-1]["pick_attempt"] == 2,
          str([(c["phase"], c.get("verdict"), c["pick_attempt"]) for c in checks]))
    checks, world, events = run("grasp_check_lock_blocks_reopen")
    blocked = [c for c in checks if c["phase"] == "release_blocked"]
    check("grasp check lock: every release_blocked line carries the gripper load and the count of blocked commands "
          "since the last line (first line 1, then one line a second)",
          [b["load"] for b in blocked] == [480.0, 480.0] and blocked[0]["count"] == 1
          and 30 <= blocked[1]["count"] <= 31, str(blocked))
    # Slow arm (exp1 R3): record what the grasp check sends against where the arm is measured.
    scenario = next(s for s in SCENARIOS if s.name == "grasp_check_miss_slow_arm_backs_off_from_measured")
    arm, skills, sensors, settings, abort, clock, world = build(scenario)
    sent, original = [], arm.send
    arm.send = lambda action: (sent.append((pose_of(action), arm.last_obs and pose_of(arm.last_obs))), original(action))
    log = TransitionLog(log_path, echo=False)
    Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(1)
    log.file.close()
    events = [json.loads(line) for line in open(log_path)]
    open(log_path, "w").close()
    first = next(i for i, (pose, _) in enumerate(sent) if pose[5] < 10)
    settle = sent[first:first + int(GraspCheckSettings().settle_s * settings.fps)]
    measured = settle[0][1][:5]
    still = all(np.abs(pose[:5] - measured).max() <= 0.5 for pose, _ in settle)
    picks = [e for e in events if e["event"] == "transition" and e["source"] == "PICK"]
    check("grasp check, slow arm: during the settle the arm is held where it is measured (not at the policy's raw "
          "target), the miss opens and backs off and PICK re-queries at once (no PICK timeout)",
          still and picks[0]["target"] == "PICK" and picks[0]["skill"] == "grasp_miss"
          and "backoff_timeout" not in picks[0],
          f"settle sent {[round(float(p[1]), 2) for p, _ in settle[:3]]} measured {round(float(measured[1]), 2)} {picks[:1]}")
    checks, _, events = run("grasp_check_backoff_timeout_repicks")
    picks = [e for e in events if e["event"] == "transition" and e["source"] == "PICK"]
    timeouts = [c for c in checks if c["phase"] == "release_timeout"]
    check("grasp check: an arm that cannot back off -> after --grasp-backoff-timeout (3 s) grasp_miss with "
          "backoff_timeout: true, re-pick as normal",
          picks[0]["target"] == "PICK" and picks[0]["skill"] == "grasp_miss" and picks[0].get("backoff_timeout") is True
          and len(timeouts) == 1 and timeouts[0]["backoff_timeout"] is True and 3.0 <= timeouts[0]["release_s"] < 3.1,
          f"{picks[:1]} {timeouts}")
    checks, world, events = run("grasp_check_policy_releases_while_locked")
    pick_end = next(e for e in events if e["event"] == "transition" and e["source"] == "PICK")
    released = [c for c in checks if c["phase"] == "released_locked"]
    end = next(e for e in events if e["event"] == "round_end")
    check("grasp check lock: the policy commands open above the lock for 1.5 s in all -> PICK ends 'policy released "
          "while locked', arm to rest with the gripper still locked, VERIFY_HOLD (no 40 s PICK timeout)",
          pick_end["target"] == "VERIFY_HOLD" and pick_end["reason"] == "pick attempt 1: policy released while locked"
          and pick_end.get("move") == "arrived" and len(released) == 1 and released[0]["open_s"] >= 1.5
          and released[0]["sent"] == 5.0 and end["duration_s"] < 20 and not world.violations,
          f"{pick_end} {released} {end['duration_s']}")
    checks, _, _ = run("grasp_check_low_load_is_miss")
    check("grasp check: reading in the band but load 120 < 200 -> miss; then load 480 -> held",
          [(c["verdict"], c["in_band"], c["stalled"], c["load"], c["load_ok"]) for c in checks]
          == [("miss", True, True, 120.0, False), ("held", True, True, 480.0, True)], str(checks))
    checks, _, _ = run("grasp_check_load_unreadable_reading_rule")
    check("grasp check: load read fails -> reading rule alone decides, logged as load_rule unavailable",
          [(c["verdict"], c["load"], c["load_rule"]) for c in checks] == [("held", None, "unavailable")]
          and "no Present_Load" in checks[0].get("load_error", ""), str(checks))
    from grasp_check import load_magnitude
    check("grasp check: load magnitude = abs(), and bit 10 (sign-magnitude direction) dropped",
          (load_magnitude(-480), load_magnitude(1024 + 480), load_magnitude(28), load_magnitude(None)) == (480, 480, 28, None))
    checks, _, _ = run("grasp_check_off_old_flow_carries_empty")
    check("grasp check off -> nothing logged, an empty close is only caught at VERIFY_HOLD", checks == [])


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
    grasp = settings.grasp_check
    check("robot defaults: grasp check on (trigger 10, settle 0.4 s, min error 3, band 6-20, min load 200, "
          "locked after a hold, tip height off); --no-grasp-check turns it off",
          grasp is not None and (grasp.trigger, grasp.settle_s, grasp.min_error, grasp.hold_range, grasp.min_load,
                                 grasp.lock_after_hold, grasp.max_tip_height)
          == (10.0, 0.4, 3.0, (6.0, 20.0), 200.0, True, None)
          and settings_from(parsed("--no-grasp-check")).grasp_check is None, str(grasp))
    check("--no-grasp-lock-after-hold restores the unlocked flow; --grasp-lock-margin (default 4)",
          settings_from(parsed("--no-grasp-lock-after-hold")).grasp_check.lock_after_hold is False
          and settings_from(parsed("--grasp-lock-margin", "3")).grasp_check.lock_margin == 3.0
          and settings_from(parsed()).grasp_check.lock_margin == 4.0)
    check("--grasp-backoff-timeout (default 3 s) and --grasp-open-intent-s (default 1.5 s)",
          (settings_from(parsed()).grasp_check.backoff_timeout_s, settings_from(parsed()).grasp_check.open_intent_s)
          == (3.0, 1.5)
          and settings_from(parsed("--grasp-backoff-timeout", "2", "--grasp-open-intent-s", "1")).grasp_check.backoff_timeout_s == 2.0
          and settings_from(parsed("--grasp-open-intent-s", "1")).grasp_check.open_intent_s == 1.0)
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


TWO_PALMS = [{"x": 0.20, "y": 0.03, "z": 0.13, "palm_up": True}, {"x": 0.20, "y": 0.09, "z": 0.13, "palm_up": True}]


def send_hands(port: int, stop: threading.Event, no_hands_for_s: float = 0.0) -> None:
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", port)) as sock:
        while not stop.is_set():
            hands = TWO_PALMS if time.monotonic() - started >= no_hands_for_s else []
            sock.sendall((json.dumps({"t": time.time(), "hands": hands}) + "\n").encode())
            time.sleep(1 / 30)


class TipRecorder(FakeArm):
    def __init__(self, world, abort, kinematics, rest=None):
        super().__init__(world, abort) if rest is None else super().__init__(world, abort, rest)
        self.kinematics, self.tips, self.grippers = kinematics, [], []

    def send(self, action):
        if self.world.holding and action["gripper.pos"] > 30:
            self.world.holding = False
        self.tips.append(self.kinematics.forward_kinematics(pose_of(action)[:5]))
        self.grippers.append(action["gripper.pos"])
        self.last_sent = dict(action)


# Balazs's folded rest pose (board 7064702491) with an open-ish gripper, the rest Tian's loop folds to on the arm.
TABLE_REST = np.array([-3.69, -103.43, 97.01, -100.26, 1.45, 30.0])


def chain_pick_give_rest(handover_dir: Path, log_path: str) -> None:
    """PICK (scripted) -> VERIFY_HOLD -> present pose -> GIVE by #13's loop with Tian's tested settings (hover 0.28,
    floor 0.20, half speed, 3 celebrate cycles) -> REST, on one simulated connection."""
    hand_socket = import_from(handover_dir, "hand_socket")
    kinematics = import_from(handover_dir, "kinematics")
    port = 18767
    receiver = hand_socket.HandReceiver("127.0.0.1", port)
    stop = threading.Event()
    threading.Thread(target=send_hands, args=(port, stop), daemon=True).start()

    abort, world = threading.Event(), World(hand=True)
    arm = TipRecorder(world, abort, kinematics, TABLE_REST)
    settings = Settings()
    give = GiveSettings()
    skills = Skills(pick=scripted("pick", world, ["ok"], pick_effect(world), ticks=10),
                    put_down=scripted("put_down", world, ["ok"], release_effect(world, "ok")),
                    give=lambda: give_sequence(arm, handover_dir=handover_dir, receiver=receiver, abort=abort,
                                                timeout_s=settings.give_timeout_s, gripper_open=60.0, settings=give))
    sensors = Sensors(palms_seen=lambda: len(receiver.newest_hands()) >= 2)
    rest = pose_of(arm.last_sent)
    log = TransitionLog(log_path, echo=False)
    started = time.monotonic()
    result = Supervisor(arm, skills, sensors, settings, rest, log, abort, RealClock(settings.fps)).run_round(98)
    stop.set()
    events = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    give_end = next(e for e in events if e["event"] == "transition" and e["source"] == "GIVE")
    tips = np.array(arm.tips)
    # Gripper closes after the drop opened it: the celebration (3 cycles -> 3 closes to the loop's CELEBRATE_CLOSED).
    grips = np.array(arm.grippers)
    opened = int(np.argmax(grips >= 59.5))
    closes = int(np.sum((grips[opened:-1] > 30) & (grips[opened + 1:] < 30)))
    check("chain PICK -> GIVE (Tian's loop) -> REST, table rest pose", result.outcome == "given"
          and [s.value for s in result.states] == ["REST", "PICK", "VERIFY_HOLD", "GIVE", "REST", "DONE"]
          and give_end["skill"] == "dropped" and np.allclose(pose_of(arm.last_sent), rest),
          f"{' -> '.join(s.value for s in result.states)}; GIVE {give_end['skill']}; {time.monotonic() - started:.1f} s")
    check("chain: 3 celebrate cycles inside GIVE, then the gripper stays open to REST", closes == 3
          and np.all(grips[-5:] >= 30), f"closes after the drop {closes}")
    check("chain: tip never below 0.20 m on the whole round (rest, present pose, follow, fold)",
          tips[:, 2].min() >= 0.20 - 1e-3, f"tip z {tips[:, 2].min():.3f}-{tips[:, 2].max():.3f} m")


def give_survives_lost_hand(handover_dir: Path, log_path: str) -> None:
    hand_socket = import_from(handover_dir, "hand_socket")
    kinematics = import_from(handover_dir, "kinematics")
    lost_to_home_s = import_from(handover_dir, "following").FollowSettings.lost_to_home_s
    port, no_hands_for_s = 18768, 14.0
    receiver = hand_socket.HandReceiver("127.0.0.1", port)
    stop = threading.Event()
    hands_from = time.time() + no_hands_for_s
    threading.Thread(target=send_hands, args=(port, stop, no_hands_for_s), daemon=True).start()

    abort, world = threading.Event(), World(hand=True)
    arm = TipRecorder(world, abort, kinematics, TABLE_REST)
    settings = Settings()
    skills = Skills(pick=scripted("pick", world, ["ok"], pick_effect(world), ticks=10),
                    put_down=scripted("put_down", world, ["ok"], release_effect(world, "ok")),
                    give=lambda: give_sequence(arm, handover_dir=handover_dir, receiver=receiver, abort=abort,
                                                timeout_s=settings.give_timeout_s, gripper_open=60.0,
                                                settings=GiveSettings()))
    sensors = Sensors(palms_seen=lambda: len(receiver.newest_hands()) >= 2, destination=lambda: "hand")
    result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), TransitionLog(log_path, echo=False),
                        abort, RealClock(settings.fps)).run_round(97)
    stop.set()
    events = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
    give_start = next(e for e in events if e["event"] == "transition" and e["target"] == "GIVE")
    give_end = next(e for e in events if e["event"] == "transition" and e["source"] == "GIVE")
    no_hand_in_give = hands_from - datetime.fromisoformat(give_start["time"]).timestamp()
    check("GIVE: no hand for longer than lost_to_home_s, then two palms -> dropped, not hand_lost",
          result.outcome == "given" and give_end["skill"] == "dropped" and no_hand_in_give > lost_to_home_s + 1.0,
          f"no hand for {no_hand_in_give:.1f} s inside GIVE (lost_to_home_s {lost_to_home_s}); GIVE {give_end['skill']}; "
          f"{' -> '.join(s.value for s in result.states)}")


def abort_over_socket_during_give(handover_dir: Path, log_path: str) -> None:
    link = import_from(SUPERVISOR.parent / "voice", "link")
    hand_socket = import_from(handover_dir, "hand_socket")
    kinematics = import_from(handover_dir, "kinematics")
    commands = Commands()
    commands.listen_voice("127.0.0.1", 18769)
    receiver = hand_socket.HandReceiver("127.0.0.1", 18770)
    world = World(hand=True)
    arm = TipRecorder(world, commands.abort, kinematics, TABLE_REST)
    settings = Settings()
    skills = Skills(pick=scripted("pick", world, ["ok"], pick_effect(world), ticks=10),
                    put_down=scripted("put_down", world, ["ok"], release_effect(world, "ok")),
                    give=lambda: give_sequence(arm, handover_dir=handover_dir, receiver=receiver, abort=commands.abort,
                                                timeout_s=settings.give_timeout_s, gripper_open=60.0,
                                                settings=GiveSettings()))
    sensors = Sensors(destination=lambda: "hand")
    threading.Timer(6.0, lambda: link.send({"abort": True}, port=18769)).start()
    result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), TransitionLog(log_path, echo=False),
                        commands.abort, RealClock(settings.fps)).run_round(96)
    states = [s.value for s in result.states]
    check("dashboard ABORT ({\"abort\": true} on the command port) stops GIVE, which the tick poll never reaches",
          result.outcome == "aborted" and "GIVE" in states and states[-1] == "ABORT", " -> ".join(states))


def status_datagram() -> None:
    from status import Status

    port = 18771
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(2.0)
    status = Status()
    status.update(gripper=13.2, give_mode="follow")
    status.add_reader(lambda: {"palms": 1})
    status.start(port=port, hz=20.0)
    message = json.loads(sock.recv(65536))
    sock.close()
    check("live status: one UDP datagram with the stored values and the reader's", message["gripper"] == 13.2
          and message["give_mode"] == "follow" and message["palms"] == 1 and "time" in message, str(message))


class _WristBus:
    def sync_read(self, name):
        return {motor: 0.0 for motor in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}


class _WristRobot:
    bus = _WristBus()

    def __init__(self):
        self.count = 0

    def get_observation(self):
        self.count += 1
        return {**{key: 0.0 for key in KEYS}, "top": np.zeros((480, 640, 3), np.uint8),
                "wrist": np.full((480, 640, 3), self.count, np.uint8)}


def wrist_frames_shared(handtrack_dir: Path) -> None:
    import run_supervisor

    frame_link = import_from(handtrack_dir, "frame_link")
    name = f"wrist_test_{int(time.time())}"
    writer = frame_link.FrameWriter(name)
    reader = frame_link.FrameReader(name)
    arm = run_supervisor.RobotArm(_WristRobot(), None, writer)
    for _ in range(4):
        arm.observe()
    frame, _ = reader.latest()
    reader.close()
    writer.close()
    check("wrist frames: every second observe lands in the shared block the dashboard reads",
          int(frame[0, 0, 0]) == 4, f"pixel {int(frame[0, 0, 0])} after 4 observes")


class _TickBus:
    def sync_read(self, name, motors=None):
        if name == "Present_Load":
            return {"gripper": 450}
        return {motor: 0.0 for motor in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}


class _TickRobot:
    bus = _TickBus()

    def get_observation(self):
        return {**{key: 1.0 for key in KEYS}, "top": np.zeros((4, 4, 3), np.uint8), "wrist": np.zeros((4, 4, 3), np.uint8)}

    def send_action(self, action):
        return action


def tick_log(work: str) -> None:
    from types import SimpleNamespace

    import run_supervisor
    from machine import State
    from ticks import TICKS

    path = run_supervisor.tick_log_path(f"{work}/chain_test.jsonl")
    TICKS.open(path)
    TICKS.context = lambda: {"round": 1, "state": "PICK"}
    arm = run_supervisor.RobotArm(_TickRobot(), 5.0)
    arm.observe()
    TICKS.note(policy=[20.0, 0, 0, 0, 0, 30.0])
    arm.send(dict(zip(KEYS, [20.0, 0, 0, 0, 0, 3.0])))
    arm.observe()
    TICKS.close()
    lines = [json.loads(line) for line in Path(path).read_text().splitlines()]
    first, second = lines
    check("tick log: one line per observation, paired with the policy action, command and capped goal sent from it",
          len(lines) == 2 and first["state"] == "PICK" and first["load"] == 450 and first["measured"][0] == 1.0
          and first["policy"][5] == 30.0 and first["command"][0] == 20.0 and first["sent"][0] == 5.0
          and "command" not in second and path.endswith("chain_test.ticks.jsonl"), str(lines))
    supervisor = SimpleNamespace(round=SimpleNamespace(number=3, states=[State.REST, State.PICK]))
    context = run_supervisor.tick_context(supervisor)
    running = context()
    supervisor.round.states.append(State.DONE)
    check("tick log: round and state while a round runs, idle between rounds; ticks.jsonl next to supervisor.jsonl",
          running == {"round": 3, "state": "PICK"} and context() == {"round": None, "state": "idle"}
          and Path(run_supervisor.tick_log_path("exp/supervisor.jsonl")).as_posix() == "exp/ticks.jsonl",
          f"{running} / {context()}")


class _StallingRobot(_TickRobot):
    def __init__(self, stalls: int):
        self.stalls = stalls

    def get_observation(self):
        if self.stalls:
            self.stalls -= 1
            raise TimeoutError("OpenCVCamera(top) latest frame is too old: 519.0 ms (max allowed: 500 ms).")
        return super().get_observation()


def camera_stall() -> None:
    import run_supervisor

    arm = run_supervisor.RobotArm(_StallingRobot(stalls=3), 5.0)
    obs = arm.observe()
    check("camera stall: a stale-frame TimeoutError is read again, not an ABORT (VLA 5K freeze, 519 ms)",
          obs["gripper.pos"] == 1.0 and arm.camera_stalls == 1, f"stalls counted {arm.camera_stalls}")
    arm = run_supervisor.RobotArm(_StallingRobot(stalls=10**6), 5.0)
    arm.stall_limit_s = 0.1
    try:
        arm.observe()
        raised = False
    except TimeoutError:
        raised = True
    check("camera stall: a camera silent past the limit still raises, so the round ends in ABORT", raised)


def hold_needs_load(log_path: str) -> None:
    def run(load: str, **settings):
        scenario = Scenario("pick_then_give", [], "", load=load, settings=settings)
        arm, skills, sensors, settings_, abort, clock, world = build(scenario)
        log = TransitionLog(log_path, echo=False)
        result = Supervisor(arm, skills, sensors, settings_, pose_of(arm.last_sent), log, abort, clock).run_round(1)
        log.file.close()
        events = [json.loads(line) for line in open(log_path)]
        open(log_path, "w").close()
        return result, events

    empty = lambda: 30.0  # noqa: E731
    result, events = run("physics")
    holds = [e for e in events if e["event"] == "transition" and e["source"] == "VERIFY_HOLD"]
    check("hold rule, grasp check off: a squeezed ball (load 480) still counts as held, the load is logged",
          result.outcome == "given" and holds[0].get("load") == 480 and holds[0].get("min_load") == 200.0, str(holds[0]))

    scenario = Scenario("pick_then_give", [], "", picks=["ok", "ok", "ok"])
    arm, skills, sensors, settings, abort, clock, world = build(scenario)
    sensors.gripper_load = empty
    log = TransitionLog(log_path, echo=False)
    result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(1)
    log.file.close()
    events = [json.loads(line) for line in open(log_path)]
    open(log_path, "w").close()
    states = [s.value for s in result.states]
    check("hold rule, grasp check off: reading in the band but load 30 (VLA 5K take 2) -> not held, re-pick, "
          "gave_up after 3 picks, no REST <-> VERIFY_HOLD loop",
          result.outcome == "gave_up" and result.pick_attempts == 3 and states.count("VERIFY_HOLD") == 3,
          " -> ".join(states))


def abort_details(log_path: str) -> None:
    scenario = Scenario("pick_then_give", [], "")
    arm, skills, sensors, settings, abort, clock, world = build(scenario)

    def broken_palms():
        raise TimeoutError("OpenCVCamera(top) latest frame is too old: 519.0 ms (max allowed: 500 ms).")

    sensors.palms_seen = broken_palms
    log = TransitionLog(log_path, echo=False)
    result = Supervisor(arm, skills, sensors, settings, pose_of(arm.last_sent), log, abort, clock).run_round(1)
    log.file.close()
    events = [json.loads(line) for line in open(log_path)]
    to_abort = next(e for e in events if e["event"] == "transition" and e["target"] == "ABORT")
    check("ABORT on an error logs the error, where it was raised, the state, gripper reading and load",
          result.outcome == "aborted" and to_abort["source"] == "VERIFY_HOLD" and "519.0 ms" in to_abort["error"]
          and "broken_palms" in to_abort["error_at"] and to_abort["load"] == 480 and to_abort["gripper"] is not None
          and to_abort["pick_attempts"] == 1, str(to_abort))


def pick_end_rule() -> None:
    rest = np.array([-5.6, -102.8, 96.6, -100.1, -5.2, 30.0])
    never_left = np.array([*rest[:5], 17.8])
    settled = Settled(rest, (6.0, 20.0), 8.0, 1.0, 3.0)
    early = [i / 30 for i in range(300) if settled(dict(zip(KEYS, never_left)), i / 30)]
    check("PICK end: an arm that never left rest does not count as back holding (training rest gripper 17.8 is in the band)",
          not early, f"ended at {early[0]:.1f} s" if early else "")
    away = np.array([*rest[:2], rest[2] - 40.0, *rest[3:5], 40.0])
    back_off_pan_roll = np.array([rest[0] + 15.0, *rest[1:4], rest[4] - 15.0, 13.0])
    settled = Settled(rest, (6.0, 20.0), 8.0, 1.0, 3.0)
    poses = [away] * 30 + [back_off_pan_roll] * 120
    ended = next((i / 30 for i, pose in enumerate(poses) if settled(dict(zip(KEYS, pose)), i / 30)), None)
    check("PICK end: back at rest holding with pan and roll 15 deg off -> ends at the 3 s minimum",
          ended is not None and abs(ended - 3.0) < 0.05, f"ended at {ended} s")


class FakeBus:
    def __init__(self, log):
        self.log, self.goal = log, {"shoulder_pan": 2048}

    def connect(self):
        self.log.append("bus connect")

    def sync_read(self, name, normalize=True):
        assert name == "Present_Position" and normalize is False
        return {"shoulder_pan": 2100}

    def sync_write(self, name, values, normalize=True):
        assert name == "Goal_Position" and normalize is False
        self.goal = dict(values)
        self.log.append("write goal")

    def disconnect(self, disable_torque=True):
        self.log.append(f"bus disconnect torque_off={disable_torque}")


class _Stop(Exception):
    pass


class FakeRobot:
    def __init__(self):
        self.log = []
        self.bus = FakeBus(self.log)

    def connect(self):
        self.log.append("robot connect")
        raise _Stop


def goal_sync_before_connect() -> None:
    import run_supervisor

    robot = FakeRobot()
    try:
        run_supervisor.run_connected(None, robot, None, None, None, False, None, None, None, None)
    except _Stop:
        pass
    check("robot mode: Goal_Position set to the present position before connect() turns torque on (Tian's fix)",
          robot.log == ["bus connect", "write goal", "bus disconnect torque_off=False", "robot connect"]
          and robot.bus.goal == {"shoulder_pan": 2100}, str(robot.log))


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
    p.add_argument("--handtrack-dir", default=str(SUPERVISOR.parent / "handtrack"),
                   help="directory with frame_link.py (on main, not on this branch); the wrist-block test is skipped if missing")
    args = p.parse_args()
    # ignore_cleanup_errors: on Windows the transition logs are still open when the folder is removed.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        fake_scenarios(f"{work}/fake.jsonl")
        voice_container()
        round_hygiene(f"{work}/hygiene.jsonl")
        hold_needs_load(f"{work}/load.jsonl")
        abort_details(f"{work}/abort_details.jsonl")
        held_ball_check(f"{work}/held.jsonl")
        grasp_check_log(f"{work}/grasp.jsonl")
        wrist_frames(work)
        free_names()
        robot_flags()
        speed_cap()
        pick_end_rule()
        status_datagram()
        handtrack_dir = Path(args.handtrack_dir)
        if (handtrack_dir / "frame_link.py").exists():
            wrist_frames_shared(handtrack_dir)
        else:
            print(f"SKIP  wrist frames in the shared block: no frame_link.py in {handtrack_dir}")
        tick_log(work)
        camera_stall()
        goal_sync_before_connect()
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
            chain_pick_give_rest(handover_dir, f"{work}/chain.jsonl")
            give_survives_lost_hand(handover_dir, f"{work}/lost.jsonl")
            abort_over_socket_during_give(handover_dir, f"{work}/abort.jsonl")
        else:
            print(f"SKIP  GIVE integration: no handover_loop.py in {handover_dir}")
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
