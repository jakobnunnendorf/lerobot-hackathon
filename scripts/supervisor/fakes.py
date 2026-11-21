import threading
from dataclasses import dataclass, field

import numpy as np
from machine import Sensors, Settings, Skills
from skills import PRESENT_POSE, PresentAndDrop, action_of, holding_move, pose_of

DRY_RUN_REST = np.array([0.0, 20.0, 60.0, 90.0, 0.0, 30.0])
HELD_READING, EMPTY_CLOSED_READING, CLOSED_GOAL = 13.0, 0.5, 2.0


class FakeClock:
    def __init__(self, fps: int, abort: threading.Event, abort_at_tick: int | None = None):
        self.t, self.period, self.ticks = 0.0, 1 / fps, 0
        self.abort, self.abort_at_tick = abort, abort_at_tick

    def now(self) -> float:
        return self.t

    def tick(self) -> None:
        self.t += self.period
        self.ticks += 1
        if self.ticks == self.abort_at_tick:
            self.abort.set()


@dataclass
class World:
    buried: int = 0
    hand: bool = True
    destination: str | None = None
    holding: bool = False
    hold_goal: float = CLOSED_GOAL
    release_ok: bool = False
    block_after_pick: bool = False
    blocked_at: np.ndarray | None = None
    # Ball each successful pick ends up holding, in order; the last one repeats.
    held_balls: list[str] = field(default_factory=lambda: ["striped"])
    held_ball: str | None = None
    violations: list[str] = field(default_factory=list)


class FakeArm:
    def __init__(self, world: World, abort: threading.Event, rest: np.ndarray = DRY_RUN_REST):
        self.world, self.abort = world, abort
        self.last_sent = action_of(rest)
        self.last_obs = None
        self.sends_after_abort = 0

    def observe(self) -> dict:
        goal = pose_of(self.last_sent)
        if self.world.blocked_at is not None:
            goal[:5] = self.world.blocked_at
        gripper = HELD_READING if self.world.holding else (EMPTY_CLOSED_READING if goal[5] < 5 else goal[5])
        self.last_obs = {**action_of(np.concatenate([goal[:5], [gripper]])), "top": np.zeros((4, 4, 3), np.uint8),
                         "wrist": np.zeros((4, 4, 3), np.uint8)}
        return self.last_obs

    def send(self, action: dict) -> None:
        if self.abort.is_set():
            self.sends_after_abort += 1
        gripper = action["gripper.pos"]
        if self.world.holding and gripper > self.world.hold_goal + 0.5:
            if self.world.release_ok:
                self.world.holding = False
            else:
                self.world.violations.append(f"gripper goal {gripper:.1f} sent while holding")
        self.last_sent = dict(action)


class ScriptedSkill:
    def __init__(self, name: str, world: World, outcome: str, effect, ticks: int = 30):
        self.name, self.world, self.outcome, self.effect, self.ticks = name, world, outcome, effect, ticks

    def start(self, obs, last_sent, now):
        self.count, self.last_sent = 0, dict(last_sent)

    def step(self, obs, now):
        self.count += 1
        if self.count == self.ticks and self.outcome != "hang":
            return self.effect(self.outcome, self.last_sent)
        return None

    def done(self, obs, now):
        return self.outcome != "hang" and self.count >= self.ticks

    def result(self):
        return self.outcome


def scripted(name: str, world: World, outcomes: list[str], effect, ticks: int = 30):
    remaining = list(outcomes)
    return lambda: ScriptedSkill(name, world, remaining.pop(0) if len(remaining) > 1 else remaining[0], effect, ticks)


def _with_gripper(last_sent: dict, value: float) -> dict:
    return {**last_sent, "gripper.pos": value}


def pick_effect(world: World):
    def effect(outcome, last_sent):
        world.holding = outcome == "ok"
        if world.block_after_pick:
            world.blocked_at = pose_of(last_sent)[:5]
        if world.holding:
            world.buried = 0
            world.held_ball = world.held_balls.pop(0) if len(world.held_balls) > 1 else world.held_balls[0]
        return _with_gripper(last_sent, CLOSED_GOAL)
    return effect


def release_effect(world: World, success: str):
    def effect(outcome, last_sent):
        if outcome != success:
            return None
        world.holding, world.held_ball = False, None
        return _with_gripper(last_sent, 60.0)
    return effect


def clear_effect(world: World):
    def effect(outcome, last_sent):
        world.buried = max(world.buried - 1, 0)
    return effect


class Releasing:
    def __init__(self, skill, world: World):
        self.skill, self.world, self.name = skill, world, skill.name

    def start(self, obs, last_sent, now):
        self.world.release_ok = True
        self.skill.start(obs, last_sent, now)

    def step(self, obs, now):
        return self.skill.step(obs, now)

    def done(self, obs, now):
        finished = self.skill.done(obs, now)
        self.world.release_ok = not finished
        return finished

    def result(self):
        return self.skill.result()


@dataclass
class Scenario:
    name: str
    expected: list[str]
    outcome: str
    picks: list[str] = field(default_factory=lambda: ["ok"])
    gives: list[str] = field(default_factory=lambda: ["dropped"])
    put_downs: list[str] = field(default_factory=lambda: ["ok"])
    hand: bool = True
    buried: int = 0
    destination: str | None = None
    target: str | None = None
    abort_at_tick: int | None = None
    shake: bool = False
    give_mode: str = "scripted"
    block_after_pick: bool = False
    containers: bool = False
    colour_pick: bool = False
    settings: dict = field(default_factory=dict)
    held_balls: list[str] = field(default_factory=lambda: ["striped"])
    # Held-ball check: "off" (no hook), "on" (fake wrist detector, threshold 300 px), "unavailable" (hook raises).
    held_check: str = "off"
    colour_sensor: bool = False


FAKE_HELD_AREA_PX, FAKE_HELD_THRESHOLD_PX = 900, 300


def fake_held_ball_area(world: World, mode: str):
    if mode == "off":
        return None

    def area(frame, wanted):
        if mode == "unavailable":
            raise RuntimeError("no wrist frame")
        return FAKE_HELD_AREA_PX if world.holding and world.held_ball == wanted else 0
    return area


def build(scenario: Scenario):
    abort = threading.Event()
    world = World(buried=scenario.buried, hand=scenario.hand, destination=scenario.destination,
                  block_after_pick=scenario.block_after_pick, held_balls=list(scenario.held_balls))
    arm = FakeArm(world, abort)
    containers = {c: scripted("put_down_" + c, world, ["ok"], release_effect(world, "ok")) for c in ("red", "blue")}
    skills = Skills(
        pick=scripted("pick", world, scenario.picks, pick_effect(world), ticks=60),
        put_down=scripted("put_down", world, scenario.put_downs, release_effect(world, "ok")),
        give=(lambda: Releasing(PresentAndDrop(lambda: world.hand, 60.0), world)) if scenario.give_mode == "present"
        else scripted("give", world, scenario.gives, release_effect(world, "dropped")),
        present=lambda: holding_move("to_present", arm, PRESENT_POSE),
        clear_gray=scripted("clear_gray", world, ["ok"], clear_effect(world)),
        put_down_container=containers if scenario.containers else {},
        pick_colour=(lambda target: scripted("pick_" + target, world, scenario.picks, pick_effect(world), ticks=60)())
        if scenario.colour_pick else None,
        shake=scripted("shake", world, ["ok"], lambda outcome, last_sent: None) if scenario.shake else None,
    )
    sensors = Sensors(target=lambda: scenario.target, visible_area=lambda frame: 0 if world.buried else 800,
                      palms_seen=lambda: world.hand, destination=lambda: world.destination,
                      held_ball_area=fake_held_ball_area(world, scenario.held_check),
                      ball_colour=(lambda: world.held_ball) if scenario.colour_sensor else None)
    held = {"min_held_target_area_px": FAKE_HELD_THRESHOLD_PX} if scenario.held_check != "off" else {}
    settings = Settings(**{**held, **scenario.settings})
    clock = FakeClock(settings.fps, abort, scenario.abort_at_tick)
    return arm, skills, sensors, settings, abort, clock, world


R, P, V, G, U, D, A = "REST", "PICK", "VERIFY_HOLD", "GIVE", "PUT_DOWN", "DONE", "ABORT"
L, C, S = "LOOK", "CLEAR_GRAY", "SHAKE"

SCENARIOS = [
    Scenario("pick_then_give", [R, P, V, G, R, D], "given"),
    Scenario("no_hand_put_down", [R, P, V, U, R, D], "put_down", hand=False),
    Scenario("pick_fails_once", [R, P, V, R, P, V, G, R, D], "given", picks=["miss", "ok"]),
    Scenario("pick_times_out_once", [R, P, V, R, P, V, G, R, D], "given", picks=["hang", "ok"]),
    Scenario("pick_always_fails", [R, P, V, R, P, V, R, P, V, R, D], "gave_up", picks=["miss"]),
    Scenario("buried_look_only_policy_digs", [R, L, P, V, G, R, D], "given", buried=2, settings={"look": True}),
    Scenario("buried_twice_clear_gray_on", [R, L, C, R, L, C, R, L, P, V, G, R, D], "given", buried=2,
             settings={"clear_gray": True}),
    Scenario("clear_limit_then_pick", [R, L, C, R, L, C, R, L, P, V, G, R, D], "given", buried=5,
             settings={"clear_gray": True, "max_clear_attempts": 2}),
    Scenario("hand_lost_during_give", [R, P, V, G, U, R, D], "put_down", gives=["hand_lost"]),
    Scenario("give_times_out", [R, P, V, G, U, R, D], "put_down", gives=["hang"]),
    Scenario("voice_says_container", [R, P, V, U, R, D], "put_down", destination="container"),
    Scenario("put_down_fails_still_holding", [R, P, V, U, A], "aborted", hand=False, put_downs=["stuck"]),
    Scenario("abort_key_during_pick", [R, P, A], "aborted", abort_at_tick=40),
    Scenario("present_mode_drop_on_two_palms", [R, P, V, G, R, D], "given", give_mode="present"),
    Scenario("present_mode_voice_hand_but_no_palms", [R, P, V, G, U, R, D], "put_down", give_mode="present",
             hand=False, destination="hand"),
    Scenario("arm_blocked_on_way_to_present_pose", [R, P, V, A], "aborted", block_after_pick=True),
    Scenario("red_target_without_pick_skill_is_refused", [], "unsupported_target", target="red"),
    Scenario("red_target_container", [R, P, V, U, R, D], "put_down", target="red", destination="container",
             containers=True, colour_pick=True),
    Scenario("striped_container_uses_generic_pose", [R, P, V, U, R, D], "put_down", target="striped",
             destination="container", containers=True),
    Scenario("shake_hook", [R, P, V, G, S, R, D], "given", shake=True),
    # Held-ball check: the gripper reads the same for the striped target and a red/blue ball.
    Scenario("held_check_target_is_given", [R, P, V, G, R, D], "given", held_check="on"),
    Scenario("held_check_red_ball_put_down_then_retry", [R, P, V, U, R, P, V, G, R, D], "given",
             held_balls=["red", "striped"], held_check="on"),
    Scenario("held_check_red_ball_every_pick", [R, P, V, U, R, P, V, U, R, P, V, U, R, D], "wrong_ball",
             held_balls=["red"], held_check="on"),
    Scenario("held_check_red_ball_sorted_into_red_container", [R, P, V, U, R, P, V, G, R, D], "given",
             held_balls=["red", "striped"], held_check="on", containers=True, colour_sensor=True),
    Scenario("held_check_red_ball_voice_container_still_retries", [R, P, V, U, R, P, V, U, R, D], "put_down",
             held_balls=["red", "striped"], held_check="on", destination="container"),
    Scenario("held_check_red_target_held_red_is_accepted", [R, P, V, U, R, D], "put_down", target="red",
             destination="container", containers=True, colour_pick=True, held_balls=["red"], held_check="on"),
    Scenario("held_check_unmeasured_threshold_logs_only", [R, P, V, G, R, D], "given", held_balls=["red"],
             held_check="on", settings={"min_held_target_area_px": None}),
    Scenario("held_check_detector_unavailable_accepts", [R, P, V, G, R, D], "given", held_balls=["red"],
             held_check="unavailable"),
]
