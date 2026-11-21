import threading
from dataclasses import dataclass, field

import numpy as np
from machine import Sensors, Settings, Skills
from skills import action_of, pose_of

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
    violations: list[str] = field(default_factory=list)


class FakeArm:
    def __init__(self, world: World, abort: threading.Event, rest: np.ndarray = DRY_RUN_REST):
        self.world, self.abort = world, abort
        self.last_sent = action_of(rest)
        self.last_obs = None
        self.sends_after_abort = 0

    def observe(self) -> dict:
        goal = pose_of(self.last_sent)
        gripper = HELD_READING if self.world.holding else (EMPTY_CLOSED_READING if goal[5] < 5 else goal[5])
        self.last_obs = {**action_of(np.concatenate([goal[:5], [gripper]])), "top": np.zeros((4, 4, 3), np.uint8)}
        return self.last_obs

    def send(self, action: dict) -> None:
        if self.abort.is_set():
            self.sends_after_abort += 1
        gripper = action["gripper.pos"]
        if self.world.holding and gripper > self.world.hold_goal + 0.5:
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
        if world.holding:
            world.buried = 0
        return _with_gripper(last_sent, CLOSED_GOAL)
    return effect


def release_effect(world: World, success: str):
    def effect(outcome, last_sent):
        if outcome != success:
            return None
        world.holding = False
        return _with_gripper(last_sent, 60.0)
    return effect


def clear_effect(world: World):
    def effect(outcome, last_sent):
        world.buried = max(world.buried - 1, 0)
    return effect


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
    abort_at_tick: int | None = None
    shake: bool = False
    settings: dict = field(default_factory=dict)


def build(scenario: Scenario):
    abort = threading.Event()
    world = World(buried=scenario.buried, hand=scenario.hand, destination=scenario.destination)
    arm = FakeArm(world, abort)
    skills = Skills(
        pick=scripted("pick", world, scenario.picks, pick_effect(world), ticks=60),
        put_down=scripted("put_down", world, scenario.put_downs, release_effect(world, "ok")),
        give=scripted("give", world, scenario.gives, release_effect(world, "dropped")),
        clear_gray=scripted("clear_gray", world, ["ok"], clear_effect(world)),
        shake=scripted("shake", world, ["ok"], lambda outcome, last_sent: None) if scenario.shake else None,
    )
    sensors = Sensors(visible_area=lambda frame: 0 if world.buried else 800,
                      hands_present=lambda: world.hand, destination=lambda: world.destination)
    settings = Settings(**scenario.settings)
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
    Scenario("shake_hook", [R, P, V, G, S, R, D], "given", shake=True),
]
