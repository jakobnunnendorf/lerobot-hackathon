import threading
from dataclasses import dataclass, field

import numpy as np
from machine import Sensors, Settings, Skills
from grasp_check import GraspCheckSettings
from skills import PRESENT_POSE, PresentAndDrop, Replay, action_of, dance_actions, holding_move, pose_of, step_towards

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
    # Grasp physics for FakeGraspPick, one outcome per close (the last repeats): "ball" (held), "nothing" (empty),
    # "tips" (held, slips out after 6 ticks, during the settle), "late_slip" (slips after 25 ticks, during the lift),
    # "light" (in the band but a low servo load, e.g. a soft or tip contact; let go when the gripper opens).
    # A pick that re-opens (FakeGraspPick reopen) drops a held ball when the command opens above 20.
    # Empty list: off (the scripted picks set `holding` themselves).
    grasps: list[str] = field(default_factory=list)
    slip_in: int | None = None
    light: bool = False
    queries: int = 0
    # A slow arm (exp1, --max-step-deg 1.0): the measured arm joints move at most lag_deg per tick towards the goal,
    # so the policy's raw target runs ahead of the arm. lift_floor: the table, the measured shoulder_lift stays at or
    # below it whatever the goal. None: the arm is where it was sent (the default).
    lag_deg: float | None = None
    lift_floor: float | None = None
    # The arm stops for this many ticks after a close on nothing (a jaw caught on the bowl): the back-off cannot arrive.
    stuck_after_miss_ticks: int = 0
    stuck_for: int = 0


class FakeArm:
    def __init__(self, world: World, abort: threading.Event, rest: np.ndarray = DRY_RUN_REST):
        self.world, self.abort = world, abort
        self.last_sent = action_of(rest)
        self.position = np.asarray(rest, dtype=float)[:5].copy()
        self.last_obs = None
        self.sends_after_abort = 0

    def observe(self) -> dict:
        if self.world.slip_in is not None:
            self.world.slip_in -= 1
            if self.world.slip_in <= 0:
                self.world.holding, self.world.held_ball, self.world.slip_in = False, None, None
        goal = pose_of(self.last_sent)
        if self.world.blocked_at is not None:
            goal[:5] = self.world.blocked_at
        if self.world.stuck_for > 0:
            self.world.stuck_for -= 1
            goal[:5] = self.position
        if self.world.lag_deg is not None:
            goal[:5] = step_towards(self.position, goal[:5], self.world.lag_deg)
        if self.world.lift_floor is not None:
            goal[1] = min(goal[1], self.world.lift_floor)
        self.position = goal[:5].copy()
        gripper = HELD_READING if self.world.holding else (EMPTY_CLOSED_READING if goal[5] < 5 else goal[5])
        self.last_obs = {**action_of(np.concatenate([goal[:5], [gripper]])), "top": np.zeros((4, 4, 3), np.uint8),
                         "wrist": np.zeros((4, 4, 3), np.uint8)}
        return self.last_obs

    def send(self, action: dict) -> None:
        if self.abort.is_set():
            self.sends_after_abort += 1
        gripper = action["gripper.pos"]
        closing = gripper < 10 <= self.last_sent["gripper.pos"]
        if self.world.grasps and closing and not self.world.holding:
            outcome = self.world.grasps.pop(0) if len(self.world.grasps) > 1 else self.world.grasps[0]
            if outcome == "nothing" and self.world.stuck_after_miss_ticks:
                self.world.stuck_for = self.world.stuck_after_miss_ticks
            if outcome != "nothing":
                self.world.holding, self.world.held_ball = True, self.world.held_balls[0]
                self.world.slip_in = {"tips": 6, "late_slip": 25}.get(outcome)
                self.world.light = outcome == "light"
        # Grasp-check worlds: a goal below the ball's reading keeps squeezing it (the lock goal after a hold).
        limit = HELD_READING - 1.0 if self.world.grasps else self.world.hold_goal + 0.5
        if self.world.holding and gripper > limit:
            if self.world.release_ok or self.world.light or (self.world.grasps and gripper > 20):
                self.world.holding, self.world.held_ball, self.world.light = False, None, False
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


# A pick policy as the grasp check sees it: approach with the gripper open, close (crossing 10), lift back. One
# instance per PICK attempt; requery() counts the re-queries the grasp check asks for. reopen: like ACT on the arm
# (2026-11-21 r1, r6), after the close it opens to 30 for 40 ticks (releasing the ball) and closes again.
# drop_at_hole: like the red/blue sorter in exp1, after the close it carries the ball on and commands open (30) from
# tick 30 for good, and never settles at rest (done stays False).
class FakeGraspPick:
    name = "pick"
    CLOSE = (30.0, 20.0, 12.0, 1.0)
    REOPEN = (30, 70)

    def __init__(self, world: World, ticks: int = 60, reopen: bool = False, drop_at_hole: bool = False):
        self.world, self.reopen, self.drop_at_hole = world, reopen, drop_at_hole
        self.ticks = 100 if reopen else ticks

    def start(self, obs, last_sent, now):
        self.count, self.base = 0, pose_of(last_sent)

    def step(self, obs, now):
        self.count += 1
        pose = self.base.copy()
        down = min(self.count, 20) - max(self.count - 25, 0) * 20 / (self.ticks - 25)
        pose[1] += 0.5 * down
        k = self.count - 20
        pose[5] = 40.0 if k < 0 else self.CLOSE[min(k, len(self.CLOSE) - 1)]
        if self.reopen and self.REOPEN[0] <= self.count < self.REOPEN[1]:
            pose[5] = 30.0
        if self.drop_at_hole and self.count >= self.REOPEN[0]:
            pose[5] = 30.0
        return action_of(pose)

    def done(self, obs, now):
        return not self.drop_at_hole and self.count >= self.ticks

    def result(self):
        return "ok"

    def requery(self):
        self.world.queries += 1


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
    # Grasp check: outcomes per close (World.grasps); non-empty = FakeGraspPick as PICK and the grasp check on.
    grasps: list[str] = field(default_factory=list)
    tip_z: float | None = None
    reopen: bool = False
    drop_at_hole: bool = False
    # Slow arm, table and a stuck arm after a miss (World.lag_deg, lift_floor, stuck_after_miss_ticks).
    lag_deg: float | None = None
    lift_floor: float | None = None
    stuck_after_miss_ticks: int = 0
    # Gripper servo load: "physics" (480 holding, 120 for a "light" hold, 30 empty), "raises" (read fails).
    load: str = "physics"
    # A dance round (run_round(dance=True)) with the synthetic dance below, or none (no --dance-episode).
    dance: bool = False
    dance_available: bool = True
    dance_max_step_deg: float | None = 3.0
    holding_at_start: bool = False
    # A bite round (run_round(bite=True)) with a scripted bite skill, or none (no --bite).
    bite: bool = False
    bite_available: bool = True
    bites: list[str] = field(default_factory=lambda: ["bit"])


DANCE_SPEED = 0.5


# Two recorded seconds at 30 fps: pan swings faster than the 3 deg/tick cap at half speed, wrist flex goes up to 110,
# past our envelope (102.1).
def synthetic_dance_recording() -> np.ndarray:
    t = np.arange(60)
    return np.column_stack([30 * np.sin(2 * np.pi * t / 30), 20 + 10 * np.sin(2 * np.pi * t / 60), np.full(60, 60.0),
                            95 + 15 * np.sin(2 * np.pi * t / 20), np.zeros(60), np.zeros(60)])


def synthetic_dance(cap: float | None):
    frames, _ = dance_actions(synthetic_dance_recording(), DANCE_SPEED, DRY_RUN_REST[5])
    return lambda: Replay("dance synthetic:0", frames, play_max_step_deg=cap)


FAKE_HELD_AREA_PX, FAKE_HELD_THRESHOLD_PX = 900, 300


def fake_held_ball_area(world: World, mode: str):
    if mode == "off":
        return None

    def area(frame, wanted):
        if mode == "unavailable":
            raise RuntimeError("no wrist frame")
        return FAKE_HELD_AREA_PX if world.holding and world.held_ball == wanted else 0
    return area


def fake_gripper_load(world: World, mode: str):
    def load():
        if mode == "raises":
            raise RuntimeError("no Present_Load")
        return (120.0 if world.light else 480.0) if world.holding else 30.0
    return load


def build(scenario: Scenario):
    abort = threading.Event()
    world = World(buried=scenario.buried, hand=scenario.hand, destination=scenario.destination,
                  block_after_pick=scenario.block_after_pick, held_balls=list(scenario.held_balls),
                  grasps=list(scenario.grasps), lag_deg=scenario.lag_deg, lift_floor=scenario.lift_floor,
                  stuck_after_miss_ticks=scenario.stuck_after_miss_ticks)
    if scenario.holding_at_start:
        world.holding, world.held_ball = True, "striped"
    arm = FakeArm(world, abort)
    containers = {c: scripted("put_down_" + c, world, ["ok"], release_effect(world, "ok")) for c in ("red", "blue")}
    skills = Skills(
        pick=(lambda: FakeGraspPick(world, reopen=scenario.reopen, drop_at_hole=scenario.drop_at_hole)) if scenario.grasps
        else scripted("pick", world, scenario.picks, pick_effect(world), ticks=60),
        put_down=scripted("put_down", world, scenario.put_downs, release_effect(world, "ok")),
        give=(lambda: Releasing(PresentAndDrop(lambda: world.hand, 60.0), world)) if scenario.give_mode == "present"
        else scripted("give", world, scenario.gives, release_effect(world, "dropped")),
        present=lambda: holding_move("to_present", arm, PRESENT_POSE),
        clear_gray=scripted("clear_gray", world, ["ok"], clear_effect(world)),
        put_down_container=containers if scenario.containers else {},
        pick_colour=(lambda target: scripted("pick_" + target, world, scenario.picks, pick_effect(world), ticks=60)())
        if scenario.colour_pick else None,
        shake=scripted("shake", world, ["ok"], lambda outcome, last_sent: None) if scenario.shake else None,
        dance=synthetic_dance(scenario.dance_max_step_deg) if scenario.dance_available else None,
        bite=scripted("bite", world, scenario.bites, lambda outcome, last_sent: _with_gripper(last_sent, 60.0))
        if scenario.bite_available else None,
    )
    sensors = Sensors(target=lambda: scenario.target, visible_area=lambda frame: 0 if world.buried else 800,
                      palms_seen=lambda: world.hand, destination=lambda: world.destination,
                      held_ball_area=fake_held_ball_area(world, scenario.held_check),
                      ball_colour=(lambda: world.held_ball) if scenario.colour_sensor else None,
                      tip_height=(lambda joints: scenario.tip_z) if scenario.tip_z is not None else None,
                      gripper_load=fake_gripper_load(world, scenario.load))
    held = ({"min_held_target_area_px": FAKE_HELD_THRESHOLD_PX, "min_held_area_px": {"red": FAKE_HELD_THRESHOLD_PX}}
            if scenario.held_check != "off" else {})
    grasp = {"grasp_check": GraspCheckSettings()} if scenario.grasps else {}
    settings = Settings(**{**held, **grasp, **scenario.settings})
    clock = FakeClock(settings.fps, abort, scenario.abort_at_tick)
    return arm, skills, sensors, settings, abort, clock, world


R, P, V, G, U, D, A = "REST", "PICK", "VERIFY_HOLD", "GIVE", "PUT_DOWN", "DONE", "ABORT"
L, C, S, DA, BI = "LOOK", "CLEAR_GRAY", "SHAKE", "DANCE", "BITE"

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
             destination="container", containers=True, colour_pick=True, held_balls=["red"], held_check="on",
             settings={"held_check_colours": ("striped", "red")}),
    Scenario("held_check_red_trusted_striped_held_in_red_round_is_wrong", [R, P, V, U, R, P, V, U, R, D], "put_down",
             target="red", destination="container", containers=True, colour_pick=True, held_balls=["striped", "red"],
             held_check="on", settings={"held_check_colours": ("striped", "red")}),
    Scenario("held_check_red_per_colour_threshold_above_area_is_wrong", [R, P, V, U, R, D], "wrong_ball", target="red",
             destination="container", containers=True, colour_pick=True, held_balls=["red"], held_check="on",
             settings={"held_check_colours": ("striped", "red"), "min_held_area_px": {"red": 2000},
                       "max_pick_attempts": 1}),
    Scenario("held_check_blue_untrusted_by_default_logs_only", [R, P, V, U, R, D], "put_down", target="blue",
             destination="container", containers=True, colour_pick=True, held_balls=["striped"], held_check="on"),
    # A deep grasp: the ball sits above the wrist region, every ball reads 0 px ("deep" matches no colour).
    Scenario("held_check_deep_grasp_unseen_is_kept", [R, P, V, G, R, D], "given", held_balls=["deep"], held_check="on"),
    Scenario("held_check_unmeasured_threshold_logs_only", [R, P, V, G, R, D], "given", held_balls=["red"],
             held_check="on", settings={"min_held_target_area_px": None}),
    # Grasp check inside PICK (GraspChecked): pause at the close, read the gripper, re-query at once on a miss.
    Scenario("grasp_check_ball_continues", [R, P, V, G, R, D], "given", grasps=["ball"]),
    Scenario("grasp_check_nothing_requery_then_ball", [R, P, P, V, G, R, D], "given", grasps=["nothing", "ball"]),
    Scenario("grasp_check_tips_slip_in_settle_is_miss", [R, P, P, V, G, R, D], "given", grasps=["tips", "ball"]),
    Scenario("grasp_check_slip_during_lift_is_miss", [R, P, P, V, G, R, D], "given", grasps=["late_slip", "ball"]),
    Scenario("grasp_check_misses_hit_pick_limit", [R, P, P, P, V, R, D], "gave_up", grasps=["nothing"]),
    Scenario("grasp_check_too_high_miss_before_reading", [R, P, P, P, V, R, D], "gave_up", grasps=["nothing"],
             tip_z=0.30, settings={"grasp_check": GraspCheckSettings(max_tip_height=0.16)}),
    # r1 pattern: held, the policy re-opens above 20 and closes again. Locked: the open is blocked, the ball kept.
    Scenario("grasp_check_lock_blocks_reopen", [R, P, V, G, R, D], "given", grasps=["ball", "nothing"], reopen=True),
    # Unlocked (today's flow): the re-open drops the ball, re-arms, the empty close is a fresh check and a miss.
    Scenario("grasp_check_unlocked_reopen_rearms", [R, P, P, V, G, R, D], "given", grasps=["ball", "nothing", "ball"],
             reopen=True, settings={"grasp_check": GraspCheckSettings(lock_after_hold=False)}),
    # exp1 R3: a slow arm lags the policy's raw target, which lies below the table. The miss must hold the arm where
    # it is, open, back off along where the arm really was, and re-pick at once, not hang until the PICK timeout.
    Scenario("grasp_check_miss_slow_arm_backs_off_from_measured", [R, P, P, V, G, R, D], "given",
             grasps=["nothing", "ball"], lag_deg=0.5, lift_floor=25.0),
    # The arm cannot move for 5 s after the miss: the back-off gives up after --grasp-backoff-timeout (3 s), re-picks.
    Scenario("grasp_check_backoff_timeout_repicks", [R, P, P, V, G, R, D], "given", grasps=["nothing", "ball"],
             stuck_after_miss_ticks=150),
    # exp1 blue holds: the sorter carries the held ball to its hole and commands open; the lock clamps it and the
    # policy never settles. After 1.5 s of open commands PICK ends, the arm goes to rest still locked, VERIFY_HOLD.
    Scenario("grasp_check_policy_releases_while_locked", [R, P, V, G, R, D], "given", grasps=["ball"],
             drop_at_hole=True),
    Scenario("grasp_check_low_load_is_miss", [R, P, P, V, G, R, D], "given", grasps=["light", "ball"]),
    Scenario("grasp_check_load_unreadable_reading_rule", [R, P, V, G, R, D], "given", grasps=["ball"], load="raises"),
    Scenario("grasp_check_off_old_flow_carries_empty", [R, P, V, R, P, V, G, R, D], "given", grasps=["nothing", "ball"],
             settings={"grasp_check": None}),
    Scenario("held_check_detector_unavailable_accepts", [R, P, V, G, R, D], "given", held_balls=["red"],
             held_check="unavailable"),
    Scenario("dance_from_rest", [R, DA, R, D], "danced", dance=True),
    Scenario("dance_refused_while_holding", [R, D], "refused_holding", dance=True, holding_at_start=True),
    Scenario("dance_unavailable_refused", [], "unsupported_dance", dance=True, dance_available=False),
    Scenario("abort_key_during_dance", [R, DA, A], "aborted", dance=True, abort_at_tick=80),
    Scenario("bite_from_rest", [R, BI, R, D], "bit", bite=True),
    Scenario("bite_no_finger_in_window", [R, BI, R, D], "no_finger", bite=True, bites=["no_finger"]),
    Scenario("bite_times_out", [R, BI, R, D], "bite_timeout", bite=True, bites=["hang"], settings={"bite_timeout_s": 2.0}),
    Scenario("bite_refused_while_holding", [R, D], "refused_holding", bite=True, holding_at_start=True),
    Scenario("bite_unavailable_refused", [], "unsupported_bite", bite=True, bite_available=False),
    Scenario("abort_key_during_bite", [R, BI, A], "aborted", bite=True, bites=["hang"], abort_at_tick=40),
]
