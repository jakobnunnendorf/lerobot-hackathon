import json
import statistics
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy as np
from drops import DropCounter, DropRule, UntilDrops
from grasp_check import GraspChecked, GraspCheckSettings, load_magnitude
from skills import MoveTo, Skill, pose_of


class State(Enum):
    REST = "REST"
    STEP = "STEP"
    LOOK = "LOOK"
    CLEAR_GRAY = "CLEAR_GRAY"
    PICK = "PICK"
    VERIFY_HOLD = "VERIFY_HOLD"
    GIVE = "GIVE"
    SHAKE = "SHAKE"
    PUT_DOWN = "PUT_DOWN"
    DANCE = "DANCE"
    BITE = "BITE"
    DONE = "DONE"
    ABORT = "ABORT"


@dataclass
class Settings:
    fps: int = 30
    look: bool = False
    look_frames: int = 5
    min_visible_area_px: int = 150
    clear_gray: bool = False
    max_clear_attempts: int = 3
    max_pick_attempts: int = 3
    pick_timeout_s: float = 40.0
    clear_timeout_s: float = 30.0
    give_timeout_s: float = 45.0
    shake_timeout_s: float = 15.0
    put_down_timeout_s: float = 20.0
    dance_timeout_s: float = 120.0
    bite_timeout_s: float = 45.0
    bite_gripper_open: float = 60.0
    move_timeout_s: float = 15.0
    hold_range: tuple[float, float] = (6.0, 20.0)
    gripper_hold: float | None = None
    verify_s: float = 0.5
    hand_wait_s: float = 3.0
    frame_every_ticks: int = 2
    # Held-ball check (Sensors.held_ball_area): the wanted ball must cover at least this many px of the wrist
    # frame at rest. None = not measured yet: the area is logged at every VERIFY_HOLD but decides nothing.
    # MEASURE AT THE TABLE: wrist frame at rest holding the striped target vs holding a red or a blue ball
    # (the gripper reads the same for both, median 13.4 vs 12.7), and set it between the two. Suggested start
    # from offline data inside the default wrist region (kb/held-ball-wrist-areas.md, PR #30): 120 (12.5% of
    # targets rejected, 4.5% of covering balls accepted). This is the striped ball's threshold.
    min_held_target_area_px: int | None = None
    # Thresholds of the other balls (and striped, overriding the one above). Red 2000 is a candidate from the same
    # offline data (<= 529 with the target held, median ~15,000 with a red ball held): confirm with one red ball
    # held at rest. Blue has none: not usable on the wrist camera (up to 27,466 px inside the region with the
    # target held, the bluish ceiling).
    min_held_area_px: dict[str, int] = field(default_factory=lambda: {"red": 2000})
    held_check_frames: int = 5
    # Every ball below this many px in the wrist region = the held ball is unseen, not wrong: a deep grasp holds it
    # above the region (board 7065772934: one hold read striped 0, red 0, blue 0 and was rejected). The gripper
    # band says held, so it is kept (held_check=unseen). Real readings that day: striped held 274, red 21,295,
    # blue 1632 and 2443.
    held_unseen_floor_px: int = 30
    # Balls whose wrist-frame area decides. Any other wanted ball is "untrusted": the area is logged, the ball is
    # accepted as without the check. Red joins with one flag (--held-check-colours striped red) once confirmed at
    # the table; blue stays out (see min_held_area_px).
    held_check_colours: tuple[str, ...] = ("striped",)
    # Grasp check inside PICK (skills.GraspChecked): pause when the policy closes, read the gripper, re-query on a
    # miss. None = off (PICK runs to its end and only VERIFY_HOLD judges, the old behaviour).
    grasp_check: GraspCheckSettings | None = None
    # With the grasp check off, "holding" also needs |Present_Load| >= this (the grasp check's min_load). The band
    # alone counted an empty gripper at the policy's half-closed goal as a hold (VLA 5K take 2: reading 15, load 32;
    # board 7067314601). Empty reads 28-44, a squeezed ball the torque cap (200, kb/gripper-torque-limit.md). None: the band
    # alone, as before.
    hold_min_load: float | None = 120.0
    # A plan's red/blue step: one prompt of the colour's clear policy runs until the drop count or this budget; after a
    # timeout with an empty gripper it is prompted again, at most max_step_reprompts times (decisions/multistep-step-state.md).
    step_timeout_s: float = 30.0
    max_step_reprompts: int = 3
    # With the grasp check on, an empty close in a step is opened in the bowl and re-prompted at once, at most this
    # many times per step (2026-11-22: each empty close otherwise cost a ~7 s trip to the hole).
    max_step_grasp_misses: int = 8
    drop_rule: DropRule = field(default_factory=DropRule)


@dataclass
class Skills:
    pick: Callable[[], Skill]
    put_down: Callable[[], Skill]
    # One container per ball colour ("red", "blue"); a colour without an entry uses put_down.
    put_down_container: dict[str, Callable[[], Skill]] = field(default_factory=dict)
    # Picks one named target ("red", "blue"). None: only the default pick (striped ball) exists, so a round
    # asking for a colour is refused before the arm moves. The colour-pick work plugs in here.
    pick_colour: Callable[[str], Skill] | None = None
    give: Callable[[], Skill] | None = None
    clear_gray: Callable[[], Skill] | None = None
    shake: Callable[[], Skill] | None = None
    present: Callable[[], Skill] | None = None
    # A replayed dance, the next one each call. None: no --dance-episode, a dance round is refused before the arm moves.
    dance: Callable[[], Skill] | None = None
    # One clear policy per hole colour ("red", "blue") for a plan's steps; a plan naming a colour without one is refused.
    clear_colour: dict[str, Callable[[], Skill]] = field(default_factory=dict)
    # The finger-biting skill (bite.BiteSkill). None: no --bite, a bite round is refused before the arm moves.
    bite: Callable[[], Skill] | None = None


@dataclass
class Sensors:
    visible_area: Callable[[np.ndarray], int] | None = None
    palms_seen: Callable[[], bool] = lambda: False
    destination: Callable[[], str | None] = lambda: None
    # What was asked for this round: "striped", "red", "blue" or None (no preference: the default pick).
    target: Callable[[], str | None] = lambda: None
    # A multi-step task this round: ((colour, count), ...), red/blue steps then optionally ("striped", 1). None: no plan.
    plan: Callable[[], tuple[tuple[str, int], ...] | None] = lambda: None
    # HOOK for colour detection (not built here): the colour of the ball in the gripper, "red", "blue" or None
    # (striped or unknown). It picks the container. Default: the colour that was asked for. Replace it with a
    # detector on the wrist/top frame once scripts/vision/colour_detector.py exists.
    ball_colour: Callable[[], str | None] | None = None
    # HOOK for the held-ball check: the area in px of the wanted ball ("striped", "red", "blue") in the wrist
    # frame, called at rest after the gripper band says holding. None: no detector, the gripper band alone
    # decides (it reads the same for the target and a red/blue ball). Robot mode plugs in #24's ColourVisibility
    # (scripts/vision/supervisor_adapter.py) when it is there. An error or a missing wrist frame counts as
    # "unavailable": the ball is accepted as before and the log says held_check=unavailable.
    held_ball_area: Callable[[np.ndarray, str], int] | None = None
    # Saves one wrist frame per VERIFY_HOLD check (frame, name) and returns where; None: not saved. Best effort.
    save_wrist_frame: Callable[[np.ndarray, str], str] | None = None
    # Grasp check: |Present_Load| of the gripper servo (logged; decides only with min_load) and the tip height (m)
    # of 5 commanded joints (only with max_tip_height). None: not available.
    gripper_load: Callable[[], float | None] | None = None
    tip_height: Callable[[np.ndarray], float] | None = None
    # Called when a round ends, whatever the outcome: clears the one-round command (destination, target).
    finish_round: Callable[[], None] = lambda: None


COLOURS = ("red", "blue")
HELD_BALLS = ("striped", *COLOURS)


class Aborted(Exception):
    pass


class RealClock:
    def __init__(self, fps: int):
        self.period = 1 / fps
        self.next_tick = time.monotonic()

    def now(self) -> float:
        return time.monotonic()

    def tick(self) -> None:
        self.next_tick = max(self.next_tick + self.period, time.monotonic())
        time.sleep(max(self.next_tick - time.monotonic(), 0.0))


class TransitionLog:
    def __init__(self, path: str | None, echo: bool = True):
        self.file = open(path, "a", encoding="utf-8") if path else None
        self.echo = echo

    def write(self, **event) -> None:
        event = {"time": datetime.now(UTC).isoformat(timespec="milliseconds"), **event}
        line = json.dumps(event, default=float)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()
        if self.echo:
            print(line, flush=True)


@dataclass
class Round:
    number: int
    started: float
    pick_attempts: int = 0
    clear_attempts: int = 0
    wrong_balls: int = 0
    putting_back_wrong_ball: bool = False
    outcome: str | None = None
    after_rest: State = State.LOOK
    target: str | None = None
    colour: str | None = None
    dance: Skill | None = None
    plan: tuple[tuple[str, int], ...] | None = None
    step: int = 0
    prompts: int = 0
    step_misses: int = 0
    drops: DropCounter | None = None
    bite: Skill | None = None
    states: list[State] = field(default_factory=list)


# See decisions/supervisor.md for the flow and why each limit exists. The supervisor sends nothing between
# skills, so the servos keep their last goal and a held ball stays held.
class Supervisor:
    def __init__(self, arm, skills: Skills, sensors: Sensors, settings: Settings, rest_pose: np.ndarray,
                 log: TransitionLog, abort: threading.Event, clock=None, frame_writer=None,
                 on_tick: Callable[[], None] = lambda: None):
        self.arm, self.skills, self.sensors, self.s = arm, skills, sensors, settings
        self.rest_pose = np.asarray(rest_pose, dtype=float)
        self.log, self.abort = log, abort
        self.clock = clock or RealClock(settings.fps)
        self.frame_writer, self.ticks, self.on_tick = frame_writer, 0, on_tick
        self.hold_load: float | None = None
        self.handlers = {State.REST: self._rest, State.STEP: self._step, State.LOOK: self._look, State.CLEAR_GRAY: self._clear_gray,
                         State.PICK: self._pick, State.VERIFY_HOLD: self._verify_hold, State.GIVE: self._give,
                         State.SHAKE: self._shake, State.PUT_DOWN: self._put_down, State.DANCE: self._dance,
                         State.BITE: self._bite}

    def run_round(self, number: int, dance: bool = False, bite: bool = False) -> Round:
        try:
            return self._run_round(number, dance, bite)
        finally:
            self.sensors.finish_round()

    def _run_round(self, number: int, dance: bool, bite: bool) -> Round:
        if dance:
            self.round = Round(number, self.clock.now(), after_rest=State.DANCE,
                               dance=self.skills.dance() if self.skills.dance else None)
            self.log.write(event="round_start", round=number, target=None, destination=None,
                           dance=self.round.dance.name if self.round.dance else "unavailable")
        elif bite:
            self.round = Round(number, self.clock.now(), after_rest=State.BITE,
                               bite=self.skills.bite() if self.skills.bite else None)
            self.log.write(event="round_start", round=number, target=None, destination=None, bite=True)
        else:
            plan = self.sensors.plan()
            self.round = Round(number, self.clock.now(), target=self.sensors.target(), plan=plan,
                               drops=DropCounter(self.s.drop_rule) if plan else None)
            self.round.after_rest = self._plan_first_state() if plan else self._first_state()
            self.log.write(event="round_start", round=number, target=self.round.target, destination=self.sensors.destination(),
                           **({"plan": [list(step) for step in plan]} if plan else {}))
        state = State.REST
        refusal = self._refusal(dance, bite)
        if refusal:
            self.round.outcome, reason = refusal
            self.log.write(event="round_end", round=number, outcome=self.round.outcome, pick_attempts=0, clear_attempts=0,
                           duration_s=0.0, states=[], reason=reason)
            return self.round
        while state not in (State.DONE, State.ABORT):
            self.round.states.append(state)
            try:
                state, reason, details = self.handlers[state]()
            except Aborted:
                state, reason, details = State.ABORT, "abort requested", {}
            except Exception as error:  # noqa: BLE001 - any skill crash ends in the safe ABORT state
                state, reason, details = State.ABORT, f"error: {error!r}", self._error_details(error)
            self._transition(self.round.states[-1], state, reason, **details)
        self.round.states.append(state)
        if state == State.ABORT:
            self.round.outcome = "aborted"
        self.log.write(event="round_end", round=number, outcome=self.round.outcome,
                       pick_attempts=self.round.pick_attempts, max_pick_attempts=self.s.max_pick_attempts,
                       clear_attempts=self.round.clear_attempts, wrong_balls=self.round.wrong_balls, duration_s=round(self.clock.now() - self.round.started, 2),
                       states=[s.value for s in self.round.states],
                       **({"drops": dict(self.round.drops.counts)} if self.round.drops else {}))
        return self.round

    def _refusal(self, dance: bool, bite: bool) -> tuple[str, str] | None:
        missing = [c for c, _ in self.round.plan or () if c != "striped" and c not in self.skills.clear_colour]
        if not (dance or bite) and missing:
            return "unsupported_plan", f"no clear policy for {missing[0]!r} (--clear-{missing[0]}-policy): plan refused"
        if not (dance or bite) and self.round.target in COLOURS and self.skills.pick_colour is None:
            return "unsupported_target", f"no pick skill for {self.round.target!r} yet"
        if dance and self.round.dance is None:
            return "unsupported_dance", "no dance episode in this robot program (--dance-episode)"
        if bite and self.round.bite is None:
            return "unsupported_bite", "no bite in this robot program: start it with --bite"
        return None

    def _empty_gripper_round(self) -> str | None:
        return "dance" if self.round.dance else "bite" if self.round.bite else None

    # The state's own details are lost when it raises (VLA 5K take 2: the held-check verdict), so log what is known.
    def _error_details(self, error: Exception) -> dict:
        frame = traceback.extract_tb(error.__traceback__)[-1]
        return {"error": repr(error), "error_at": f"{Path(frame.filename).name}:{frame.lineno} {frame.name}",
                "load": self._gripper_load(), "hold_load": self.hold_load, "pick_attempts": self.round.pick_attempts}

    def _first_state(self) -> State:
        return State.LOOK if self.s.look or self.s.clear_gray else State.PICK

    def _plan_first_state(self) -> State:
        if self.round.plan[0][0] == "striped":
            self.round.target = "striped"
            return self._first_state()
        return State.STEP

    # Counts add up per colour over the whole plan, so "2 red, 1 blue, 1 red" wants 3 red by the third step.
    def _step_wanted(self) -> tuple[str, int]:
        colour = self.round.plan[self.round.step][0]
        return colour, sum(count for c, count in self.round.plan[: self.round.step + 1] if c == colour)

    # One prompt of the step's clear policy. No REST between prompts or steps: the clear demonstrations start where the
    # previous drop ended, at a hole. A wrong-colour drop counts for its own hole and is logged.
    def _step(self):
        colour, wanted = self._step_wanted()
        details = {"colour": colour, "wanted": wanted, "dropped": dict(self.round.drops.counts)}
        if self.round.drops.counts[colour] >= wanted:
            return self._next_step(f"{colour}: {wanted} already dropped", details)
        self.round.prompts += 1
        skill = UntilDrops(self._grasp_checked_clear(colour), self.round.drops, colour, wanted,
                           lambda hole, pan, obs: self._log_drop(colour, hole, pan, obs))
        result = self.run_skill(skill, self.s.step_timeout_s)
        details.update(skill=result, prompt=self.round.prompts, dropped=dict(self.round.drops.counts))
        if result == "count_reached":
            return self._next_step(f"{colour}: {wanted} dropped", details)
        if result == "grasp_miss":
            self.round.step_misses += 1
            details["grasp_misses"] = self.round.step_misses
            if self.round.step_misses <= self.s.max_step_grasp_misses:
                return State.STEP, f"{colour} step: empty close, opened in the bowl: re-prompt", details
            self.round.outcome, self.round.after_rest = "step_gave_up", State.DONE
            return State.REST, f"{colour} step: {self.round.step_misses} empty closes, giving up", details
        holding, reading = self.holding()
        details["reading"] = round(reading, 2)
        timeouts = self.round.prompts - self.round.step_misses
        if timeouts <= self.s.max_step_reprompts:
            # Holding at the timeout is a ball on its way to the hole (2026-11-22: red ran out mid-trip): a new
            # prompt lets the policy finish the drop instead of aborting.
            state = "still holding" if holding else "gripper empty"
            return State.STEP, f"{colour} step {result}, {state}: re-prompt {timeouts}", details
        if holding:
            return State.ABORT, f"{colour} step: still holding after {timeouts} prompts", details
        self.round.outcome, self.round.after_rest = "step_gave_up", State.DONE
        return State.REST, f"{colour} step: {timeouts} prompts without reaching {wanted}, giving up", details

    # Unlocked: the clear policy has to open at its hole, which the pick's lock after a hold would block.
    def _grasp_checked_clear(self, colour: str):
        skill = self.skills.clear_colour[colour]()
        if self.s.grasp_check is None:
            return skill
        checked = GraspChecked(skill, replace(self.s.grasp_check, lock_after_hold=False), self.s.fps,
                               self.sensors.gripper_load, self.sensors.tip_height)
        prompt = self.round.prompts
        checked.on_check = lambda record: self.log.write(event="grasp_check", round=self.round.number,
                                                         step_colour=colour, prompt=prompt, **record)
        return checked

    def _next_step(self, reason: str, details: dict):
        self.round.step, self.round.prompts, self.round.step_misses = self.round.step + 1, 0, 0
        if self.round.step == len(self.round.plan):
            self.round.outcome, self.round.after_rest = "cleared", State.DONE
            return State.REST, f"{reason}; plan done", details
        if self.round.plan[self.round.step][0] == "striped":
            self.round.target, self.round.after_rest = "striped", self._first_state()
            return State.REST, f"{reason}; now the striped ball", details
        return State.STEP, reason, details

    def _log_drop(self, step_colour: str, hole: str | None, pan: float, obs: dict) -> None:
        self.log.write(event="drop" if hole else "release", round=self.round.number, step_colour=step_colour, hole=hole,
                       wrong_colour=hole is not None and hole != step_colour, pan=round(pan, 1),
                       gripper=round(float(obs["gripper.pos"]), 2), closed_min=round(self.round.drops.last_closed_min, 2),
                       dropped=dict(self.round.drops.counts))

    def _transition(self, source: State, target: State, reason: str, **details) -> None:
        self.log.write(event="transition", round=self.round.number, source=source.value, target=target.value,
                       reason=reason, gripper=self.last_gripper_reading(), **details)

    def last_gripper_reading(self) -> float | None:
        obs = getattr(self.arm, "last_obs", None)
        return None if obs is None else round(float(obs["gripper.pos"]), 2)

    def _observe(self) -> dict:
        if self.abort.is_set():
            raise Aborted
        self.on_tick()
        obs = self.arm.observe()
        if self.frame_writer is not None and self.ticks % self.s.frame_every_ticks == 0:
            self.frame_writer.write(obs["top"], time.perf_counter_ns())
        self.ticks += 1
        return obs

    def _wait(self, seconds: float, until: Callable[[dict], bool] = lambda obs: False) -> list[dict]:
        seen, start = [], self.clock.now()
        while self.clock.now() - start < seconds:
            seen.append(self._observe())
            if until(seen[-1]):
                break
            self.clock.tick()
        return seen

    def idle_until(self, event: threading.Event) -> None:
        while not event.is_set():
            self._observe()
            self.clock.tick()

    def run_skill(self, skill: Skill, timeout_s: float) -> str:
        obs = self._observe()
        start = self.clock.now()
        skill.start(obs, self.arm.last_sent, start)
        while True:
            action = skill.step(obs, self.clock.now())
            if action is not None:
                self.arm.send(action)
            if skill.done(obs, self.clock.now()):
                return skill.result()
            if self.clock.now() - start >= timeout_s:
                return "timeout"
            self.clock.tick()
            obs = self._observe()

    def holding(self) -> tuple[bool, float]:
        readings = [float(obs["gripper.pos"]) for obs in self._wait(self.s.verify_s)]
        reading = statistics.median(readings)
        low, high = self.s.hold_range
        self.hold_load = None
        return low <= reading <= high and self._squeezing(), reading

    # Every caller shares this rule: with it only in VERIFY_HOLD, REST would see a hold again and loop back to it.
    def _squeezing(self) -> bool:
        if self.s.grasp_check is not None or self.s.hold_min_load is None or self.sensors.gripper_load is None:
            return True
        try:
            self.hold_load = statistics.median(load_magnitude(self.sensors.gripper_load()) for _ in range(3))
        except Exception:  # noqa: BLE001 - an unreadable load leaves the band alone, as in the grasp check
            return True
        return self.hold_load >= self.s.hold_min_load

    def _gripper_load(self) -> float | None:
        try:
            return load_magnitude(self.sensors.gripper_load()) if self.sensors.gripper_load else None
        except Exception:  # noqa: BLE001 - best effort, only logged
            return None

    def _held_gripper(self) -> float:
        return float(self.arm.last_sent["gripper.pos"])

    def _move_to_rest(self, holding: bool) -> str:
        # While holding, the gripper goal stays at the last commanded value: never loosen a held ball.
        gripper = self._held_gripper() if holding else self.rest_pose[5]
        target = np.concatenate([self.rest_pose[:5], [gripper]])
        return self.run_skill(MoveTo("to_rest", target), self.s.move_timeout_s)

    def _rest(self):
        holding, reading = self.holding()
        # A dance swings the arm at speed and a bite snaps the jaws, so both start only from an empty gripper, refused
        # before anything moves.
        special = self._empty_gripper_round()
        if holding and special and self.round.outcome is None:
            self.round.outcome = "refused_holding"
            return State.DONE, f"the gripper holds something: empty it before a {special}", {"reading": reading}
        # After a bite a hold can only be a finger (a ball refuses the round): open, never ABORT, which would keep it.
        if holding and self.round.bite:
            self.log.write(event="bite_jaws_opened", round=self.round.number, reading=reading, move=self._open_bite_jaws())
            holding = False
        if holding and special:
            return State.ABORT, f"gripper reads holding after the {special}", {"reading": reading}
        if holding:
            result = self._move_to_rest(holding=True)
            return State.VERIFY_HOLD, "gripper reads holding at REST", {"move": result, "reading": reading}
        result = self._move_to_rest(holding=False)
        if result == "timeout":
            return State.ABORT, "could not reach the rest pose", {}
        target = self.round.after_rest
        if target == State.DONE:
            return State.DONE, f"round over: {self.round.outcome}", {}
        return target, "at rest", {}

    def _look(self):
        frames = [obs["top"] for obs in self._wait(self.s.look_frames / self.s.fps)][: self.s.look_frames]
        area = int(statistics.median(self.sensors.visible_area(frame) for frame in frames)) if self.sensors.visible_area else None
        visible = area is not None and area >= self.s.min_visible_area_px
        details = {"visible_area_px": area, "visible": visible}
        can_clear = self.s.clear_gray and self.skills.clear_gray is not None
        if not visible and can_clear and self.round.clear_attempts < self.s.max_clear_attempts:
            return State.CLEAR_GRAY, "target not visible", details
        if not visible and can_clear:
            return State.PICK, "clear limit reached, the pick policy digs", details
        return State.PICK, "target visible" if visible else "target not visible, the pick policy digs", details

    def _clear_gray(self):
        self.round.clear_attempts += 1
        result = self.run_skill(self.skills.clear_gray(), self.s.clear_timeout_s)
        holding, reading = self.holding()
        if holding:
            return State.ABORT, "still holding something after CLEAR_GRAY", {"skill": result, "reading": reading}
        self.round.after_rest = State.LOOK
        return State.REST, f"clear attempt {self.round.clear_attempts}", {"skill": result}

    def _pick(self):
        self.round.pick_attempts += 1
        skill = self.skills.pick_colour(self.round.target) if self.round.target in COLOURS else self.skills.pick()
        if self.s.grasp_check is not None:
            skill = GraspChecked(skill, self.s.grasp_check, self.s.fps, self.sensors.gripper_load, self.sensors.tip_height)
            attempt = self.round.pick_attempts
            skill.on_check = lambda record: self.log.write(event="grasp_check", round=self.round.number,
                                                           pick_attempt=attempt, **record)
        result = self.run_skill(skill, self.s.pick_timeout_s)
        details = {"skill": result}
        if getattr(skill, "backoff_timed_out", False):
            details["backoff_timeout"] = True
        if result == "released_while_locked":
            # The policy wants to drop the held ball (a sorter at its hole) and never settles: take the arm to rest
            # with the gripper still at the lock goal, then VERIFY_HOLD as after any pick.
            details["move"] = self._move_to_rest(holding=True)
            if details["move"] == "timeout":
                return State.ABORT, "arm did not reach the rest pose after the policy released while locked", details
            return State.VERIFY_HOLD, f"pick attempt {self.round.pick_attempts}: policy released while locked", details
        if result == "grasp_miss" and self.round.pick_attempts < self.s.max_pick_attempts:
            # Jaws closed on nothing (or the ball slipped): opened and backed off already, re-query from here.
            return State.PICK, f"pick attempt {self.round.pick_attempts}: grasp check miss, re-query", details
        return State.VERIFY_HOLD, f"pick attempt {self.round.pick_attempts}: {result}", details

    def _verify_hold(self):
        holding, reading = self.holding()
        details = {"reading": round(reading, 2)}
        if self.hold_load is not None:
            details["load"], details["min_load"] = round(self.hold_load), self.s.hold_min_load
        if holding:
            if self.s.gripper_hold is not None:
                self.arm.send({**self.arm.last_sent, "gripper.pos": self.s.gripper_hold})
            verdict = self._check_held_ball(details)
            if verdict == "timeout":
                return State.ABORT, "arm did not reach the rest pose for the held-ball check", details
            if verdict == "wrong":
                return self._wrong_ball(details)
            return self._destination(details)
        if self.round.pick_attempts < self.s.max_pick_attempts:
            self.round.after_rest = self._first_state()
            return State.REST, "not holding, retry", details
        self.round.outcome, self.round.after_rest = "gave_up", State.DONE
        return State.REST, f"not holding after {self.round.pick_attempts} picks, giving up", details

    def _wanted_ball(self) -> str:
        return self.round.target if self.round.target in COLOURS else "striped"

    # The gripper reads the same for the striped target and a red/blue ball, so the wrist camera decides.
    # Verdicts: off (no hook), unavailable (hook failed: accept), unseen (every ball below held_unseen_floor_px:
    # accept, the gripper band says held), unmeasured (no threshold: log only, accept),
    # untrusted (this ball's wrist detection is not trusted: log only, accept), target, wrong, timeout (could not
    # settle at rest).
    def _check_held_ball(self, details: dict) -> str:
        if self.sensors.held_ball_area is None:
            details["held_check"] = "off"
            return "off"
        if self._move_to_rest(holding=True) == "timeout":
            details["held_check"] = "timeout"
            return "timeout"
        wanted = self._wanted_ball()
        try:
            frames = [obs["wrist"] for obs in self._wait(self.s.held_check_frames / self.s.fps)][: self.s.held_check_frames]
            self._save_wrist_frame(frames, details)
            # Every ball's area is logged, so the table log also measures the thresholds of the balls not asked for.
            areas = {ball: int(statistics.median(self.sensors.held_ball_area(frame, ball) for frame in frames))
                     for ball in HELD_BALLS}
        except Exception as error:  # noqa: BLE001 - a failed detector must not end the round
            details.update(held_check="unavailable", held_check_error=repr(error))
            return "unavailable"
        area, threshold = areas[wanted], self.held_threshold(wanted)
        details["held_areas_px"] = areas
        if all(value < self.s.held_unseen_floor_px for value in areas.values()):
            # No ball seen at all: the camera cannot tell, so the gripper band decides (held), as without the check.
            verdict = "unseen"
            details["held_unseen_floor_px"] = self.s.held_unseen_floor_px
        elif wanted not in self.s.held_check_colours:
            verdict = "untrusted"
        elif threshold is None:
            verdict = "unmeasured"
        else:
            verdict = "target" if area >= threshold else "wrong"
        details.update(held_check=verdict, held_ball=wanted, held_area_px=area, held_threshold_px=threshold)
        return verdict

    def _save_wrist_frame(self, frames: list, details: dict) -> None:
        if self.sensors.save_wrist_frame is None or not frames:
            return
        try:
            name = f"verify_hold_r{self.round.number}_p{self.round.pick_attempts}"
            details["wrist_frame"] = self.sensors.save_wrist_frame(frames[len(frames) // 2], name)
        except Exception as error:  # noqa: BLE001 - a frame that cannot be saved must not end the round
            details["wrist_frame_error"] = repr(error)

    def held_threshold(self, ball: str) -> int | None:
        if ball in self.s.min_held_area_px:
            return self.s.min_held_area_px[ball]
        return self.s.min_held_target_area_px if ball == "striped" else None

    def _wrong_ball(self, details: dict):
        # Never handed over. Sorted by the colour sensor if there is one, else the generic put-down pose (the
        # target's colour would be a guess). PUT_DOWN then retries PICK within the pick budget.
        self.round.wrong_balls += 1
        self.round.putting_back_wrong_ball = True
        colour = self.sensors.ball_colour() if self.sensors.ball_colour else None
        self.round.colour = colour if colour in COLOURS else None
        details["colour"] = self.round.colour
        return State.PUT_DOWN, f"holding, but not the {self._wanted_ball()} ball; put it down", details

    # The hand decision is made in the present pose, where the tracker sees the hands that GIVE will serve.
    def _ball_colour(self) -> str | None:
        colour = self.sensors.ball_colour() if self.sensors.ball_colour else self.round.target
        return colour if colour in COLOURS else None

    def _destination(self, details: dict):
        wanted = self.sensors.destination()
        self.round.colour = self._ball_colour()
        details["destination"], details["colour"] = wanted, self.round.colour
        if wanted == "container" or self.skills.give is None:
            return State.PUT_DOWN, "holding; put down" if wanted else "holding; no GIVE skill", details
        if self.skills.present is not None:
            details["present"] = self.run_skill(self.skills.present(), self.s.move_timeout_s)
            if details["present"] == "timeout":
                return State.ABORT, "arm did not reach the present pose while holding", details
        if wanted == "hand":
            return State.GIVE, "holding; voice or key says hand", details
        seen = self._wait(self.s.hand_wait_s, until=lambda obs: self.sensors.palms_seen())
        if self.sensors.palms_seen():
            return State.GIVE, f"holding, two palms seen after {len(seen) / self.s.fps:.1f} s", details
        return State.PUT_DOWN, f"holding, no two palms in {self.s.hand_wait_s} s", details

    def _give(self):
        result = self.run_skill(self.skills.give(), self.s.give_timeout_s)
        if result == "aborted":
            raise Aborted
        holding, reading = self.holding()
        details = {"skill": result, "reading": round(reading, 2)}
        if holding:
            self._move_to_rest(holding=True)
            return State.PUT_DOWN, "GIVE ended still holding", details
        self.round.outcome, self.round.after_rest = "given", State.DONE
        if self.skills.shake is not None:
            return State.SHAKE, "ball given", details
        return State.REST, "ball given", details

    def _dance(self):
        result = self.run_skill(self.round.dance, self.s.dance_timeout_s)
        self.round.outcome = "danced" if result == "replayed" else f"dance_{result}"
        self.round.after_rest = State.DONE
        return State.REST, f"dance: {result}", {"skill": result, "dance": self.round.dance.name}

    def _bite(self):
        result = self.run_skill(self.round.bite, self.s.bite_timeout_s)
        self.round.outcome = result if result in ("bit", "no_finger") else f"bite_{result}"
        self.round.after_rest = State.DONE
        details = {"skill": result}
        if result not in ("bit", "no_finger"):
            details["jaws_opened"] = self._open_bite_jaws()
        return State.REST, f"bite: {result}", details

    # Cut off mid-bite, the jaws may be shut on a finger: open them where the arm is, in one step, before it moves.
    def _open_bite_jaws(self) -> str:
        opened = np.concatenate([pose_of(self.arm.last_sent)[:5], [self.s.bite_gripper_open]])
        return self.run_skill(MoveTo("bite_open", opened, gripper_step=100.0), self.s.move_timeout_s)

    def _shake(self):
        result = self.run_skill(self.skills.shake(), self.s.shake_timeout_s)
        return State.REST, f"shake: {result}", {"skill": result}

    def _put_down(self):
        # The container of the ball's colour; the generic put-down pose for the striped ball or an unknown colour.
        container = self.skills.put_down_container.get(self.round.colour)
        result = self.run_skill((container or self.skills.put_down)(), self.s.put_down_timeout_s)
        holding, reading = self.holding()
        details = {"skill": result, "reading": round(reading, 2), "container": self.round.colour if container else "default"}
        if holding:
            return State.ABORT, "still holding after PUT_DOWN", details
        if self.round.putting_back_wrong_ball:
            self.round.putting_back_wrong_ball = False
            if self.round.pick_attempts < self.s.max_pick_attempts:
                self.round.after_rest = self._first_state()
                return State.REST, "wrong ball put down, retry", details
            self.round.outcome, self.round.after_rest = "wrong_ball", State.DONE
            return State.REST, f"wrong ball put down after {self.round.pick_attempts} picks, giving up", details
        self.round.outcome, self.round.after_rest = "put_down", State.DONE
        return State.REST, "ball put down", details
