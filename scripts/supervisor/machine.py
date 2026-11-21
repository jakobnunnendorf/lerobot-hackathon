import json
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import numpy as np
from skills import MoveTo, Skill


class State(Enum):
    REST = "REST"
    LOOK = "LOOK"
    CLEAR_GRAY = "CLEAR_GRAY"
    PICK = "PICK"
    VERIFY_HOLD = "VERIFY_HOLD"
    GIVE = "GIVE"
    SHAKE = "SHAKE"
    PUT_DOWN = "PUT_DOWN"
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
    # Balls whose wrist-frame area decides. Any other wanted ball is "untrusted": the area is logged, the ball is
    # accepted as without the check. Red joins with one flag (--held-check-colours striped red) once confirmed at
    # the table; blue stays out (see min_held_area_px).
    held_check_colours: tuple[str, ...] = ("striped",)


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


@dataclass
class Sensors:
    visible_area: Callable[[np.ndarray], int] | None = None
    palms_seen: Callable[[], bool] = lambda: False
    destination: Callable[[], str | None] = lambda: None
    # What was asked for this round: "striped", "red", "blue" or None (no preference: the default pick).
    target: Callable[[], str | None] = lambda: None
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
        self.handlers = {State.REST: self._rest, State.LOOK: self._look, State.CLEAR_GRAY: self._clear_gray,
                         State.PICK: self._pick, State.VERIFY_HOLD: self._verify_hold, State.GIVE: self._give,
                         State.SHAKE: self._shake, State.PUT_DOWN: self._put_down}

    def run_round(self, number: int) -> Round:
        try:
            return self._run_round(number)
        finally:
            self.sensors.finish_round()

    def _run_round(self, number: int) -> Round:
        self.round = Round(number, self.clock.now(), after_rest=self._first_state(), target=self.sensors.target())
        state = State.REST
        self.log.write(event="round_start", round=number, target=self.round.target, destination=self.sensors.destination())
        if self.round.target in COLOURS and self.skills.pick_colour is None:
            self.round.outcome = "unsupported_target"
            self.log.write(event="round_end", round=number, outcome=self.round.outcome, pick_attempts=0, clear_attempts=0,
                           duration_s=0.0, states=[], reason=f"no pick skill for {self.round.target!r} yet")
            return self.round
        while state not in (State.DONE, State.ABORT):
            self.round.states.append(state)
            try:
                state, reason, details = self.handlers[state]()
            except Aborted:
                state, reason, details = State.ABORT, "abort requested", {}
            except Exception as error:  # noqa: BLE001 - any skill crash ends in the safe ABORT state
                state, reason, details = State.ABORT, f"error: {error!r}", {}
            self._transition(self.round.states[-1], state, reason, **details)
        self.round.states.append(state)
        if state == State.ABORT:
            self.round.outcome = "aborted"
        self.log.write(event="round_end", round=number, outcome=self.round.outcome,
                       pick_attempts=self.round.pick_attempts, clear_attempts=self.round.clear_attempts,
                       wrong_balls=self.round.wrong_balls, duration_s=round(self.clock.now() - self.round.started, 2),
                       states=[s.value for s in self.round.states])
        return self.round

    def _first_state(self) -> State:
        return State.LOOK if self.s.look or self.s.clear_gray else State.PICK

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
        return low <= reading <= high, reading

    def _held_gripper(self) -> float:
        return float(self.arm.last_sent["gripper.pos"])

    def _move_to_rest(self, holding: bool) -> str:
        # While holding, the gripper goal stays at the last commanded value: never loosen a held ball.
        gripper = self._held_gripper() if holding else self.rest_pose[5]
        target = np.concatenate([self.rest_pose[:5], [gripper]])
        return self.run_skill(MoveTo("to_rest", target), self.s.move_timeout_s)

    def _rest(self):
        holding, reading = self.holding()
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
        result = self.run_skill(skill, self.s.pick_timeout_s)
        return State.VERIFY_HOLD, f"pick attempt {self.round.pick_attempts}: {result}", {"skill": result}

    def _verify_hold(self):
        holding, reading = self.holding()
        details = {"reading": round(reading, 2)}
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
    # Verdicts: off (no hook), unavailable (hook failed: accept), unmeasured (no threshold: log only, accept),
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
            # Every ball's area is logged, so the table log also measures the thresholds of the balls not asked for.
            areas = {ball: int(statistics.median(self.sensors.held_ball_area(frame, ball) for frame in frames))
                     for ball in HELD_BALLS}
        except Exception as error:  # noqa: BLE001 - a failed detector must not end the round
            details.update(held_check="unavailable", held_check_error=repr(error))
            return "unavailable"
        area, threshold = areas[wanted], self.held_threshold(wanted)
        details["held_areas_px"] = areas
        if wanted not in self.s.held_check_colours:
            verdict = "untrusted"
        elif threshold is None:
            verdict = "unmeasured"
        else:
            verdict = "target" if area >= threshold else "wrong"
        details.update(held_check=verdict, held_ball=wanted, held_area_px=area, held_threshold_px=threshold)
        return verdict

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
