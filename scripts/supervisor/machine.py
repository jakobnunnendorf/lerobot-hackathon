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
    hold_range: tuple[float, float] = (10.0, 17.0)
    verify_s: float = 0.5
    hand_wait_s: float = 3.0
    frame_every_ticks: int = 2


@dataclass
class Skills:
    pick: Callable[[], Skill]
    put_down: Callable[[], Skill]
    give: Callable[[], Skill] | None = None
    clear_gray: Callable[[], Skill] | None = None
    shake: Callable[[], Skill] | None = None


@dataclass
class Sensors:
    visible_area: Callable[[np.ndarray], int] | None = None
    hands_present: Callable[[], bool] = lambda: False
    destination: Callable[[], str | None] = lambda: None


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
    outcome: str | None = None
    after_rest: State = State.LOOK
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
        self.round = Round(number, self.clock.now(), after_rest=self._first_state())
        state = State.REST
        self.log.write(event="round_start", round=number)
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
                       duration_s=round(self.clock.now() - self.round.started, 2),
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
        result = self.run_skill(self.skills.pick(), self.s.pick_timeout_s)
        return State.VERIFY_HOLD, f"pick attempt {self.round.pick_attempts}: {result}", {"skill": result}

    def _verify_hold(self):
        holding, reading = self.holding()
        details = {"reading": round(reading, 2)}
        if holding:
            return self._destination(details)
        if self.round.pick_attempts < self.s.max_pick_attempts:
            self.round.after_rest = self._first_state()
            return State.REST, "not holding, retry", details
        self.round.outcome, self.round.after_rest = "gave_up", State.DONE
        return State.REST, f"not holding after {self.round.pick_attempts} picks, giving up", details

    def _destination(self, details: dict):
        wanted = self.sensors.destination()
        details["destination"] = wanted
        if wanted == "container" or self.skills.give is None:
            return State.PUT_DOWN, "holding; put down" if wanted else "holding; no GIVE skill", details
        seen = self._wait(self.s.hand_wait_s, until=lambda obs: self.sensors.hands_present())
        if self.sensors.hands_present():
            return State.GIVE, f"holding, hand seen after {len(seen) / self.s.fps:.1f} s", details
        return State.PUT_DOWN, f"holding, no hand in {self.s.hand_wait_s} s", details

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
        result = self.run_skill(self.skills.put_down(), self.s.put_down_timeout_s)
        holding, reading = self.holding()
        details = {"skill": result, "reading": round(reading, 2)}
        if holding:
            return State.ABORT, "still holding after PUT_DOWN", details
        self.round.outcome, self.round.after_rest = "put_down", State.DONE
        return State.REST, "ball put down", details
