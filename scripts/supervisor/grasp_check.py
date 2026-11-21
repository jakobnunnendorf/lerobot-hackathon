import statistics
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from skills import MoveTo, Sequence, action_of, pose_of
from ticks import TICKS


# Re-query a pick skill: drop the rest of its queued chunk so the next step predicts from the current observation.
# PolicySkill: LoadedPolicy.reset() (ACT 0.7.0: policy.reset() empties _action_queue). A fake: its requery().
def requery(skill) -> None:
    if hasattr(skill, "requery"):
        skill.requery()
    elif hasattr(skill, "loaded"):
        skill.loaded.reset()


# |Present_Load| (0-1000, 0.1% of max torque). The STS3215 register is sign-magnitude with the direction in bit 10;
# 2026-11-21 it read 28-500, all positive, so the bus decodes it or the close is always one way. abs() covers a
# signed decode, and bit 10 is dropped in case a raw value (1024 + magnitude) ever comes through.
def load_magnitude(value):
    if value is None:
        return None
    value = abs(float(value))
    return value - 1024 if value >= 1024 else value


# Grasp check (Jakob, 2026-11-21): ACT often closes on the jaw tips, misses the ball and, because the chunk keeps
# running, carries on to the target as if holding. GraspChecked pauses the pick the moment the policy closes the
# gripper, holds the arm still, squeezes to close_goal and reads the gripper after settle_s:
#   held = hold_low <= reading <= hold_high and reading - goal >= min_error (a ball stalls the jaw short of its goal)
#   and |Present_Load| >= min_load (both rules must agree; a load that cannot be read leaves the reading rule alone,
#   logged as "load_rule": "unavailable")
# Measured on 606 teleoperated closes in molnrb's four pick datasets (rb_mixed, ball, blue_ball, buried_ball),
# 0.5 s after the leader closed below 10: closes carried to the end of the episode (ball held, 218, excluding
# so101_pick_ball's empty end closes) read p1 10.9, median 12.9; 217 of 218 pass the rule. All 78 closes reading
# below 4 (empty: 0.4-3.9) fail it, and so do 13 closes resting at 6-8 with the jaw at its goal (error < 3), which
# the band alone would call holding. After the settle the reading moved by <0.02/frame (p95).
# A miss opens the gripper, backs the arm off along the measured path of the last backoff_s before the close and
# ends the skill as "grasp_miss": the supervisor re-runs PICK at once (a fresh policy query), within the pick limit.
# Hold and back-off use the measured arm, not the policy's raw action (exp1 R3, 2026-11-21): with --max-step-deg 1.0
# the raw target ran far ahead of the arm, below the table, the release never arrived within 4 deg and the arm stood
# still until the 40 s PICK timeout. The release also gives up after backoff_timeout_s ("release_timeout", then
# grasp_miss with backoff_timeout: true).
# A hold re-queries the policy and keeps watching: a ball that slips out of the tips later (median of watch_frames
# readings below hold_low while the policy still commands closed) is a miss too, opened in place.
# Lock after a hold (calibration run 2026-11-21, molnrb/arm-logs 4e20743, r1 and r6): ACT later re-opened on the held
# ball, as its demos end, and closed again on nothing, which the check judged a fresh miss. With lock_after_hold the
# gripper sent is min(policy, lock goal) for the rest of the pick (lock goal: lock_goal_for), a policy command
# >= trigger is logged as "release_blocked" (at most once a second, with the gripper load and the count of blocked
# commands since the last line) and the slip watch keeps running. Unlocked, a re-open is logged as
# "release" and the trigger re-arms only after the command is >= armed_above again.
# Policy release while locked (exp1, the red/blue sorter carried held balls to their hole and commanded open): the
# policy never settles at rest then, so once it has commanded open above the lock for open_intent_s in all, the pick
# ends as "released_while_locked" (logged "released_locked") and the supervisor takes the arm to rest, still locked.
@dataclass
class GraspCheckSettings:
    trigger: float = 10.0          # a commanded gripper falling below this = "the policy closes"
    armed_above: float = 20.0      # only after the policy opened above this in this pick (rest may read closed)
    close_goal: float = 1.0        # gripper goal during the check (teleop holds: leader median 1.1)
    settle_s: float = 0.4
    window_frames: int = 5         # median of the last frames of the settle
    min_error: float = 3.0
    hold_range: tuple[float, float] = (6.0, 20.0)
    # |Present_Load| of the gripper servo (0.1% of max torque) for a hold; None = logged only. 2026-11-21 on the arm:
    # ball 440-500 (500 = the gripper's Max_Torque_Limit), empty 28-44.
    min_load: float | None = 200.0
    backoff_s: float = 0.5
    open_goal: float | None = None  # None: the widest gripper command of this pick, at least trigger + 10
    watch_frames: int = 5
    requery_on_hold: bool = True
    lock_after_hold: bool = True
    lock_margin: float = 4.0       # lock goal = held reading - this (see lock_goal_for)
    release_log_interval_s: float = 1.0
    backoff_timeout_s: float = 3.0       # a miss's open + back-off that has not arrived by then ends the pick anyway
    open_intent_s: float | None = 1.5    # locked: open commands for this long in all end the pick; None = never
    # "Too high": a close commanded with the tip (FK of the commanded joints, #13's kinematics) above this is a miss
    # before the gripper is read. None = off: on the datasets the model's tip z at a holding close is 0.019-0.156 m
    # (p95 0.11) and empty closes overlap it (median 0.07), so it can only catch closes far above the bowl.
    max_tip_height: float | None = None


class GraspChecked:
    def __init__(self, inner, settings: GraspCheckSettings, fps: int = 30,
                 read_load: Callable[[], float | None] | None = None,
                 tip_height: Callable[[np.ndarray], float] | None = None):
        self.inner, self.s, self.fps, self.read_load = inner, settings, fps, read_load
        self.tip_height = tip_height
        self.name = inner.name
        self.on_check: Callable[[dict], None] = lambda record: None
        self.checks: list[dict] = []

    def start(self, obs, last_sent, now):
        self.inner.start(obs, last_sent, now)
        self.phase, self.armed, self.watching, self.missed = "run", False, False, False
        self.widest = float(last_sent["gripper.pos"])
        self.last_gripper = self.widest
        self.history = deque(maxlen=max(int(self.s.backoff_s * self.fps), 1))
        self.history.append(pose_of(obs))
        self.watch = deque(maxlen=self.s.watch_frames)
        self.load_failed = False
        self.last_release_log = None
        self.blocked_count, self.open_s, self.released_locked, self.backoff_timed_out = 0, 0.0, False, False

    def _open_goal(self) -> float:
        return self.s.open_goal if self.s.open_goal is not None else max(self.widest, self.s.trigger + 10.0)

    def _load(self):
        if self.read_load is None or self.load_failed:
            return None
        try:
            return load_magnitude(self.read_load())
        except Exception as error:  # noqa: BLE001 - a failed load read must not stop the pick
            self.load_failed = True
            self.load_error = repr(error)
            return None

    def step(self, obs, now):
        TICKS.note(grasp="watch" if self.phase == "run" and self.watching else self.phase)
        if self.phase == "settle":
            return self._settle(obs, now)
        if self.phase == "release":
            if now - self.release_start >= self.s.backoff_timeout_s:
                self.backoff_timed_out = True
                self._record({"phase": "release_timeout", "backoff_timeout": True,
                              "release_s": round(now - self.release_start, 2), "part": self.release.parts[self.release.index].name})
                return None
            action = self.release.step(obs, now)
            self.release_sent = action or self.release_sent
            return action
        action = self.inner.step(obs, now)
        if action is None:
            return None
        gripper = float(action["gripper.pos"])
        if self.watching:
            self.watch.append(float(obs["gripper.pos"]))
            if self.s.lock_after_hold:
                if gripper >= self.s.trigger:
                    self.blocked_count += 1
                    self.open_s += 1.0 / self.fps
                    if self.last_release_log is None or now - self.last_release_log >= self.s.release_log_interval_s:
                        self.last_release_log = now
                        self._record({"phase": "release_blocked", "command": round(gripper, 2),
                                      "sent": round(self.lock_goal, 2), "reading": round(float(obs["gripper.pos"]), 2),
                                      "load": self._load(), "count": self.blocked_count})
                        self.blocked_count = 0
                    if self.s.open_intent_s is not None and self.open_s >= self.s.open_intent_s - 1e-9:
                        self.released_locked = True
                        self._record({"phase": "released_locked", "open_s": round(self.open_s, 2),
                                      "sent": round(self.lock_goal, 2), "reading": round(float(obs["gripper.pos"]), 2)})
                gripper = min(gripper, self.lock_goal)
                action = {**action, "gripper.pos": gripper}
            elif gripper >= self.s.trigger:
                self.watching, self.armed = False, False  # re-armed once the command is >= armed_above again
                self._record({"phase": "release", "command": round(gripper, 2),
                              "reading": round(float(obs["gripper.pos"]), 2)})
            if self.watching and len(self.watch) == self.watch.maxlen and statistics.median(self.watch) < self.s.hold_range[0]:
                return self._miss(obs, now, {"phase": "watch", "reading": round(statistics.median(self.watch), 2),
                                             "readings": [round(r, 2) for r in self.watch]}, back_off=False)
        self.armed |= gripper >= self.s.armed_above
        self.widest = max(self.widest, gripper)
        closing = self.armed and not self.watching and gripper < self.s.trigger <= self.last_gripper
        self.last_gripper = gripper
        if closing:
            self.goal = min(gripper, self.s.close_goal)
            # Hold the arm where it is measured: the raw action can be far ahead of a speed-capped arm.
            self.hold_pose = np.concatenate([pose_of(obs)[:5], [self.goal]])
            self.back_pose = self.history[0].copy()
            self.phase, self.settle_start, self.readings, self.loads = "settle", now, [], []
            self.command_at_trigger = gripper
            self.tip_z = float(self.tip_height(pose_of(action)[:5])) if self.tip_height else None
            if self.s.max_tip_height is not None and self.tip_z is not None and self.tip_z > self.s.max_tip_height:
                return self._miss(obs, now, {"phase": "trigger", "reason": "too_high", "tip_z": round(self.tip_z, 3),
                                             "max_tip_height": self.s.max_tip_height,
                                             "command_at_trigger": round(gripper, 2)}, back_off=True)
            return action_of(self.hold_pose)
        self.history.append(pose_of(obs))
        return action

    # Lock goal after a hold: lock_margin below the held reading (less stall current than squeezing at close_goal:
    # at goal 1 the load sat at 500, the gripper's torque limit), at least close_goal, and at most hold_low - 1 so an
    # empty jaw at the lock goal still reads below hold_low and the slip watch is unchanged.
    def lock_goal_for(self, reading: float) -> float:
        return min(max(self.s.close_goal, reading - self.s.lock_margin), self.s.hold_range[0] - 1.0)

    def _settle(self, obs, now):
        self.readings.append(float(obs["gripper.pos"]))
        load = self._load()
        if load is not None:
            self.loads.append(float(load))
        if now - self.settle_start < self.s.settle_s or len(self.readings) < self.s.window_frames:
            return action_of(self.hold_pose)
        window = self.readings[-self.s.window_frames:]
        reading = statistics.median(window)
        error = reading - self.goal
        low, high = self.s.hold_range
        load = statistics.median(abs(v) for v in self.loads[-self.s.window_frames:]) if self.loads else None
        in_band, stalled = low <= reading <= high, error >= self.s.min_error
        if self.s.min_load is None:
            load_ok, load_rule = True, "off"
        elif load is None:
            load_ok, load_rule = True, "unavailable"  # fall back to the reading rule alone
        else:
            load_ok, load_rule = load >= self.s.min_load, "on"
        record = {"phase": "settle", "reading": round(reading, 2), "goal": round(self.goal, 2), "error": round(error, 2),
                  "spread": round(max(window) - min(window), 2), "first_reading": round(self.readings[0], 2),
                  "command_at_trigger": round(self.command_at_trigger, 2), "load": load, "in_band": in_band,
                  "stalled": stalled, "load_ok": load_ok, "load_rule": load_rule, "min_load": self.s.min_load,
                  "settle_s": round(now - self.settle_start, 2),
                  "tip_z": None if self.tip_z is None else round(self.tip_z, 3)}
        if getattr(self, "load_error", None):
            record["load_error"] = self.load_error
        if in_band and stalled and load_ok:
            if self.s.lock_after_hold:
                self.lock_goal = self.lock_goal_for(reading)
                self.hold_pose = np.concatenate([self.hold_pose[:5], [self.lock_goal]])
            else:
                self.lock_goal = None
            lock = None if self.lock_goal is None else round(self.lock_goal, 2)
            self._record({**record, "verdict": "held", "lock_goal": lock})
            self.phase, self.watching = "run", True
            self.watch.clear()
            self.last_gripper = self.goal if self.lock_goal is None else self.lock_goal
            if self.s.requery_on_hold:
                requery(self.inner)
            return action_of(self.hold_pose)
        return self._miss(obs, now, record, back_off=True)

    def _miss(self, obs, now, record, back_off: bool):
        self.missed = True
        self._record({**record, "verdict": "miss", "back_off": back_off, "open_goal": self._open_goal()})
        here = pose_of(obs)
        held = self.hold_pose if back_off else np.concatenate([here[:5], [self.last_gripper]])
        opened = np.concatenate([held[:5], [self._open_goal()]])
        parts = [MoveTo("grasp_open", opened, gripper_step=4.0)]
        if back_off:
            parts.append(MoveTo("grasp_back_off", np.concatenate([self.back_pose[:5], [self._open_goal()]])))
        self.release, self.release_sent = Sequence("grasp_release", parts), action_of(held)
        self.release.start(obs, action_of(held), now)
        self.phase, self.release_start = "release", now
        return self.step(obs, now)

    def _record(self, record: dict) -> None:
        self.checks.append(record)
        self.on_check(record)

    def done(self, obs, now):
        if self.phase == "release":
            return self.backoff_timed_out or self.release.done(obs, now)
        return self.phase == "run" and (self.released_locked or self.inner.done(obs, now))

    def result(self):
        if self.missed:
            return "grasp_miss"
        return "released_while_locked" if self.released_locked else self.inner.result()
