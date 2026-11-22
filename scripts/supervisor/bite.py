from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from skills import action_of, pose_of, step_towards


@dataclass
class BiteSettings:
    mode: str = "aim"
    gripper_open: float = 60.0
    # The gripper is position-controlled: the squeeze comes from the gap between this goal and the finger's width.
    gripper_close: float = 5.0
    window_s: float = 30.0
    max_bites: int = 3
    standoff_m: float = 0.06
    overshoot_m: float = 0.01
    zone_m: float = 0.08
    zone_ahead_m: float = 0.04
    min_height_m: float = 0.15
    min_radius_m: float = 0.20
    max_radius_m: float = 0.38
    max_pan_deg: float = 45.0
    seen_frames: int = 5
    seen_of: int = 8
    steady_m: float = 0.03
    steady_s: float = 0.3
    freeze_within_m: float = 0.10
    lost_s: float = 1.0
    step_m: float = 0.008
    lunge_step_m: float = 0.02
    max_step_deg: float = 1.5
    lunge_step_deg: float = 4.0
    lunge_timeout_s: float = 0.6
    snap_s: float = 0.5
    release_s: float = 0.4
    cooldown_s: float = 1.5
    limit_margin_deg: float = 3.0
    arrive_m: float = 0.005
    arrive_tol_deg: float = 4.0
    # Joint interpolation between two pitch-0 poses dips a few mm below the straight tip line; without this slack a
    # mouth pose right at the floor would block every step.
    guard_slack_m: float = 0.003


class Phase(StrEnum):
    READY = "READY"
    WATCH = "WATCH"
    AIM = "AIM"
    LUNGE = "LUNGE"
    SNAP = "SNAP"
    RELEASE = "RELEASE"
    COOLDOWN = "COOLDOWN"


FILTER_TIPS = 5
LATCH_M = 0.01


def radial(point: np.ndarray) -> np.ndarray:
    return point[:2] / np.hypot(point[0], point[1])


def bearing_deg(point: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(point[1], point[0])))


def toward(point: np.ndarray, target: np.ndarray, distance: float) -> np.ndarray:
    delta = target - point
    length = np.linalg.norm(delta)
    return target.copy() if length <= distance else point + delta * distance / length


# Phases, workspace and defaults: decisions/finger-biting.md. The arm works on its commanded joints (as the
# handover loop does), so a lagging servo never makes the next waypoint jump.
class BiteSkill:
    name = "bite"

    def __init__(self, kinematics, receiver, settings: BiteSettings, ready_pose: np.ndarray,
                 on_event: Callable[..., None] = lambda **event: None):
        self.k, self.receiver, self.s, self.on_event = kinematics, receiver, settings, on_event
        self.ready = np.asarray(ready_pose, dtype=float)[:5]
        self.mouth = kinematics.forward_kinematics(self.ready)
        self.roll = float(kinematics.urdf_degrees(self.ready)[4])
        if not self._in_workspace(self.mouth, slack=settings.guard_slack_m):
            raise ValueError(f"bite mouth pose tip {np.round(self.mouth, 3).tolist()} is outside the bite workspace "
                             f"(at least {settings.min_height_m} m up, {settings.min_radius_m}-{settings.max_radius_m} m out)")

    def start(self, obs, last_sent, now):
        self.commanded = pose_of(last_sent)[:5]
        self.gripper = self.s.gripper_open
        self.bites, self.watch_started, self.finished, self.last_message = 0, None, False, None
        self._enter(Phase.READY, now)

    def step(self, obs, now):
        self._advance(pose_of(obs)[:5], now)
        if not self.finished:
            self._command()
        return action_of(np.concatenate([self.commanded, [self.gripper]]))

    def done(self, obs, now):
        return self.finished

    def result(self):
        return "bit" if self.bites else "no_finger"

    def _enter(self, phase: Phase, now: float) -> None:
        self.phase, self.phase_since, self.latched = phase, now, False
        if phase in (Phase.READY, Phase.WATCH):
            self.seen, self.tips = deque(maxlen=self.s.seen_of), deque(maxlen=FILTER_TIPS)
            self.finger, self.frozen, self.arrived_since = None, False, None

    def _advance(self, measured: np.ndarray, now: float) -> None:
        elapsed = now - self.phase_since
        if self.phase in (Phase.READY, Phase.COOLDOWN) and self._at_ready(measured):
            if self.phase == Phase.READY or elapsed >= self.s.cooldown_s:
                self._after_mouth(now)
        elif self.phase == Phase.WATCH:
            self._watch(now)
        elif self.phase == Phase.AIM:
            self._aim(now)
        elif self.phase == Phase.LUNGE and (self._tip_at(self.lunge_to) or elapsed >= self.s.lunge_timeout_s):
            self._snap(now)
        elif self.phase == Phase.SNAP and elapsed >= self.s.snap_s:
            self.gripper = self.s.gripper_open
            self._enter(Phase.RELEASE, now)
        elif self.phase == Phase.RELEASE and elapsed >= self.s.release_s:
            self._enter(Phase.COOLDOWN, now)

    def _after_mouth(self, now: float) -> None:
        if self.bites >= self.s.max_bites:
            self.finished = True
            return
        self.watch_started = now if self.watch_started is None else self.watch_started
        self._enter(Phase.WATCH, now)

    def _watch(self, now: float) -> None:
        new, tip = self._new_candidate(self.mouth)
        if new:
            self.seen.append(tip is not None)
        if tip is not None:
            self.tips.append(tip)
        if now - self.watch_started >= self.s.window_s:
            self.finished = True
        elif sum(self.seen) >= self.s.seen_frames:
            self.finger = np.median(self.tips, axis=0)
            self.on_event(event="bite_seen", finger=np.round(self.finger, 3).tolist(), mode=self.s.mode)
            if self.s.mode == "zone":
                self._lunge(self.mouth, self._zone_lunge_point(self.finger), now)
            else:
                self._enter(Phase.AIM, now)
                self._sighted(now)

    # Frozen from here on: near the jaws the arm hides the finger from the top camera. Only on a sighting, since a
    # finger that vanished while still far from the jaws was withdrawn, not hidden.
    def _sighted(self, now: float) -> None:
        self.last_seen = now
        self.frozen = bool(np.linalg.norm(self._tip() - self.finger) <= self.s.freeze_within_m)

    def _aim(self, now: float) -> None:
        if not self.frozen:
            _, tip = self._new_candidate(self.finger)
            if tip is not None:
                self.tips.append(tip)
                self.finger = np.median(self.tips, axis=0)
                self._sighted(now)
            elif now - self.last_seen > self.s.lost_s:
                self.on_event(event="bite_lost", finger=np.round(self.finger, 3).tolist())
                self._enter(Phase.READY, now)
                return
        aim = self._aim_point(self.finger)
        if not (self.frozen and self._tip_at(aim)):
            self.arrived_since = None
        elif self.arrived_since is None or np.linalg.norm(self.finger - self.anchor) > self.s.steady_m:
            self.arrived_since, self.anchor = now, self.finger.copy()
        elif now - self.arrived_since >= self.s.steady_s:
            self._lunge(aim, self._lunge_point(self.finger), now)

    def _lunge(self, start: np.ndarray, target: np.ndarray, now: float) -> None:
        self.lunge_from, self.lunge_to = start, target
        self._enter(Phase.LUNGE, now)

    def _snap(self, now: float) -> None:
        self.bites += 1
        self.gripper = self.s.gripper_close
        self.on_event(event="bite", bite=self.bites, tip=np.round(self._tip(), 3).tolist(),
                      target=np.round(self.lunge_to, 3).tolist())
        self._enter(Phase.SNAP, now)

    def _command(self) -> None:
        if self.phase in (Phase.READY, Phase.COOLDOWN):
            self._go_to_mouth()
        elif self.phase == Phase.AIM:
            self._move_tip(self._aim_point(self.finger), self.s.step_m, self.s.max_step_deg)
        elif self.phase == Phase.LUNGE:
            self._move_tip(self.lunge_to, self.s.lunge_step_m, self.s.lunge_step_deg)
        elif self.phase == Phase.RELEASE:
            self._move_tip(self.lunge_from, self.s.step_m, self.s.max_step_deg)

    # From the folded rest pose (tip behind the pan axis) only a joint move gets there; once near the mouth pose the
    # last centimetre is a joint move too, so it ends on the taught joints even if they are not exactly pitch 0.
    # A Cartesian step that cannot move (no pitch-0 solution 35 cm up, 32-35 cm out, on the way from the demo rest)
    # falls back to the joint move: without it the arm hung mid-air until the bite timeout.
    def _go_to_mouth(self) -> None:
        tip = self._tip()
        self.latched |= np.linalg.norm(tip - self.mouth) < LATCH_M
        cartesian = not self.latched and self._in_workspace(tip, slack=self.s.guard_slack_m)
        if not (cartesian and self._move_tip(self.mouth, self.s.step_m, self.s.max_step_deg)):
            self.commanded = self._guarded(self.ready, self.s.max_step_deg, base=self.latched)

    def _move_tip(self, target: np.ndarray, step_m: float, max_step_deg: float) -> bool:
        goal = self._solve(self._clamp(toward(self._tip(), target, step_m)))
        if goal is None:
            return False
        moved = self._guarded(goal, max_step_deg)
        changed = not np.allclose(moved, self.commanded)
        self.commanded = moved
        return changed

    def _solve(self, point: np.ndarray) -> np.ndarray | None:
        try:
            urdf = self.k.solve(point, 0.0, self.roll, margin_deg=self.s.limit_margin_deg)
        except self.k.Unreachable:
            return None
        return self.k.lerobot_degrees(urdf)

    # As handover_loop.guarded_step: a shorter step whenever the full one would bring the tip lower (or, with base,
    # closer to the pan axis) than the floor allows; the violation never grows past where the arm already is.
    def _guarded(self, goal: np.ndarray, max_step_deg: float, base: bool = True) -> np.ndarray:
        allowed = max(self._violation(self.commanded, base), self._violation(goal, base), self.s.guard_slack_m)
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = step_towards(self.commanded, self.commanded + fraction * (goal - self.commanded), max_step_deg)
            if self._violation(candidate, base) <= allowed:
                return candidate
        return self.commanded

    def _violation(self, joints: np.ndarray, base: bool) -> float:
        tip = self.k.forward_kinematics(joints)
        below = max(self.s.min_height_m - tip[2], 0.0)
        return below + (max(self.s.min_radius_m - np.hypot(tip[0], tip[1]), 0.0) if base else 0.0)

    def _tip(self) -> np.ndarray:
        return self.k.forward_kinematics(self.commanded)

    def _tip_at(self, point: np.ndarray) -> bool:
        return bool(np.linalg.norm(self._tip() - point) <= self.s.arrive_m)

    def _at_ready(self, measured: np.ndarray) -> bool:
        return bool(np.allclose(self.commanded, self.ready) and np.abs(measured - self.ready).max() <= self.s.arrive_tol_deg)

    # The receiver hands out its newest tracker message for up to 0.5 s; only a message not looked at before counts,
    # or one false detection fills "5 of the last 8" and the median on its own.
    def _new_candidate(self, near: np.ndarray) -> tuple[bool, np.ndarray | None]:
        message, tips = self.receiver.newest_pointing()
        if message == self.last_message:
            return False, None
        self.last_message = message
        valid = [tip for tip in map(self._as_point, tips) if tip is not None and self._targetable(tip)]
        return True, min(valid, key=lambda tip: np.linalg.norm(tip - near), default=None)

    @staticmethod
    def _as_point(tip) -> np.ndarray | None:
        point = np.asarray(tip, dtype=float).ravel()
        return point if point.shape == (3,) and np.all(np.isfinite(point)) else None

    def _targetable(self, finger: np.ndarray) -> bool:
        if not self._in_workspace(finger):
            return False
        if self.s.mode == "zone":
            zone = np.array([*(self.mouth[:2] + radial(self.mouth) * self.s.zone_ahead_m), self.mouth[2]])
            return bool(np.linalg.norm(finger - zone) <= self.s.zone_m)
        return self._solve(self._aim_point(finger)) is not None and self._solve(self._lunge_point(finger)) is not None

    def _in_workspace(self, point: np.ndarray, slack: float = 0.0) -> bool:
        radius = np.hypot(point[0], point[1])
        pan_off = abs((bearing_deg(point) - bearing_deg(self.mouth) + 180.0) % 360.0 - 180.0)
        return bool(point[2] >= self.s.min_height_m - slack - 1e-9
                    and self.s.min_radius_m - slack <= radius <= self.s.max_radius_m + slack
                    and pan_off <= self.s.max_pan_deg)

    def _clamp(self, point: np.ndarray) -> np.ndarray:
        radius = np.clip(np.hypot(point[0], point[1]), self.s.min_radius_m, self.s.max_radius_m)
        return np.array([*(radial(point) * radius), max(point[2], self.s.min_height_m)])

    def _along(self, finger: np.ndarray, distance: float) -> np.ndarray:
        radius = np.hypot(finger[0], finger[1]) + distance
        return self._clamp(np.array([*(radial(finger) * radius), finger[2]]))

    def _aim_point(self, finger: np.ndarray) -> np.ndarray:
        return self._along(finger, -self.s.standoff_m)

    # Radial, because pitch 0 keeps the gripper in the vertical plane through the pan axis: the lunge is along the jaws.
    def _lunge_point(self, finger: np.ndarray) -> np.ndarray:
        return self._along(finger, self.s.overshoot_m)

    def _zone_lunge_point(self, finger: np.ndarray) -> np.ndarray:
        forward = radial(self.mouth)
        depth = max(float(np.dot(finger[:2] - self.mouth[:2], forward)), 0.0) + self.s.overshoot_m
        return self._clamp(np.array([*(self.mouth[:2] + forward * depth), self.mouth[2]]))
