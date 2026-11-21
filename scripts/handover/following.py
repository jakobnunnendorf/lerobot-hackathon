from dataclasses import dataclass
from enum import Enum

import numpy as np

import kinematics


class Mode(Enum):
    FOLLOW = "follow"
    HOLD = "hold"
    HOME = "home"
    DROP = "drop"
    DONE = "done"


@dataclass
class Decision:
    mode: Mode
    target_xyz: np.ndarray | None = None


# handover_loop.py builds this positionally from its first four fields; new fields go at the end.
@dataclass
class FollowSettings:
    hover_height: float = 0.22
    lost_to_home_s: float = 3.0
    drop_radius: float = 0.03
    drop_dwell_s: float = 1.0
    open_time_s: float = 1.0
    out_of_reach_to_home_s: float = 3.0
    filter_time_constant_s: float = 0.25
    dead_band_m: float = 0.015
    settle_m: float = 0.005
    max_speed_m_s: float = 0.15
    jump_m: float = 0.15
    jump_confirm_frames: int = 5


# The loop clamps the workspace, solves the IK, limits joint speed and owns the gripper. This class decides the
# mode and hands it a Cartesian target that is already smoothed, speed-limited and reachable.
class Follower:
    def __init__(self, settings: FollowSettings):
        self.s = settings
        self.drop_started: float | None = None
        self.last_now: float | None = None
        self.lost_since: float | None = None
        self.unreachable_since: float | None = None
        self.over_hands_since: float | None = None
        self._restart_approach()

    def _restart_approach(self) -> None:
        self.filtered: np.ndarray | None = None
        self.jump_frames = 0
        self.goal: np.ndarray | None = None
        self.tracking = False

    def step(self, hands: list[np.ndarray], tip_xyz: np.ndarray, now: float) -> Decision:
        dt = 0.0 if self.last_now is None else min(max(now - self.last_now, 0.0), 0.1)
        self.last_now = now
        if self.drop_started is not None:
            return Decision(Mode.DROP if now - self.drop_started < self.s.open_time_s else Mode.DONE)

        if not hands:
            self.over_hands_since = None
            self.lost_since = now if self.lost_since is None else self.lost_since
            if now - self.lost_since >= self.s.lost_to_home_s:
                self._restart_approach()
                return Decision(Mode.HOME)
            return Decision(Mode.HOLD)
        self.lost_since = None

        point = np.mean(hands, axis=0)
        # z from one camera is too rough to follow, so the tip hovers at a fixed height above the table.
        target = self._smooth(np.array([point[0], point[1], self.s.hover_height]), dt)
        start = tip_xyz if self.goal is None else self.goal
        reachable, complete = self._reachable_toward(start, target)
        if not complete:
            self.unreachable_since = now if self.unreachable_since is None else self.unreachable_since
            if now - self.unreachable_since >= self.s.out_of_reach_to_home_s:
                self._restart_approach()
                return Decision(Mode.HOME)
            if reachable is None:
                return Decision(Mode.HOLD)
        else:
            self.unreachable_since = None

        if self.goal is None:
            self.goal = np.asarray(tip_xyz, dtype=float).copy()
        self._advance_goal(reachable, dt)

        if self._dwelled_over_hands(len(hands), tip_xyz, target, now):
            self.drop_started = now
            return Decision(Mode.DROP)
        return Decision(Mode.FOLLOW, self.goal.copy())

    def _smooth(self, raw: np.ndarray, dt: float) -> np.ndarray:
        if self.filtered is None:
            self.filtered = raw.copy()
            return self.filtered
        # a far-off detection for a frame or two is usually the tracker swapping hands; follow it only if it persists
        if np.linalg.norm(raw - self.filtered) > self.s.jump_m:
            self.jump_frames += 1
            if self.jump_frames < self.s.jump_confirm_frames:
                return self.filtered
            self.filtered = raw.copy()
        self.jump_frames = 0
        alpha = 1.0 - np.exp(-dt / self.s.filter_time_constant_s)
        self.filtered = self.filtered + alpha * (raw - self.filtered)
        return self.filtered

    def _reachable_toward(self, start: np.ndarray, target: np.ndarray) -> tuple[np.ndarray | None, bool]:
        if self._solvable(target):
            return target, True
        if not self._solvable(start):
            return None, False
        reachable, unreachable = np.asarray(start, dtype=float), target
        for _ in range(8):
            middle = (reachable + unreachable) / 2
            if self._solvable(middle):
                reachable = middle
            else:
                unreachable = middle
        return reachable, False

    # The real wrist_roll is only known to the loop; roll 0 moves the tool point by at most 8 mm.
    @staticmethod
    def _solvable(point: np.ndarray) -> bool:
        return kinematics.inverse_kinematics(point, np.zeros(len(kinematics.ARM_JOINTS))) is not None

    def _advance_goal(self, point: np.ndarray, dt: float) -> None:
        distance = np.linalg.norm(point - self.goal)
        # dead band with hysteresis: a trembling hand leaves the arm still, a real move is followed to the end
        if distance > self.s.dead_band_m:
            self.tracking = True
        if not self.tracking:
            return
        # rise to the hover height before moving sideways, so an approach from a low pose does not sweep the table
        rising = self.goal[2] < point[2] - self.s.dead_band_m
        waypoint = np.array([self.goal[0], self.goal[1], point[2]]) if rising else point
        to_waypoint = np.linalg.norm(waypoint - self.goal)
        if to_waypoint > 1e-9:
            self.goal = self.goal + (waypoint - self.goal) * (min(to_waypoint, self.s.max_speed_m_s * dt) / to_waypoint)
        if distance < self.s.settle_m:
            self.tracking = False

    def _dwelled_over_hands(self, hand_count: int, tip_xyz: np.ndarray, target: np.ndarray, now: float) -> bool:
        over = hand_count >= 2 and np.hypot(*(np.asarray(tip_xyz)[:2] - target[:2])) <= self.s.drop_radius
        if not over:
            self.over_hands_since = None
            return False
        self.over_hands_since = now if self.over_hands_since is None else self.over_hands_since
        return now - self.over_hands_since >= self.s.drop_dwell_s
