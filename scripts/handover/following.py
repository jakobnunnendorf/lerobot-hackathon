from dataclasses import dataclass
from enum import Enum

import numpy as np

# PLACEHOLDER until TianK003/hand-following lands: Tian's agent owns the following/drop logic
# (board comment 7061160180). The loop only needs Follower.step() returning a Decision.
PLACEHOLDER = True


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


@dataclass
class FollowSettings:
    hover_height: float = 0.18
    lost_to_home_s: float = 3.0
    drop_radius: float = 0.03
    drop_dwell_s: float = 1.0
    open_time_s: float = 1.0


class Follower:
    def __init__(self, settings: FollowSettings):
        self.s = settings
        self.lost_since: float | None = None
        self.over_hands_since: float | None = None
        self.drop_started: float | None = None

    def step(self, hands: list[np.ndarray], tip_xyz: np.ndarray, now: float) -> Decision:
        if self.drop_started is not None:
            return Decision(Mode.DROP if now - self.drop_started < self.s.open_time_s else Mode.DONE)

        if not hands:
            self.over_hands_since = None
            self.lost_since = self.lost_since if self.lost_since is not None else now
            return Decision(Mode.HOME if now - self.lost_since >= self.s.lost_to_home_s else Mode.HOLD)
        self.lost_since = None

        point = np.mean(hands, axis=0)
        # z from one camera is too rough to follow, so the tip hovers at a fixed height above the table.
        target = np.array([point[0], point[1], self.s.hover_height])

        over = len(hands) >= 2 and np.hypot(*(tip_xyz[:2] - point[:2])) <= self.s.drop_radius
        if not over:
            self.over_hands_since = None
        elif self.over_hands_since is None:
            self.over_hands_since = now
        elif now - self.over_hands_since >= self.s.drop_dwell_s:
            self.drop_started = now
            return Decision(Mode.DROP)
        return Decision(Mode.FOLLOW, target)
