from collections.abc import Callable
from dataclasses import dataclass, field

from skills import Skill

HOLES = ("red", "blue")


# Measured on teleoperated molnrb/so101_pick_rb_mixed, not on policy runs: red hole at pan -35.4 to -28.7, blue at
# +22.3 to +28.2, home -13 to +9. The rule gives Jakob's labelled hole sequence on 53 of 58 episodes; a partial open
# that never reaches open_at is missed, and the step's time budget is the fallback (decisions/multistep-step-state.md).
# held_above: the lowest reading while closed must stay at or above it. On 2026-11-22 the first arm test counted
# two blue "drops" that were empty closes carried to the hole (gripper 1.6-4 while closed). Empty closes read
# 0.4-3.9 and held balls 5.5+ (grasp_check.py, 606 teleop closes).
@dataclass
class DropRule:
    holding_below: float = 16.5
    open_at: float = 18.0
    red_pan_below: float = -25.0
    blue_pan_above: float = 18.0
    held_above: float = 5.0

    def hole(self, pan: float) -> str | None:
        return "red" if pan < self.red_pan_below else "blue" if pan > self.blue_pan_above else None


# The measured gripper closes below holding_below, then opens past open_at: a drop if it held something and the pan
# is over a hole, else a release (an empty close, or a ball let go elsewhere). The colour is the hole's, whatever step
# is running.
@dataclass
class DropCounter:
    rule: DropRule
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(HOLES, 0))
    closed: bool = False
    closed_min: float = float("inf")
    last_closed_min: float | None = None

    def update(self, obs: dict) -> tuple[str | None, float] | None:
        gripper, pan = float(obs["gripper.pos"]), float(obs["shoulder_pan.pos"])
        if gripper < self.rule.holding_below:
            self.closed, self.closed_min = True, min(self.closed_min, gripper)
            return None
        if not self.closed or gripper < self.rule.open_at:
            return None
        self.closed, self.last_closed_min, self.closed_min = False, self.closed_min, float("inf")
        hole = self.rule.hole(pan) if self.last_closed_min >= self.rule.held_above else None
        if hole:
            self.counts[hole] += 1
        return hole, pan


# Runs a clear skill until `colour` has `wanted` drops in all; the skill itself never says it is done (CLAUDE.md).
class UntilDrops:
    def __init__(self, skill: Skill, counter: DropCounter, colour: str, wanted: int,
                 on_open: Callable[[str | None, float, dict], None]):
        self.skill, self.counter, self.colour, self.wanted, self.on_open = skill, counter, colour, wanted, on_open
        self.name = skill.name

    def reached(self) -> bool:
        return self.counter.counts[self.colour] >= self.wanted

    def start(self, obs, last_sent, now):
        self.skill.start(obs, last_sent, now)

    def step(self, obs, now):
        return self.skill.step(obs, now)

    def done(self, obs, now):
        opened = self.counter.update(obs)
        if opened is not None:
            self.on_open(*opened, obs)
        return self.reached() or self.skill.done(obs, now)

    def result(self):
        return "count_reached" if self.reached() else self.skill.result()
