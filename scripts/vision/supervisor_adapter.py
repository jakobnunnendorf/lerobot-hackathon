import statistics

import numpy as np
from colour_detector import ColourDetector, load_detector

# The voice contract sends the ball as "target" (scripts/voice/link.py). Words the team says for the striped ball.
ALIASES = {"striped": "striped", "stripy": "striped", "yellow": "striped", "green": "striped", "red": "red", "blue": "blue"}
CONTAINER_OF = {"red": "red_container", "blue": "blue_container"}


def colour_of(target: str | None) -> str | None:
    return ALIASES.get((target or "").strip().lower())


# Plugs into Sensors(visible_area=...) of the supervisor (PR #17): the LOOK step takes the median of this
# over its look frames and compares it with Settings.min_visible_area_px. The colour comes from the
# voice command; the supervisor itself stays unaware of colours.
class ColourVisibility:
    def __init__(self, detector: ColourDetector | None = None, target: str = "striped"):
        self.detector = detector or load_detector()
        self.target = target

    def set_target(self, command_target: str | None) -> bool:
        colour = colour_of(command_target)
        if colour is not None:
            self.target = colour
        return colour is not None

    def __call__(self, rgb: np.ndarray) -> int:
        return self.detector.area(rgb, self.target)

    # Before starting a round for "hand me the red ball": is there any red ball at all? LOOK alone cannot
    # tell "buried" from "not in the bowl", and clearing balls for a colour that is absent only wastes time.
    def in_bowl(self, frames: list[np.ndarray], min_area_px: int) -> bool:
        return statistics.median(self(frame) for frame in frames) >= min_area_px

    def container_visible(self, rgb: np.ndarray, colour: str | None = None) -> bool:
        container = CONTAINER_OF.get(colour or self.target)
        return container is not None and self.detector.detect(rgb)[container].present
