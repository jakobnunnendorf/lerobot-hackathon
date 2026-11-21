import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handover"))

from following import Follower, FollowSettings, Mode  # noqa: E402

FPS = 30
PALM = np.array([0.25, 0.0, 0.20])
START_TIP = np.array([0.20, 0.0, 0.28])
failures = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def drop_time(palm_at, seen, seconds: float = 12.0) -> float | None:
    """Simulated arm that reaches each goal at once; returns when the follower drops, or None."""
    follower, tip = Follower(FollowSettings()), START_TIP.copy()
    for i in range(int(seconds * FPS)):
        now = i / FPS
        decision = follower.step([palm_at(now)] if seen(now) else [], tip, now)
        if decision.mode == Mode.DROP:
            return now
        if decision.mode == Mode.FOLLOW:
            tip = decision.target_xyz
    return None


def main() -> None:
    still = drop_time(lambda now: PALM, lambda now: True)
    check("one still palm: drop after reaching it and 1.5 s still", still is not None and 1.5 <= still <= 4.0,
          f"dropped at {still} s")

    def moving_hand(now):
        return PALM + np.array([0.0, 0.06 * np.sin(2 * np.pi * now / 3.0), 0.0])

    moving = drop_time(moving_hand, lambda now: True)
    check("palm moving back and forth by 6 cm (3 s period): no drop", moving is None, f"dropped at {moving} s")

    def flicker(now):
        return (now % 0.6) < 0.3

    flickering = drop_time(lambda now: PALM, flicker)
    check("still palm seen only half the time (0.3 s gaps): still drops", flickering is not None,
          f"dropped at {flickering} s")

    def long_gaps(now):
        return (now % 2.0) < 1.2

    gaps = drop_time(lambda now: PALM, long_gaps, seconds=8.0)
    check("still palm with 0.8 s gaps every 2 s: the dwell restarts, no drop", gaps is None, f"dropped at {gaps} s")
    print("FAIL" if failures else "PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
