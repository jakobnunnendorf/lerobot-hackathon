import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "handover"))
from hand_socket import HandReceiver  # noqa: E402


def hand(open_=True, **extra):
    return {"x": 0.15, "y": 0.0, "z": 0.15, "open": open_, "palm_up": open_, **extra}


def pixel(open_=True):
    return {"u": 300.0, "v": 200.0, "open": open_, "palm_up": open_}


def pointing(**extra):
    return hand(False, **{"pointing": True, "tip_x": 0.30, "tip_y": 0.05, "tip_z": 0.15, **extra})


def feed(receiver, **message):
    receiver._store(json.dumps({"t": 0, **message}))
    return len(receiver.newest_hands()), receiver.newest_palm_count()


def main():
    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {got}" + ("" if ok else f" (want {want})"), flush=True)
        if not ok:
            failures.append(name)

    r = HandReceiver("127.0.0.1", 0)
    check("two open hands (3D) count", feed(r, hands=[hand(), hand()]), (2, 2))
    check("two fists (3D) do not count", feed(r, hands=[hand(False), hand(False)]), (0, 0))
    check("one palm and one fist: one palm", feed(r, hands=[hand(), hand(False)]), (1, 1))
    check("a hand without the open key counts (old tracker)", feed(r, hands=[{"x": 0, "y": 0, "z": 0}]), (1, 1))
    check("uncalibrated: two open pixel palms count", feed(r, hands=[], pixels=[pixel(), pixel()]), (0, 2))
    check("uncalibrated: two fists do not count", feed(r, hands=[], pixels=[pixel(False), pixel(False)]), (0, 0))
    check("uncalibrated: a pixel palm gives no 3D hand for the follower", feed(r, hands=[], pixels=[pixel()])[0], 0)
    check("garbage line leaves the last frame alone", feed(r, hands="x"), (0, 1))
    tips = lambda: [tuple(t) for t in r.newest_pointing_tips()]  # noqa: E731
    feed(r, hands=[pointing()])
    check("pointing hand with tip: tip is there", tips(), [(0.30, 0.05, 0.15)])
    check("pointing hand is not an open palm", (len(r.newest_hands()), r.newest_palm_count()), (0, 0))
    feed(r, hands=[pointing(), hand()])
    check("pointing hand and open palm in one line", (tips(), len(r.newest_hands())), ([(0.30, 0.05, 0.15)], 1))
    feed(r, hands=[hand(False, pointing=True)])
    check("pointing hand without tip_x: no tip", tips(), [])
    feed(r, hands=[hand(False, pointing=True, tip_x=0.3)])
    check("pointing hand with a partial tip: no tip", tips(), [])
    feed(r, hands=[hand(False, pointing=False, tip_x=0.30, tip_y=0.05, tip_z=0.15)])
    check("tip of a hand that is not pointing: no tip", tips(), [])
    feed(r, hands=[hand(pointing=True, tip_x="garbage", tip_y=0.0, tip_z=0.15), hand()])
    check("garbage tip: skipped, the open palm still arrives", (tips(), len(r.newest_hands())), ([], 2))
    feed(r, hands=[pointing(tip_z=None)])
    check("null tip field: skipped", tips(), [])
    feed(r, hands=[pointing(tip_x=float("nan"))])
    check("nan tip field: skipped", tips(), [])
    r.open_only = False
    feed(r, hands=[pointing()])
    check("open_only off: tips unchanged", tips(), [(0.30, 0.05, 0.15)])
    r.open_only = True
    check("open_only on: pointing tip still counts", tips(), [(0.30, 0.05, 0.15)])
    r.open_only = False
    check("open_only off: fists count again", feed(r, hands=[hand(False), hand(False)]), (2, 2))
    r.open_only = True
    feed(r, hands=[pointing()])
    first, _ = r.newest_pointing()
    same = r.newest_pointing()
    feed(r, hands=[hand()])
    after_palm = r.newest_pointing()
    feed(r, hands="x")
    check("message number: the same message twice keeps it, every stored line (pointing or not) adds one, garbage none",
          (same[0] - first, [tuple(t) for t in same[1]], after_palm[0] - first, after_palm[1], r.newest_pointing()[0] - first),
          (0, [(0.30, 0.05, 0.15)], 1, [], 1))
    r.max_age_s = -1
    check("stale frame counts nothing", (len(r.newest_hands()), r.newest_palm_count(), tips(), r.newest_pointing()[1]),
          (0, 0, [], []))
    print("PASS" if not failures else f"FAIL: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
