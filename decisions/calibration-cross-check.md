# How is the top camera's calibration checked independently?

- Status: DECIDED (2026-11-21)

## Decision

`scripts/calib/cross_check.py` recomputes intrinsics (same detector flags, same two-pass outlier rule, as Tian's script, so numbers compare) and the table pose from the same files, reads his YAML, and ends in GOOD / CHECK / BAD. It imports nothing from his scripts.

## Why

- Two implementations that agree catch a bug in either; a script reusing his modules would not.
- A flat board cannot expose a wrong focal length (the pose from a board of known size scales with it); only the multi-view RMS and the rough lens position can. The report says so.
- The 180 degree ambiguity of the 7 x 5 pattern is resolved by the red tape (found in the picture), else by the lens position; with neither the verdict is CHECK. The two mirror-image labelings are rejected by a proper (no mirror) rigid fit in the table plane.

## Alternatives

- Scale-fitting the ruler to the board: rejected, it would hide a wrong square size; the ruler-implied square size is printed instead.
- Checking only against the ruler via four-point solvePnP: rejected, two degrees of freedom of check and the same K dependence.

Tested on synthetic pictures only (`tests/calib_cross_check.py`); nothing real yet.
