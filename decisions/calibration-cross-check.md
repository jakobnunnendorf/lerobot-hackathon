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

## Result on the real calibration (2026-11-21, board 7064157627, 7064225867)

- Verdict GOOD, with caveats: same camera pose as Tian's script to 0.0 mm and 0.01°, corner residuals 0.8-2.0 mm, board reprojection 0.25 px. Expected position error about 2-3 mm at the bowl, 5-7 mm at the edges of the work area and at hand height.
- Intrinsics recomputed from the 40 pushed frames matched `intrinsics.yaml`. The image centre (cy about 95) is poorly fixed by the narrow 28° view; fixing it at the image centre puts the camera at 840 mm, matching the tape measure, and moves positions by at most 4.5 mm.
- Bug found in this tool: its tape finder took the orange arm for the red tape. Giving the tape pixel by hand (`--tape-px`) works around it.
- Intersect rays at the ball-centre height (about 20 mm), not the table plane: the plane costs about 14 mm, the largest error found.

Synthetic tests: `tests/calib_cross_check.py`.
