# Calibration cross-check (independent of Tian's scripts)

Recomputes the top camera's intrinsics and pose from the same pictures and ruler values that `scripts/calib/calibrate_*.py` (PR #21) used, and prints the differences plus one verdict line: GOOD, CHECK or BAD. Needs only OpenCV, numpy and PyYAML. Opens no camera and sends nothing to the robot, so it runs on the Mac or on the robot laptop (the repo's `.venv` has everything; on Windows use `uv run python` instead of `.venv/bin/python`).

Inputs: `calib_frames/intrinsics/` (25 frames), `calib_frames/table/` (the table frame), the ruler values from the board (`1: x, y` ... `4: x, y`, mm, robot frame), board thickness, rough lens position, and Tian's two YAML files once they exist.

```bash
.venv/bin/python scripts/calib/cross_check.py \
  --intrinsics-frames calib_frames/intrinsics --table-frame calib_frames/table \
  --corners-mm '1: 215, 82' '2: 216, -90.5' '3: 331, -89' '4: 330, 83.5' \
  --thickness-mm 3 --lens-mm 300 -60 560 \
  --tian-intrinsics scripts/calib/intrinsics.yaml --tian-extrinsics scripts/calib/extrinsics.yaml
```

(The four corner values are placeholders; use the measured ones. `--corners-file corners.txt` takes the same lines from a file. A line `1: x, y | <read_pose.py line>` adds the jaw-tip touch for that corner and is compared as well; that needs `scripts/handover/` from PR #13.)

Reading the output:

- `VERDICT: GOOD` means every check passed; `CHECK` means look at the lines marked CHECK before trusting it; `BAD` means do not use the calibration. The exit code is 0, 1 or 2. Thresholds and their reasons are in the script's docstring.
- Lines marked `[own K]` use intrinsics recomputed here, `[Tian's K]` use his file. The comparison with his extrinsics uses his K, so a difference there is the method, not the lens.
- "corner labels 1-4 sit ... px from Tian's" BAD means his script and this one gave the labels to different corners of the board (corner 1 versus 3 swapped). Look at `calib_check/table_overlay.png`: corner 1 must be the one by the red tape.
- Without red tape in the picture and without `--lens-mm`, the board orientation cannot be told apart (the pattern looks the same turned by 180 degrees); the verdict is CHECK. Give `--tape-px X Y` or `--lens-mm X Y Z`.
- A flat board on the table cannot show a wrong focal length. Only the intrinsics RMS and the lens position can.

Results land in `calib_check/` (overlay image, `result.yaml`). Self-test with synthetic pictures: `.venv/bin/python tests/calib_cross_check.py`.
