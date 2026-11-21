# Demo chain rounds, 2026-11-21 17:15–17:35 CET

These are the rounds reported in 7065772934.

## Files
- `chain_20261121_*.jsonl`: the `--log` files, copied unchanged from `~/so101/demo/`. The 171516 file is empty: that start ended before any line was written.
- `chain_robot_r1r2.log` (rounds 1–2) and `chain_robot.log` (the later run): the robot program's terminal output, from `tee`.
- `chain_tracker.log`: the hand tracker's terminal output.
- `chain_robot.sh` and `chain_tracker.sh`: the exact commands used.
  - Robot: `bash scripts/supervisor/demo_chain.sh --step 015000 --min-held-target-area 120`.
  - Tracker: `hand_tracker.py --hand-z 0.20`.
- `local_patch.diff`: the local crash fix, the same as 115ccbb.

## Checkout
`TianK003/demo-chain` at **a13cacfafbbe92713590cd7a5d0429b343ec2639**, plus `local_patch.diff`, in `C:\Users\molib\Documents\elrobot\demo`.

The policy is `Space0code/act_redblue_from_foil@015000`.

No wrist JPEGs: this checkout does not save them.

## Table notes, one line per round
_To be added in Balazs's words: ball visible, half-covered or buried, and anything he saw._
