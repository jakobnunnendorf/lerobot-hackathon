# PICK benchmark, 2026-11-22 (robot laptop, molnrb)

- **Benchmark CSV:** `benchmark_eval_20261122_083731.csv`.
  - Layouts L1-L4: 28 runs, 7 models, `--n-action-steps 30`, 90 s cap, gripper torque cap 200.
  - Stopped before L5 for time.
- **Every `start_state` reads `hidden`:** the table keyed `h` for all four layouts. By the plan, L2 is partly covered and L4 is visible.
- **No recordings:** the `--record` dataset stayed empty. In `eval_pick.py` (main 58644fc), `buffer = EpisodeBuffer() if dataset else None` tests a `LeRobotDataset` for truth, and its `__len__` is the frame count. A new, empty dataset is therefore falsy, so no buffer was made and nothing was recorded.
  - The fix is `if dataset is not None`.
  - `recording_info_empty.json` is that empty `meta/info.json`.
  - The private Hub repo `molnrb/so101_eval_pick_20261122` is empty, tagged `v3.0`.
- **`fails.csv`:** the pre-filled run order. Failure codes were not filled in at the table.
- **`runsheet.md`:** the table instructions used, with layout cards and model order.
- **`benchmark_launch_log_filtered.txt`:** the eval terminal log, with LeRobot's goal-clamp warnings removed.
- **`earlier_evals/`:** every other `eval_20261122_*.csv` from tonight, all on the new gripper servo from about 05:00 CET. The board posts summarise them: 7074277776, 7074960583, 7075616730.
- **`demo_chain_e2e/`:** the demo-chain round logs from 06:00 CET on (transitions, grasp checks, round ends). The ticks files are not included.
