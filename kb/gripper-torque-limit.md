# Follower gripper torque: capped at 200 of 1000. Do not exceed 200.

- Date: 2026-11-22
- Verified by: molnrb at the table (force step test on the arm, servo registers read back)

**Never set the follower gripper's `Torque_Limit` or `Max_Torque_Limit` above 200**, in code, on the command line or by hand. Not to "grip harder", not for a test, not for one dataset.

- **Why:** LeRobot 0.7.0 caps the gripper at 500 (50%), and our grasp check drives it to goal 1 on the ball, so every grasp stalled at 500. On 2026-11-22 that stripped the gears of the follower's gripper servo (clicking, the ball slipping). The organisers' replacement servo is in; it has no spare.
- **200 is enough:** at 200 the gripper holds a ball firmly (Balazs, step test 150-250: the jaws stop on the ball at 12-14% open, load = the cap).
- **Where it is enforced:** `scripts/so101/gripper_limit.py` (`GRIPPER_TORQUE_LIMIT = 200`). It runs right after every connect in `run_supervisor.py`, `eval_pick.py`, `handover_loop.py` and the dashboard recorder (`record_session.py`), and once before `lerobot-record` / `lerobot-teleoperate` / `lerobot-rollout` in `record.sh`, `teleop.sh` and `run_policy.sh`.
- **Why after every connect:** LeRobot's `connect()` writes `Max_Torque_Limit` 500 (EEPROM) each time. The live `Torque_Limit` (RAM) keeps the last value written, survives torque off/on and a new LeRobot connect, and only reloads from `Max_Torque_Limit` (500) at power-up. A program that connects the follower without our scripts (a bare `lerobot-*` command) after a power cycle runs at 500: run `python scripts/so101/gripper_limit.py /dev/so101_follower` first.
- **Load readings:** a held ball reads the cap, 200 (was 440-500 at 500); empty closes read 28-44. The hold thresholds `--grasp-min-load` and `--hold-min-load` are 120 for that reason. Raising the cap would not fix a weak grip; lowering those thresholds below about 60 would count empty closes as holds.
- **Replacement servo, 2026-11-22:** gripper recalibrated on its own (homing -549, range 2047-3480, span 1433; the old one was 1979-3432). The previous calibration file is kept next to it as `mi_follower.before_gripper_swap_20261122_045746.json`.
