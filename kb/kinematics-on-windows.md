# LeRobot's kinematics does not install on Windows

- Date: 2026-11-20
- Verified by: reading `uv.lock` and the LeRobot v0.7.0 file tree on GitHub (2026-11-20); the hand-written replacement was checked on the arm with a four-point touch test by Balazs (2026-11-21, board 7064451117).

## The fact

- LeRobot 0.7.0 has an arm model for forward and inverse kinematics in `lerobot/model/kinematics.py`, used by `lerobot/robots/so_follower/robot_kinematic_processor.py`. It depends on the `placo` package.
- In our `uv.lock`, `placo` 0.9.15 and `pin` 3.4.0 (pinocchio) have wheels for macOS and Linux only. There is no Windows wheel for either.
- So on a Windows laptop the built-in kinematics cannot be imported. It can on Linux (`placo` installed on Kamel's WSL2 laptop, 2026-11-21, `kb/machines.md`). On macOS `uv.lock` has no torch wheel, so it was not tried there.
- The LeRobot package does not ship an SO-101 URDF. The team took `so101_new_calib.urdf` from TheRobotStudio/SO-ARM100 (`research/so101-inverse-kinematics.md`).

## What to do instead on Windows

Write the forward kinematics by hand with numpy: the SO-101 is a short chain of five joints plus the gripper, so joint angles to link positions is a few dozen lines once the link lengths are known. Measure the link lengths on the real arm or take them from the URDF.

That is what the team did: closed-form kinematics in `scripts/handover/kinematics.py` (`decisions/hand-following-geometry.md`). Mapping LeRobot's degrees onto the URDF's zero pose needed fitted zero offsets, found by touching four measured points with the arm (2026-11-21): the joint signs matched, the offsets were up to 10 degrees (elbow_flex +9.9).
