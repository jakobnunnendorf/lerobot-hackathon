# LeRobot's kinematics does not install on Windows

- Date: 2026-11-20
- Verified by: reading `uv.lock` and the LeRobot v0.7.0 file tree on GitHub. Nothing was run.

## The fact

- LeRobot 0.7.0 has an arm model for forward and inverse kinematics in `lerobot/model/kinematics.py`, used by `lerobot/robots/so_follower/robot_kinematic_processor.py`. It depends on the `placo` package.
- In our `uv.lock`, `placo` 0.9.15 and `pin` 3.4.0 (pinocchio) have wheels for macOS and Linux only. There is no Windows wheel for either.
- So on a Windows laptop the built-in kinematics cannot be imported. It can on Linux, and on macOS once the torch pin problem is solved (see the message board).
- The LeRobot package does not ship an SO-101 URDF. It has to come from the SO-ARM100 hardware repository (not checked).

## What to do instead on Windows

Write the forward kinematics by hand with numpy: the SO-101 is a short chain of five joints plus the gripper, so joint angles to link positions is a few dozen lines once the link lengths are known. Measure the link lengths on the real arm or take them from the URDF.

Not checked: how LeRobot's joint values (degrees with `use_degrees`, otherwise normalized) map onto the zero pose of a hand-written model. Verify on the arm by putting it in a known pose and comparing.
