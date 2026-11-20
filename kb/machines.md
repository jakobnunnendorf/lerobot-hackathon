# Machines: what trains, what records, what runs the demo

- Date: 2026-11-20
- Verified by: the first two facts were told to Tian at the venue. The rest is marked.

## Facts

- **The final demo has to run inference on our own laptops**, not on an organizers' machine.
- **Kamel's laptop has a mobile RTX 4070.** That card normally has 8 GB of video memory; Kamel to confirm, together with the operating system.
- Tian's laptop (Windows 11): no NVIDIA GPU found and `uv` is not installed yet (checked 2026-11-20). It can record and run code, it cannot train.

## What follows from that

- 8 GB is enough for ACT (about 2-6 GB by the LeRobot hardware guide) and tight or too small for the larger policies. Unless cloud credits appear, ACT is the policy.
- Kamel's laptop is both the only training machine and the demo machine. While it trains, another laptop should drive the robot for recording, which needs no GPU.
- Whichever laptop drives the robot needs the same calibration files and the same `--robot.id` that were used when recording. A policy run with a different calibration behaves differently. Copy the calibration files between laptops instead of recalibrating. Where 0.7.0 stores them is not checked yet.
- A second laptop should be able to run the whole demo in case Kamel's fails on Sunday. ACT plans many steps per inference, so running it on a CPU may be fast enough. Not tested: `uv run python tests/lerobot_check.py` on a laptop without a GPU prints the inference time.

## To confirm on the hardware

- Video memory and operating system of Kamel's laptop.
- Time for one ACT training run on it (the smoke test gives this).
- Whether a CPU-only laptop can run the policy at full speed.
