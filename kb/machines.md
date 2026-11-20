# Machines: what trains, what records, what runs the demo

- Date: 2026-11-20
- Verified by: the first two facts were told to Tian at the venue. The rest is marked.

## Facts

- **The final demo has to run inference on our own laptops**, not on an organizers' machine.
- **Kamel's laptop: NVIDIA RTX 4070 Laptop GPU with 8 GB, running Ubuntu 24.04 under WSL2 on Windows.** Confirmed by Kamel on the message board (2026-11-20). `tests/cuda.py` and `tests/lerobot_check.py` both pass there: torch 2.11.0+cu128, CUDA available, ACT builds, peak GPU memory in the check about 1 GB.
- **WSL2 does not see USB devices by default.** The arm's serial ports and the cameras have to be passed into WSL (usbipd-win), and cameras are the usual trouble spot. Not yet tried on Kamel's laptop. If it does not work, the robot and the demo run from a native Windows environment on the same laptop (`uv.lock` has Windows wheels with CUDA) and WSL is kept for training. WSL and Windows have separate calibration files and separate dataset caches.
- In Kamel's WSL environment the LeRobot kinematics dependencies can install, because Linux wheels exist (`kb/kinematics-on-windows.md`). That only helps if the robot is also driven from WSL.
- Tian's laptop (Windows 11): no NVIDIA GPU found and `uv` is not installed yet (checked 2026-11-20). It can record and run code, it cannot train.

- **About 30 EUR of compute on Hugging Face.** Tian believes this is the amount (2026-11-20). Not confirmed: the exact amount, which account holds it, and whether it covers training jobs.

## What follows from that

- 8 GB is enough for ACT (about 2-6 GB by the LeRobot hardware guide) and tight or too small for the larger policies. On the laptop alone, ACT is the policy.
- Kamel's laptop is the only local training machine and the demo machine. While it trains, another laptop should drive the robot for recording, which needs no GPU.
- If the Hugging Face credit can pay for training jobs, the skills can train in the cloud at the same time and Kamel's laptop stays free for trying policies on the arm. `lerobot-train` takes `--job.target=<hardware>` for this (see `research/lerobot-training-feasibility.md`), and the dataset has to be on the Hub first.
- 30 EUR is a small budget. The current price per hour is not checked; `hf jobs hardware` lists the options. By the LeRobot hardware guide one ACT run takes about 1-2 hours on a mid-range cloud GPU, so a handful of ACT runs should fit and one attempt at a larger policy might. Every paid job is started by a person, and the running total is written on the message board.
- Whichever laptop drives the robot needs the same calibration files and the same `--robot.id` that were used when recording. A policy run with a different calibration behaves differently. Copy the calibration files between laptops instead of recalibrating. Where 0.7.0 stores them is not checked yet.
- A second laptop should be able to run the whole demo in case Kamel's fails on Sunday. ACT plans many steps per inference, so running it on a CPU may be fast enough. Not tested: `uv run python tests/lerobot_check.py` on a laptop without a GPU prints the inference time.

## To confirm on the hardware

- Video memory and operating system of Kamel's laptop.
- Time for one ACT training run on it (the smoke test gives this).
- Whether a CPU-only laptop can run the policy at full speed.
