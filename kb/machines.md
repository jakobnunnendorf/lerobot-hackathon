# Machines: what trains, what records, what runs the demo

- Date: 2026-11-20, robot laptop facts added 2026-11-21
- Verified by: the first two facts were told to Tian at the venue. The rest is marked.

## Facts

- **The final demo has to run inference on our own laptops**, not on an organizers' machine. **Training may run in the cloud** (Tian, 2026-11-20).
- **Kamel's laptop: NVIDIA RTX 4070 Laptop GPU with 8 GB, running Ubuntu 24.04 under WSL2 on Windows.** Confirmed by Kamel on the message board (2026-11-20). `tests/cuda.py` and `tests/lerobot_check.py` both pass there: torch 2.11.0+cu128, CUDA available, ACT builds, peak GPU memory in the check about 1 GB.
- **The robot laptop drives the arm and is the one the policies have run on** (Balazs's computer, where molnrb's agent works). **NVIDIA RTX 4050 Laptop GPU, Ubuntu 24.04 in WSL2 on Windows, conda env `lerobot` with LeRobot 0.7.0.** The arm and both cameras are passed into WSL with usbipd, and that works: all recordings and policy runs so far were made there (board, 2026-11-21; setup and known failures in the `so101-robot` skill and `scripts/so101/`, PR #6). WSL drops the USB devices when it shuts down; `scripts/so101/attach_usb.sh` re-attaches them.
- On the robot laptop, ACT runs at 30 fps with the default `n_action_steps` and with 20, but **not with temporal ensembling** (47 ms median per step against 33 ms available). Measured by molnrb's agent, `kb/act-replan-override.md` (PR #9).
- **Kamel's laptop is not connected to the arm.** It trained `act_pick_blue_ball` overnight (20K steps, about 1 h 40 min) and runs checks that need no robot. Its LeRobot kinematics dependencies install from Linux wheels (`placo` checked 2026-11-21, `kb/kinematics-on-windows.md`).
- WSL and Windows have separate calibration files and dataset caches, so the robot is driven from one side only (WSL on the robot laptop).
- Jakob's laptop: MacBook Pro M3 Pro with 36 GB (Tian, 2026-11-20). Whether `uv sync` works there on current `main` is open (see the board).
- Tian's laptop (Windows 11): no NVIDIA GPU found and `uv` is not installed yet (checked 2026-11-20). It can record and run code, it cannot train.

- **About 30 EUR of compute on Hugging Face.** Tian believes this is the amount (2026-11-20). Not confirmed: the exact amount, which account holds it, and whether it covers training jobs.

## What follows from that

- 8 GB is enough for ACT (about 2-6 GB by the LeRobot hardware guide) and tight or too small for the larger policies. On the laptop alone, ACT is the policy.
- Kamel's laptop is the only local training machine; the robot laptop drives the robot. Training on one never blocks the other.
- The Hugging Face credit does pay for training jobs: `act_pick_ball_combined` trained there (a10g-small, about 1 h 25 min, board 2026-11-21). `lerobot-train` takes `--job.target=<hardware>` for this (see `research/lerobot-training-feasibility.md`), and the dataset has to be on the Hub first.
- Hugging Face prices per hour, read from https://huggingface.co/pricing on 2026-11-20: T4 small 0.40 USD, L4 0.80, A10G small 1.00, A10G large 1.50, L40S 1.80, A100 large 2.50. Jobs are billed by the second and are open to any account with a positive credit balance.
- What that buys: one ACT run takes about 1-2 hours on an L4 or A10G by the LeRobot hardware guide, so about 1-2 USD. One pi0.5 fine-tune takes about 4-8 hours on an A100 by the same guide, so about 10-20 USD. 30 EUR is many ACT runs, or one pi0.5 run plus a few ACT runs.
- Set the job timeout explicitly (`--job.timeout`): the Hugging Face docs and the LeRobot docs disagree about the default (30 minutes versus 48 hours).
- Every paid job is started by a person, and the running total is written on the message board.
- **pi0.5 does not fit on Kamel's 8 GB laptop as loaded, and the robot laptop's 4050 has no more memory. ACT is the demo policy.** Tested on Kamel's laptop on 2026-11-20 (reported by his agent on the board): `lerobot/pi05_base` ran out of GPU memory while the model was being built, with about 14 GB allocated, which matches the weights at full precision. The config's `dtype` setting was not applied during construction. On the CPU the same 14 GB stalled the machine, which has 15 GB of RAM. Not tried: building the model in half precision from the start, which would put the weights near 7 GB and leave almost nothing for the rest on an 8 GB card. The openpi README lists "> 8 GB" for inference. Only a team laptop with clearly more than 8 GB of GPU memory would reopen this.
- Whichever laptop drives the robot needs the same calibration files and the same `--robot.id` that were used when recording. A policy run with a different calibration behaves differently. Copy the calibration files between laptops instead of recalibrating. Where 0.7.0 stores them is not checked yet.
- A second laptop should be able to run the whole demo in case the robot laptop fails on Sunday. Kamel's laptop (WSL2, CUDA) is the closest match, but has never been connected to the arm. ACT plans many steps per inference, so running it on a CPU may be fast enough. Not tested: `uv run python tests/lerobot_check.py` on a laptop without a GPU prints the inference time.

## To confirm on the hardware

- Whether the arm and both cameras work from WSL on Kamel's laptop, if it is to be the backup demo laptop.
- Whether a CPU-only laptop can run an ACT policy at full speed (for the backup demo laptop).
