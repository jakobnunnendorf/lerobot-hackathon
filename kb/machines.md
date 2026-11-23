# Machines: what trains, what records, what runs the demo

- Date: 2026-11-20, robot laptop facts added 2026-11-21, state at the end of the hackathon 2026-11-22
- Verified by: the first fact was told to Tian at the venue. The rest is marked.

## Facts

- **The final demo has to run inference on our own laptops**, not on an organizers' machine. **Training may run in the cloud** (Tian, 2026-11-20).
- **The robot laptop drives the arm and ran every policy and the whole demo** (Balazs's computer, where molnrb's agent works). **NVIDIA RTX 4050 Laptop GPU, Ubuntu 24.04 in WSL2 on Windows, conda env `lerobot` with LeRobot 0.7.0.** The arm and both cameras are passed into WSL with usbipd, and that works: all recordings, all policy runs and the demo chain ran there (2026-11-21 to 22; setup and known failures in the `so101-robot` skill and `scripts/so101/`). WSL drops the USB devices when it shuts down; `scripts/so101/attach_usb.sh` re-attaches them. usbip dropouts can end a recording session (`decisions/data-collection-runner.md`).
- On the robot laptop, ACT runs at 30 fps with the default `n_action_steps` and with 20, but **not with temporal ensembling** (47 ms median per step against 33 ms available). Measured by molnrb's agent, `kb/act-replan-override.md`.
- **Kamel's laptop: NVIDIA RTX 4070 Laptop GPU with 8 GB, Ubuntu 24.04 under WSL2 on Windows.** `tests/cuda.py` and `tests/lerobot_check.py` both pass there: torch 2.11.0+cu128, CUDA available, ACT builds, peak GPU memory in the check about 1 GB. It was never connected to the arm. It trained `act_pick_blue_ball` overnight (20K steps, about 1 h 40 min) and ran checks that need no robot. LeRobot's kinematics dependencies install there from Linux wheels (`placo` checked 2026-11-21, `kb/kinematics-on-windows.md`).
- WSL and Windows have separate calibration files and dataset caches, so the robot is driven from one side only (WSL on the robot laptop).
- Jakob's laptop: MacBook Pro M3 Pro with 36 GB (Tian, 2026-11-20). `uv.lock` has no macOS torch wheel (CLAUDE.md).
- Tian's laptop (Windows 11, AMD Ryzen 5 5500U): no NVIDIA GPU. A full ACT chunk took a median 563 ms on its CPU (`decisions/dashboard.md`), so it ran analysis, voice benchmarks and the dashboard against a fake robot, not the arm.
- **Cloud training:** `act_pick_ball_combined` trained on Hugging Face Jobs (a10g-small, about 1 h 25 min, board 2026-11-21), paid from the team's Hugging Face credit (about 30 EUR by Tian's belief; the exact amount was never confirmed). Most later runs, ACT and SmolVLA, ran on Modal (L40S and H100) under Tomi's, Kamel's and Tian's accounts (`scripts/modal_train.py`; costs per run in `decisions/act-redblue-generalisation.md`).

## What follows from that

- 8 GB is enough for ACT (about 2-6 GB by the LeRobot hardware guide) and tight or too small for the larger policies. On the laptops, ACT is the policy.
- Hugging Face prices per hour, read from https://huggingface.co/pricing on 2026-11-20: T4 small 0.40 USD, L4 0.80, A10G small 1.00, A10G large 1.50, L40S 1.80, A100 large 2.50. Jobs are billed by the second and are open to any account with a positive credit balance. `lerobot-train` takes `--job.target=<hardware>` (see `research/lerobot-training-feasibility.md`), and the dataset has to be on the Hub first.
- Set the job timeout explicitly (`--job.timeout`): the Hugging Face docs and the LeRobot docs disagree about the default (30 minutes versus 48 hours).
- **pi0.5 does not fit on Kamel's 8 GB laptop as loaded, and the robot laptop's 4050 has no more memory. ACT is the demo policy.** Tested on Kamel's laptop on 2026-11-20 (reported by his agent on the board): `lerobot/pi05_base` ran out of GPU memory while the model was being built, with about 14 GB allocated, which matches the weights at full precision. The config's `dtype` setting was not applied during construction. On the CPU the same 14 GB stalled the machine, which has 15 GB of RAM. Not tried: building the model in half precision from the start, which would put the weights near 7 GB and leave almost nothing for the rest on an 8 GB card. The openpi README lists "> 8 GB" for inference. Only a team laptop with clearly more than 8 GB of GPU memory would reopen this.
- SmolVLA does fit for inference: one 50-action chunk took 0.43 s warm on Kamel's RTX 4070 (`decisions/hand-handover-policy.md`).
- Whichever laptop drives the robot needs the same calibration files and the same `--robot.id` (`mi_follower`) that were used when recording. A policy run with a different calibration behaves differently. Copy the calibration files between laptops instead of recalibrating. Where 0.7.0 stores them on disk was not written down.

## Never tested

- A backup demo laptop: the arm and cameras were never connected to Kamel's laptop or any other machine.
- Whether a CPU-only laptop can run an ACT policy at full speed. On Tian's CPU a full chunk took 563 ms, so the arm would stall about 0.5 s every chunk.
