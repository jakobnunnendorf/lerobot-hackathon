# research/ - external research

Findings from outside the team, each claim with its source link. One file per question. Mark anything that could not be verified as unverified.

Once something from here is confirmed on our own hardware, write the confirmed version into `kb/`.

| File | About | Still current? |
| --- | --- | --- |
| `ball-picking-and-handover.md` | The chosen idea: skills versus one policy, detection, balls in a bowl, handover safety, pitch | partly: written for a blue target among foil balls; the scene became a striped target among red and blue balls |
| `lerobot-training-feasibility.md` | LeRobot 0.7.0: policies, VRAM, commands, bimanual support, data collection, dataset tools, Windows notes | yes |
| `lerobot-multi-skill-runner.md` | Running several skills in one program with LeRobot 0.7.0, from the source | yes |
| `continue-training-from-checkpoint.md` | Fine-tuning versus resuming an ACT run after recording more episodes | yes |
| `voice-commands.md` | Wake word, speech to text, turning speech into a command, calling Claude from a script | yes |
| `multi-colour-targets.md` | Ways to make the pick skill work for a ball colour chosen at run time | yes |
| `hand-handover-policies.md` | ACT and SmolVLA for a learned hand-following handover | yes; the demo used rule-based hand following instead (`decisions/hand-handover-policy.md`) |
| `hand-tracking-and-camera-calibration.md` | MediaPipe hand tracking, hand position from one camera, checkerboard calibration, camera-to-robot transform | yes |
| `so101-inverse-kinematics.md` | SO-101 kinematics and Cartesian motion for the handover | yes |
| `rl-post-training.md` | Whether reinforcement learning can improve the trained policies in the time left; human corrections (DAgger) as the realistic step | yes |
| `label-patterns.md` | Our own pick data: wrist blindness before the striped grab, clearing patterns, inconsistent demos, and episode vs segment curation per policy (with `label-patterns/*.csv`) | yes |
| `DATASET_GUIDE.md` | The striped-ball datasets: labels, episode patterns and curation per architecture (summarised in the `so101-data` skill) | yes |
| `smolvla-clear-first.md` | Why the red/blue SmolVLA goes straight for the striped ball instead of clearing first (offline analysis) | yes |
| `act-capacity-and-hidden-targets.md` | Whether the ACT model is too small, and what the literature says limits picking a hidden target (observability, memory, chunk length, demonstrations) | yes |
