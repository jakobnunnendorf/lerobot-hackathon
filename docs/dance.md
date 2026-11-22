# Dance: the arm replays a recorded dance

For the people at the robot laptop. The arm plays back a dance that someone else recorded on their SO-101 (`NikeSong/so101_gestures_dance`, episode 1, then 2). It starts from rest with an empty gripper, folds out over its base, dances, and folds back to rest.

It runs inside the demo chain program (`demo_chain.sh`). The dashboard has a **💃 Dance** tile, and the chat or 🎤 accepts "dance", "let's dance" or "show me a dance". In terminal 1, `d` + Enter works too. Why it works this way: `decisions/dance.md`.

## 0. Once: code and the dance data (WSL, about 5 min)

Update the demo checkout to `main` (the dance is part of the demo chain since #47):

```bash
cd ~/demo && git fetch origin && git checkout --detach origin/main
```

Fetch the dance before the first run, so the robot program does not download about 400 MB of video while the arm waits:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; D('NikeSong/so101_gestures_dance', episodes=[1, 2])"
```

## 1. Safety checklist, before every start

- [ ] Clear the space around the arm: about 30 cm in every direction, and the bowl, balls and cables out of that circle. The dance sways over the base, at most 20 cm out from it, and turns the base up to about 35 degrees either way.
- [ ] One person's hand on the power switch for the whole dance.
- [ ] The gripper is empty. The program refuses to dance while the gripper reads 6-20, and the dashboard greys the tile out.
- [ ] The arm is folded at rest, as for a pick round.
- [ ] Nobody leans over the arm.

## 2. First try: episode 1 at half speed

Terminal 1, the robot program, the usual demo chain plus the dance flags:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
bash scripts/supervisor/demo_chain.sh \
  --dance-episode NikeSong/so101_gestures_dance:1 --dance-speed 0.5 --dance-max-step-deg 4
```

Check the start lines:

```text
dance NikeSong/so101_gestures_dance:1: 599 frames at speed 0.5 -> 39.9 s, speed cap 4.0 deg/tick
```

It stops with `the model puts the tip at ... below --dance-min-tip-height` when an episode would come too low. Do not lower the limit: choose another episode.

Terminals 2 and 3 (hand tracker, dashboard) as in `docs/demo-chain-test.md`. The dashboard header must say `Robot program: running`, and the **💃 Dance** tile is green.

Click **💃 Dance**, type "let's dance", or press `d` + Enter in terminal 1. Expect:

1. **Rest**: a small move into the rest pose.
2. **Dance**:
   - About 4 s of slow fold-out to the first pose: arm up over the base, gripper pointing up, like the present pose before a handover.
   - Then about 40 s of dancing, the gripper half open and still.
3. **Rest**: it folds back. The round shows under **Rounds** as `danced`, and the dashboard films it like any round.

**Watch the fold-out.** If the arm leans towards the table or the bowl, or the wrist bends the opposite way from the dataset video, press **ABORT**: the recording does not match our arm.

**Stop:** ABORT on the page, or `a` + Enter in terminal 1. The arm stops where it is with torque on. Hold the arm, then switch the power off.

## 3. Afterwards

- Episode 1 looked safe → restart with both dances; each dance command plays the next one in turn:
  `--dance-episode NikeSong/so101_gestures_dance:1 --dance-episode NikeSong/so101_gestures_dance:2 --dance-speed 0.5 --dance-max-step-deg 4`
- Both looked safe → try `--dance-speed 0.75`. Raise `--dance-max-step-deg` to 6 only if the dance looks smoothed out.
- Post on the board (issue #1) how it looked, with the round video.
