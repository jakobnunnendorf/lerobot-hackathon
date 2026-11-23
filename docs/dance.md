# Dance

The arm plays back a dance recorded on another SO-101 (`NikeSong/so101_gestures_dance`, episodes 1 and 2). It starts from rest with an empty gripper, folds out over its base, dances, and folds back to rest. It runs inside the demo program (`docs/demo.md`). Why it works this way: `decisions/dance.md`.

## Once: fetch the dance

Download it before the first run, so the robot program does not fetch about 400 MB of video while the arm waits:

```bash
conda activate lerobot
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; D('NikeSong/so101_gestures_dance', episodes=[1, 2])"
```

## Safety, before every start

- Clear about 30 cm around the arm: bowl, balls and cables out of that circle. The dance sways over the base, at most 20 cm out, and turns the base up to about 35 degrees either way.
- Someone's hand on the power switch for the whole dance. Nobody leans over the arm.
- The gripper is empty and the arm is folded at rest. The program refuses to dance while the gripper reads 6-20, and the dashboard greys the tile out.

## Start the robot program with the dance

The dance is only offered when the program is started with `--dance-episode`. Append the dance flags to the usual demo command; each dance command plays the next episode in turn:

```bash
PICK_POLICY=TianK003/act_hidden_from_v2 bash scripts/supervisor/demo_chain.sh --step 002000 --n-action-steps 25 \
  --dance-episode NikeSong/so101_gestures_dance:1 --dance-episode NikeSong/so101_gestures_dance:2 \
  --dance-speed 0.5 --dance-max-step-deg 4
```

The start lines show each dance, e.g.:

```text
dance NikeSong/so101_gestures_dance:1: 599 frames at speed 0.5 -> 39.9 s, speed cap 4.0 deg/tick
```

- `--dance-speed` is the playback pace, at most 1 (the recorded pace). Start at 0.5; try 0.75 once it looks safe.
- `--dance-max-step-deg` caps each joint's change per tick. Raise it to 6 only if the dance looks smoothed out.
- The program refuses an episode whose tip would come below `--dance-min-tip-height` (0.20 m) by our kinematic model. Do not lower the limit: choose another episode.

Start the hand tracker and the dashboard as in `docs/demo.md`. The **💃 Dance** tile turns green.

## Run it

Click **💃 Dance**, say or type "let's dance", or press `d` + Enter in terminal 1.

1. **Rest**: a small move into the rest pose.
2. **Dance**: about 4 s of slow fold-out to the first pose (arm up over the base, gripper pointing up), then the dance with the gripper half open.
3. **Rest**: it folds back. The round shows as `danced` and the dashboard films it.

**Watch the fold-out.** If the arm leans towards the table or the bowl, or the wrist bends the opposite way from the dataset video, press ABORT: that episode does not match our arm.

**Stop:** ABORT on the page, or `a` + Enter in terminal 1. The arm stops where it is with torque on. Hold the arm, then switch the power off.

## Music

When the dance starts, the dashboard page plays a 10 s chiptune on the speakers of the computer that shows it. Click anywhere on the page once after loading it, or the browser keeps it muted. **🔊 Dance music** in the header turns it off. To play your own clip, save it as `scripts/dashboard/static/music/dance.mp3` (git-ignored) and reload the page. Why: `decisions/dance-music.md`.
