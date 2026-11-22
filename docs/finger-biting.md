# Finger biting: point at the gripper and it bites

For the people at the robot laptop. Experimental add-on: if it works badly, we drop it. Why it works this way: `decisions/finger-biting.md`.

When a BITE round runs, the arm holds its gripper horizontal and open like a mouth. Point at it (index finger straight, the other fingers curled) and it turns to your finger, stops short, lunges and snaps. Up to 3 bites in 30 s, then it folds back to rest. It runs inside the demo chain program, only when started with `--bite`. Start a round with the dashboard's **🦷 Bite** tile, by saying or typing "bite the finger", or with `f` + Enter in terminal 1.

**Team fingers only.** The roadmap says the gripper never closes on a visitor's hand. The bite closes at the gripper's normal strength.

## Before every run

- [ ] One person's hand on the power switch for the whole test.
- [ ] Gripper empty: a BITE round is refused while the gripper reads holding (6-20), and the tile is greyed out.
- [ ] Bowl, balls and cables out of the area the mouth faces. The arm stays within 45° of the mouth pose's direction and never goes below 15 cm above the table.

## 1. Tracker only, the arm does not move (about 5 min)

Stop the robot program. Terminal 2, the tracker with its own camera, as in `docs/hand-tracking-test.md` stage 1, plus the finger height:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate handtrack && cd ~/demo
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml \
    --hand-z 0.20 --finger-z 0.15 --camera
```

- Point with one hand, the finger about 15 cm above the table, in front of the arm where the mouth will face. Twice a second it prints the hand with `pointing`. An open palm or a fist must not say `pointing`.
- Check that the camera sees the finger there: no `no hand` while you point. If it only sees it elsewhere, choose the mouth direction (step 2) towards where it sees fingers.
- Hold the fingertip over a point you can measure. The tracker prints the palm's x and y. The bite uses the fingertip at `--finger-z`, so hold it at that height. If people naturally point higher or lower, change `--finger-z` to that height (in m) and keep it for step 3.

Ctrl+C.

## 2. Read the mouth pose (about 5 min, torque off)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

The torque goes off: hold the arm. Put it where it waits for fingers:

- gripper **horizontal**, jaws opening top and bottom (wrist roll as when it picks);
- the jaw end about **25-30 cm** out from the base axis and **15-20 cm** above the table;
- facing the play area where the tracker saw fingers, **away from the bowl**.

Write down the five joints it prints (`shoulder_pan` … `wrist_roll`). Ctrl+C, hold the arm and put it back to rest by hand.

## 3. First real run: zone mode, half speed

Terminal 1, the usual demo chain plus the bite flags. `<5 joints>` from step 2, then `60` (gripper open):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/demo
bash scripts/supervisor/demo_chain.sh --step 015000 \
  --bite --bite-mode zone --bite-ready <5 joints> 60 --bite-max 1 --bite-lunge-step-deg 2
```

- With a bad mouth pose it stops before the torque goes on: `--bite-ready: ...`. Read the pose again, higher or closer.
- The start line lists `f + Enter: bite`.

Terminal 2: the tracker reading the robot program's frames (step 1's command **without** `--camera`). Terminal 3: the dashboard as in `docs/demo-chain-test.md`; the **🦷 Bite** tile is green.

Start a round (tile, "bite the finger", or `f` + Enter). Expect:

1. The arm moves from rest to the mouth pose, gripper open.
2. An adult points a finger right in front of the jaws, about 4 cm ahead of them. The arm lunges straight forward, snaps, opens, backs off.
3. After the bite (or 30 s with no finger) it folds back to rest. The round shows as `bit` or `no_finger`.

**Stop:** ABORT on the page, or `a` + Enter in terminal 1. The arm stops where it is with torque on.

## 4. Aim mode

Same command with `--bite-mode aim` and without `--bite-max 1`. Point from 10-20 cm away, inside the area the mouth faces: the arm turns to the finger, stops about 6 cm short, waits until the finger is still for a moment, then lunges. Once it behaves, drop `--bite-lunge-step-deg 2` (default 4, faster snap).

Tuning, all flags of `run_supervisor.py`: `--bite-standoff` (stop distance, m), `--bite-window` (s), `--bite-max` (bites per round), `--bite-close` (gripper value of the snap, 0-100; higher means the jaws stop wider), `--bite-min-height`, `--bite-max-pan`.

## 5. Post on the board (issue #1)

- How often it reached the finger in zone and aim mode, and how far off it ended (cm, which direction).
- Whether the tracker lost the finger while the arm came near it (`bite_lost` lines in the log).
- The `--finger-z` and `--bite-ready` values that worked.
