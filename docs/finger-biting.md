# Finger biting (experimental)

**Experimental: it did not work reliably on our arm.** The dashboard shows the **🦷 Bite** tile greyed out with "coming soon: not working on the arm yet". The round can still be started with `f` + Enter in terminal 1, or by saying or typing "bite the finger", when the robot program runs with `--bite`. Why it works this way: `decisions/finger-biting.md`.

In a bite round the arm holds its gripper horizontal and open like a mouth. Point at it (index finger straight, the other fingers curled) and it turns to your finger, stops short, lunges and snaps. Up to 3 bites in 30 s, then it folds back to rest.

**Team fingers only.** The bite closes at the gripper's normal strength. Never offer it to visitors.

## Before every run

- Someone's hand on the power switch.
- Gripper empty: a bite round is refused while the gripper reads 6-20.
- Bowl, balls and cables out of the area the mouth faces. The arm stays within 45 degrees of pan from the mouth pose and never targets below 15 cm above the table.

## 1. Tracker only, the arm does not move

Stop the robot program. Run the tracker with its own camera:

```bash
conda activate handtrack
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task \
    --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml \
    --hand-z 0.20 --finger-z 0.15 --camera
```

- Point with one hand, the finger about 15 cm above the table, where the mouth will face. Twice a second it prints the hand with `pointing`. An open palm or a fist must not say `pointing`.
- Check that the camera sees the finger there (no `no hand`). Face the mouth towards where it sees fingers.
- The bite aims at the fingertip at `--finger-z`. If people naturally point higher or lower, set `--finger-z` to that height (m). It must be at least `--bite-min-height` (0.15), or every finger is outside the bite workspace.

Ctrl+C.

## 2. Read the mouth pose (torque off)

```bash
conda activate lerobot
python scripts/handover/read_pose.py --port /dev/so101_follower --robot-id mi_follower
```

Torque goes off: hold the arm. Put it where it waits for fingers:

- gripper **horizontal**, jaws opening top and bottom (wrist roll as when it picks);
- the jaw end about **25-30 cm** out from the base axis and **15-20 cm** above the table;
- facing the play area, **away from the bowl**.

Write down the five joints it prints. Ctrl+C, hold the arm and fold it back to rest by hand.

## 3. First run: zone mode, slow lunge

Terminal 1, the demo program plus the bite flags. `<5 joints>` from step 2, then `60` (gripper open):

```bash
conda activate lerobot
bash scripts/supervisor/demo_chain.sh \
  --bite --bite-mode zone --bite-ready <5 joints> 60 --bite-max 1 --bite-lunge-step-deg 2
```

With a bad mouth pose it stops before torque goes on (`--bite-ready: ...`). Read the pose again, higher or closer.

Terminal 2: the tracker reading the robot program's frames (step 1's command **without** `--camera`). Terminal 3: the dashboard, as in `docs/demo.md`.

Start a round (`f` + Enter, or "bite the finger"):

1. The arm moves from rest to the mouth pose, gripper open.
2. Point a finger right in front of the jaws, about 4 cm ahead of them. The arm lunges straight forward, snaps, opens, backs off.
3. After the bite, or 30 s with no finger, it folds back to rest. The round shows as `bit` or `no_finger`.

**Stop:** ABORT on the page, or `a` + Enter in terminal 1. The arm stops where it is with torque on, and the jaws stay as they are. **If a finger is caught, switch the power off**: with torque off the jaws let go.

## 4. Aim mode

The same command with `--bite-mode aim` (the default) and without `--bite-max 1`. Point from 10-20 cm away, inside the area the mouth faces: the arm turns to the finger, stops about 6 cm short, waits until the finger is still, then lunges. Once it behaves, drop `--bite-lunge-step-deg 2` (default 4, a faster snap).

Tuning flags of `run_supervisor.py`: `--bite-standoff` (stop distance, m), `--bite-window` (s), `--bite-max` (bites per round), `--bite-close` (gripper goal of the snap, 0-100; higher stops the jaws wider), `--bite-min-height`, `--bite-max-pan`. The log has a `bite_lost` line whenever the tracker lost the finger as the arm came near.
