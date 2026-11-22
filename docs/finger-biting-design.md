# Finger biting: design

- Status: DRAFT for Tian's approval (2026-11-22). Experimental add-on: it is tested at the table and dropped if it works badly.
- Branch: `TianK003/finger-biting`.

## Intent

A funny extra for the demo, aimed at kids: "don't point your finger at people". When someone points at the gripper (index finger extended, the other fingers curled), the arm turns to face the finger and snaps at it with the gripper, at the gripper's normal, already reduced strength. It is started like the dance: a **🦷 Bite** tile on the dashboard, the words "bite the finger" said to the 🎤 or typed in the chat, or a key in the robot program's terminal.

It is not important to the main demo. Everyone else is training, so there is time to build it, but it must never put the PICK → GIVE demo at risk.

## Goals

- A BITE round in the demo chain program (`scripts/supervisor/run_supervisor.py`), next to DANCE: from REST with an empty gripper, watch for a pointing finger, turn to face it, lunge and close, open, back to REST.
- Started from the dashboard tile, from voice/chat ("bite the finger"), and from the terminal key `b`.
- Pointing detected with MediaPipe in the existing hand tracker, by the same landmark geometry used for "open palm".
- A fallback mode without aiming (bite zone, see Approaches), chosen by a flag, in case aiming is poor at the table.

## Non-goals

- Accurate biting. "Tries to bite" is enough; a miss that snaps next to the finger is still funny.
- Checking that the finger points *at the gripper*. The rule is the one Tian gave: index extended and the other three fingers not. The direction check is listed under "Later".
- A learned policy, new datasets, or a new speech model.
- Choosing the strength in software. Grip force was judged not important (Tian).

## Anti-goals (a failure even if biting works)

- Any change to how PICK, GIVE, PUT DOWN, DANCE or the palm receiver behave. All new tracker fields are additive; GIVE must ignore them.
- A bite started by accident: a pointing finger seen during a PICK or GIVE round must do nothing. Biting happens only inside a BITE round somebody asked for.
- The arm sweeping low over the bowl, or hitting the table.

## Constraints (from the repo)

- **One robot connection.** The demo chain owns the arm and the top camera; anything that moves the arm runs inside it as a `Skill` (`skills.Skill`: `start`, `step`, `done`, `result`), like `Replay` for the dance (`decisions/dance.md`, `decisions/supervisor.md`).
- **Tracker in its own process and environment** (MediaPipe's OpenCV clashes with LeRobot's). It reads the top-camera frames through shared memory and sends one JSON line per frame to `127.0.0.1:8766` (`scripts/handtrack/hand_tracker.py`). In the demo it runs **without** `--only-open`, so pointing hands already reach the robot side; `HandReceiver(open_only=True)` filters them out for GIVE.
- **Single camera, no depth.** Positions come from intersecting the pixel ray with an assumed height. The top camera is at about (0.52, -0.56, 0.88) m in the robot frame (`scripts/calib/extrinsics.yaml`), about 47° from vertical, so a fingertip 5 cm higher than assumed lands about 5 cm off along the camera's line of sight (`research/hand-tracking-and-camera-calibration.md`, section 2).
- **Kinematics.** `scripts/handover/kinematics.py` has `solve(target, pitch_deg, wrist_roll_deg, ...)` with a fixed gripper pitch. With the gripper horizontal (pitch 0, roll 0) the tip reaches 20-42 cm from the pan axis at 10-20 cm above the table, 13-39 cm at 25 cm (computed 2026-11-22 with the team's IK, 3° limit margin). At roll 0 the jaws open top and bottom (the moving jaw is the upper one), so a horizontal gripper looks like a mouth to a finger pointed at it.
- **Strength.** LeRobot 0.7.0 already limits the gripper motor at connect (`Max_Torque_Limit` 500 of 1000, `Protection_Current` 250; `so_follower.py`). That is the "50%".
- **The gripper is position-controlled.** The squeeze comes from the gap between the commanded close value and the finger's width.
- **Never run commands that move the robot** (CLAUDE.md). Agents build and dry-run; the person at the table runs it on the arm.
- **After the feature freeze** `main` is the demo branch: merging needs Tian's yes, and code stops at 10:00 on Sunday (`docs/roadmap.md`). This work stays on its branch until then.

## Approaches

| | How | Trade-off | Verdict |
| --- | --- | --- | --- |
| **A. Bite zone, no aiming** | Arm waits in a fixed "mouth" pose (gripper horizontal, open). Bites when a pointing fingertip is within a few cm in front of the jaws. | Hardly depends on calibration; the arm never goes after the finger, the kid has to come to it. | Kept as a flag (`--bite-mode zone`). |
| **B. Turn, then lunge** | Wait in the mouth pose; when a pointing fingertip is seen, turn to face it, stop short at its estimated height, then lunge to the last seen position and close. | Looks alive; depends on the calibration and the assumed finger height. The lunge is blind, which matches reality: the arm hides the finger at the end anyway. | **Chosen** (`--bite-mode aim`, the default). |
| C. Track the finger continuously | The palm follower aimed at the fingertip, bite when close. | The follower is built for hovering over a palm from above; it goes blind on the final approach anyway; most work. | Dropped. |

B is a superset of A: with aiming switched off, B becomes A. Tian handed the choice to the agent.

## How it works

### 1. Tracker: pointing and the fingertip (`scripts/handtrack/`)

- `hand_features.py`: `is_pointing(hand)`: index straightness above 0.9 (the existing `extended_fingers` threshold), middle, ring and pinky all below a "curled" threshold (start at 0.75, tune at the table), thumb ignored. `describe()` also returns the index fingertip pixel (landmark 8) and the index direction in the image (landmark 5 → 8).
- `hand_tracker.py`: per hand, additive fields `pointing` (bool), `tip_u`, `tip_v`, and with calibration `tip_x`, `tip_y`, `tip_z`, the fingertip intersected with the plane `--finger-z` (default 0.15 m, tune at the table). Existing fields and the `--only-open` behaviour stay as they are.
- `hand_socket.py`: `HandReceiver` keeps one more list, `newest_pointing_tips()`, fresh under the same 0.5 s rule. `newest_hands()` is unchanged.

### 2. BITE round (`scripts/supervisor/`)

A new state `BITE` and round flag, built like DANCE:

- `commands.py`: `request_bite()` / `take_bite()`, key `b`, link message `{"bite": true}`; cleared by `Commands.clear` like the dance, so a request during another round never starts later.
- `run_supervisor.py`: the bite skill exists only when the robot program is started with `--bite` (off by default, like the dance needs `--dance-episode`), plus the calibration and `kinematics.py`. Without it the tile is greyed out and voice, chat and the key are refused, so the main demo cannot start a bite by accident.
- `machine.py`: `run_round(number, bite=True)` → `REST → BITE → REST → DONE`. Refused before anything moves when there is no bite skill or the gripper reads holding (same check as the dance). Outcomes: `bit` (at least one bite), `no_finger` (window ran out), `bite_<result>` otherwise.
- `skills.py`: `BiteSkill`, one `Skill` with its own small state machine, in the 30 Hz supervisor loop:

| Phase | What the arm does | Leaves when |
| --- | --- | --- |
| READY | Speed-limited joint move from REST to the mouth pose (`--bite-ready`, 5 joints; default to be read at the table: gripper horizontal, about 28 cm out, 15 cm up, facing the play area). Gripper opens to `--bite-open` (default 60). | arrived |
| WATCH | Holds still. | a pointing fingertip seen in at least 5 of the last 8 frames, inside the bite workspace (below); or the window `--bite-window` (default 30 s) runs out → done |
| AIM (mode `aim` only) | Moves the tip in straight Cartesian waypoints (the handover loop's `Mover` idea: waypoint, clamp, IK at pitch 0, per-joint limit) to the **aim point**: the fingertip pulled `--bite-standoff` (default 6 cm) back toward the pan axis, at the fingertip's height. The fingertip is smoothed (median of the last 5) and **frozen** once the tip is within 10 cm of it, because the arm then hides it from the camera. | arrived at the aim point and the finger stayed (moved less than 3 cm) for 0.3 s; finger lost for over 1 s → back to READY |
| LUNGE | Fast straight move along the gripper axis (radial, since pitch 0 keeps the gripper in the vertical plane through the pan axis) to `--bite-overshoot` (default 1 cm) past the frozen fingertip. Joint limit raised to `--bite-lunge-step-deg` (default 4°/tick, 120°/s). | arrived, or 0.6 s |
| SNAP | Gripper to `--bite-close` (default 5) in one step; arm holds. | 0.5 s |
| RELEASE | Gripper back to `--bite-open`; arm backs off radially by the standoff. | 0.4 s |
| COOLDOWN | Back to the mouth pose (mode `zone`: stays there). | arrived and 1.5 s passed; then WATCH again, up to `--bite-max` bites (default 3) per round |

In mode `zone`, AIM is skipped. WATCH fires only when the fingertip is within `--bite-zone` (default 8 cm) of the point 4 cm in front of the jaws, and LUNGE goes straight forward from the mouth pose.

**Bite workspace (all targets clamped or refused):** tip at least `--bite-min-height` above the table (default 0.15 m; the bowl and balls are below that, grasps are at 0.02-0.16 m), 0.20-0.38 m from the pan axis, pan within `--bite-max-pan` of the mouth pose (default 45°). The person at the table chooses the mouth pose facing away from the bowl and containers, so this pan limit is the keep-out: the arm never swings round over the bowl. A target outside it is ignored (no aim, no lunge). Each joint step is guarded against the table and the base like `handover_loop.guarded_step`.

**ABORT** works as in every skill: `_observe` raises, nothing more is sent, the arm holds with torque on.

### 3. Dashboard and voice

- `scripts/voice/commands.py`: `wants_bite(text)`. Whole words, as for the dance (a short word scores high inside longer ones with fuzzy matching): `bite`, `bites`, `biting`, plus Whisper's likely mishearing `byte`. "Bite the finger", "bite my finger" and "bite" all match. A sentence that is also a ball command or a dance is refused, as with the dance. `parse()` returns None for it.
- `scripts/voice/link.py`: `decode` accepts `{"bite": true}`. `listen.py` sends it.
- `scripts/dashboard/server.py`: `send_bite`, `/api/bite`, `bite_refusal()` (no bite skill in this robot program, or the gripper holds a ball), routing in the text/voice handler next to `wants_dance`. Status `available.bite` comes from the robot program.
- `scripts/dashboard/static/app.js`: a **🦷 Bite** tile, greyed out with the reason like the dance tile; flow `REST → BITE → DONE`; the round list shows `🦷 bite`.
- `scripts/dashboard/fake_robot.py`: a fake bite round, so the dashboard can be shown without the arm.

### 4. Tests (no arm)

- `tests/finger_pointing.py`: `is_pointing` on synthetic landmark sets (pointing, open palm, fist, victory) and, if they are in `data/handtrack/downloads/`, the MediaPipe sample `pointing_up` photo.
- `tests/finger_bite.py`: `BiteSkill` against a simulated arm and scripted fingertip messages: aims to the standoff point, lunges past the fingertip, closes, releases, returns; ignores targets outside the workspace; never commands a tip below the minimum height; the finger lost mid-aim → READY; `--bite-max` respected; zone mode.
- Extend `tests/voice_parse.py`, `tests/supervisor_scenarios.py` (bite round, refusal with a held ball, request during a PICK round cleared) and `tests/dashboard_smoke.py` (tile, `/api/bite`, chat "bite the finger").

### 5. At the table (runbook `docs/finger-biting.md`, run by the person at the arm)

1. Tracker only, arm still: hold a pointing hand where the mouth pose will face, at bite height; check the top camera sees it there (not outside the frame, not behind the arm). Point at the camera; the tracker prints `pointing` and the fingertip position. Check it against a ruler at 2-3 spots; adjust `--finger-z` to where people actually hold their finger.
2. Read the mouth pose: put the arm by hand (torque off, `read_pose.py`) into a horizontal, open gripper facing the play area; note the joints for `--bite-ready`.
3. First run in mode `zone`, half speed, someone with a hand on the power switch, an adult finger first.
4. Then mode `aim`.
5. Post on the board: how often it reached the finger, how far off, any lost tracking.

## Decided by Tian

- The feature: pointing (index extended, others not) triggers a bite (2026-11-22).
- Grip force is not a concern; the existing gripper limit is fine.
- It is an optional demo add-on, kept only if it works well enough at the table.
- It is in the dashboard and can be started by voice, "bite the finger".
- There is time to build it while the others train; it will be tested at the arm when there is a slot. It stays on its branch and is merged only if Tian says so after the test (feature freeze rules).

## Calls made by the agent (Tian can overrule)

- Approach B with A as a flag.
- A BITE *round* started on request, like the dance, rather than always-on biting whenever someone points (always-on would fire during PICK and GIVE).
- Up to 3 bites per round within a 30 s window.
- Gripper horizontal ("mouth") with a radial lunge, not a top-down bite.
- Thresholds, distances and speeds above are starting values to tune at the table.
- Off by default behind `--bite`; minimum bite height 0.15 m; the pan limit around the mouth pose as the bowl keep-out (after the builder check).

## Left to the builder

- How `BiteSkill` reuses `handover_loop.Mover`/`guarded_step` (import from `--handover-dir` like `GiveSkill`, or a small copy).
- Exact names, log events, the tile's wording and emoji.
- Whether `pointing` hands without calibration are sent under `pixels` too (only for display; biting needs calibration).

## Later, if it works

- Only bite when the finger points at the gripper: compare the image direction of the index (5 → 8) with the direction from the fingertip to the gripper's projected position.
- Wrist-camera fine alignment before the lunge.
- A sound or LED "growl" before the snap.
