# Finger Biting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A BITE round in the demo chain: a pointing finger seen by the top camera makes the arm turn to it and snap the gripper, started from the dashboard tile, voice/chat ("bite the finger") or key `b`.

**Architecture:** The MediaPipe tracker adds a `pointing` flag and the index fingertip's position to each hand line; `HandReceiver` exposes them. A new `BiteSkill` (a supervisor `Skill`, like the dance's `Replay`) runs in the supervisor's 30 Hz loop on its one robot connection. Voice, the link, the dashboard and the fake robot learn a `{"bite": true}` command exactly the way they learned `{"dance": true}`.

**Tech Stack:** Python 3.12, numpy, MediaPipe Tasks (tracker env), LeRobot 0.7.0 (supervisor), FastAPI + vanilla JS (dashboard), rapidfuzz (voice).

**Spec:** `docs/finger-biting-design.md` (read it first; every default below comes from it).

## Global Constraints

- Follow `CLAUDE.md`: no comments except about decisions, no docstrings that repeat the signature, no commented-out code; clear names, small functions.
- Never run a command that moves the robot. No `lerobot-*` CLIs, no `--robot`.
- All tracker fields are **additive**; `HandReceiver.newest_hands()`, `newest_palm_count()`, GIVE and `--only-open` behave exactly as before.
- A bite can only happen inside a BITE round; the robot program offers one only when started with `--bite`.
- Never `git add -A`; add only the files you own. Commit on your own branch.
- Test runners (no uv on this laptop):
  - tracker env (numpy + mediapipe): `C:/Users/Tian/Desktop/lerobot-hackathon/data/handtrack/.venv/Scripts/python.exe -I`
  - dashboard/voice env: `C:/Users/Tian/.claude/jobs/012257a3/tmp/dash-venv/Scripts/python.exe -I`
  - LeRobot env (supervisor tests): `C:/Users/Tian/.claude/jobs/012257a3/tmp/lerobot-venv/Scripts/python.exe -I`
- Baseline (all PASS before any change): `tests/voice_parse.py`, `tests/handover_hand_socket.py`, `tests/dashboard_smoke.py` (dash env), `tests/supervisor_scenarios.py` (LeRobot env). They must still pass.

## Shared contracts (every task relies on these exact names)

- **Tracker JSON, per hand (additive):** `"pointing": bool`, `"tip_u": float`, `"tip_v": float` always; `"tip_x"`, `"tip_y"`, `"tip_z"` (metres, robot frame, fingertip intersected with plane `--finger-z`, default 0.15) only when calibrated and the ray hits the plane. Pointing hands are not open, so they appear in `hands` with `"open": false`.
- **`HandReceiver.newest_pointing_tips() -> list[np.ndarray]`** (`scripts/handover/hand_socket.py`): xyz of every hand with `pointing` true and all three `tip_*` present, from the newest message, `[]` when older than `max_age_s`. Ignores `open_only`.
- **Command line message:** `{"bite": true, "transcript": str, "source": str}` on the command port (8765).
- **Robot status / log:** `STATUS` `available["bite"]: bool`; `round_start` event carries `bite=True`; state name `"BITE"`; round outcomes `"bit"` (≥1 bite), `"no_finger"` (window ran out with no bite), `f"bite_{result}"` otherwise.
- **Voice:** `commands.wants_bite(text: str) -> bool`.

## Review Focus

1. A pointing finger seen during a PICK or GIVE round must do nothing, and a bite asked for during a round must not start after it (cleared like the dance) — Task 3 scenario test.
2. A tracker line with a pointing hand but no calibration (no `tip_x`) or with garbage tip fields must not break palms or the receiver — Task 1 socket test.
3. A finger outside the bite workspace (too low, too far, behind the pan limit) must never move the arm — Task 3 `finger_bite` test.
4. The finger vanishing mid-aim (the arm hides it) must not freeze the round forever — Task 3 test: lost > 1 s before the freeze → READY; after the freeze the lunge goes on.
5. "Bite" in a sentence that also asks for a ball or a dance must be refused, not guessed — Task 2 voice test.

---

### Task 1: Tracker sends pointing and the fingertip; receiver exposes them

**Files:**
- Modify: `scripts/handtrack/hand_features.py`, `scripts/handtrack/hand_tracker.py`, `scripts/handover/hand_socket.py`
- Create: `tests/finger_pointing.py`
- Modify: `tests/handover_hand_socket.py`

**Interfaces:**
- Produces: `hand_features.describe()` adds `"straightness": {"index","middle","ring","pinky": float}`, `"index_tip_px": (u, v)`, `"index_direction_px": (dx, dy)` (unit vector landmark 5 → 8). `hand_features.is_pointing(hand: dict, curled_below: float = 0.75) -> bool`. The tracker JSON fields and `HandReceiver.newest_pointing_tips()` from "Shared contracts".

- [ ] **Step 1: Write `tests/finger_pointing.py`** (standalone script, prints PASS lines and a final `PASS`, exits 1 on failure, like the other `tests/*.py`). Build fake MediaPipe results (objects with `.x .y .z`, handedness with `.category_name .score`) from synthetic 21-point hands where each finger chain is straight or curled. Assert: pointing (index straight, others curled) → `is_pointing` True; open palm (all straight) → False; fist (all curled) → False; victory (index + middle straight) → False; thumb straight or curled does not matter; `describe()` returns `index_tip_px` equal to landmark 8 in pixels. If `data/handtrack/downloads/images/` holds a `pointing_up` photo and `data/handtrack/downloads/models/hand_landmarker.task` exists, run MediaPipe on it and assert `is_pointing` True; otherwise print `SKIP`.
- [ ] **Step 2: Run it** with the tracker env: expect FAIL (no `is_pointing`).
- [ ] **Step 3: Implement** `is_pointing` (index straightness > 0.9, the `extended_fingers` threshold; middle, ring, pinky each < `curled_below`) and the new `describe()` keys in `hand_features.py`.
- [ ] **Step 4: Tracker fields.** In `hand_tracker.py`: `--finger-z` (float, default 0.15, "assumed fingertip height above the table, metres"); `pixel_message` adds `pointing`, `tip_u`, `tip_v`; `hand_message` adds `tip_x/y/z` from `geometry.pixel_to_base(*hand["index_tip_px"], finger_z)` when not None (the palm stays mandatory as today). `summary()` says `pointing` for pointing hands. `--only-open` unchanged.
- [ ] **Step 5: Extend `tests/handover_hand_socket.py`** (keep its style): a line with a pointing, not-open hand with tip fields → `newest_pointing_tips()` has its tip and `newest_hands()` (open_only) is empty; a pointing hand without `tip_x` → no tip; a hand with `"tip_x": "garbage"` → that tip is skipped and an open palm in the same line still arrives in `newest_hands()`; stale message → `[]`.
- [ ] **Step 6: Implement `newest_pointing_tips()`** in `hand_socket.py`: parse tips in `_store` separately from the palms, so a bad tip never drops the palms.
- [ ] **Step 7: Run** `tests/finger_pointing.py` (tracker env), `tests/handover_hand_socket.py` (dash env): PASS.
- [ ] **Step 8: Commit** `feat(handtrack): pointing flag and fingertip position for finger biting`.

### Task 2: Voice, link, dashboard and fake robot learn the bite command

**Files:**
- Modify: `scripts/voice/commands.py`, `scripts/voice/link.py`, `scripts/voice/listen.py`
- Modify: `scripts/dashboard/server.py`, `scripts/dashboard/static/app.js`, `scripts/dashboard/fake_robot.py`
- Modify: `tests/voice_parse.py`, `tests/dashboard_smoke.py`

**Interfaces:**
- Consumes: the command message and status contract from "Shared contracts" (Task 3 produces them in the real robot program; `fake_robot.py` produces them here).
- Produces: `commands.wants_bite(text) -> bool`; `POST /api/bite`; chat/🎤 routing.

- [ ] **Step 1: Voice tests** in `tests/voice_parse.py` (its table style): `wants_bite` True and `parse` None for "bite the finger", "Bite my finger!", "bite", "byte the finger", "biting time"; False for "give it to me", "put it in the container", "let's dance"; "bite and give it to me" and "dance and bite" → `wants_bite` False, `wants_dance` False, `parse` None.
- [ ] **Step 2: Run** (dash env): FAIL.
- [ ] **Step 3: Implement** in `commands.py`: `BITE_WORDS = {"bite", "bites", "biting", "byte"}` (whole words, same reason as `DANCE_WORDS`); `wants_bite` = a bite word, no dance word, no destination match; `parse` returns None when a bite word is present; `wants_dance` returns False when a bite word is present. `link.decode` accepts `{"bite": true}`. `listen.py` sends `{"bite": true, "transcript", "source": "voice"}` when `wants_bite`, mirroring its dance path.
- [ ] **Step 4: Run** voice tests: PASS.
- [ ] **Step 5: Dashboard tests** in `tests/dashboard_smoke.py` (existing fake-robot harness): fake robot with bite → `POST /api/bite` ok and the fake robot runs a round with states `REST, BITE, REST, DONE` and outcome `bit`; chat text "bite the finger" → delivered `{"bite": true}`; fake robot started `--no-bite` → `/api/bite` refused with a reason containing `--bite`; a busy arm refuses as for the dance.
- [ ] **Step 6: Implement** `server.py`: `bite_refusal()` ("no bite in this robot program: start it with --bite"; gripper holding → "the gripper holds a ball: empty it before biting", same check as `dance_refusal`), `send_bite(transcript, source)`, `wants_bite` routing before `parse` in the text handler, `@app.post("/api/bite")`, round summary label `"bite"` when the round's start has `bite`. `app.js`: `BITE_FLOW = ["REST", "BITE", "DONE"]`, `STATE_NAMES.BITE = "Bite"`, a **🦷 Bite** tile greyed with `biteReason()` like the dance tile (title "Point at the gripper and it bites, from rest, gripper empty"), `startBite()`, round label `🦷 bite`, reply `▶ Starting: bite`. `fake_robot.py`: `run_bite()` like `run_dance()` (`round_start` with `bite=True`, a few fake poses, outcome `bit`), `BUDGETS["BITE"] = 30.0`, `--no-bite`, `available["bite"]`, `{"bite": true}` dispatch.
- [ ] **Step 7: Run** `tests/voice_parse.py` and `tests/dashboard_smoke.py` (dash env): PASS.
- [ ] **Step 8: Commit** `feat(dashboard,voice): bite command, tile and "bite the finger"`.

### Task 3: BiteSkill and the BITE round in the supervisor

**Files:**
- Create: `scripts/supervisor/bite.py` (the skill; keeps `skills.py` from growing)
- Modify: `scripts/supervisor/commands.py`, `scripts/supervisor/machine.py`, `scripts/supervisor/run_supervisor.py`, `scripts/supervisor/fakes.py`
- Create: `tests/finger_bite.py`
- Modify: `tests/supervisor_scenarios.py`

**Interfaces:**
- Consumes: `receiver.newest_pointing_tips()` (Task 1; in tests use a fake receiver with the same method). From `--handover-dir` via `skills.import_from`: `kinematics.forward_kinematics(lerobot_deg) -> xyz`, `kinematics.solve(xyz, pitch_deg, wrist_roll_deg, margin_deg=...) -> URDF deg (5,)` raising `kinematics.Unreachable`, `kinematics.lerobot_degrees(urdf)`, `kinematics.urdf_degrees(lerobot)`. **`solve` works in URDF degrees; convert.**
- Produces: `BiteSettings` (dataclass) and `BiteSkill(kinematics, receiver, settings: BiteSettings, ready_pose: np.ndarray)` implementing `skills.Skill` (`name`, `start`, `step`, `done`, `result`; `result()` returns `"bit"` or `"no_finger"`). `Skills.bite: Callable[[], Skill] | None`. `Commands.request_bite()`, `take_bite() -> bool`. `Machine.run_round(number, dance=False, bite=False)`.

`BiteSettings` fields and defaults (from the spec): `mode="aim"` (`"aim"|"zone"`), `ready_pose` comes separately (6 values incl. gripper), `gripper_open=60.0`, `gripper_close=5.0`, `window_s=30.0`, `max_bites=3`, `standoff_m=0.06`, `overshoot_m=0.01`, `zone_m=0.08`, `zone_ahead_m=0.04`, `min_height_m=0.15`, `min_radius_m=0.20`, `max_radius_m=0.38`, `max_pan_deg=45.0`, `seen_frames=5`, `seen_of=8`, `steady_m=0.03`, `steady_s=0.3`, `freeze_within_m=0.10`, `lost_s=1.0`, `step_m=0.008`, `max_step_deg=1.5`, `lunge_step_deg=4.0`, `lunge_timeout_s=0.6`, `snap_s=0.5`, `release_s=0.4`, `cooldown_s=1.5`, `limit_margin_deg=3.0`.

Phases (spec table): READY → WATCH → AIM (aim mode) → LUNGE → SNAP → RELEASE → COOLDOWN → WATCH …, done when `max_bites` reached or the window runs out (window counts from the first WATCH). Every arm target goes through: pitch 0 IK with the ready pose's wrist roll, the workspace check (tip z ≥ `min_height_m`, radius in range, pan within `max_pan_deg` of the ready pose's pan) and a per-tick joint step limit; a step that would bring the tip below `min_height_m` or inside `min_radius_m` is shortened like `handover_loop.guarded_step`. Fingertip filter: median of the last 5 tips; frozen once the tip is within `freeze_within_m`. Aim point = fingertip pulled `standoff_m` toward the pan axis (radially, same height); lunge point = fingertip pushed `overshoot_m` past it radially. Zone mode: WATCH fires only for a tip within `zone_m` of the point `zone_ahead_m` in front of the ready-pose tip along its radial direction; LUNGE then goes radially from the ready pose to that tip + overshoot.

- [ ] **Step 1: Write `tests/finger_bite.py`** (standalone, numpy only plus `scripts/handover/kinematics.py`; drive `BiteSkill` with a simulated arm that follows commands exactly, `now` advanced 1/30 s per tick, and a fake receiver whose tips you script). Ready pose: pick one with the tip horizontal about (0.28, 0, 0.15) using `kinematics.solve` + `lerobot_degrees` + gripper 60. Assert:
  - fingertip at (0.30, 0.05, 0.15) held still: the tip passes within 1 cm of the aim point, then reaches within 1.5 cm of the lunge point, the gripper command goes to 5 then back to 60, `result()` is `"bit"` after `max_bites=1`.
  - fingertip at z 0.08 (below the floor), at radius 0.45, or at pan 80° from the ready pose: the arm never leaves the ready pose; after `window_s` (set to 2 s in the test) `result()` is `"no_finger"`.
  - no commanded tip ever below `min_height_m - 0.005` and no joint step above the phase's limit (+1e-6), over all runs.
  - finger disappears during AIM before the freeze for > 1 s → back to READY/WATCH; disappears after the freeze → the lunge still happens.
  - tips seen in only 3 of 8 frames → no aim.
  - `max_bites=3` with the finger held → exactly 3 snaps, then done.
  - zone mode: a tip 4 cm ahead of the jaws → bite without any pan change; a tip 20 cm to the side → no motion.
- [ ] **Step 2: Run** (dash env has numpy): FAIL (no module `bite`).
- [ ] **Step 3: Implement `scripts/supervisor/bite.py`** to pass. Keep the phase logic in small methods; the `# why` comments only where the spec gives a reason (freeze: the arm hides the finger; radial lunge: pitch 0 keeps the gripper in the pan plane; gripper close value: position control squeezes by the gap).
- [ ] **Step 4: Run** `tests/finger_bite.py`: PASS.
- [ ] **Step 5: Scenario tests** in `tests/supervisor_scenarios.py` (use `fakes.py`, add a fake bite skill or a `BiteSkill` on a fake receiver): a bite round `REST → BITE → REST → DONE`, outcome `bit`; refused with `"unsupported_bite"` before moving when `Skills.bite` is None; refused when the gripper reads holding; a `request_bite()` during a PICK round is cleared at the round end and starts nothing; key `b` and link `{"bite": true}` both call `request_bite`.
- [ ] **Step 6: Wire it.** `commands.py`: `_bite`, `request_bite`, `take_bite`, clear in `clear`, key `"b"`, link `{"bite": true}` in `poll_voice`, comment line updated. `machine.py`: `State.BITE`, `Settings.bite_timeout_s = 45.0`, `Skills.bite`, `Round.bite: Skill | None`, `run_round(..., bite=False)` mirroring the dance path (round_start `bite=True`, refusal `"unsupported_bite"`, empty-gripper check at REST before and holding check after, `_bite` state: `run_skill(self.round.bite, self.s.bite_timeout_s)`, outcome from `result()`: `"bit"`, `"no_finger"`, else `f"bite_{result}"`, then REST). Wherever the main loop calls `take_dance()` to start a dance round, do the same for `take_bite()`. `run_supervisor.py`: `--bite` (store_true), `--bite-mode {aim,zone}`, `--bite-ready` (6 floats: 5 joints + gripper; required with `--bite`), `--finger-z` is the tracker's flag, not here; flags for `--bite-window`, `--bite-max`, `--bite-close`, `--bite-open`, `--bite-standoff`, `--bite-min-height`, `--bite-max-pan`, `--bite-lunge-step-deg`; build `Skills.bite` only with `--bite` and a receiver and `kinematics.py` (otherwise exit with a clear message when `--bite` was given); `STATUS.update(available={..., "bite": bool})` next to `"dance"`; log the bite settings in the `start` event. Also cap the bite's steps by `--max-step-deg` when it is lower (as `dance_cap`). `fakes.py`: whatever the scenarios need.
- [ ] **Step 7: Run** `tests/finger_bite.py` (dash env) and `tests/supervisor_scenarios.py` (LeRobot env): PASS.
- [ ] **Step 8: Commit** `feat(supervisor): BITE round with BiteSkill (aim and zone modes)`.

### Task 4: Merge, runbook, decision record, final checks (main session)

**Files:**
- Create: `docs/finger-biting.md` (runbook for the person at the table, from the spec's section 5, with the exact demo-chain command plus `--bite --bite-mode zone --bite-ready <values> --max-step-deg ...` and the tracker command plus `--finger-z`)
- Create: `decisions/finger-biting.md` (the Approaches table and the calls from the spec, in `decisions/README.md` format)
- Delete before the PR: `docs/finger-biting-design.md` and this plan (folded into the decision record; the repo keeps decisions, not design docs)

- [ ] **Step 1:** Merge the three task branches into `TianK003/finger-biting`; resolve conflicts.
- [ ] **Step 2:** Run every test listed under Global Constraints plus `tests/finger_pointing.py` and `tests/finger_bite.py`: all PASS.
- [ ] **Step 3:** Write the runbook and the decision record; remove the design and plan files.
- [ ] **Step 4:** Commit, push, whole-branch review, then tell Tian; PR after his go.
