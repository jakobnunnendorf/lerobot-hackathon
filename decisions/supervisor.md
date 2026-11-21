# How does the demo chain its skills, and who owns the robot while it does?

- Status: OPEN (2026-11-21, Jakob's agent, `scripts/supervisor/`). Built and tested with fakes and a simulated arm only; nothing has run on the arm, a camera or a real policy checkpoint.

## Decision

A thin state machine in one process that owns the robot connection (both cameras included) for the whole demo:

REST -> [LOOK, optional] -> PICK -> VERIFY_HOLD -> GIVE (hand) or PUT_DOWN (no hand, or voice says "container") -> REST -> DONE, with ABORT reachable from every state.

- **PICK is one run of `molnrb/act_pick_ball_combined`, also when the ball is buried.** That policy was trained on visible and buried episodes, and the buried ones dig and pick in one episode (molnrb's agent, board 7061614734). Balazs (board 7061650334): the combined policy grabs and digs end to end, "so no supervisor is needed for that part". So the supervisor does not split picking into look/dig steps.
- **REST is the folded rest pose, not a HOME above the bowl.** Every pick episode starts and ends there, holding the ball at the end (board 7061614734). In robot mode it is the pose at start, or `--rest`.
- **VERIFY_HOLD** takes the median gripper reading over 0.5 s with the arm still: 10-17 on the 0-100 scale means holding (the ball reads about 11-16, empty about 0; board 7061614734, not measured by us). Not holding: back to REST and PICK again, at most `--max-pick-attempts` (3) picks, then the round ends as `gave_up`.
- **Destination:** voice or keyboard "container" -> PUT_DOWN. Otherwise wait up to 3 s for a hand from the tracker; hand -> GIVE, none -> PUT_DOWN.
- **GIVE** moves in joint space (gripper held) to `GIVE_START` = `[0, -55.3, 5.6, 91.8, 0]` (tip about (0.15, 0, 0.20) m, board 7061460387), then runs the #13 hand-following loop. It ends when the loop has opened the gripper (`dropped`), when the loop gives up on the hand (`hand_lost`, after its own 3 s) or on timeout (45 s). Still holding after GIVE -> back to REST holding -> PUT_DOWN.
- **PUT_DOWN** is a replay of a recorded episode (`--put-down-episode REPO:EP`) or, until one exists, a placeholder: carry the ball in joint space to `--put-down-pose` and open there. Still holding afterwards -> ABORT.
- **LOOK** (`--look`) is a log-only pre-check: the median mask area of `scripts/recolour/target_mask.py` over 5 top-camera frames, threshold 150 px (`decisions/target-visibility-check.md`). It never stops a pick. With `--clear-gray-policy` it also routes "not visible" to CLEAR_GRAY (REST to REST), at most `--max-clear-attempts` (3) times, then PICK digs anyway. Off by default: no CLEAR GRAY dataset exists (board 7061614734).
- **SHAKE** has a hook (a state after GIVE, `Skills.shake`) but no skill. The team put the handshake after everything else (board 7061650334).
- **Limits:** a time budget per skill (PICK 40 s, CLEAR GRAY 30 s, GIVE 45 s, PUT DOWN 20 s, moves 15 s). A policy never says it is done, so PICK also ends early when the arm has been within 8 degrees of REST with the gripper in the holding range for 1 s (after at least 3 s).
- **Safe states:** the gripper goal is never raised while the gripper reads holding, except inside GIVE (the #13 loop's drop) and PUT_DOWN. ABORT (key `a` + Enter, Ctrl-C, or any skill error) sends nothing more: the servos keep their last goal, torque stays on (`disable_torque_on_disconnect=False`), and the person at the table takes the arm.
- **Log:** one JSONL line per transition and one `round_end` line per round with `outcome` (`given`, `put_down`, `gave_up`, `aborted`), pick and clear attempts and the state list, for the H5/H7 counts.

## Robot ownership

- **One process, one connection, for the whole demo.** The supervisor connects the follower with both cameras once; every skill (policies, replays, the #13 loop) is called in that process and uses that connection. The hand tracker gets top-camera frames from the supervisor through `FrameWriter` (every second tick, also while waiting for a hand), as #13 does.
- **Handing the arm to the #13 loop:** the supervisor calls `handover_loop.run()` in its own thread with an adapter for the arm, passing the GIVE start pose as the loop's HOME and the last commanded gripper value as its hold value. It stops the loop by raising from the loop's `on_tick` callback. The supervisor sends nothing while the loop runs, and the loop sends nothing after it returns. #13's files are imported, not changed.
- **Not a subprocess:** handing over to a second process means `disconnect()` here and `connect()` there, and `connect()` runs `configure()` with torque off on all six motors (`research/lerobot-multi-skill-runner.md`, section 1). That drops the ball held at that moment.
- **The loop's HOME and REST are reconciled by the GIVE start pose.** The loop moves the tip in clamped Cartesian steps (0.12-0.38 m from the pan axis, 0.08-0.30 m high). Tried in the simulation (2026-11-21, `tests/supervisor_scenarios.py`) with the loop's HOME set to a folded rest pose (`[0, 20, 60, 90, 0]`, tip 3.4 cm behind the pan axis, 6.3 cm high): the arm never moved and the round ran into the 45 s timeout. With the joint-space move to `GIVE_START` first, the loop follows and drops, and the supervisor moves back to REST in joint space: given in 8.5 s.

## Alternatives

- **GIVE as a subprocess running `handover_loop.py --robot`** - rejected: drops the ball (above).
- **Copying the loop's tick body into the supervisor** - rejected: the safety rules would exist twice and drift. Raising from `on_tick` keeps the loop unchanged.
- **LOOK -> CLEAR GRAY -> LOOK as the default flow** (the first plan, `ideas/ball-picking-and-handover.md`) - replaced by one PICK run (above). Kept behind `--clear-gray-policy`.
- **Voice through our own stand-in listener on port 8765** - replaced by Tian's `CommandReceiver` from `scripts/voice/link.py` once #16 was merged; polled once per tick.

## Still unknown

- Whether PICK succeeds often enough to chain anything: no counts yet (molnrb's 20 runs are running).
- The real rest pose, and whether the joint-space moves between REST and `GIVE_START` clear the table, bowl and base. With the guessed folded rest pose the tip passes 2.9-3.4 cm above the table, over the base. Watch the first REST -> GIVE_START move at the table with a hand on the power switch, and change `--give-start` (or add a waypoint) if it comes close.
- The holding range on our gripper (10-17 is from board 7061614734). A gray foil ball may read in the same range.
- Whether the top camera sees the bowl with the arm at REST (LOOK assumes so).
- PUT_DOWN: no recorded episode yet; the placeholder pose needs joint values from the table.
- GIVE was only run on a simulated arm with fake hands; the #13 loop itself is tested in dry run only (`decisions/hand-following-safety.md`).
- The policy runner was tested with a randomly initialised ACT checkpoint saved and loaded on the Mac, not with `molnrb/act_pick_ball_combined` or on a GPU.

## Revisit if

- The 20-run counts show that the combined policy cannot dig: record CLEAR GRAY and enable it.
- A handshake skill is recorded: plug it into `Skills.shake`.
- The table check shows the REST <-> GIVE_START move comes close to the table or the bowl.
