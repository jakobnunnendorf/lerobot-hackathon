# How does the arm bite a pointing finger?

- Status: DECIDED (2026-11-22), built and tested in simulation only, not yet run on the arm. Experimental: dropped if it works badly at the table.

## Decision

A BITE round in the demo chain, offered only when the robot program starts with `--bite`, and started like the dance: the dashboard's 🦷 Bite tile, "bite the finger" by voice or chat, or `f` + Enter in the robot program's terminal (`b` already selects the blue ball). The hand tracker flags a hand as pointing when the index finger is straight and the middle, ring and pinky fingers are curled, and sends the fingertip's position. The arm waits with the gripper horizontal and open (the "mouth" pose). It turns to face the fingertip, stops 6 cm short, lunges radially to 1 cm past the last seen fingertip and closes. That repeats up to 3 times in a 30 s window, then the arm goes back to REST. Runbook: `docs/finger-biting.md`.

## Why

- **A round on request, not always on** (Tian asked for the feature, the agent chose the form): pointing hands are seen during PICK and GIVE as well, and a bite there would wreck the demo. A request made during another round is cleared when it ends, as for the dance (`decisions/dance.md`).
- **Inside the demo chain**: it owns the only robot connection and the top camera (`decisions/supervisor.md`).
- **Horizontal gripper, radial lunge**: at wrist_roll 0 the jaws open top and bottom, like a mouth facing a finger pointed at it. At pitch 0 the gripper stays in the vertical plane through the pan axis, so "forward" is radial. Reach at pitch 0 is 20-42 cm from the pan axis at 10-20 cm height (team IK, `scripts/handover/kinematics.py`, computed 2026-11-22).
- **The lunge is blind, from a frozen fingertip**: the arm hides the finger from the top camera on the final approach (the same occlusion GIVE has, `decisions/hand-following-safety.md`). The fingertip is frozen once the tip is within 10 cm of it.
- **One camera, assumed finger height (`--finger-z`)**: the top camera is about 47° from vertical (`scripts/calib/extrinsics.yaml`), so a finger 5 cm off the assumed height lands about 5 cm off along the line of sight (`research/hand-tracking-and-camera-calibration.md`). Accepted: "tries to bite" is the goal (Tian).
- **Strength**: the gripper's torque cap, 200 of 1000 since 2026-11-22 (`kb/gripper-torque-limit.md`); Tian had asked for "50%", LeRobot's 500, which stripped the gripper's gears. Grip force was judged not important (Tian, 2026-11-22). The close command is 5; who may be bitten defaults to the team only, since `docs/roadmap.md` says the gripper never closes on a visitor's hand.
- **Floor 0.15 m and pan within 45° of the mouth pose**: grasps happen at 0.02-0.16 m (`decisions/dance.md`). The mouth pose is chosen at the table facing away from the bowl, so the pan limit keeps the arm off it.
- **Voice by whole word** (`bite`, `bites`, `biting`, `byte`): a short word matches inside longer ones with fuzzy matching, the same reason as the dance words (`scripts/voice/commands.py`).

## Alternatives

- **Bite zone, no aiming** - kept as `--bite-mode zone`: the arm stays in the mouth pose and bites only a fingertip within 8 cm of the point 4 cm in front of the jaws. It hardly depends on the calibration, but the arm never goes after the finger. Use it if aiming misses at the table.
- **Track the finger continuously, bite when close** (the palm follower aimed at the fingertip) - considered, not built: the follower is made for hovering over a palm from above, it goes blind on the final approach anyway, and it is the most work.
- **A top-down bite** - considered, not built: the gripper cannot point straight down at finger height (`research/so101-inverse-kinematics.md`, section 3), and the jaws would close across the finger from above, which looks less like a bite.
- **A separate bite program** - considered, not built: the dashboard and the voice would then need a second robot program, and only one process can own the arm.
- **Biting only when the finger points at the gripper** - left for later: the rule Tian gave is "index extended, the others not".

## Revisit if

- At the table, aim mode misses by more than a jaw width most of the time: try `--bite-mode zone`, or adjust `--finger-z` to where people hold their finger.
- The team wants visitors or kids bitten: set `--bite-close` to a gap wider than a finger first.
