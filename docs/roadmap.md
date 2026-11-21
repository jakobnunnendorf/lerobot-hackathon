# Roadmap: stop points and human check-ins

Rewritten on the night of 2026-11-20 for the ball-picking idea (`ideas/ball-picking-and-handover.md`), including the change from stirring to clearing gray balls. All times are local Bilbao time.

Changed on 2026-11-21 because of board comments 7061614734, 7061650334 and 7061772563: the build order, S3, H5, the demo tiers and the Saturday view below follow the current plan (one dig-and-pick ACT policy, hand following with a two-flat-palm drop or PUT DOWN, handshake last). The earlier plan, with CLEAR GRAY and the gray hole, is kept in the lines marked "earlier plan".

## Fixed by the organizers

| When | What |
| --- | --- |
| Sat 09:30-21:00 | Hacking (night work from 22:00 is optional on both nights) |
| **Sun 11:00** | **Deadline: presentation, dataset and code** |
| Sun 11:30-12:30 | Presentations, 5 minutes per team, live demo in front of the jury |
| Sun 13:00 | Awards |

Told to us at the venue: the solution needs at least two cameras, and the demo has to run inference on our own laptops (`kb/judges-requirements.md`, `kb/machines.md`). Source for the schedule: `research/hackathon-logistics.md`.

## Build order

The demo is built in this order. Each line is a working demo on its own, so whatever is not finished is cut from the bottom.

1. PICK with `act_pick_ball_combined` (one dig-and-pick policy for a visible and a buried ball; it ends at the folded rest pose holding the ball), then PUT DOWN into the container.
2. The supervisor chains PICK, a gripper check (holding band 6-20), and GIVE or PUT DOWN (PR #17, draft).
3. GIVE by hand following (MediaPipe tracker, camera calibration, inverse kinematics, PRs #13 and #15): the arm follows the hand and drops the ball between two flat palms.
4. SHAKE, after everything else (Balazs, 7061650334).
5. Optional, only if the run counts show the policy cannot dig: a separate CLEAR GRAY skill (one gray ball into the gray hole). No dataset exists for it.
6. Parked: a pick skill that can be told the ball colour (`ideas/ball-picking-and-handover.md`, "Other target colours").

Earlier plan: step 3 was a GIVE replay, step 4 was CLEAR GRAY for a hidden ball, step 5 was SHAKE and a learned GIVE.

Alongside, by one person, without blocking the steps above: the voice command for the destination (hand or container), PR #16 (merged). It ends at the feature freeze like everything else, and the keyboard always sends the same commands.

## Stop points

A stop point is a time after which something no longer changes. If the work is not there at that time, we take the fallback. We do not move the stop point.

| # | When | What stops | If we are not there |
| --- | --- | --- | --- |
| S1 | done, Fri evening | **Idea chosen.** It can shrink (build order above) but not change again. | - |
| S2 | Sat 11:00 | **Scene freeze.** Bowl fixed to the table, ball sizes, camera positions, lighting, the HOME and CARRY poses, the hand spot and the marked positions for the gray hole are fixed, photographed and written into `kb/`. | Recording does not start until it is frozen. Every hour of delay costs about 30 episodes. |
| S3 | Sat 16:00 | **Data freeze** for the base datasets (the combined pick dataset; earlier plan: PICK, CLEAR GRAY, and one take each of GIVE, PUT DOWN, SHAKE). | Train on what exists. Fewer skills, not later training. |
| S4 | **Sat 21:00** | **Development stops** (feature freeze). No new capability after this. Allowed: bug fixes, correction episodes, fine-tuning, the demo flow, the presentation. | Whatever is not working is cut from the bottom of the build order. |
| S5 | Sun 08:00 | **Training stops.** The last checkpoint we will consider exists. Overnight runs must be started by Sat 23:00. | Use the best checkpoint from Saturday. |
| S6 | **Sun 10:00** | **Testing stops.** Demo line-up is chosen at 09:45. After 10:00 no code, config, camera or robot changes, only run-throughs of the exact demo. | Demo the fallback tier (below). |
| S7 | Sun 10:30 | **Submission is in**: code on `main`, datasets on the Hub, presentation uploaded. 30 minutes of margin before the deadline. | Submit what is there at 10:30 and fix nothing after. |

## Human check-ins

Agents stop at each of these and wait for a human decision on the message board (issue #1). One human writes the outcome there in one or two lines.

| # | When | A human decides | On what evidence |
| --- | --- | --- | --- |
| H1 | Fri night, at the latest Sat 10:30 | Whether the hardware can do the task, and which bowl and balls | By teleoperation: the target ball picked 10 times in a row from different places in the bowl, including touching other balls and against the wall. The whole chain once: record 10 episodes, train briefly, run the policy once. Cameras and arm working on the demo laptop (WSL or Windows, `kb/machines.md`) |
| H2 | Sat 11:00 | Scene freeze (S2) | Photo of the setup, camera views saved, HOME and CARRY joint values written down |
| H3 | Sat 13:00 | Whether the recorded data is good enough to keep recording the same way | A person watches 10 random recorded episodes of each skill, checks the recording really ran at 30 fps, and checks that ball and hole positions are spread out |
| H4 | Sat 16:00 | What to train, where, and what it may cost | Episode counts per skill, prices, credit left |
| H5 | **Sat 20:00** | **The demo tier** (below), and what trains overnight | PICK x/10 and full-run x/10 on the real arm, counted with `docs/eval-counting-sheet.md` and written on the board as raw counts, e.g. "PICK: 7/10". No episodes were held out of the training data, so "positions not in the training data" is not tested by these runs. (Earlier plan: 10 runs of PICK and of CLEAR GRAY, CLEAR GRAY 8/10; CLEAR GRAY no longer exists as a skill.) |
| H6 | Sat 21:00 | Feature freeze (S4), who does what at night, who sleeps | H5 result. First full run-through of the 5-minute demo with whatever exists |
| H7 | Sun 09:00-09:45 | The final demo line-up, including which checkpoint runs each skill | 10 runs of each skill with the final checkpoints |
| H8 | Sun 10:30 | That the submission is complete | A person opens every submitted link in a private browser window |
| H9 | Sun 10:30-11:30 | The pitch | Two timed run-throughs of the 5 minutes |

## What always needs a human, whatever the time

- **Anything that moves the robot.** Teleoperation, recording, replay and policy runs are started by a person at the table, with the arm's power switch in reach. An agent may prepare the command, it does not run it.
- **A hand near the robot.** Handover and handshake are first tried by a team member who knows where the power switch is. A judge's or visitor's hand only after it has worked ten times in a row. The gripper never closes on a hand.
- **Merging a pull request into `main`**, unless the person said at the start of that task that the agent may merge it.
- **Deleting or overwriting** datasets, episodes or checkpoints, force-pushing, and changing the `lerobot` or `torch` pins.
- **Spending money or credits** (cloud training jobs). The running total is written on the board after every job.
- **Changing the scene after S2.**
- **Choosing the demo line-up, submitting, and what we say to the jury.**

## How agents work between check-ins

- Agents may on their own: write code and scripts, analyse datasets and logs, prepare commands, watch a training run, write docs, post on the board.
- An agent does not work for more than about 90 minutes on one task without a human having read its output. At 90 minutes it posts where it is and asks.
- Results are reported as raw counts ("7/10"), never as "works" or "mostly works".
- Agents never commit to `main`. They commit regularly on their own branch and open a pull request (rules in `CLAUDE.md`). An agent merges only after the person it works for says yes, or when that person told it at the start of the task to merge.
- At every check-in a human looks through the open pull requests, so none of them waits longer than one check-in.
- After the feature freeze (S4) `main` is the demo branch: every merge is asked, whatever was said earlier.

## Demo tiers

Chosen at H5 and confirmed at H7 from the measured counts.

| Tier | What the jury sees | Condition |
| --- | --- | --- |
| 1 | The full loop live: the target ball is hidden under gray balls, the arm digs and picks it, follows a hand and drops the ball between two flat palms (or puts it down in the container) | Thresholds are not yet re-agreed for the new plan; the proposal is PICK at least 8 out of 10 and full run at least 8 out of 10 (7061772563), and the whole loop has worked end to end several times. Earlier plan: PICK and CLEAR GRAY each at least 8 out of 10, ending with a handshake |
| 2 | The target ball visible: the arm picks it and drops it into a hand held out, or puts it down when there is no hand | PICK at 6 or 7 out of 10 |
| 3 | The supervisor's decisions live on screen, the arm motions replayed from recordings with the ball in a fixed place, and we say so | PICK below 6 out of 10 |

A video of the best full run is recorded on Saturday evening for every tier, in case the robot fails on Sunday morning.

## Saturday in one view

| Time | Work |
| --- | --- |
| Fri night | First PICK batch: 60 episodes with the target on top. Train it overnight |
| 09:30-11:00 | Try the overnight PICK policy. Finish H1, fix the scene, freeze (S2, H2). Earlier plan: also fix the gray-hole positions |
| 11:00-13:00 | Count PICK runs (`docs/eval-counting-sheet.md`). Earlier plan: record CLEAR GRAY, 50-60 episodes; not done, no CLEAR GRAY dataset exists |
| 13:00 | Data check over lunch (H3). Earlier plan: the CLEAR GRAY training run starts straight away; only if the counts show the policy cannot dig |
| 14:00-16:00 | More PICK where the overnight policy missed (30-40); PUT DOWN pose taught at the table. Push datasets to the Hub (S3, H4) |
| 16:00-20:00 | Train the second PICK. Meanwhile: supervisor, hand following, voice, presentation draft |
| 20:00 | Count PICK and full runs, x/10 each (H5). Decide what trains overnight |
| 20:30-21:00 | First full run-through, feature freeze (S4, H6) |
| 22:00 onwards | Correction episodes on the weakest skill and fine-tuning overnight, started by 23:00 |

## Machines and money

Facts and open checks are in `kb/machines.md`.

- **Training runs in the cloud** (Hugging Face Jobs) where the credit allows, so the laptops stay free for the robot. One ACT run costs about 1-2 USD, so about 30 EUR covers many runs.
- **Inference runs on a team laptop.** Kamel's laptop (RTX 4070, 8 GB) is the demo machine. The skills are ACT policies: pi0.5 ran out of memory while loading there.
- **Recording laptop:** any laptop can drive the robot for recording. It must use the same calibration files and robot id as the demo machine.
- **Backup demo laptop:** a second laptop that can run the full demo. Set up and tried on Saturday evening, not on Sunday morning.
- The demo laptop stays on its charger with sleep and automatic updates switched off for the weekend.

## Not yet known, and it changes this plan

- Whether the arm and both cameras work from WSL on the demo laptop, or the demo runs from native Windows there.
- The exact Hugging Face credit and which account holds it.
- Whether the gripper can take the target ball when it touches other balls or the bowl wall (H1).
- This year's judging criteria.
