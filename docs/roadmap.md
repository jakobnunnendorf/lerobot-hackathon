# Roadmap: stop points and human check-ins

Written 2026-11-20 (Friday evening). All times are local Bilbao time.

The project idea is still a **placeholder** (`ideas/bomb-defusal.md`). The stop points and check-ins below hold for any idea of the shape "teach one SO-101 a short manipulation skill and demo it live". The lines marked *(current idea)* only apply if the wire-cutting idea is kept.

## Fixed by the organizers

| When | What |
| --- | --- |
| Sat 09:30-21:00 | Hacking (night work from 22:00 is optional on both nights) |
| **Sun 11:00** | **Deadline: presentation, dataset and code** |
| Sun 11:30-12:30 | Presentations, 5 minutes per team, live demo in front of the jury |
| Sun 13:00 | Awards |

Source and open questions: `research/hackathon-logistics.md`.

## Stop points

A stop point is a time after which something no longer changes. If the work is not there at that time, we take the fallback. We do not move the stop point.

| # | When | What stops | If we are not there |
| --- | --- | --- | --- |
| S1 | Fri 21:00, at the latest Sat 10:00 | **Idea lock.** After this the idea can shrink but not change. | Sat 10:00 passes without agreement: we build the smallest version of the placeholder (one arm, one skill, fixed position). |
| S2 | Sat 11:00 | **Scene freeze.** Camera positions, lighting, table, fixture and tool are fixed and photographed into `kb/`. | Recording does not start until it is frozen. Every hour of delay costs about 30 episodes. |
| S3 | Sat 16:00 | **Data freeze** for the base dataset. Training starts by 17:00. | Train on what exists. Fewer skills, not later training. |
| S4 | **Sat 21:00** | **Development stops** (feature freeze). No new capability after this. Allowed: bug fixes, correction episodes, fine-tuning, the demo flow, the presentation. | Whatever is not working is cut from the demo. |
| S5 | Sun 08:00 | **Training stops.** The last checkpoint we will consider exists. Overnight runs must be started by Sat 23:00. | Use the best checkpoint from Saturday. |
| S6 | **Sun 10:00** | **Testing stops.** Demo line-up is chosen at 09:45. After 10:00 no code, config, camera or robot changes, only run-throughs of the exact demo. | Demo the fallback tier (below). |
| S7 | Sun 10:30 | **Submission is in**: code on `main`, datasets on the Hub, presentation uploaded. 30 minutes of margin before the deadline. | Submit what is there at 10:30 and fix nothing after. |

## Human check-ins

Agents stop at each of these and wait for a human decision on the message board (issue #1). One human writes the outcome there in one or two lines.

| # | When | A human decides | On what evidence |
| --- | --- | --- | --- |
| H0 | Fri evening, before S1 | The idea | Answers from the mentors: is a cutting tool allowed, can we get a second arm, are there GPU or cloud credits, what are the judging criteria, is the demo on a stage or at our table |
| H1 | Fri night, at the latest Sat 10:30 | Whether the hardware can do the task at all | Two bench tests by teleoperation: the task done 10 times in a row by a person, and the whole chain once (record 10 episodes, train briefly, run the policy once). *(current idea: the tool cuts the chosen "wire" material 10 out of 10)* |
| H2 | Sat 11:00 | Scene freeze (S2) | Photo of the setup, camera views saved |
| H3 | Sat 13:00 | Whether the data is good enough to keep recording the same way | A person watches 10 random recorded episodes and checks that the recording really ran at the target frame rate |
| H4 | Sat 16:00 | What to train, on which machine, and what it may cost | Episode counts per skill, which GPU is free, price of any cloud job |
| H5 | **Sat 20:00** | **The demo tier** (below) and what runs overnight | 10 runs of each skill on the real arm, counted and written on the board as raw counts, e.g. "slot 2: 7/10" |
| H6 | Sat 21:00 | Feature freeze (S4), who does what at night, who sleeps | H5 result. First full run-through of the 5-minute demo with whatever exists |
| H7 | Sun 09:00-09:45 | The final demo line-up | 10 runs of each skill with the final checkpoints |
| H8 | Sun 10:30 | That the submission is complete | A person opens every submitted link in a private browser window |
| H9 | Sun 10:30-11:30 | The pitch | Two timed run-throughs of the 5 minutes |

## What always needs a human, whatever the time

- **Anything that moves the robot.** Teleoperation, recording, replay and policy runs are started by a person at the table, with the arm's power switch in reach. An agent may prepare the command, it does not run it.
- **Any tool with an edge on the robot.** Mounted, removed and reset by a person, and only after a mentor has said it is allowed.
- **Deleting or overwriting** datasets, episodes or checkpoints, force-pushing, and changing the `lerobot` or `torch` pins.
- **Spending money or credits** (cloud training jobs).
- **Changing the scene after S2.**
- **Locking the idea, choosing the demo line-up, submitting, and what we say to the jury.**

## How agents work between check-ins

- Agents may on their own: write code and scripts, analyse datasets and logs, prepare commands, watch a training run, write docs, post on the board.
- An agent does not work for more than about 90 minutes on one task without a human having read its output. At 90 minutes it posts where it is and asks.
- Results are reported as raw counts ("7/10"), never as "works" or "mostly works".
- Commits are small and frequent on `main`. At every check-in a human looks through what the agents pushed since the last one.

## Demo tiers

Chosen at H5 and confirmed at H7 from the measured counts.

| Tier | What the jury sees | Condition |
| --- | --- | --- |
| 1 | The full chain live: the target is chosen (by a person or by the camera), a learned policy does the task, the result is checked | The skill succeeds at least 8 times out of 10 |
| 2 | A learned policy on the positions where it is reliable, the others shown by replaying a recorded motion | 6 or 7 out of 10 |
| 3 | Target selection live, the motion replayed from a recording, and we say so | Below 6 out of 10 |

A recorded video of the best run is made on Saturday evening for every tier, in case the robot fails on Sunday morning.

## Saturday in one view

| Time | Work |
| --- | --- |
| 09:30-11:00 | Finish bench tests (H1), fix the scene, freeze it (S2, H2) |
| 11:00-13:00 | Record base episodes |
| 13:00 | Data check over lunch (H3) |
| 14:00-16:00 | Record the rest, delete bad episodes, push datasets to the Hub (S3, H4) |
| 16:00-20:00 | Training runs. Meanwhile: target selection code, demo flow, presentation draft |
| 20:00 | Count 10 runs per skill (H5) |
| 20:30-21:00 | First full run-through, feature freeze (S4, H6) |
| 22:00 onwards | Correction episodes on the weakest skill, fine-tuning overnight, started by 23:00 |

## Not yet known, and it changes this plan

- Which machine trains, and whether there are cloud credits. Without a strong GPU only the small policy (ACT) is realistic. See `research/lerobot-training-feasibility.md`.
- Whether a second arm is available. The organizers list one SO-101 per team.
- Whether a cutting tool is allowed at the venue. *(current idea)*
- This year's judging criteria.
