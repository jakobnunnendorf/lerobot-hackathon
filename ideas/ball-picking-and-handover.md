# Find the target ball, clear the gray ones if it is hidden, hand it over

**Status: CHOSEN.** The team switched to this idea on the evening of 2026-11-20 (reported by Tian). It replaces the bomb-defusal placeholder and the cable-cutting candidate. The structure below is agreed with Tian. The learned skills run as ACT policies, because pi0.5 did not fit on the demo laptop.

## The idea, as the team stated it

Two things changed during the evening of 2026-11-20, both reported by Tian:

- The target ball was first blue. It is now a ball with yellow and green stripes. Below it is called the target ball.
- The first plan for a hidden target was to stir the balls (MIX). Testing on the arm showed that does not work in this bowl, for lack of room. It is replaced by clearing gray balls out of the way.

The idea now:

1. A brown bowl holds the target ball and gray balls made of tinfoil. If the target ball can be seen, the arm picks it up wherever it lies. A computer-vision model finds it.
2. If no target ball is found, the arm picks up one gray ball and drops it into the gray hole. Then it looks again. The position of the hole is changed between recordings so the skill does not depend on one place.
3. Once it holds the target ball: if a hand is detected, it puts the ball into the hand and then tries to shake hands. If no hand is detected, it puts the ball down safely.

Pitch line from the research: "Robots can pick what they can see. Ours goes looking for what it cannot see, and then hands it to a person." The research name for the first half is mechanical search: removing what is in the way until the target shows.

## Structure: a decision tree on top, separate skills below

This was the team's own plan, and the analysis supports it over one end-to-end policy for the whole task.

- Target ball seen well enough to grasp? Yes: PICK. No: CLEAR GRAY, then look again. Stop when no gray ball is left or after a fixed number of rounds, and say that the target was not found.
- Holding the target ball: hand seen? Yes: GIVE, then SHAKE. No: PUT DOWN.

Why not one policy for everything:

- The task has two branches and a loop of unknown length. A single policy would have to learn the branching from whole-task episodes, and 100-200 episodes spread over four paths through the task is too few.
- Errors multiply in one long policy. The ACT paper itself reports 92% at the first stage of one task and 20% at the end, and one SO-100 hackathon team got about 10% on a long single-policy task with 175 episodes. Separate skills can be retried one at a time.
- The decisions are made by code that can be tested without the robot and shown on a screen.
- Skills can be recorded and trained in parallel, and cut without retraining the rest.

No source shows this exact design on an SO-101, working or failing. The case is assembled from adjacent evidence (`research/ball-picking-and-handover.md`).

## The skills

Two anchor poses make the skills chain. Every demonstration and replay starts and ends in one of them:

- HOME: above the bowl, gripper open. The arm is out of the cameras' view of the bowl.
- CARRY: above the bowl, gripper closed on the target ball.

| Skill | From, to | How | Episodes |
| --- | --- | --- | --- |
| PICK | HOME to CARRY | learned (the target's position varies) | 80-120; the first 60 on the night of 2026-11-20 with the target on top |
| CLEAR GRAY | HOME to HOME | learned (which gray ball, and where the hole is, both vary): pick one gray ball and drop it into the gray hole | 50-60 |
| GIVE | CARRY to HOME | first version a replay, with the hand held over a marked spot; learned (30-40 episodes) if time allows | 0, later 30-40 |
| PUT DOWN | CARRY to HOME | a replayed recording (fixed spot) | 0 |
| SHAKE | HOME to HOME | a replayed recording; first thing to drop | 0 |

- CLEAR GRAY is proposed as one continuous skill, pick and drop together, so there is no handoff in the middle while a gray ball is held. It can be split into two skills later if needed.
- The gray hole: use a small number of marked positions, four or five, with at least 10 episodes at each. Moving it freely with only 50-60 episodes spreads the data too thin for the policy to learn any of the positions. The hole has to be visible in the top camera in every position.
- GIVE and PUT DOWN start with the target ball already in the gripper, so the first action of the next skill does not open it.
- One dataset per skill, each with its own task sentence, same camera names, resolution and robot id. They can be merged later for a single instruction-following model.
- A policy never says "finished". Each skill gets a time budget, then the supervisor checks: after PICK, does the gripper reading show it closed on something; after CLEAR GRAY, is the gripper empty and the arm back at HOME; after GIVE, is the gripper empty.

MIX is dropped. PICK and CLEAR GRAY together need 130-180 episodes, which is most of the recording budget; that is why GIVE starts as a replay.

## Which policy runs the learned skills: ACT

Training can run in the cloud. Inference has to run on a team laptop, and the only GPU laptop has 8 GB (`kb/machines.md`).

| | One ACT policy per skill | One pi0.5 model for all skills |
| --- | --- | --- |
| SO-101 benchmark at 50 demonstrations per task | 19.2% | 47.6% |
| Cost of one training run | about 1-2 USD | about 10-20 USD |
| Inference on the 8 GB laptop | yes, about 1 GB | no: out of memory while loading (test on 2026-11-20) |
| Switching skills | our own loop, one policy per skill | change the task sentence |
| Training attempts before Sunday | many | one, overnight |

The benchmark numbers were read by an automated tool and need re-checking before anyone quotes them. SmolVLA scored 15.0% in the same benchmark, below ACT, so it is not planned.

Result: Tested on Kamel's laptop on 2026-11-20 (reported by his agent on the board): `lerobot/pi05_base` ran out of GPU memory while the model was being built, with about 14 GB allocated, which matches the weights at full precision. The config's `dtype` setting was not applied during construction. On the CPU the same 14 GB stalled the machine, which has 15 GB of RAM. Not tried: building the model in half precision from the start, which would put the weights near 7 GB and leave almost nothing for the rest on an 8 GB card.

So the skills are ACT, one policy per learned skill, and the cloud credit goes to ACT training runs. The recording still uses one dataset per skill with a task sentence, so a larger model stays possible if a team laptop with clearly more than 8 GB of GPU memory turns up.

## The supervisor

- It decides only while the arm is parked at HOME. The arm hides the bowl from the top camera while it works.
- Target ball: start with a colour threshold for its yellow and green inside a fixed mask of the bowl, with a minimum area, stable over several frames. Yellow lies next to brown in hue, so the bowl can fool a yellow threshold: require high saturation and brightness, and lean on the green stripes. If the bowl or reflections on the foil still fool it, train a small detector on 150-200 frames from our own cameras. The research file discusses a blue ball; its colour advice needs re-checking for the new ball.
- "Seen well enough to grasp" is a minimum visible area, set once on the real scene. A target that is half under a gray ball counts as not graspable, and a gray ball is cleared first.
- It also counts the gray balls it can see, so it knows when there is nothing left to clear.
- Two cameras: the top camera decides whether the target can be grasped. The wrist camera, from a fixed look-in pose, can only upgrade "not visible" to "visible". This is the use of the second camera the judges require (`kb/judges-requirements.md`).
- The vision model decides and shows its result on a screen. It does not steer the arm: turning a pixel position into arm motion needs camera calibration and an arm model, and the policies learn that mapping from the demonstrations. Each policy always goes for the same kind of ball, so it needs no extra input.
- Hand: MediaPipe's hand tracker in its own environment and process, reporting "hand present, open, where" to the supervisor.
- Vision never runs inside the control loop. One report saw a policy fall from 9/10 to 4/10 when its loop dropped from 30 to 15 Hz.

## Running the skills

Read from the LeRobot 0.7.0 source, not yet run (`research/lerobot-multi-skill-runner.md`):

- `lerobot-rollout` handles one policy and one robot connection, and cannot switch. The demo is our own 30 Hz loop: connect once, load every policy up front, and each tick read the observation, ask the active skill's policy, send the action.
- Switching skills sends nothing, so the arm holds its pose and the gripper stays closed.
- Never reset, disconnect or reconnect while holding the ball. All three open or relax the gripper.
- A replay is the same loop sending a recorded episode's actions.

## Handover and handshake

- The hand is held in the air over a marked spot, not resting on the table.
- The arm comes from above, stops a centimetre or two over the palm and opens. Nothing closes on the person.
- Handshake: the person grips the robot's closed gripper and the arm makes a small slow up-and-down.
- A team member is the hand. A judge's hand only after it has worked ten times in a row.

## Recording notes

- PICK, first batch: 60 episodes as 6 areas of the bowl with 10 each, target on top, ball count fixed, the other balls rearranged and the target turned between episodes. Ten arrangements are kept aside, not recorded, for testing.
- CLEAR GRAY: 50-60 episodes over four or five marked hole positions. Include bowls with and without the target visible under the gray balls, so the policy does not learn to grab the target.
- Every episode starts at HOME. PICK ends at CARRY with a one-second pause. CLEAR GRAY ends at HOME.
- `--dataset.no_stamp=true`, a full task sentence per skill, one operator, no hands or leader arm in view, and a check after the first episodes that the loop really runs at 30 fps.

## Open questions

- Whether any team laptop has clearly more than 8 GB of GPU memory. Only that would reopen pi0.5.
- Whether the target ball also goes into a hole when no hand is there, or is put down on a fixed spot.
- Where the hand zone is, and how many gray balls are in the demo bowl.
- Which closed-gripper reading means "holding a ball". Measure on the arm.
