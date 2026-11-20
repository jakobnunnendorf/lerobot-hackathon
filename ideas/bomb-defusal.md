# Bomb defusal with computer vision

**Status: PLACEHOLDER.** This is the working idea while the team ideates. It is not finalized and may change or be dropped.

## The idea as it stands (2026-11-20)

- An SO-101 arm looks at a mock device with several coloured wires, decides which wire is the right one (for example the green one) with computer vision, and cuts it.
- Cutting tool: a blade (razor blade was suggested) mounted at the gripper tip.
- Possible extension: two arms, one holds the object steady and the other cuts.
- Pitch angle: in a conflict zone there are far more explosive hazards than trained operators, so automatic or assisted disposal could scale where people cannot.

## Open questions

- Is two-arm operation recordable and trainable in the time we have?
- How do we train with 100-200 recorded episodes and combine that with a vision step that picks the wire?
- Is a razor blade acceptable at the venue, and is there a safer tool that cuts just as reliably?
- Does the "bomb" framing help or hurt with the judges?

## Critique after first research (2026-11-20)

Sources for every point are in `research/eod-domain-and-prior-art.md`, `research/lerobot-training-feasibility.md` and `research/hackathon-logistics.md`. None of this has been tested on our hardware yet.

What holds up:

- The problem is real and large: 6,279 mine and explosive-remnant casualties in 2024, 90% of them civilians. Ukraine has about 4,500 deminers against a stated need for tens of thousands.
- Bomb-disposal robots are still driven by hand, joint by joint, and fine work is hard: on a purpose-built research robot, novice operators cut the wire on the first try only 46-71% of the time. Help with the fine motion has been a research goal since 2008 and is not in fielded robots.
- The task reads instantly for an audience, and no SO-101 project doing it was found (the search was not exhaustive).

What does not hold up:

- Choosing a wire by its colour is a film trope. Real technicians do not cut wires by colour, and the humanitarian standard prefers destroying a device in place. A judge who knows the field will say so.
- "Not enough operators, so the robot defuses on its own" skips a step. The sources all describe help for the operator, with a person approving each action. Nothing supports autonomous defusal.
- Picking a colour is a few lines of classical vision. We need a clear answer to "what did the robot learn?".
- A razor blade is the wrong tool. The software limits the gripper to half of its torque, so real copper wire is out by estimate, and a bare blade sits in a workspace that hands reach into 150 times or more. No venue rule on blades was found either way.
- A prop that looks like a bomb at a public event is an avoidable risk.

Two arms:

- The organizers list one SO-101 per team.
- LeRobot 0.7.0 supports two arms (`bi_so_follower`, `bi_so_leader`), and one person can drive both leaders, so recording is possible.
- Learning it in two days is not supported by the evidence: in a July 2026 benchmark the small policy that fits a laptop (ACT) scored 2.5% on two-arm tasks with 50 demonstrations each.
- If a second arm appears, it should hold the object in a fixed or replayed pose and not be learned. A clamp does the same job.

Training from 100-200 episodes plus vision:

- Do not ask one policy to both choose the wire and cut it. Let vision (or a person) choose which position to cut, then run a small learned skill for that position.
- Proposed budget: 3 positions x 40 episodes, about 30 correction episodes on day two, the rest in reserve.

Proposed direction, not yet decided: keep the robot work, change the story to "the operator chooses, the robot does the fine motion", use scissors or a small cutter on soft material in place of a blade, and make the prop look like a game.

Decisions are taken on the message board (issue #1).
