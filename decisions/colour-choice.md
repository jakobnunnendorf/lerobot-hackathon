# How does the robot pick or sort a ball by colour ("choose by colour", "hand me the red ball")?

- Status: OPEN (2026-11-21). Options only. The detector in `scripts/vision/colour_detector.py` was tuned on the OLD scene (see Why).
- Where it stood at the deadline: the robot could not pick a ball by colour. The dashboard's red and blue tiles stayed greyed out and a colour command is refused before the arm moves (`decisions/supervisor.md`). The detector shipped only as the wrist-camera check that the held ball is the striped one (threshold 120, `kb/held-ball-wrist-areas.md`). For spoken plans ("2 blue, then the striped ball") the team chose one clearing ACT per colour, close to option A (`decisions/multistep-voice-policy.md` on the unmerged branch `TianK003/multistep-act`); on the arm those made only empty grasps (`decisions/multistep-step-state.md`). A SmolVLA on the red/blue data (option B's model, not language-tested) scored 0/4 at picking the striped ball (`decisions/hidden-ball-picking.md`).

## Decision

Nothing was decided for picking by colour. Recommendation of Jakob's agent, in order:

1. **Sorting into the matching container (stage 4): no colour input is needed.** The ~80 new red/blue episodes show colour -> container in the image itself, so one ACT policy can learn it, as long as its recordings really sort both colours. Use the detector only as sensors: "any red or blue ball left in the bowl?" (stop condition) and "are both containers in view?" (scene check).
2. **"Hand me the red/blue/striped ball" (stage 5): option A, one ACT per colour, if the new episodes can be split by the colour that was picked** (about 40 each, other colour present as distractor). Add option C as the gate in front of it: the detector refuses ("no red ball in the bowl") instead of clearing balls for a colour that is not there.
3. Keep D (IK pick) and B (SmolVLA) as named fallbacks, not plans.

## Why

What is known about the setup (board 7062034477, `decisions/target-visibility-check.md`): `act_pick_ball_combined` picks 9/10 with the target visible and 0-1/10 buried; the gray foil was replaced by 11 red and 11 blue balls plus the striped target, red and blue sorting containers are fixed, and about 80 new episodes are being recorded. How those episodes choose their ball (a fixed colour per episode? either colour, operator's choice?) is **not known to this note**; every cost below depends on it.

| Option | Data / recording | Training | Risk before Sun 10:30 | What it shows the judges |
| --- | --- | --- | --- | --- |
| **A. one ACT per colour**, voice `target` switches the checkpoint (the team loop already holds several policies) | Split the ~80 episodes by picked colour (about 40 each, no new recording) and keep the striped policy. If an episode has no colour label, labelling costs one look per episode. Research says other colours must be in the bowl in training (`research/multi-colour-targets.md`, 39% pick success with colour-matched competitors for ACT in one study) | Kamel's laptop: 20-30K steps = 1.5-2.25 h each, free (board offer); HF Jobs in parallel | Medium. 40 episodes per colour is thin; 2 policies on one 8 GB laptop at once for inference is untested | "Say a colour, the arm takes that colour", honest and robust to the judges' choice |
| **B. one SmolVLA conditioned on the task text**, mixed-colour data | All ~80 episodes with a per-episode sentence ("pick the red ball"); same data as A | Cloud, hours (`scripts/modal_train.py` has SmolVLA recipes). ArmnetBench: SmolVLA 15% strict success at 50 demos, the lowest of seven policies (`research/lerobot-training-feasibility.md`) | High. Fits the "needs a different pick skill" line only if language following actually works at 40 demos per colour; one training attempt | A single model that follows a sentence, the most impressive if it works |
| **C. detector + the existing policy**, refuse when the colour is absent | None | None | Low for the gate; **does not choose**: the policy still picks "whatever it picks". Works for the demo only when the bowl holds ONE ball of the asked colour, which hides the choice | Safe failure: "there is no red ball", plus a visible detector overlay on screen |
| **D. detector + kinematics pick** (pixel -> table point -> #15's IK) for visible balls | No demos, but a camera-to-table calibration at the table and a pick routine for a deep bowl with 23 balls | None | High: needs table time we do not have, and balls lie at different heights in the bowl | A classical vision pipeline, nothing learned; any colour for free |
| **E. mark the commanded ball in the image** (research option 1) | Rewrite episodes offline | Cloud | Dropped for the striped ball (`decisions/other-ball-colours.md`: top-camera mask 59% in the grasp window). Real red/blue episodes make it unnecessary | - |

- **Detector, measured, old scene.** Frames: `molnrb/so101_pick_blue_ball`, top camera, episodes 48, 49 (file-003), 69, 70 (file-005), 95-99 (file-007), one frame per second (140 frames), plus 16 frames of episode 0 of `molnrb/so101_pick_ball` (a different scene, only used as a stress test). Judged by eye by this agent. Striped ball: 79 frames with the ball clearly visible in the bowl, 79 found; 42 frames with it clearly not in the bowl (picked up, carried away), 0 false hits; 19 ambiguous frames (ball under or in the gripper), 10 found. Blue and red balls: **no such ball exists in these videos**, so they are tested on synthetic images (balls, foil, thin cable, tinted specks) and the blue tape roll stands in for a blue container (found in 66 of 156 frames, not checked frame by frame). The thresholds were tuned on the same episodes: expect worse on the new scene.
- **What fools it.** The orange arm carries a red cable and a dark-red motor housing: the red detector fires on them in 14 of 156 frames, all with the arm in the picture (none of the 9 parked-arm start frames). The supervisor looks only while the arm is parked, so this matters mostly for any overlay shown during motion. Thin things are removed by a 9 px opening; the housing is not. Foil: coloured reflections are small and low in chroma, so no colour fires on them (0 of 140 frames for blue; striped and red only on the real objects above), and bright foil is simply the "gray" class.
- **Hue augmentation must stay off** for any option above that depends on colour (Kamel's check, board 7062043167).
- Option D needs numbers nobody has: the camera moved a few centimetres between the old episodes, so a calibration must be redone after the scene freeze.

## Alternatives

- Considered, not tried: everything in the table. Nothing was run on the arm or with the new red/blue frames.
- Already rejected elsewhere: pi0.5 on the 8 GB laptop (`decisions/skill-policies.md`), repainting the striped ball red (`decisions/other-ball-colours.md`).

## Revisit if

- The new red/blue dataset is on the Hub: re-tune `scripts/vision/colour_detector.py` on its frames (`calibrate` mode, 2 frames per colour, then `tests/colour_detector.py`) and measure red/blue by eye the way the striped ball was.
- The new episodes turn out to be "pick any ball, sort by its colour": then A needs new per-colour recordings, or B/C change.
