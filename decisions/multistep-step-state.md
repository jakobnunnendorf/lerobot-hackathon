# How does the supervisor run and end the colour steps of a spoken plan?

- Status: DECIDED (2026-11-22, agent of TianK003). Fake scenarios only; no clear-red/clear-blue ACT exists yet and nothing has run on the arm. Which policy runs a step (option (b), one ACT per colour) is `decisions/multistep-voice-policy.md` on `TianK003/multistep-research`.

## Decision

A plan round goes REST -> STEP (one per prompt) -> ... -> REST -> PICK chain. A STEP runs the colour's ACT until the drop counter reaches the plan's count for that colour, then the next STEP starts from where the arm is. Only the striped step goes back to rest first.

## Why

- **"Done" comes from an absolute joint rule** (`scripts/supervisor/drops.py`): the measured gripper below 16.5, then at 18 or above, with shoulder pan below -25 (red hole) or above 18 (blue hole). On teleoperated `molnrb/so101_pick_rb_mixed` it gives Jakob's labelled hole sequence on **53 of 58** episodes (0-57; misses 11, 28, 38, 49, 51; main session's measurement, 2026-11-21). Episode 11 is a partial open at the red hole (12.5 -> 16.8 at pan -31, never 18): such a drop is missed, and the step's time budget is the fallback. Re-grasps in the bowl open at a pan between the holes and are logged as `release`, not counted.
- **No REST between colour steps**: the clear segments in the training data start at the previous drop, at a hole.
- **REST before the striped PICK**: the PICK policy and its end rule (`Settled`, "left rest first") start from rest.
- **Counts add up per colour over the round.** A wrong-colour drop lands in its own hole, so it counts there and is logged `wrong_colour: true`; a later step of that colour that is already met is skipped.
- **Timeout: re-prompt** (fresh skill, ACT queue reset, no robot reset), at most 3 times, then the round ends at rest as `step_gave_up`, like `gave_up` after the pick limit. **Still holding after the last re-prompt: ABORT**, like CLEAR_GRAY still holding: carrying a ball to rest would be a guess about where it should go. Before the first arm test a timeout while holding aborted at once; see the arm test below.
- **A drop needs a held ball**: the lowest measured reading while closed must be 5 or more (`held_above`). Empty closes read 0.4-3.9, held balls 5.5 and up (`grasp_check.py`, 606 teleop closes).
- **The grasp check runs inside each step** when it is on (the default), the same `GraspChecked` as PICK but **unlocked**, so the policy can still open at its hole. An empty close is opened in the bowl and the step is re-prompted at once, at most 8 times per step (`--max-step-grasp-misses`).

## First arm test (2026-11-22, board 7072005462, `p 2b 2r s`, both clear ACTs at step 4000)

- **No ball was grabbed**, blue or red, according to Tian, who watched the run. Balazs's agent had reported "blue 2 / 2", but those were empty closes (gripper 1.6-4) that the policy still carried to the blue hole and opened there. The counter had no held-ball check and counted them. That is why the counter now has `held_above`.
- **Red**: three cycles of an empty close, a swing to the red hole and an open. The fourth close read 14 (a ball, probably) when the 30 s budget ran out mid-trip, which aborted the round. That is why a timeout while holding now re-prompts, and why the grasp check runs inside a step.
- These fixes make the step honest and let it retry. They do not make the clear ACTs grasp; that is a policy question, still open.
- **Voice: a number makes a sentence a plan.** Any count word or digit, except a bare "one" ("the striped one"). "a"/"an" count as one inside a plan but never start one, so "give me a red ball" stays a single command.

## Alternatives

- **Relative release rule** ("opens by at least N from the held minimum") - measured by the main session on the same 58 episodes: rise 2 matches 43, rise 3 49, rise 4 52, all below the absolute rule's 53.
- **Top-camera colour counts to end a step** - not built: the arm hides the bowl while it works. Kept as the check to add if the counter miscounts on the arm.
- **Park at HOME between clears** - rejected: the policy would start from a pose its segments never start from.

## Revisit if

- Policy runs on the arm release over the bowl, open only partly at a hole, or miss holes the teleop pans do not cover: tune `--drop-gripper` / `--drop-pan` from the `drop` and `release` log lines, or add the camera check.
- A step often times out while holding, even after the re-prompts: give it a carry-to-hole fallback instead of ABORT.
- The step grasp check calls real holds misses (log `grasp_check` with `step_colour`): tune it, or run with `--no-grasp-check` and rely on `held_above`.
