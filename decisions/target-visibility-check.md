# Can a colour check on the top camera tell "target visible, pick" from "target buried, clear a gray ball"?

- Status: OPEN (2026-11-21). Measured on the recordings only. Not tested in the demo hall or on live camera frames.

## Decision

Nothing decided yet. The measured candidate: run `scripts/recolour/target_mask.py` (top camera, one frame, no tracking) while the arm is parked, and answer "visible" when the mask covers at least about 150 px. Any threshold from 130 to 200 px gave the same result.

## Why

All numbers: Tian's agent, 2026-11-21, on the top-camera videos of `molnrb/so101_pick_blue_ball` (100 pick episodes, target on top) and `molnrb/so101_pick_buried_ball` (45 episodes, target buried at the start). Sheets and CSVs are in the git-ignored `data/recolour-work/visibility/`.

- **First 5 frames of every episode, arm parked.** Judged by eye, the dataset labels are not the ground truth:
  - Pick episodes: 81 fully visible, 16 half hidden (at the front rim or behind a foil ball), 3 hidden (e25: a person's arm over the bowl; e58, e81).
  - Buried episodes: 17 partly visible, 15 with only a sliver, 13 hidden.
- **Mask area per frame for each group.**

  | by-eye group | frames | mask area (median) |
  | --- | --- | --- |
  | fully visible | 405 | 764-1716 px (1356) |
  | half hidden | 80 | 382-819 px (602) |
  | partly visible, buried | 85 | 204-849 px (418) |
  | sliver | 75 | 0-127 px; 65 frames are 0 |
  | hidden | 80 | 0 px in every frame |

- **Threshold.**
  - Any threshold from 130 to 200 px separates the 570 visible frames from the 80 hidden and 75 sliver frames without an error.
  - With a threshold of 1 px, 10 sliver frames count as visible (e2, e21, e38).
  - At 220 px, 3 visible frames are missed; at 300 px, 22.
  - The margin is small: the smallest visible area is 204 px and the largest sliver 127 px.
- **Tuned and untuned episodes.** The mask was tuned on 10 pick episodes. Their smallest area was 407 px, against 382 px for the 90 untuned pick episodes. All buried episodes are untuned. The threshold does not change.
- **Later frames of 14 buried episodes (13,882 frames).**
  - False masks: 13 frames had a mask on the moving arm, outside the bowl (79-115 px; e1, e15, e41). Both the 130 px threshold and a bowl region remove them.
  - Gray balls only: in 2,711 frames before the target first showed (7 episodes; the arm was in view in 1,607 of them), no frame had a mask inside the bowl.
- **When the target first shows with the arm out of the way.** "Out of the way" means fewer than 3,000 orange pixels in the bowl region for at least 5 frames. 10 of the 14 episodes start hidden or with a sliver:
  - 6 had such a moment before the pick: e7, e10, e13, e15, e22, e41. At the first moment where more than a sliver showed, the mask was present in every frame (median 274-1294 px).
  - In e10 and e13 the first arm-out moment showed only a sliver: the mask was present in 17 of 115 frames (66-86 px) and in 79 of 83 frames (median 91 px). Both stay below the threshold.
  - 4 had no such moment: e1, e6, e16 and e23. The operator uncovered the ball and grasped it without leaving the bowl.

## What this does not show

- Lighting, bowl, camera position and ball are the same as in the recordings. Nothing here says anything about the demo hall, a moved camera or another ball.
- The frames are decoded AV1 video; live camera frames were not tested.
- In the buried recordings the arm never parks between gray balls (it goes straight on to the next one), so the "arm out of the way" moments are the arm passing over to the gray hole. They are not a parked HOME pose.
- The by-eye labels were not made blind: the sheets the frames were judged on show the mask outline and area. Hidden (mask always 0) against clearly visible is solid; where "sliver" ends and "partly visible" begins is a judgement, so the gap between 127 and 204 px is partly drawn by the labeller.
- "Visible" is not "graspable". Whether the pick policy can take a ball that shows only 200-400 px (half hidden) is not tested.
- Areas are in pixels at 640x480. They change if the camera moves or zooms, so the threshold has to be measured again.

## Alternatives

- **Threshold fitted to the dataset labels** (pick means visible, buried means hidden) - tried 2026-11-21: the best value is 360-380 px, with 15 misses (the 3 hidden pick episodes) and 49 false "visible" frames. Dropped, because 17 of the 45 buried episodes start with the target partly visible.
- **Tracking across the 5 frames** instead of single frames - looked at 2026-11-21: it keeps sliver masks of 42-62 px alive where a single frame drops to 0 (e21, e38). Below the threshold, so it changes nothing.
- **Learned classifier or a vision-language model** - considered, not tried: the colour check separated the recorded cases without an error.

## Revisit if

- The camera is moved, the hall lighting is different, or the target ball changes.
- The supervisor needs "graspable" rather than "visible", for example because the pick policy fails on half-hidden balls.
- A check in the hall gives a mask above about 130 px with no target in view.
