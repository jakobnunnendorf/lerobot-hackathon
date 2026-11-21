# Can we get a red-ball training set by repainting the striped ball in the recorded episodes?

- Status: DECIDED (2026-11-21): not with mask version 2. No red dataset was written.

## Decision

No red-ball dataset is written with mask version 2 (Tian's agent, 2026-11-21, after looking at the preview frames; Tian can overrule). In the top camera the half of the ball in a jaw's shadow stays striped, so during the grasp the ball is half red and half striped and the paint switches on and off between frames. A policy trained on that would see a ball that changes colour exactly when it matters.

The bar a mask has to pass before a training run is worth its time, per camera, judged by eye on episodes not used for tuning (about 150 frames with the ball visible): at least 95% correct where the ball is visible, at least 90% in the grasp window, false repaints below 1%.

`scripts/recolour/recolour_dataset.py` keeps the repaint and the checks of the output. Its preview mode (`--episodes`) was run; **writing a whole dataset was never run end to end**. Packages as used: `scripts/recolour/requirements.txt`.

## Why

All numbers: Tian's agent, 2026-11-21, on `molnrb/so101_pick_blue_ball` (100 episodes, 47,401 frames, 640x480 AV1; the ball is striped despite the name). Work files and judgement CSVs are in the git-ignored `data/recolour-work/`.

- **Episodes.** Mask tuned on episodes 0, 11, ..., 99. Version 1 checked on 7, 10, 14, 21, 46, 51, 56, 75, 82, 92. Version 2 checked on 16, 24, 29, 32, 37, 39, 40, 61, 64, 95 ("set B"), which were not used for tuning or for the version-1 check.
- **By-eye check of version 2 on set B**: 360 sampled frames. "Correct" means the mask covers at least about two thirds of the visible ball and nothing else. The grasp window runs from 3 s before to 1.5 s after the gripper closes.

  | camera, samples | ball visible | correct | partial | missed | spill | mask without ball |
  | --- | --- | --- | --- | --- | --- | --- |
  | top, 150 spread over the episodes | 116 | 93 (80%) | 19 | 4 | 0 | 0 of 34 |
  | top, the 39 of those in the grasp window | 39 | 23 (59%) | 13 | 3 | 0 | - |
  | wrist, 150 spread over the episodes | 135 | 123 (91%) | 0 | 3 | 9 | 0 of 15 |
  | wrist, 60 in the grasp window | 60 | 57 (95%) | 0 | 0 | 3 | - |

- **Top camera failure.** When a jaw is next to or across the ball, the shaded part of the ball is dark olive and falls below the core chroma, so it is not masked. This happened in 5 of the 10 episodes. When the ball lies mostly under the jaw it is missed altogether (e61, e64).
- **Wrist camera failures.**
  - Spill: the mask runs over the dark gap and the foil between the ball and the right jaw, adding 10-30% to the ball. 7 of the 12 failures are in one episode (e40).
  - Miss: the ball lies partly under a foil ball (e61).
- **Holes in the held ball (wrist camera).** Lit cream on the held ball falls into the gripper colour range (hue 15-21, saturation 50 or more, value 200 or more) and is cut out. This was not counted as a failure, but it shows as yellow patches on the red ball. 22 of the 132 masked wrist frames (spread sample) have holes covering 2-10% of the ball. In 4 held-ball frames, 78-91% of the pixels inside the hull but outside the mask were gripper-coloured.
- **Automatic counts on all set-B frames.**
  - The mask is present in 75.4% of top frames and 81.9% of wrist frames.
  - Dropouts of 3 frames or less: 4 (top), 15 (wrist).
  - Centre jumps over 60 px: 0 (top), 11 (wrist).
  - Area jumps over 2.5x: 51 (top), 11 (wrist).
- **Red previews of e24, e40 and e95.**
  - Light and shadow on the ball stay.
  - In the wrist camera the stripes are mostly gone; thin lines remain where the stripes had their edges. In the top camera the stripes stay as dark red stripes.
  - Highlights turn light red, not white. There is a soft red fringe about 2 px wide around the ball.
  - In e40 (top camera), the paint switches on and off 21 times between frames 257 and 441, while the ball is mostly behind the jaw.
- **Speed, single process, quiet laptop.** Decode, mask, paint and AV1 encode run at 44.5 fps (top) and 22.1 fps (wrist).
  - All 2 x 47,401 frames take about 54 min one after another.
  - One file is one job, and the largest wrist file holds 20,866 frames, so even in parallel the run takes at least about 16 min.
- **Mask cost.** 7.1 ms (top) and 9.9 ms (wrist) per frame, measured in the version-1 check. The set-B figures (17 and 26 ms) were taken while two other decoding jobs were running.
- **Metadata.** `meta/` is copied unchanged, so its image statistics still describe the striped ball. According to `research/multi-colour-targets.md`, training normalises camera images with ImageNet statistics by default (read from the source, not tested).

## Alternatives

- **Hue shift of the masked pixels** - tried 2026-11-21 (earlier agent, `paint_try1.png`, `paint_try2.png`): the stripes keep their light-dark pattern and the ball comes out striped in two tones of red. Replaced by repainting from luminance, with green stripes raised to the brightness of the cream.
- **Mask version 1** (no tracking, wrist core hue up to 95, no roundness or paleness check) - tried 2026-11-21. The earlier agent's by-eye check on the 10 held-out episodes found 9 false masks in the 25 wrist frames without the ball: yellow objects on a shelf, lines on the table, a strip at the image bottom.
  - Version 2 therefore stops the wrist core at hue 86 (green mats lie above hue 90), requires roundness of at least 0.55, and rejects pale yellow (overexposed white tables and labels).
  - Version 2 also accepts a smaller seed within 40 px of the last position, so that a ball mostly hidden by the gripper stays tracked.
  - The earlier top judgements (111 of 116 OK) were not re-checked and look more lenient than the ones above.
- **Seed search at full resolution** (before version 1) - tried 2026-11-21 (`perframe_tune1.csv` against `perframe_tune2.csv`): 13.4 and 17.8 ms per frame, against 7.1 and 9.7 ms at half resolution. Detection on the tuning episodes stayed the same: top 75.8%, wrist 86.0% against 86.4%.
- **Fill the holes after the gripper cut instead of before** - tried 2026-11-21 on 190 set-B wrist frames (single frames, no tracking). It adds a median of 0.4% and at most 11% to the mask, all of it enclosed cream; in the 12 frames that changed most, no jaw was added. Not adopted, because it needs its own by-eye check.
- **Other routes to other colours** (one ACT per colour, a one-hot input, a marked ball): `research/multi-colour-targets.md`.

## Revisit if

- The top-camera mask covers the shaded part next to the jaw (for example a circle fitted to the visible part of the ball), and a fresh by-eye check passes this bar in both cameras:
  - at least 95% correct where the ball is visible,
  - at least 90% correct in the grasp window,
  - false repaints below 1%.
- Recording real red-ball episodes turns out cheaper than fixing the mask.
