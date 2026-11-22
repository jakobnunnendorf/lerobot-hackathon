# Can a learned top-camera classifier decide clear-or-pick better than a mint-pixel rule?

- Status: OPEN (2026-11-22). Built and scored on the hold-out; not wired into the supervisor.

## Decision

Ship the classifier (`scripts/scene_clf/`) as an optional second opinion, not as the supervisor's rule. On the hold-out it did **not** beat the mint-pixel baseline.

## What was built

- Input: one top frame at 160x120 plus the 6-dim state. Output: striped visibility (buried / partly / visible) and next action (clear_red / clear_blue / pick_striped).
- Model: ImageNet ResNet18, layers 1-3 frozen, layer4 fine-tuned, two linear heads on a 128-unit trunk. 8 epochs on MPS, about 1 minute.
- Labels: `labels.jsonl` (human, wins) over `prelabels.jsonl` (AI, latest line wins). Visibility = latest `s_*` code at or before t (`start_state` at t=0). Next action = the next `*_target` at or after t.
- Frames: 3 fps cache; kept the 1.5 s before each `*_target` plus t=0. 2585 train frames (191 episodes), 99 hold-out frames (the 10 hold-out episodes, never trained on).
- Baseline: mint fraction (hue 50-160, sat > 0.3, val > 0.35) inside a fixed bowl region of the top image; two thresholds for visibility and one for pick-vs-clear, tuned on train. It cannot tell red from blue, so it clears with the commonest train colour (red).

## Hold-out result (99 frames; one frame is 1 point, so differences under about 8 points are noise)

| | visibility acc | next-action acc |
|---|---|---|
| majority class | 0.354 | 0.535 |
| mint-pixel rule | **0.667** | **0.687** |
| ResNet18 + state | 0.566 | 0.667 |

Confusion matrices (rows true, columns predicted; visibility order buried, partly, visible; next action order clear_red, clear_blue, pick_striped):

- Model visibility `[[19,18,0],[5,19,11],[4,5,18]]`; mint rule `[[20,13,4],[0,35,0],[0,16,11]]`.
- Model next action `[[12,9,11],[1,11,2],[4,6,43]]`; mint rule `[[15,0,17],[5,0,9],[0,0,53]]`.

The model is the only one that ever says clear_blue (11 of 14 right), and it picks the striped ball as often as the rule does when the ball is visible (43 of 53). It loses on buried versus partly: 18 buried frames called partly. The rule never misses a pick_striped but also never recommends clear_blue.

## Why it is weaker than hoped

- At 160x120 the ball is about 20 pixels, and the label depends on how much of it is covered. Training accuracy keeps rising while hold-out stays at 0.57-0.65 from epoch 1 on: it overfits episodes (frames within one episode look alike).
- The labels are partly AI pre-labels (episodes 58-200), so visibility noise is part of the ceiling.
- Not tried (each costs a re-extraction or a longer run): a bowl crop at higher resolution, which gives the ball more pixels at the same input size; combining the model's probabilities with the mint fraction; hold-out-free early stopping.

## How the supervisor would call it (proposal, supervisor code untouched)

At each decision point, while the arm is parked:

```python
from scene_clf.predict import load_model, predict   # scripts/ on sys.path
loaded = load_model()                                # once, at start-up; CPU is fast enough (ResNet18, 160x120)
out = predict(top_frame_rgb_uint8, state=obs_state, loaded=loaded)
# out = {"visibility": "partly", "next_action": "clear_blue", "probs": {...}}
```

- Run it on the last 3 parked frames and average `probs`, to damp single-frame noise.
- Use it only to pick the clearing colour: when the existing rule says "not visible enough", `next_action` chooses clear_red or clear_blue (the mint rule cannot). Keep the mint rule for pick vs clear until a better number exists.
- CLI: `python -I scripts/scene_clf/predict.py frame.png --state 0 -20 30 60 0 10` prints the same JSON. Weights: `outputs/scene_clf/scene_clf.pt` (22 MB, fp16, committed with `git add -f`); retrain with `scripts/scene_clf/train.py --cache <dir>` where the cache is built by `scripts/label_data.py <dir>` (about 2 minutes, 1.5 GB).

## Revisit if

- A larger or human-only labelled set exists, or the cache is rebuilt as a bowl crop at 2x resolution: rerun `train.py`.
