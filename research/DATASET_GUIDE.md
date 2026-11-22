# Striped-ball datasets: labels, episode patterns, and curation per architecture

**Load this before you build a training set.** Written 2026-11-22 by Jakob's agent, from:
- Jakob's hand labels (rb_mixed 0–57);
- the v2 AI pre-labels (the other 288 episodes, about 4% ball-event errors);
- Tomi's agent's analysis in PR #54 (`research/label-patterns.md`, `segments.csv`, `episodes.csv`);
- Tian's agent's analysis in PR #53 (`research/smolvla-clear-first.md`).

The guide describes **what happens in the data and why it matters**. The choice for your model is yours: §5 gives defaults, not rules.

## 1. Datasets and where the labels live
| dataset | episodes | scene | label source |
|---|---|---|---|
| `molnrb/so101_pick_rb_mixed` | 201 | red + blue balls, one striped | 0–57 Jakob, 58–200 AI v2 |
| `molnrb/so101_pick_buried_ball` | 45 | grey **foil** balls, striped often buried | AI v2 |
| `molnrb/so101_pick_blue_ball` | 100 | foil balls, striped mostly visible | AI v2 |

- **Hand labels:** branch `labels`, `labels/jakob.jsonl`. When an episode has several lines, the last one wins.
- **AI labels:** branch `Space0code/prelabel-v2-claude`, `prelabel-kit/prelabels_v2_claude.jsonl`.
- **Segments and per-episode flags:** PR #54, `research/label-patterns/{segments,episodes}.csv`.
- **Read-only:** never write to the `labels` branch.

## 2. Label vocabulary (one record = timestamped `steps`)
| code | meaning |
|---|---|
| `r_/b_/s_` + `target` | the arm starts heading for the red / blue / striped ball |
| `…grab` | the gripper closes on it and it is held |
| `…miss` | the gripper closes on nothing, or on the wrong thing |
| `…slip` | the ball was held, then dropped before reaching its place |
| `…hole` | the red / blue ball is dropped into its matching hole (a "clear") |
| `…putback` | the ball is put back into the bowl (a bad demo pattern) |
| `s_vis` / `s_partly` / `s_buried` | **striped visibility from the top camera, starting at t** (it changes during an episode) |
| `wrist_blind` | the striped ball is hidden from the **wrist** camera (other balls in front) while the operator still sees it |
| `home` / `end_away` / `end_empty` | the episode ends at the home pose holding the striped ball / ends elsewhere / ends without the ball |

- **Start state:** `start_state` holds visibility at t=0.
- **Foil balls** in buried_ball and blue_ball are logged with `r_` codes. They are **not** red clears, because they go to a different hole. Keep them out of any clear-red set.

## 3. Sequences that occur (all 346 labelled episodes)
Shorthand: `SG` = striped grab, `SM` = striped miss, `SS` = striped slip, `RH`/`BH` = a red/blue ball cleared into its hole.

**Striped sub-sequence:**
| pattern | episodes |
|---|---|
| `SG` (first try) | 281 |
| `SM SG` | 37 |
| `SM SM SG` | 7 |
| `SG SS SG` | 7 |
| 3–7 misses before the grab, or mixed slip/miss | 11 |
| no striped grab | 1 (blue_ball) |

**Scene paths (start → clears → striped pick):**
- **A. Visible, direct:** `[vis] SG home`. 130 clean + 14 with misses. The easy case.
- **B. Buried, clear-until-uncovered:** `[buried] RH [partly] (BH …) SG home`.
  - The first drop uncovers the stripe in 20 of 34 of Jakob's buried starts, the second in 9.
  - 2–3 clears is typical; 56 episodes clear 3 or more.
- **C. Partly covered: a coin flip.** In rb_mixed, 19 clear first and 16 go direct. In buried_ball partly-covered starts always clear first; in blue_ball they almost never do. "Partly" therefore means different things across datasets.
- **D. Contradicting starts:** buried but picked direct (8 rb_mixed), or visible but cleared first (11 rb_mixed).
- **E. Keeps clearing after the stripe is uncovered:** 10 of Jakob's 31 buried episodes clear 1–3 more balls after `s_partly`; the other 21 go straight for the striped ball.
- **F. Failure → recovery:** each of these is followed by a recovery:
  - striped miss: rb 32/201, buried 13/45, blue 10/100;
  - striped slip: 15;
  - red/blue miss: 36;
  - put-back: 7.

  The striped ball always ends up held at home, with one exception.
- **G. Pushed back under:** a clear buries the stripe again (rb_mixed 22). Rare; not a pattern to curate for.

## 4. The key failure pattern: grabbing the striped ball while it is still covered
**Jakob's observation (and the main lesson of the labelling).** The operator often goes for the striped ball while it is only **partly** uncovered:
- the gripper reaches into the pile and pushes balls around, including the striped ball itself;
- other balls fill the wrist camera, so the camera is effectively blind at the moment that matters.

**Visibility at the moment of the striped grab:**
| top-camera state at `SG` | first try, wrist sees it | first try, wrist blind | after a miss or slip |
|---|---|---|---|
| visible | 146 | 11 | 17 |
| partly | 67 | 36 | 27 |
| buried (AI labels mostly) | 25 | 8 | 9 |

- In **Jakob's own 58 episodes**, 43 grabs happen while the stripe is only partly visible.
- **Wrist blindness sits almost entirely in partly or buried grabs:** 27 of 31 in his labels, against 1 of 12 blind grabs when it was visible.
- **Measured from frames (PR #54):** 72 of 345 episodes have a blind final approach. 36 never show more than a sliver of stripe; the other 36 lose sight of it 0.6 s or more before the grab.

**Why it trips models.** A wrist view full of red or blue looks exactly like a normal clearing grab. In a blind approach the right action is "keep going down to the striped ball underneath". In a clear it is "grab this ball". Only the joint state and a few top-camera pixels separate the two. The striped ball covers at most about 11% of one SmolVLA visual token.

**The behaviour we want instead:** clear the covering balls one by one until the striped ball lies **bare** (`s_vis`), then pick it with the wrist camera seeing it. The data has few clean examples of the full path `[buried] … clears … [vis] SG(seen)`, so either filter for them, or record more (§6).

## 5. Curation defaults per architecture
| | SmolVLA / single-frame VLA (one frame, re-plans every 1.7 s) | ACT (chunk 100 = 3.3 s open loop) | multi-skill (supervisor + per-colour ACT) |
|---|---|---|---|
| unit | whole episodes **or** segments | segments preferred | segments (`segments.csv`: `clear_r`, `clear_b`, `pick_s`) |
| blind striped approach | drop the `pick_s` part of all 72 | keep blind < 1.7 s (26); drop the 36 sliver-only ones and blind > 1.7 s (10) | pick skill as for ACT |
| grab while `partly` | prefer `vis`-at-grab picks; keep partly only if the wrist sees it | keep if the wrist sees it | the supervisor sends "pick" only when visible; train pick on visible-at-grab |
| contradicting starts (D) | drop (19 rb_mixed), or condition on two task sentences | harmless for single skills | harmless; the supervisor decides |
| keeps clearing after uncover (E) | drop, or use a separate sentence: it teaches the opposite of the other 21 | fine as clearing segments | fine |
| failure + recovery (F) | clean segments only: one frame can't tell a first try from a retry | start the segment **after** the failed close (the recovery tail is useful) | same as ACT |
| put-backs | drop | drop | drop |
| foil balls | not red; separate the datasets or the sentence | don't mix into clear-red | own skill, or excluded |
| buried share | keep about 40–50% buried starts (a strict filter drops it to 26%) | — | balance per skill |

- **Segments vs episodes:** dropping whole episodes for one bad moment throws away 62% of the clean clearing footage. Cut only at the boundaries in `segments.csv`; never splice two ranges together.
- **Hold-out:** rb_mixed was recorded in blocks: 50–74 mostly buried, 75–124 mostly direct, 150–174 mostly buried. Hold out e.g. every 10th episode, **never the last N**. The old held-out 114–119 are all direct picks, so their eval loss can't rank clearing behaviour.
- **Task sentences:** one fixed sentence carries no information. If the policy must choose "clear or pick", give scene-specific sentences, or let a supervisor choose (Tian's fix 1).
- **Spot-check AI-labelled cut points** (about 10 per set) before training on episodes 58+ or the foil sets.

## 6. Recording rules these findings suggest
1. Clear until the striped ball is **bare**, then pick it. Never reach into a partly covered pile.
2. Approach from a side where the wrist camera keeps the striped ball in view.
3. Open the gripper wide (40 or more) before the close.
4. Never put balls back into the bowl.
5. End at the home pose holding the striped ball.
6. Mix scenarios within each batch instead of recording them in blocks.

## 7. Known label weaknesses
- AI pre-labels:
  - start state is right about 78% of the time; "partly" and "visible" are hard to tell apart;
  - visibility changes and `wrist_blind` are found only about half the time. For blindness, use the frame-measured columns in `episodes.csv`.
- Quick failed pinches are often left unlabelled, by Jakob as well. Use the gripper trace in `observation.state` if you need them.
- Statistics that depend on visibility changes should use Jakob's labels (rb_mixed 0–57) only.
