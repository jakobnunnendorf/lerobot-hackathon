# Per-episode curation table

`episodes.csv` has one row per episode of molnrb/so101_pick_rb_mixed (0–119), so101_pick_blue_ball and so101_pick_buried_ball: 265 rows.

- **verdict:** KEEP, DROP or LOOK.
- **KEEP** means:
  - the episode ends holding the striped ball near the start pose;
  - every cleared ball went into a hole, and none was put back;
  - there is at most 1 regrasp on the striped ball.
- **label_source:**
  - `stage1` = automatic checks from joint data;
  - `look_part*` / `drop_part*` = re-labelled from top and wrist video by agents;
  - `new40` = automatic checks on episodes 80–119.
- **scene:** rb_buried, rb_visible, foil_buried or foil_visible.
- **Caveat:** the datasets record no gripper load. The automatic colour labels were unreliable and were overridden from video where it mattered.
