# ACT on the cleanest rb_mixed episodes: from the all-data ACT vs from scratch

**Decided 2026-11-22 by Jakob.** Two runs on the same clean episodes, so only the starting weights differ.
- `act_clean_from_all`: starts from Kamel's ACT pretrained on all 314 striped-ball episodes (`brouthen/act_so101_pick_striped_all_pretrain_50k`, step 050000).
- `act_clean_scratch`: starts from fresh weights.

**Clean set** (so101-data skill §5, ACT rules), from Jakob's labels for episodes 0–57 and the v2 pre-labels for 58–200:

| rule | dropped |
|---|---|
| any miss, slip or put-back (§5.5) | 56 |
| wrist blind for 1.7 s or more, or never clearly seen (§5.2) | 28 |
| contradicting start: buried picked directly, or visible cleared first (§5.4) | 11 |
| automatic-check DROP | 3 |

- **Kept: 103 episodes:** 58 visible, 18 partly covered, 27 buried.
- **Hold-out:** every 10th kept episode, held out for both runs (§6): 6, 23, 65, 88, 99, 110, 126, 157, 177, 191. The other 93 train.
- **Known weakness:** buried starts are only 26%, below the skill's 40–50%, because whole episodes are dropped for one failure.
  - The fix is segments (PR #58's builder): cut after the failed close instead of dropping the episode.
  - Not done here, to keep this run simple and comparable.
- **Same settings as `act_redblue`:** augmentation, eval split, seed, lr 2e-5 at batch 32.
