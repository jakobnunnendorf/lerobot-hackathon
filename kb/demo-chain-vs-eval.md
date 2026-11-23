# The demo chain does not run a policy the way `eval_pick` does

- Date: 2026-11-22
- Verified by: agent of TianK003, read from the code on `main` at `9039757` and the eval changes on the unmerged branch `molnrb/eval-pick-multi`. The first two rows were the main cause and were fixed in #67 (2026-11-22 06:35): `demo_chain.sh` now runs v2 `@20000` at 30 action steps by default, and the demo ran the hidden model the same way (`decisions/hidden-ball-picking.md`). The step-by-step check below was never run on the arm, so the effect of the other rows is not measured.

Tian saw the same model pick worse through the dashboard than in `eval_pick`. Voice, ASR, the dashboard and the hand tracker cannot change PICK's policy or settings. The defaults of `scripts/supervisor/demo_chain.sh` do.

| | `eval_pick` on 2026-11-22 | `demo_chain.sh` by default |
|---|---|---|
| checkpoint | `Space0code/act_redblue_from_foil_v2@20000` | `Space0code/act_redblue_from_foil@015000` before #67, then v2 `@20000` |
| `n_action_steps` | 25 or 50 | the checkpoint's 100 before #67, then 30 |
| between policy and gripper | nothing | the grasp check (`grasp_check.py`): on the first close it freezes the arm, squeezes, judges hold or miss; a hold resets the policy and locks the gripper, a miss opens, backs off and restarts PICK |
| gripper at the start of a pick | 17.8, as in training | 30 |
| attempts | one, 75 to 90 s | up to 3 of 90 s (raisable live from the dashboard); a timeout sends the arm to rest and restarts with an empty action queue |
| held-ball check | none | `--min-held-target-area 120`: a striped hold reading under 120 px is put down as a wrong ball |

- The grasp check's position thresholds (holding band 6 to 20) were set on the old gripper servo. If an empty jaw reads 6 or more on the new one, the supervisor reports a hold with nothing in it.
- With the grasp check on, a ball the policy grabs in order to clear it is locked in the gripper (inferred from the code, to be confirmed at the table).
- A command during a round only sets target and destination; the dashboard refuses commands mid-round and nothing queues. Speech cannot abort; only the dashboard button and the `a` or `q` key can.
- "give me the ball" is target striped, destination hand: REST, PICK, VERIFY_HOLD, present pose, GIVE.

## Command closest to the eval

```bash
PICK_POLICY=Space0code/act_redblue_from_foil_v2 bash scripts/supervisor/demo_chain.sh --step 20000 \
  --n-action-steps 25 --no-grasp-check --max-pick-attempts 1 --min-held-target-area 0 \
  --rest -5.6 -102.8 96.6 -100.1 -5.2 17.8 --device cuda --give-mode present
```

- Tomi's v2 repos use unpadded step folders: `20000`, not `020000`.
- Rest gripper 17.8 is only safe with `--no-grasp-check`. With the grasp check on, use 21. The gripper must read outside 6 to 20 when the program starts.

## Finding the cause, one change at a time

1. The command above with no dashboard, voice or tracker. It should match the eval.
2. Add the dashboard with `--no-video --no-speech`.
3. Remove those two flags and speak while it picks.
4. Start the hand tracker and go back to `--give-mode follow`.
5. Remove `--no-grasp-check`, rest gripper 21, and read the `grasp_check` lines in `~/so101/demo/chain_<time>.jsonl`.

Slow ticks in the demo chain: gaps above 33.3 ms between consecutive `time` values with `state == "PICK"` in `chain_<time>.ticks.jsonl`. The eval's share was 1.0 to 1.6%.

## No flag removes these

- The supervisor's speed cap is 15 degrees per tick relative to the last goal and does not cap the gripper; the eval uses the robot's `max_relative_target=15`.
- `--tick-log` reads the load and writes a line every tick.
- The camera backend is OpenCV's default; the eval asks for V4L2.
- The supervisor has no temporal-ensembling option.
