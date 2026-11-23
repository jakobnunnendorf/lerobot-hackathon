# Voice commands

The dashboard is the normal way to speak to the robot: its 🎤 records in the browser and the server transcribes it (`docs/demo.md`). This page covers what you can say, how to add phrases, and the standalone voice program kept as a fallback. Why it works this way: `decisions/voice-commands.md`.

Both paths use the same speech model (Parakeet TDT 0.6b v2, English, on the CPU) and the same phrase table, `scripts/voice/commands.py`. Both send one JSON line to the robot program on TCP port 8765.

## What you can say

| Say | Does |
| --- | --- |
| "give me the ball", "hand it to me", "put it in my hand" | pick the striped ball, give it to the hand |
| "put it in the container" | pick the striped ball, put it down |
| "dance", "let's dance" | play the dance (`docs/dance.md`) |
| "bite the finger" | a bite round (`docs/finger-biting.md`) |
| "take out two red balls, then give me the striped ball" | a multi-step plan (`docs/demo.md`) |

- Naming no colour means the striped ball. Naming two colours in a single command is refused.
- A number makes a sentence a plan. A bare "one" does not ("the striped one"), and "a red ball" stays a single command.
- There is no wake phrase: the button (or Enter) marks the words as meant for the robot. A recording stops by itself after 8 s.

## Add a phrase

1. Add it to the destination's list in `DESTINATIONS` in `scripts/voice/commands.py`.
2. Add the sentence to `tests/voice_parse.py` and run it until it prints `PASS` (it needs `rapidfuzz`, which the `dashboard` env and the voice environment below have).
3. Optional, in the voice environment (below): `scripts/voice/benchmark.py --rescore` re-checks saved transcripts against the changed parser in seconds. Without `--rescore` it re-measures every recogniser on synthetic speech, which takes 1.5-2 hours.

A command needs a text-similarity score of 0.80 for one action and a lead of 0.10 over the other.

## Standalone voice program (fallback)

Use it when the browser microphone does not work. It runs on Windows, where the microphone is, in its own environment. Never install it into the robot environment.

```bash
python -m venv scripts/voice/.venv
scripts/voice/.venv/Scripts/python -m pip install -r scripts/voice/requirements.txt
scripts/voice/.venv/Scripts/python tests/voice_parse.py      # prints PASS
```

On macOS or Linux use `scripts/voice/.venv/bin/python`. The first start downloads the speech model (about 660 MB). On bad Wi-Fi, set `HF_HUB_OFFLINE=1` to load it from the cache.

Run it while the robot program runs (`demo_chain.sh` already listens on `0.0.0.0:8765` inside WSL):

```bash
scripts/voice/.venv/Scripts/python scripts/voice/listen.py
```

Press Enter, speak, press Enter. It prints what it heard, the command, and `sent {...}`. The keys `1` + Enter (give it to me) and `2` + Enter (put it in the container) always work, also without a microphone.

| You see | Do |
| --- | --- |
| `NOT UNDERSTOOD: no single command recognised` | Say one command. If the transcript is right but not matched, add the phrasing (above). |
| `ONLY SILENCE RECORDED` | Wrong or muted microphone: pick it (below). |
| `MICROPHONE FAILED` / `SPEECH-TO-TEXT FAILED TO LOAD` | Use the keys `1` and `2`. |
| `NOT SENT: nothing is listening` | Start the robot program. Right after it starts in WSL, Windows sometimes cannot reach it for 10-15 s: wait and send again. |

**Pick the microphone:** `listen.py --list-devices`, then `listen.py --device <number>`. Use the headset's `MME` line; `WASAPI` and `WDM-KS` lines fail with `Invalid sample rate`. A headset or clip-on microphone matters more than anything else in a loud hall.

**Without the robot:**

- `listen.py --dry-run` shows the transcript and the scores and sends nothing.
- `listen.py --wav some-command.wav` transcribes a file (16 kHz mono 16-bit) instead of the microphone.
- `python3 scripts/voice/link.py` prints the commands it receives on port 8765, as a stand-in for the robot program (standard library only). Use `--host 0.0.0.0` when it runs in WSL and the listener on Windows.
