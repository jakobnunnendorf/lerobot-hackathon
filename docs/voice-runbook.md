# Voice commands: runbook

For the person at the table. Commands are for Windows, from the repo root, in PowerShell or Git Bash; on macOS or Linux use `scripts/voice/.venv/bin/python` instead of `scripts/voice/.venv/Scripts/python`. Why it works this way: `decisions/voice-commands.md`.

## Install (once per laptop, on good internet)

The voice program has its own environment. Never install it into the robot environment.

```bash
python -m venv scripts/voice/.venv
scripts/voice/.venv/Scripts/python -m pip install -r scripts/voice/requirements.txt
```

With uv instead:

```bash
uv venv scripts/voice/.venv --python 3.12
uv pip install -r scripts/voice/requirements.txt --python scripts/voice/.venv/Scripts/python.exe
```

Check the parser: `scripts/voice/.venv/Scripts/python tests/voice_parse.py` must print `PASS`.

Start the listener once before the event (next section). The first start downloads the speech model (about 660 MB) into the Hugging Face cache; later starts load it from there. If a start hangs on bad Wi-Fi, set `HF_HUB_OFFLINE=1` (PowerShell: `$env:HF_HUB_OFFLINE=1`) and start again; loading from the cache with it set was tested.

## Run

Terminal 1, until the robot program receives commands itself:

```bash
scripts/voice/.venv/Scripts/python scripts/voice/link.py
```

It prints `waiting for commands on 127.0.0.1:8765`.

Terminal 2:

```bash
scripts/voice/.venv/Scripts/python scripts/voice/listen.py
```

## What a good run looks like

1. The listener prints `ready in ... s` and `microphone: <the headset you plugged in>`.
2. Press Enter, say "Hey bot, give it to me", press Enter.
3. A box shows `HEARD 'Hey bot, give it to me.'` and `>>> HAND  (striped ball)`, then `sent {...}`.
4. Terminal 1 prints the same JSON line.
5. Same with "Hey bot, put it in the container": `>>> CONTAINER`.

Say "hey bot" first, then the command, in one breath. A recording stops by itself after 8 s.

## When something goes wrong

| You see | Do |
| --- | --- |
| `NOT UNDERSTOOD: no "hey bot" at the start` | Say it again, "hey bot" first and clearly. |
| `NOT UNDERSTOOD: no single command recognised` | Say one of the two commands, not both. |
| `ONLY SILENCE RECORDED` | Wrong or muted microphone. See below. |
| `MICROPHONE FAILED` or `SPEECH-TO-TEXT FAILED TO LOAD` | Type `1` + Enter (give it to me) or `2` + Enter (put it in the container). Same commands, `"source": "keyboard"`. |
| `NOT SENT: nothing is listening` | Start the robot program or terminal 1. The listener keeps running; just say the command again. |

The keys 1 and 2 always work, also while the microphone is fine. The presenter can say the sentence aloud and press the key.

## Pick the microphone

```bash
scripts/voice/.venv/Scripts/python scripts/voice/listen.py --list-devices
scripts/voice/.venv/Scripts/python scripts/voice/listen.py --device 1
```

Use the number of the headset's line that says `MME`. `WASAPI` and `WDM-KS` lines fail with `Invalid sample rate`. A headset or clip-on microphone close to the mouth matters more than anything else in a loud hall.

## Check without a microphone

```bash
scripts/voice/.venv/Scripts/python scripts/voice/listen.py --wav some-command.wav
```

The file must be 16 kHz mono 16-bit WAV.

## Add a phrase

1. Add it to the destination's list in `DESTINATIONS` in `scripts/voice/commands.py`. A new destination also needs a key in `KEYS`, and the robot program must know its name.
2. If the screen shows the wake phrase written wrongly (for example `Hay but`) and it was rejected with `no "hey bot" at the start`, add that spelling, lower case without punctuation, to `WAKE_VARIANTS`. Not ordinary words such as `People`: then almost any sentence would pass.
3. Add the sentence to `tests/voice_parse.py` and run it until it prints `PASS`.
4. Optional: `scripts/voice/.venv/Scripts/python scripts/voice/benchmark.py --rescore` re-checks the saved transcripts against the changed parser in seconds. Without `--rescore` it re-measures all recognisers on synthetic speech, about 1.5 to 2 hours (Whisper small alone about an hour); `--latency` times a fixed subset only, about 15 minutes on mains power with nothing else running. The saved transcripts live in `scripts/voice/.venv/tmp/benchmark/` on Tian's laptop only.

## Robot program in WSL2

The listener stays on Windows, where the microphone is, and sends to `127.0.0.1:8765`; the robot program in WSL receives on `127.0.0.1:8765`, the default. Check this on the robot laptop before the demo:

1. In WSL, from the repo: `python3 scripts/voice/link.py` (plain `python3` is enough, `link.py` needs only the standard library).
2. On Windows, from the repo: `scripts/voice/.venv/Scripts/python scripts/voice/listen.py`, then type `1` + Enter.
3. The WSL terminal must print `{"target": "striped", "destination": "hand", ..., "source": "keyboard"}`.

If the listener prints `NOT SENT`, wait 15 s and press `1` + Enter again: right after the receiver starts in WSL, Windows sometimes cannot reach it for a few seconds. If it still fails, restart the receiver with `--host 0.0.0.0`. Sending the other way, from WSL to a program on Windows, did not work in our test (`decisions/voice-commands.md`).
