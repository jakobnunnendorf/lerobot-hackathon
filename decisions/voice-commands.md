# How does the robot take spoken commands?

- Status: DECIDED (2026-11-21)

## Decision

A separate program in its own environment (`scripts/voice/listen.py`) records while the person presses Enter (Enter again stops, at most 8 s), transcribes the recording on the CPU with Parakeet TDT 0.6b v2 (English) through `onnx-asr` (`--backend parakeet-en`, the default), and keeps it only if it starts with "hey bot" and matches one command in a synonym table (fuzzy matching, no LLM). Each command goes as one JSON line over TCP to 127.0.0.1:8765; the robot program receives it with `CommandReceiver` from `scripts/voice/link.py`. Keys 1 and 2 send the same commands without the microphone.

## Why

- **Push-to-talk, not always listening.** The hall is loud; one project caught 5 of 20 wake words in noise against 19 of 20 in a quiet room (`research/voice-commands.md`). Transcribing only on demand keeps the CPU free for the robot's 30 Hz loop on the same laptop (assumed, not measured with the robot running). The wake phrase is still required in the transcript, so the "wake-up call" stays visible.
- **Enter to start and Enter to stop** needs no global key hook library and behaves the same in every terminal (assumed; only tried in Git Bash on Windows, and with piped input).
- **Parakeet v2 never produced an unwanted or wrong command, and it is the fastest** in the measurement below: 0 false accepts and 0 wrong destinations over 980 clips, 475 of 560 commands understood, 0.31 s median per clip on mains power (0.53 s on battery). Parakeet v3 is as fast but understood far fewer commands (367); Whisper base without a prompt also made no mistakes but understood fewer (451) and is presumably as slow as with the prompt (not timed quietly). It is not the most accurate: Whisper with our prompt understood more commands (base.en 547, small.en 554 of 560), but the prompt leaks into its transcripts, so it accepted sentences that must be rejected (9 and 2 clips) and sent 1 and 2 commands to the wrong destination (Alternatives); it is also 3 and 10 times slower. We chose fewer wrong actions over fewer repeats: a missed command is said again, a wrong one sends the ball to the wrong place. This is a judgement on synthetic speech; Parakeet v2 missed 85 of 560 commands, most of them at 5 dB SNR, so expect repeats in a loud hall.
- **A synonym table is enough** for two fixed commands with one slot each (`research/voice-commands.md`, section 6). `tests/voice_parse.py` holds the accepted and rejected sentences.
- **TCP, voice program as client, one connection per command.** The robot program may run in WSL2 while the microphone is on Windows. Tested on Tian's laptop (2026-11-21; WSL 3.0.5, Ubuntu 24.04, default NAT networking, no `.wslconfig`): `link.py` inside WSL bound to 127.0.0.1 or to 0.0.0.0 received commands sent from Windows to 127.0.0.1:8765, also end to end from `listen.py --wav`; binding 0.0.0.0 is not required. With 127.0.0.1, 2 of 8 starts were not reachable from Windows for the first 11 s (one probe) or still at 15 s (one try with key 1); with 0.0.0.0, 3 of 3 starts were reachable within 2 s. The other direction, a receiver on Windows (bound to 127.0.0.1 or 0.0.0.0) and the sender in WSL (to 127.0.0.1 or to the Windows host address), failed in every try. The robot laptop's WSL settings were not checked. Connecting per command makes start order and restarts of either program irrelevant.
- **Own process and environment**, so speech packages (onnxruntime, numpy 2.5) never touch the robot environment's pins (`CLAUDE.md`).

## Measurement (2026-11-21, Tian's laptop, synthetic speech)

Synthetic speech from the offline Windows synthesiser is not a person in a hall. These numbers rank the recognisers; they do not predict the demo. Not tested at all: live human speech, accents, a real headset, the real hall.

- Clips: 14 sentences (8 commands starting with "hey bot", 4 phrasings per destination; 6 that must be rejected: no wake phrase, unrelated request, both commands at once) x 7 voices (David, Hazel, Zira Desktop; Zira, Susan, George, Mark OneCore) x 2 speaking rates x 5 conditions (clean; white noise and 4-talker synthetic babble at 10 dB and 5 dB SNR) = 980 clips per recogniser (196 per condition), 0.5 s silence before and after.
- Machine: AMD Ryzen 5 5500U, 6 cores, 15 GB, no GPU.
- "Right" means the full pipeline (speech-to-text plus parser) gives the expected destination, or nothing for a sentence that must be rejected. A false accept is a command from one of the 420 must-reject clips; a wrong destination is a command clip sent to the other destination.
- Re-run: `scripts/voice/benchmark.py`; `--rescore` re-parses saved transcripts, `--latency` times every 7th clip only.

All right, with the wake list in `commands.py` (last column: with only "hey bot", "heybot", "hey robot" plus fuzzy matching):

| `--backend` | clean | white 10 dB | white 5 dB | babble 10 dB | babble 5 dB | all | commands understood (of 560) | false accepts (of 420) | wrong destination | all, before the added spellings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `parakeet-en` (Parakeet TDT 0.6b v2) | 100% | 98% | 86% | 93% | 80% | 91% | 475 | 0 | 0 | 86% |
| `parakeet` (Parakeet TDT 0.6b v3) | 100% | 87% | 77% | 83% | 55% | 80% | 367 | 1 | 0 | 78% |
| `whisper-base` (base.en, with prompt) | 99% | 100% | 98% | 98% | 92% | 98% | 547 | 9 | 1 | 98% |
| `whisper-base-no-prompt` (benchmark only) | 100% | 92% | 88% | 92% | 72% | 89% | 451 | 0 | 0 | 83% |
| `whisper-small` (small.en, with prompt) | 100% | 100% | 99% | 99% | 97% | 99% | 554 | 2 | 2 | 99% |

Latency per clip after loading, first call excluded, as median / 90th percentile / worst; load is the time to build the backend in a fresh process. The quiet runs (140 clips, every 7th, the same for every backend, one backend at a time, no other job) were done once on battery and once on mains power, Balanced power plan both times; power state from the Windows event log. On battery everything is about 1.7 times slower. The tails on mains power are noisier than on battery (VS Code's Python helpers restarted during the `whisper-small` run), so trust the medians there. The full accuracy runs shared the CPU with other jobs.

| `--backend` | quiet, mains power | load | quiet, battery | load | full run, 980 clips, CPU shared | load |
| --- | --- | --- | --- | --- | --- | --- |
| `parakeet-en` | 0.31 / 0.40 / 0.60 s | 3.2 s | 0.53 / 0.67 / 0.92 s | 6.2 s | 0.30 / 0.41 / 3.80 s (mains) | 3.2 s |
| `parakeet` | 0.31 / 0.40 / 0.49 s | 3.0 s | 0.53 / 0.65 / 0.94 s | 6.9 s | 0.32 / 0.40 / 0.65 s (mains) | 3.3 s |
| `whisper-base` | 0.93 / 1.02 / 6.09 s | 1.4 s | 1.56 / 1.65 / 2.23 s | 7.4 s | 1.00 / 1.28 / 6.70 s (mains) | 0.9 s |
| `whisper-small` | 3.05 / 4.67 / 8.78 s | 2.3 s | 5.20 / 5.34 / 6.45 s | 5.0 s | 3.43 / 7.38 / 19.62 s (battery, mains from 11:10) | 8.2 s |
| `whisper-base-no-prompt` | not measured | | not measured | | 1.85 / 3.21 / 16.49 s (battery) | 5.0 s |

How the recognisers wrote "hey bot" when it was said (770 clips each: the 8 commands and the 3 rejects that start with it), read the way the parser reads the first one to three words, with the final list:

| `--backend` | exactly "hey bot" | close enough for fuzzy matching | other spellings in the list | wake phrase missed |
| --- | --- | --- | --- | --- |
| `parakeet-en` | 436 | 158 ("hey boss" 53, "hey bob" 30, "hey box" 25, "hey but" 11) | 63 ("a bot" 17, "paybot" 13, "paypot" 4) | 113 (empty transcript 24) |
| `parakeet` | 302 | 193 ("hey bob" 94, "hey boss" 22, "hey bolt" 18) | 30 ("hey bud" 7, "heybox" 4, "hey pop" 4) | 245 (empty 99, "people ..." 46, "paper ..." 11) |
| `whisper-base` | 758 | 1 | 0 | 11 |
| `whisper-base-no-prompt` | 344 | 188 ("hey boss" 67, "hey but" 37, "they both" 20) | 81 ("paybot" 19, "a bot" 19, "heybot" 8) | 157 ("they bought" 24, "people ..." 25, "payback ..." 15, "he bought" 9) |
| `whisper-small` | 765 | 0 | 0 | 5 |

Rule for `WAKE_VARIANTS` (2026-11-21): a spelling goes in if the recognisers wrote it for "hey bot" at least twice over all five runs (spaced and joined forms such as "pay box" and "paybox" count together), and it still sounds like "hey bot": a joined non-word ("paybought", "peabot") or a "hey"-like syllable followed by one syllable starting with b, p, v or f ("hey fox", "pay bought"). Ordinary sentence starts stay out even when frequent: "people", "paper", "payback", "they bought", "he bought", "a bottle", "playbook", "the robot", "take off". Times each entry matched over the five runs: "a bot" 36, "paybot" 32, "hey bud" 15, "heybot" 10, "hey bye" 7, "paypot", "hey pop", "hey buck" 5 each, "paybox" 4 + "pay box" 1, "hey bosch", "hey vaughn", "heybox" 4 each, "playbot" 3 + "play bot" 1, "hey von", "baybot", "hey fox", "paybook", "hey bach", "paybought" 3 each, "pay bought", "haybox", "k bot", "hey vaugh", "haybot", "hey folks", "hey vod", "hey books", "peabot", "pay bot" 2 each; "hey robot" 0 (kept from the start). No must-reject clip passes the wake check with this list in any of the five runs.

Effect of the added spellings (all right, then commands understood of 560): `parakeet-en` 86% to 91% (423 to 475), `parakeet` 78% to 80% (349 to 367), `whisper-base-no-prompt` 83% to 89% (392 to 451), `whisper-base` and `whisper-small` unchanged; false accepts unchanged everywhere. The spellings were picked from these same transcripts, so the gain on new speech will be smaller.

## Alternatives

- **Parakeet TDT 0.6b v3 (multilingual), the research pick** - tried, see table: worse than the English-only v2 on every noisy condition; in white noise it wrote "People" or "Paper" for "hey bot" 56 times, and it returned an empty transcript for 128 clips, mostly in babble. Kept selectable as `--backend parakeet`.
- **faster-whisper base.en with the two commands as prompt** - tried, see table: understood far more commands than Parakeet in noise (547 of 560), but writes the prompt's sentences instead of what was said. 9 of the 70 "give it to me or put it in the container" clips came out as "Hey bot, give it to me." and were accepted, and one "hand it to me" in 5 dB babble came out as "Hey bot, put it in the container." Three times Parakeet's latency. Kept as `--backend whisper-base`.
- **faster-whisper base.en without the prompt** - tried, see table: no false accepts, but fewer commands understood than Parakeet v2 (451 against 475), and presumably as slow as base.en with the prompt (its latency was not measured on a quiet machine). It writes "hey bot" as "they bought" (24 times) or "he bought" (9), which stay rejected (below).
- **faster-whisper small.en with the prompt** - tried, see table: the most accurate (554 of 560), but the same prompt leak as base.en: 2 false accepts, and 2 "hand" commands ("hand it to me" in 5 dB white noise, "give me the ball" in 5 dB babble) came out as "Hey bot, put it in the container." About 10 times Parakeet's latency. Kept as `--backend whisper-small`.
- **A Whisper prompt with only the wake phrase** ("Hey bot.") - considered, not tried: might keep Whisper's sure wake phrase without writing whole command sentences nobody said.
- **Accepting the ordinary words heard in place of "hey bot"** ("people", "paper", "payback", "they bought", "about", "table", "take off") - tried on the saved transcripts: they would recover some noisy clips, but the wake check would then pass almost any sentence that starts with them. Not done.
- **Always-listening wake word** (openWakeWord, livekit-wakeword, Porcupine) - considered, not tried: worst part in noise (above), "hey bot" needs a custom-trained model, Porcupine's free tier is reported to end (`research/voice-commands.md`, section 1).
- **Hold a key to talk** - considered, not tried: needs a global key hook library (`keyboard`, `pynput`), which needs extra permissions on macOS and Linux (assumed).
- **Claude or another LLM as parser** - considered, not tried: adds network, an API key and an open rule question (does a cloud call count as our own inference?) for two fixed commands (`research/voice-commands.md`, addendum).
- **UDP** - considered, not tried: Windows-to-WSL2 localhost forwarding carries TCP but not UDP in the default network mode (team's information, not tested).
- **Robot program as client, voice program as server** - considered, not tried: the robot loop would have to reconnect whenever the voice program restarts; and from WSL2 to Windows nothing got through in our test (above).
- **Noise suppression or voice activity detection** - considered, not tried: push-to-talk ends the recording; denoisers made recognition worse in a benchmark (`research/voice-commands.md`, section 4).
- **Spoken replies** - dropped by the team (reported by Tian, 2026-11-21): speech recognition only; the screen shows the transcript and the command.

## Revisit if

- The test in the hall with the real headset shows `NOT UNDERSTOOD` often: add the spellings shown on screen to `WAKE_VARIANTS` (same rule as above); if the transcripts themselves are wrong, try `--backend whisper-base` (more commands understood, but watch for transcripts that repeat the prompt) and a closer microphone.
- The voice laptop is much slower than a Ryzen 5 5500U, or speech-to-text visibly disturbs the control loop.
- The robot program's port is not reachable from Windows on the robot laptop (`docs/voice-runbook.md`, last section).
- The demo needs hands-free activation, or many more commands with free phrasing (then an LLM fallback behind the table).
