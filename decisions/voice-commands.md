# How does the robot take spoken commands?

- Status: DECIDED (2026-11-21; wake phrase dropped the same day)

## Decision

A separate program in its own environment (`scripts/voice/listen.py`) records while the person presses Enter (Enter again stops, at most 8 s), transcribes the recording on the CPU with Parakeet TDT 0.6b v2 (English) through `onnx-asr` (`--backend parakeet-en`, the default), and sends a command if the transcript matches exactly one command in a synonym table (fuzzy matching, no LLM). There is no wake phrase: pressing Enter is the wake-up call. Each command goes as one JSON line over TCP to 127.0.0.1:8765; the robot program receives it with `CommandReceiver` from `scripts/voice/link.py`. Keys 1 and 2 send the same commands without the microphone.

## Why

- **Push-to-talk, not always listening.** The hall is loud; one project caught 5 of 20 wake words in noise against 19 of 20 in a quiet room (`research/voice-commands.md`). Transcribing only on demand keeps the CPU free for the robot's 30 Hz loop on the same laptop (assumed, not measured with the robot running).
- **No wake phrase** (team's decision, 2026-11-21, after trying "hey bot" live with `--dry-run`: too often not recognised). The measurements agreed: Parakeet v2 wrote "hey bot" exactly in only 436 of 770 clips, and even with a list of 30 measured mis-spellings it missed the wake phrase in 113 (Alternatives). Re-scoring the same saved transcripts without the wake check: commands understood went from 475 of 560 to 660 of 700, still with 0 false accepts and 0 wrong destinations. What is lost: a filter for talk not meant for the robot while Enter is held; the person holding the microphone has to avoid that.
- **The microphone stays open and keeps the last 0.5 s.** Opening the stream on Enter, as before, delivered the first audio 124-142 ms later on Tian's laptop (built-in Realtek microphone, 5 tries), and 331 ms on the first opening, so a command spoken right after Enter lost its start; without a wake phrase that start is the command. The recording also runs 0.3 s past the second Enter (Enter is often pressed on the last syllable), and 0.5 s of silence is added at each end before transcription, as in the benchmark clips. Not yet compared live against the old recording.
- **Enter to start and Enter to stop** needs no global key hook library and behaves the same in every terminal (assumed; only tried in Git Bash on Windows, and with piped input).
- **Parakeet v2 never produced an unwanted or wrong command, and it is the fastest** in the measurement below: 0 false accepts and 0 wrong destinations over 980 clips, 660 of 700 commands understood, 0.31 s median per clip on mains power (0.53 s on battery). Parakeet v3 is as fast but understood fewer commands (553) and made 2 false accepts and 1 wrong destination. Whisper base without a prompt understood more (682) with 1 false accept ("and early AVOT to open the window." in 5 dB babble) and is slower (1.85 s median, on battery with the CPU shared; not timed quietly); it exists only in the benchmark. Whisper with our prompt understood the most (base.en 691, small.en 696 of 700), but the prompt leaks into its transcripts, so it accepted sentences that must be rejected (9 and 2 clips) and sent 1 and 2 commands to the wrong destination (Alternatives); it is also 3 and 10 times slower. We chose fewer wrong actions over fewer repeats: a missed command is said again, a wrong one sends the ball to the wrong place. This is a judgement on synthetic speech; Parakeet v2's 40 misses are all in noise (28 in 5 dB babble, 10 in 10 dB babble, 2 in 5 dB white noise), 23 of them empty transcripts and most of the rest the babble's own words ("We walked along the river."), so expect repeats in a loud hall.
- **A synonym table is enough** for two fixed commands with one slot each (`research/voice-commands.md`, section 6). `tests/voice_parse.py` holds the accepted and rejected sentences.
- **The ball's colour is a second slot** (2026-11-21, after the team replaced the foil balls with red and blue ones): "red", "blue", or "striped" (also "yellow", "green") goes into `target`. No colour means the striped target; two colours reject the command, because guessing one would send the wrong ball. Before this, "hand me the red ball" was sent as the striped ball, and "hand me ..." was not understood at all (found in a teammate's integration dry run). Colour words are matched as whole words, not fuzzily: "red" scores 100 inside "tired" or "hundred". The robot program decides what to do with a target it cannot pick yet.
- **TCP, voice program as client, one connection per command.** The robot program may run in WSL2 while the microphone is on Windows. Tested on Tian's laptop (2026-11-21; WSL 3.0.5, Ubuntu 24.04, default NAT networking, no `.wslconfig`): `link.py` inside WSL bound to 127.0.0.1 or to 0.0.0.0 received commands sent from Windows to 127.0.0.1:8765, also end to end from `listen.py --wav`; binding 0.0.0.0 is not required. With 127.0.0.1, 2 of 8 starts were not reachable from Windows for the first 11 s (one probe) or still at 15 s (one try with key 1); with 0.0.0.0, 3 of 3 starts were reachable within 2 s. The other direction, a receiver on Windows (bound to 127.0.0.1 or 0.0.0.0) and the sender in WSL (to 127.0.0.1 or to the Windows host address), failed in every try. The robot laptop's WSL settings were not checked. Connecting per command makes start order and restarts of either program irrelevant.
- **Own process and environment**, so speech packages (onnxruntime, numpy 2.5) never touch the robot environment's pins (`CLAUDE.md`).

## Measurement (2026-11-21, Tian's laptop, synthetic speech)

Synthetic speech from the offline Windows synthesiser is not a person in a hall. These numbers rank the recognisers; they do not predict the demo. Not tested at all: live human speech, accents, a real headset, the real hall.

- Clips: 14 sentences (10 commands, 5 per destination, 8 of them starting with "hey bot" from when it was required and 2 without; 4 that must be rejected: two unrelated requests after "hey bot", one statement, both commands at once) x 7 voices (David, Hazel, Zira Desktop; Zira, Susan, George, Mark OneCore) x 2 speaking rates x 5 conditions (clean; white noise and 4-talker synthetic babble at 10 dB and 5 dB SNR) = 980 clips per recogniser (196 per condition), 0.5 s silence before and after.
- Machine: AMD Ryzen 5 5500U, 6 cores, 15 GB, no GPU.
- "Right" means the full pipeline (speech-to-text plus parser) gives the expected destination, or nothing for a sentence that must be rejected. A false accept is a command from one of the 280 must-reject clips; a wrong destination is a command clip sent to the other destination.
- Re-run: `scripts/voice/benchmark.py`; `--rescore` re-parses saved transcripts, `--latency` times every 7th clip only. The table below is a re-score, without the wake check, of the transcripts recorded when "hey bot" was required; the sentences were not re-synthesised.

All right, without a wake phrase (last column: commands understood when "hey bot" was required, of the 560 clips that said it, with the final spelling list):

| `--backend` | clean | white 10 dB | white 5 dB | babble 10 dB | babble 5 dB | all | commands understood (of 700) | false accepts (of 280) | wrong destination | with "hey bot" required |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `parakeet-en` (Parakeet TDT 0.6b v2) | 100% | 100% | 99% | 95% | 86% | 96% | 660 | 0 | 0 | 475 |
| `parakeet` (Parakeet TDT 0.6b v3) | 100% | 97% | 92% | 81% | 53% | 85% | 553 | 2 | 1 | 367 |
| `whisper-base` (base.en, with prompt) | 99% | 99% | 98% | 98% | 95% | 98% | 691 | 9 | 1 | 547 |
| `whisper-base-no-prompt` (benchmark only) | 100% | 100% | 97% | 99% | 93% | 98% | 682 | 1 | 0 | 451 |
| `whisper-small` (small.en, with prompt) | 100% | 100% | 99% | 100% | 97% | 99% | 696 | 2 | 2 | 554 |

Parakeet v2 on the two commands said without "hey bot": "Give it to me." 64 of 70, "Put it in the container." 68 of 70; 5 of the 8 misses are empty transcripts, the others "Put it in the potato.", "I think Timmy is on the sergeant." and "Yeah.", all three in 5 dB babble.

Latency per clip after loading, first call excluded, as median / 90th percentile / worst; load is the time to build the backend in a fresh process. The quiet runs (140 clips, every 7th, the same for every backend, one backend at a time, no other job) were done once on battery and once on mains power, Balanced power plan both times; power state from the Windows event log. On battery everything is about 1.7 times slower. The tails on mains power are noisier than on battery (VS Code's Python helpers restarted during the `whisper-small` run), so trust the medians there. The full accuracy runs shared the CPU with other jobs.

| `--backend` | quiet, mains power | load | quiet, battery | load | full run, 980 clips, CPU shared | load |
| --- | --- | --- | --- | --- | --- | --- |
| `parakeet-en` | 0.31 / 0.40 / 0.60 s | 3.2 s | 0.53 / 0.67 / 0.92 s | 6.2 s | 0.30 / 0.41 / 3.80 s (mains) | 3.2 s |
| `parakeet` | 0.31 / 0.40 / 0.49 s | 3.0 s | 0.53 / 0.65 / 0.94 s | 6.9 s | 0.32 / 0.40 / 0.65 s (mains) | 3.3 s |
| `whisper-base` | 0.93 / 1.02 / 6.09 s | 1.4 s | 1.56 / 1.65 / 2.23 s | 7.4 s | 1.00 / 1.28 / 6.70 s (mains) | 0.9 s |
| `whisper-small` | 3.05 / 4.67 / 8.78 s | 2.3 s | 5.20 / 5.34 / 6.45 s | 5.0 s | 3.43 / 7.38 / 19.62 s (battery, mains from 11:10) | 8.2 s |
| `whisper-base-no-prompt` | not measured | | not measured | | 1.85 / 3.21 / 16.49 s (battery) | 5.0 s |

## Alternatives

- **Wake phrase "hey bot" required at the start of the transcript** - tried, dropped (2026-11-21): unreliable when tried live (team), and the measurement below shows why. It was meant to keep the "wake-up call" visible. How the recognisers wrote "hey bot" when it was said (770 clips each: the 8 commands and the 3 rejects that start with it), read the way the parser read the first one to three words:

  | `--backend` | exactly "hey bot" | close enough for fuzzy matching | other measured spellings | wake phrase missed |
  | --- | --- | --- | --- | --- |
  | `parakeet-en` | 436 | 158 ("hey boss" 53, "hey bob" 30, "hey box" 25, "hey but" 11) | 63 ("a bot" 17, "paybot" 13, "paypot" 4) | 113 (empty transcript 24) |
  | `parakeet` | 302 | 193 ("hey bob" 94, "hey boss" 22, "hey bolt" 18) | 30 ("hey bud" 7, "heybox" 4, "hey pop" 4) | 245 (empty 99, "people ..." 46, "paper ..." 11) |
  | `whisper-base` | 758 | 1 | 0 | 11 |
  | `whisper-base-no-prompt` | 344 | 188 ("hey boss" 67, "hey but" 37, "they both" 20) | 81 ("paybot" 19, "a bot" 19, "heybot" 8) | 157 ("they bought" 24, "people ..." 25, "payback ..." 15, "he bought" 9) |
  | `whisper-small` | 765 | 0 | 0 | 5 |

  A list of 30 spellings written at least twice for "hey bot" (such as "a bot", "paybot", "hey bud") raised Parakeet v2 from 423 to 475 commands understood. Accepting the ordinary words heard in its place ("people", "paper", "payback", "they bought") would have let almost any sentence through, so they stayed out. Whisper writes "hey bot" reliably only because our prompt contains it. Presumably because "hey bot" is short and ends in a plosive, which noise and a clipped recording start swallow easily (not tested separately).
- **A longer wake phrase** ("hello robot", "hey robot", "computer") - considered, not tried: the team chose push-to-talk alone. Revisit below.
- **Parakeet TDT 0.6b v3 (multilingual), the research pick** - tried, see table: worse than the English-only v2 on every noisy condition; it returned an empty transcript for 128 clips, mostly in babble. Kept selectable as `--backend parakeet`.
- **faster-whisper base.en with the two commands as prompt** - tried, see table: understood far more commands than Parakeet in noise, but writes the prompt's sentences instead of what was said. 9 of the 70 "give it to me or put it in the container" clips came out as "Hey bot, give it to me." and were accepted, and one "hand it to me" in 5 dB babble came out as "Hey bot, put it in the container." Three times Parakeet's latency. Kept as `--backend whisper-base`; its prompt still contains "Hey bot", because that is how it was measured.
- **faster-whisper base.en without the prompt** - tried, see table: more commands understood than Parakeet v2 (682 against 660) but 1 false accept, and slower (its latency was not measured on a quiet machine). Not selectable in `listen.py`.
- **faster-whisper small.en with the prompt** - tried, see table: the most accurate (696 of 700), but the same prompt leak as base.en: 2 false accepts, and 2 "hand" commands ("hand it to me" in 5 dB white noise, "give me the ball" in 5 dB babble) came out as "Hey bot, put it in the container." About 10 times Parakeet's latency. Kept as `--backend whisper-small`.
- **Always-listening wake word** (openWakeWord, livekit-wakeword, Porcupine) - considered, not tried: worst part in noise (above), needs a custom-trained model, Porcupine's free tier is reported to end (`research/voice-commands.md`, section 1).
- **Hold a key to talk** - considered, not tried: needs a global key hook library (`keyboard`, `pynput`), which needs extra permissions on macOS and Linux (assumed).
- **Claude or another LLM as parser** - considered, not tried: adds network, an API key and an open rule question (does a cloud call count as our own inference?) for two fixed commands (`research/voice-commands.md`, addendum).
- **UDP** - considered, not tried: Windows-to-WSL2 localhost forwarding carries TCP but not UDP in the default network mode (team's information, not tested).
- **Robot program as client, voice program as server** - considered, not tried: the robot loop would have to reconnect whenever the voice program restarts; and from WSL2 to Windows nothing got through in our test (above).
- **Noise suppression or voice activity detection** - considered, not tried: push-to-talk ends the recording; denoisers made recognition worse in a benchmark (`research/voice-commands.md`, section 4).
- **Spoken replies** - dropped by the team (reported by Tian, 2026-11-21): speech recognition only; the screen shows the transcript and the command.

## Revisit if

- The test in the hall with the real headset shows `NOT UNDERSTOOD` often: if the transcript on screen is right, add its phrasing to `DESTINATIONS`; if the transcripts themselves are wrong, try `--backend whisper-base` (more commands understood, but watch for transcripts that repeat the prompt) and a closer microphone.
- Talk not meant for the robot turns into commands while Enter is held: bring back a wake phrase, a longer one than "hey bot", measured first.
- The voice laptop is much slower than a Ryzen 5 5500U, or speech-to-text visibly disturbs the control loop.
- The robot program's port is not reachable from Windows on the robot laptop (`docs/voice-runbook.md`, last section).
- The demo needs hands-free activation, or many more commands with free phrasing (then an LLM fallback behind the table).
