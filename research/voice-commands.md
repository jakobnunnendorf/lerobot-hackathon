# Voice commands for the ball-picking robot

Researched 2026-11-20 (web only, nothing was installed or run). Every claim carries its source link. Anything marked **(inference)**, **(estimate)** or **(unverified)** is not backed by a primary source I opened, and is collected again in the last section. The Claude API side of intent parsing is out of scope here.

## TL;DR

1. In a loud hall with Spanish, Basque and English talk all around, use **push-to-talk plus a close-talk headset or lavalier mic** as the activation. An always-listening wake word is the part most likely to fail; keep it as an optional extra, and keep a keyboard fallback for every command.
2. STT pick for short English or Spanish commands on a laptop CPU: **NVIDIA Parakeet TDT 0.6b v3 through `onnx-asr`** (no PyTorch, only onnxruntime). Safe second choice: **faster-whisper `small`, int8, language forced**.
3. Intent parsing: **a synonym table with fuzzy matching** is enough for a two-slot command (`target`, `destination`). Use an LLM only as a second try when the rules find nothing, never as the only path, because the venue Wi-Fi may drop.
4. Speaking back: **pre-rendered audio clips** for the dozen sentences the robot can say; macOS `say` or Piper to render them. No live TTS dependency in the demo.
5. Run the voice loop as **its own process in its own environment, on native Windows or on the MacBook, not inside WSL2**, and send one JSON line to the state machine over a local socket. Keep STT off the RTX 4070 so it does not compete with the policy.
6. Build time: about **2 to 3 hours** for the minimal pipeline by one person, **4 to 6 hours** with wake word, LLM fallback and tuning in the hall (estimate).

---

## 1. Wake word and activation

### openWakeWord

- Code is Apache 2.0; the **pre-trained models are CC BY-NC-SA 4.0** (non-commercial), because some training data has unknown or restrictive licences. [README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md)
- Pre-trained phrases: "alexa", "hey mycroft", "hey jarvis", "hey rhasspy", plus weather and timer phrases. English only. [README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md)
- Stated targets: false-reject rate below 5 % and fewer than 0.5 false accepts per hour, with threshold tuning. Default threshold 0.5. [README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md)
- Custom phrases: a Colab notebook trains a basic model in "<1 hour" from synthetic TTS speech (several thousand generated examples recommended). [README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md)
- Platform: on Windows only onnxruntime is installed (no tflite); the Speex noise suppression option exists only on x86 and Arm64 Linux; a Silero VAD gate (`vad_threshold`) is built in. [README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md)
- Latest release is v0.6.0 from February 2024, so the project is slow-moving. [README](https://raw.githubusercontent.com/dscripka/openWakeWord/HEAD/README.md)
- Community results for custom models are mixed: one "Hey Atlas" training project reports 81 % accuracy and 62 % recall, and notes that a two-word phrase with a "Hey"/"Okay" prefix matters a lot. [atlas-voice-training](https://github.com/briankelley/atlas-voice-training)
- A voice-agent project that tested it found the threshold that works for built-in phrases was 0.90, rejecting a near-miss at 0.898 and accepting "hey jarvis" at 0.996. [agent-voice](https://github.com/bluetooth47/agent-voice)

### livekit-wakeword (newer, same model format)

- Announced 6 April 2026; trains custom wake words with one command and exports ONNX compatible with openWakeWord integrations. [LiveKit blog](https://livekit.com/blog/livekit-wakeword)
- Apache 2.0, Python 3.11+. Ships pre-trained `hey_livekit.onnx` and `hey_jarvis.onnx`. Default threshold 0.5. [GitHub](https://github.com/livekit/livekit-wakeword)
- Training needs a CUDA GPU or an Apple-silicon Mac, **plus a LiveKit Cloud project for the TTS that generates training speech**; about 11 minutes on a data-centre GPU, about 22 min speech generation plus 37 min training on an M4 Pro, and a one-time download of about 4 GB. [GitHub](https://github.com/livekit/livekit-wakeword)
- Vendor's own comparison against openWakeWord: 0.08 vs 8.50 false positives per hour and 86 % vs 69 % recall. Vendor claim, test conditions not in the article. [LiveKit blog](https://livekit.com/blog/livekit-wakeword)

### Picovoice Porcupine

- A Home Assistant forum post reports that Picovoice emailed users that the **Free Tier ends 30 June 2026 and existing free AccessKeys are disabled**, and quotes Picovoice: "The SDK does not run without a valid key" and "There is no non-commercial tier planned." [HA community](https://community.home-assistant.io/t/fyi-picovoice-confirmed-free-tier-accesskeys-will-stop-working-after-june-30-2026/1012744)
- I could not read Picovoice's own pricing page (it returned no content), and one third-party blog from November 2026 still lists a $0 plan. Treat Porcupine as **not available for free** unless someone on the team confirms otherwise in the console. [pricing page](https://picovoice.ai/pricing/), [third-party blog](https://www.activepieces.com/blog/picovoice-pricing-2026-free-foundation-enterprise)

### Alternatives

- **Keyword inside the ASR transcript.** One hackathon arm drops the wake word and "finds the key word anywhere in what you say". [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton). The weakness of doing this with a small constrained recogniser: phonetic neighbours pass ("hey cloud" scored 1.00 against a "hey claude" Vosk grammar, and a confidence threshold could not fix it). [agent-voice](https://github.com/bluetooth47/agent-voice)
- **sherpa-onnx open-vocabulary keyword spotting**: keywords are given as text, no training; pre-trained models exist only for English and Chinese. [sherpa-onnx KWS docs](https://k2-fsa.github.io/sherpa/onnx/kws/index.html). In one synthetic test it caught 28 of 30 wakes with 8 false wakes in 150 clips, versus 63 false wakes for Vosk. [agent-voice](https://github.com/bluetooth47/agent-voice)
- **Push-to-talk key or button.** Described by one project as "the reliable path when the room is noisy or the wake word is being stubborn". [agent-voice](https://github.com/bluetooth47/agent-voice). Picovoice's own guide frames PTT as the simpler implementation and wake word as the hands-free one. [Picovoice guide](https://picovoice.ai/blog/complete-guide-to-wake-word/)

### What is most reliable in a loud room

- Measured in one project: wake detection needed about +24 dB of headroom; at +1.2 dB signal-to-noise it caught 5 of 20 wakes versus 19 of 20 in a quiet room. [agent-voice](https://github.com/bluetooth47/agent-voice)
- The same project found that stock WebRTC noise suppression plus automatic gain control "gate the wake word into unintelligibility (0/20)". [agent-voice](https://github.com/bluetooth47/agent-voice)
- Loudspeaker output feeding the same machine's mic causes false wakes. [agent-voice](https://github.com/bluetooth47/agent-voice)
- Conclusion **(inference from the above)**: push-to-talk is the reliable activation; a wake word works only with a close mic and tuning done in the actual hall.

### What demo builders actually use

| Project | Activation | Source |
|---|---|---|
| RoboStrands (SO-101, ACT) | Press Enter, fixed 10 s recording | [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant) |
| Say-and-Grasp (SO-101) | Wake word "Wiggle", plus Enter push-to-talk, plus typed input | [GitHub](https://github.com/adoodevv/say-and-grasp) |
| Seeed SO-ARM voice controller | openWakeWord "hey jarvis", threshold 0.5, then fixed 3 s recording | [Seeed wiki](https://wiki.seeedstudio.com/respeaker_flex_soarm/) |
| hack_princeton arm | No wake word, keyword anywhere in speech | [GitHub](https://github.com/aditya-baniya-ai/hack_princeton) |
| Dum-E (SO-100/101) | Browser page with mic, Pipecat | [GitHub](https://github.com/aaronsu11/dum-e) |
| agent-voice | Wake word, with hold-F8 push-to-talk for noisy rooms | [GitHub](https://github.com/bluetooth47/agent-voice) |

---

## 2. Voice activity detection and end of utterance

- **Silero VAD**: MIT licence; one 30 ms chunk takes under 1 ms on a single CPU thread; 8 kHz and 16 kHz; model about 2 MB; `pip install silero-vad`; the pip package requires `torch>=1.12`, with ONNX extras available. [GitHub](https://github.com/snakers4/silero-vad)
- Silero defaults: `threshold=0.5`, `min_speech_duration_ms=250`, `min_silence_duration_ms=100`, `speech_pad_ms=30`; the window is 512 samples at 16 kHz; the negative threshold defaults to `threshold - 0.15`. [utils_vad.py](https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py)
- **webrtcvad** (py-webrtcvad): aggressiveness modes 0 to 3, frames of 10/20/30 ms, 16-bit mono PCM at 8/16/32/48 kHz; last version 2.0.10; little recent activity. [GitHub](https://github.com/wiseman/py-webrtcvad)
- **Typical settings in a glue library** (RealtimeSTT defaults): `silero_sensitivity=0.4`, `webrtc_sensitivity=3`, `post_speech_silence_duration=0.6` s, `min_length_of_recording=0.5` s, `pre_recording_buffer_duration=1.0` s. [configuration.md](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/docs/configuration.md)
- faster-whisper's built-in VAD filter is off by default and, when on, only removes silence longer than 2 s. [faster-whisper README](https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md)
- **TEN VAD**: claims better precision than Silero and faster end-of-speech detection, but its "Apache 2.0 with certain conditions" licence carries restrictions, and Silero's own metrics show TEN doing worse on pure-noise clips (0.42 vs 0.87 accuracy). [TEN VAD](https://github.com/TEN-framework/ten-vad), [Cekura review](https://www.cekura.ai/blogs/what-is-silero-vad)
- **Smart Turn v3** (Pipecat): a model that decides whether the speaker is finished, used on top of Silero; about 65 ms inference. [smart-turn](https://github.com/pipecat-ai/smart-turn), [Pipecat docs](https://docs.pipecat.ai/server/utilities/turn-detection/smart-turn-overview)
- Simple fixed windows are common in robot demos: 3 s after the wake word (Seeed) or 10 s after Enter (RoboStrands, which found shorter windows cut off parts of the request). [Seeed wiki](https://wiki.seeedstudio.com/respeaker_flex_soarm/), [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant)
- Known trap: a fixed energy gate suits one room; a fan (or a crowd) keeps the turn open until the hard ceiling, and a quiet room cuts off soft speech. [agent-voice](https://github.com/bluetooth47/agent-voice)

Recommendation **(inference)**: with push-to-talk the key release ends the utterance and no VAD is needed. If hands-free is used, take Silero with threshold raised to 0.6 to 0.7, 600 to 800 ms end silence, and a hard cap of 8 s per utterance, then tune in the hall.

---

## 3. Local speech-to-text, late 2026

### Overview

| Model / runtime | Languages (en / es / eu) | Size and memory | Speed | Licence | Install notes |
|---|---|---|---|---|---|
| **Parakeet TDT 0.6b v3** via `onnx-asr` | en yes, es yes, eu no (25 European languages) | 600 M params; at least 2 GB RAM | RTFx 36 on a Ryzen 9800X3D CPU (v2/v3) | CC BY 4.0 | `pip install onnx-asr[cpu,hub]`, needs only numpy and onnxruntime, Python 3.11 to 3.14, Windows/Linux/macOS |
| **faster-whisper** (CTranslate2) | en, es; Basque see below | `small` int8 about 1.5 GB RAM | `small` int8: 13 min audio in 1 m 42 s on i7-12700K, 8 threads | MIT | `pip install faster-whisper`, Python 3.9+, no FFmpeg needed; GPU needs CUDA 12 + cuDNN 9 |
| **whisper.cpp** | same models as Whisper | tiny 273 MB, base 388 MB, small 852 MB, medium 2.1 GB, large 3.9 GB | Metal on Apple silicon, CUDA, Vulkan | MIT | C++ build (MSVC/MinGW on Windows); Python bindings such as pywhispercpp |
| **Moonshine** (`moonshine-voice`) | en yes, es yes, eu no | from about 1 MB micro models upward | Tiny needs about 5x less compute than Whisper tiny.en on a 10 s clip | MIT (legacy non-streaming non-English models are non-commercial) | `pip install moonshine-voice`; Windows, macOS, Linux |
| **Nemotron 3.5 ASR Streaming 0.6b** | en yes, es yes, eu no (40 locales) | 0.6 B params | streaming chunks 80 ms to 1.12 s | OpenMDW-1.1 | NeMo, Transformers 5.13+, or a GGUF C++ runtime; card lists NVIDIA GPUs and Linux |
| **Canary-1b-v2** | en yes, es yes, eu no | 978 M params; at least 6 GB RAM | RTFx 8 on CPU via onnx-asr | CC BY 4.0 | NeMo, Linux preferred |
| **Kyutai STT** | English and French only | 1 B and 2.6 B params | 0.5 s and 2.5 s delay | CC BY 4.0 | `moshi` Python package / Rust server / MLX |
| **Vosk** | en yes, es yes, eu no | small models about 40 MB | very light | Apache 2.0 | `pip install vosk`; phrase-list grammar on small models |

Sources per row: Parakeet [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [onnx-asr](https://github.com/istupakov/onnx-asr); faster-whisper [README](https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md), [PyPI](https://pypi.org/project/faster-whisper/); whisper.cpp [GitHub](https://github.com/ggml-org/whisper.cpp); Moonshine [GitHub](https://github.com/moonshine-ai/moonshine), [paper](https://arxiv.org/abs/2410.15608); Nemotron [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b); Canary [model card](https://huggingface.co/nvidia/canary-1b-v2); Kyutai [model card](https://huggingface.co/kyutai/stt-1b-en_fr), [site](https://kyutai.org/stt); Vosk [models](https://alphacephei.com/vosk/models). The Kyutai runtime details are from a [search summary](https://github.com/kyutai-labs/delayed-streams-modeling), not the card itself.

### Details per model

**Whisper family (faster-whisper, whisper.cpp)**

- Sizes: tiny 39 M, base 74 M, small 244 M, medium 769 M, large 1550 M, turbo 809 M parameters. In OpenAI's implementation the required VRAM is about 1 GB (tiny, base), 2 GB (small), 5 GB (medium), 10 GB (large), 6 GB (turbo); relative speed 10x, 7x, 4x, 2x, 1x, 8x. [openai/whisper](https://github.com/openai/whisper)
- `large-v3-turbo` is large-v3 with the decoder cut from 32 to 4 layers, "minor quality degradation". [model card](https://huggingface.co/openai/whisper-large-v3-turbo)
- `distil-large-v3` and `distil-large-v3.5` are **English only**; v3.5 is about 1.5x faster than turbo with the same short-form error (7.10 vs 7.25 average WER); a CTranslate2 build exists. [model card](https://huggingface.co/distil-whisper/distil-large-v3.5)
- faster-whisper benchmarks (v1.1.0): `small` on CPU int8 uses 1477 MB RAM; `large-v2` on an 8 GB RTX 3070 Ti uses 2926 MB VRAM at int8 and 4525 MB at fp16. [README](https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md)
- Latest faster-whisper is 1.2.1 (31 October 2025); `Requires-Python >=3.9`; classifiers list up to 3.11 only. [PyPI](https://pypi.org/project/faster-whisper/). RealtimeSTT, whose default engine is faster-whisper, runs its CI on Python 3.11 and 3.12. [RealtimeSTT README](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/README.md)
- Whisper's performance "varies widely depending on the language". [openai/whisper](https://github.com/openai/whisper)
- If no language is given, faster-whisper detects it from the first 30 s of audio; `initial_prompt` only affects the first window; a `hotwords` option biases vocabulary. [faster-whisper repo](https://github.com/SYSTRAN/faster-whisper), [hotwords PR in openai/whisper](https://github.com/openai/whisper/pull/2070) (both via a search summary, see Unverified). Set `language="en"` or `"es"` explicitly for short commands **(inference)**.
- Hallucinations: on silence or noise Whisper produces phrases like "Thank you for watching"; the reliable fix is not to send silence (VAD gate), plus `condition_on_previous_text=False`; `no_speech_threshold` alone is not a hard gate. [dograh](https://www.dograh.com/feeds/blog/whisper-hallucination-silence), [openai/whisper discussion](https://github.com/openai/whisper/discussions/679)
- A small real-world data point for commands: one voice-agent project runs faster-whisper `tiny.en` in-process with a custom vocabulary prompt and reports 4.9 % WER versus 6.5 % for `base.en`, and rejected `small.en` for latency. [agent-voice](https://github.com/bluetooth47/agent-voice)

**NVIDIA Parakeet TDT 0.6b v3**

- 25 European languages including Spanish; Basque is not among them. Released 14 August 2025. [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- WER: English 4.85 % (Fleurs), 6.34 % average on the Open ASR Leaderboard sets; Spanish 3.45 % (Fleurs), 4.39 % (MLS), 3.41 % (CoVoST). [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **Noise robustness table** (average WER with MUSAN noise): clean 6.34 %, 10 dB SNR 7.12 %, 5 dB 8.23 %, 0 dB 11.66 %, -5 dB 19.88 %. [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- The official runtime is NeMo on Linux with NVIDIA GPUs. [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). The practical laptop route is the ONNX export: `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")`, then `model.recognize("test.wav")`, with "no need for PyTorch, Transformers, or FFmpeg". [onnx-asr](https://github.com/istupakov/onnx-asr), [ONNX model](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- RealtimeSTT also exposes Parakeet TDT through its sherpa-onnx engine on CPU. [RealtimeSTT README](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/README.md)

**Moonshine**

- Trained without zero-padding on variable-length segments, so short clips cost less. [paper](https://arxiv.org/abs/2410.15608). Moonshine v2 (February 2026) adds streaming encoders for latency-critical use. [v2 paper](https://arxiv.org/abs/2602.12241)
- Speech-to-text in English, Spanish, German, Mandarin, Japanese, Korean, Vietnamese, Tagalog, Ukrainian, Arabic. [GitHub](https://github.com/moonshine-ai/moonshine)
- All-in-one package: `moonshine-voice mic --language en` prints a live transcript, and `moonshine-voice agent` "listens for a trigger phrase, asks questions, and confirms the answers. Matching is semantic". [quickstart](https://moonshine-voice.readthedocs.io/en/stable/quickstart/)

**Nemotron, Canary, Kyutai**

- Nemotron 3.5 ASR Streaming 0.6b: released 4 June 2026, OpenMDW-1.1, Spanish is "transcription-ready" (es-US, es-ES), Fleurs Spanish WER 4.11 to 4.87 % depending on chunk size, English 7.91 to 9.43 %. [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b). RealtimeSTT 1.0.4 and later run Nemotron on CPU through sherpa-onnx. [RealtimeSTT releases](https://github.com/KoljaB/RealtimeSTT/releases)
- Canary-1b-v2: 25 languages, more accurate but much slower than Parakeet on CPU (RTFx 8 vs 36). [model card](https://huggingface.co/nvidia/canary-1b-v2), [onnx-asr](https://github.com/istupakov/onnx-asr)
- Kyutai STT: English and French only, so not useful for Spanish. Its semantic VAD exists only in the Rust server. [model card](https://huggingface.co/kyutai/stt-1b-en_fr)

**Leaderboard context**

- Open ASR Leaderboard findings (November 2025): Conformer encoders with LLM decoders are most accurate; CTC and TDT decoders (Parakeet) are 10 to 100x faster with slightly higher error. [HF blog](https://huggingface.co/blog/open-asr-leaderboard)
- Secondary 2026 guides say newer NVIDIA and Qwen models now top the English table (around 5.3 to 5.9 % average WER). Not checked against the leaderboard itself. [apidog guide](https://apidog.com/blog/best-open-source-speech-to-text-models/), [northflank guide](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)

**Vosk (closed-vocabulary option)**

- `vosk-model-small-en-us-0.15` (40 MB, 9.85 % WER on LibriSpeech clean) and `vosk-model-small-es-0.42` (39 MB, 16.02 % on Common Voice), both Apache 2.0; no Basque model. [models](https://alphacephei.com/vosk/models)
- The recogniser can be restricted to a phrase list (with `[unk]` to absorb everything else); small models support this, big static-graph models do not. [adaptation docs](https://alphacephei.com/vosk/adaptation), [grammar example](https://docs.rs/crate/vosk/latest/source/examples/grammar.rs)
- Used this way by two robot-arm projects: a 74-word offline vocabulary accepted at confidence 0.70 or higher (0.80 dropped real commands) [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton), and Vosk on Windows feeding an allow-list in WSL2 [SO101-Voice-Control](https://github.com/noittz/SO101-Voice-Control-Physical-AI).

### Basque

- Parakeet v3, Moonshine, Nemotron, Canary and Vosk do not cover Basque (sources above). The base Parakeet v3 scores about 108 % WER on Basque test sets, i.e. unusable. [itzune model card](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque)
- Basque needs a dedicated model:
  - `xezpeleta/whisper-large-v3-eu`, Apache 2.0, 4.84 % WER on Common Voice 18 (self-reported). [model card](https://huggingface.co/xezpeleta/whisper-large-v3-eu)
  - `zuazo/whisper-large-v3-eu`, 10.62 % WER on Common Voice 13. [model card](https://huggingface.co/zuazo/whisper-large-v3-eu)
  - `HiTZ/stt_eu_conformer_transducer_large` (NeMo), 2.79 % on Common Voice 16.1. [model card](https://huggingface.co/HiTZ/stt_eu_conformer_transducer_large)
  - `itzune/parakeet-tdt-0.6b-v3-basque` (NeMo format), 6.92 % on Common Voice; other languages "not the target". [model card](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque)
- These are large-v3-size or NeMo models, so Basque commands mean a second, heavier STT path. Recommendation **(inference)**: take commands in English or Spanish; if a Basque touch is wanted for the local jury, make the robot *say* a Basque sentence (section 7) instead of understanding one.

### Clear pick for short English or Spanish commands on a laptop CPU

**Parakeet TDT 0.6b v3 via `onnx-asr`.** Reasons: Spanish and English in one model with strong Spanish numbers and a published noise table; about 36x real time on a desktop CPU; installs with onnxruntime only, so it cannot disturb the pinned torch 2.11 / CUDA 12.8 robot environment; supports Python 3.12 on Windows, Linux and macOS (all sourced above).

**Second choice: faster-whisper `small`, `device="cpu"`, `compute_type="int8"`, language forced.** It is the most widely used path and the default of the glue libraries. Use `base` if the CPU is weak, and `small.en`/`base.en` if English only.

Not recommended on the policy laptop: any GPU STT. The 8 GB RTX 4070 is already running the policy; one SO-101 team had to unload its LLM from the GPU before LeRobot would launch because of GPU memory. [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant)

---

## 4. Noise and microphones

- Close-talk versus distant microphones on the same speech: error rates rise by a factor of about 2.5 to 4 with distant mics; on the AMI meeting corpus an eight-microphone distant setup reached 52.0 % WER versus 29.6 % for individual headsets. [far-field ASR survey](https://arxiv.org/pdf/2009.09395), [thesis](https://etheses.whiterose.ac.uk/id/eprint/17691/1/thesis_final_20June2017.pdf)
- Why: at a distant mic the speech level drops while room noise stays the same, so "ambient additive noise [is] much more significant". [ICSI paper](https://www1.icsi.berkeley.edu/ftp/pub/speech/papers/icslp02-noiseandreverb.pdf)
- How much SNR matters to a modern model: Parakeet v3 goes from 6.34 % WER clean to 11.66 % at 0 dB and 19.88 % at -5 dB. [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Arithmetic **(estimate)**: moving the mic from 50 cm (laptop) to 2.5 cm (headset boom) raises the speech level by about 26 dB (20·log10(50/2.5)) while the hall noise stays the same.
- Project evidence: a webcam mic picked up speech about twice as loud as a speaker-phone mic and gave confidence 0.95 versus 0.87 [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton); a tutorial advises speaking within about 1 m of even a beam-forming mic array [Seeed wiki](https://wiki.seeedstudio.com/respeaker_flex_soarm/); another recommends a wrist or collar mic because it stays close to the mouth while hands work [earshot](https://github.com/MNS1007/earshot); a student team advises keeping the mic 1 to 2 feet from the user and away from robot motors [Hackster](https://www.hackster.io/Elephant-Robotics-Official/student-team-case-of-building-voice-controlled-mobile-robot-ce75dd).
- "A speech-to-text demo in a quiet room rarely predicts performance" in a noisy place. [Deepgram](https://deepgram.com/blog/noise-robust-speech-recognition-methods-best-practices)

### Noise suppression (RNNoise, DeepFilterNet)

- Denoising in front of ASR can make recognition **worse**: in the RSA-Bench benchmark one noisy scene went from 5.27 (no denoising) to 10.06 with DeepFilterNet and 14.69 with PyRNNoise. [RSA-Bench](https://arxiv.org/pdf/2601.10384)
- Reason given in the literature: enhancement artifacts, more than residual noise, cause the ASR degradation. [arXiv 2605.12107](https://arxiv.org/html/2605.12107v1). Deepgram says the same about denoisers adding "musical noise". [Deepgram](https://deepgram.com/blog/noise-robust-speech-recognition-methods-best-practices)
- DeepFilterNet3 sounds cleaner than RNNoise on hard noise but adds about 40 ms and needs a modern CPU. [Forasoft](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/real-time-noise-suppression-krisp-rnnoise-deepfilternet)
- Recommendation **(inference)**: do not add a software denoiser this weekend. Fix the signal at the microphone.

### Microphone access under WSL2

- WSLg runs a PulseAudio server for "audio in (microphone) and out", bridged to Windows over RDP, and presets `PULSE_SERVER` in the distro. [WSLg README](https://github.com/microsoft/wslg)
- In practice the mic appears as `RDPSource` (1 channel, s16le, 44100 Hz) and shows SUSPENDED when idle; one user needed `libasound2-plugins` and `pulseaudio` installed before an application could use it. [Claude Code issue 31109](https://github.com/anthropics/claude-code/issues/31109)
- Passing a USB microphone into WSL2 with usbipd-win needs USB support in the WSL kernel and, for some device classes, a custom kernel build. [Microsoft docs](https://learn.microsoft.com/windows/wsl/connect-usb), [usbipd-win wiki](https://github.com/dorssel/usbipd-win/wiki/WSL-support)
- One SO-101 project already uses the split that avoids all of this: **speech recognition on Windows, robot control in WSL2**, connected by a small JSON hand-off. [SO101-Voice-Control](https://github.com/noittz/SO101-Voice-Control-Physical-AI)
- `sounddevice` wheels bundle PortAudio on Windows and macOS; on Linux PortAudio must be installed separately. [sounddevice docs](https://python-sounddevice.readthedocs.io/en/latest/installation.html)

### Practical tips from people who demoed voice robots

- Mute the mic while the robot talks, and for 2 s after. [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton)
- Keep the mic off while the arm moves; voice is not an emergency stop. [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant), [RoboticsHackCal](https://github.com/MaxBingham/RoboticsHackCal)
- Show on screen what was heard and why it was ignored. [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton)
- Keep a typed or key-press path next to voice. [Say-and-Grasp](https://github.com/adoodevv/say-and-grasp)
- Expect the crowd: the 2025 Bilbao winners could not reach their table during the final demo because people stood in the way. [write-up](https://www.elchinaslanli.com/projects/how-we-won-the-lerobot-hackathon-2025-meet-the-ai-bartender)
- A microphone "hears everyone near it"; tell bystanders if audio goes to a cloud service. [earshot](https://github.com/MNS1007/earshot)

---

## 5. Glue libraries

| Library | What it combines | State | Licence | Fit |
|---|---|---|---|---|
| **RealtimeSTT** | Mic capture, WebRTC + Silero VAD, wake word (Porcupine or openWakeWord), many STT engines (faster-whisper default, whisper.cpp, Moonshine, sherpa-onnx Parakeet and Nemotron) | Actively developed: 1.0.0 in May, 1.2.0 on 11 November | Apache 2.0 from 1.2.0 (MIT up to 1.1.2) | Simplest full wake + VAD + STT in one call |
| **Pipecat** | Pipeline framework: Silero VAD, Smart Turn, Whisper/Moonshine STT, Ollama, Piper/Kokoro TTS, local transport | Active | BSD 2-Clause | Good for conversations; more than needed for one command |
| **LiveKit Agents** | Agent framework with a semantic turn detector; console mode works with local audio and no server | Active | Apache 2.0 (turn-detector model under LiveKit Model License) | Overkill here |
| **speech_recognition** | `listen()` with an energy threshold, engines incl. faster-whisper, Vosk, Groq | Version 3.11, Python 3.10+ | BSD 3-Clause | Easy, but energy-based endpointing is weak in a noisy room |
| **sounddevice loop** | Your own 50 to 100 lines: record, transcribe, parse | n/a | n/a | Most control; best with push-to-talk |
| **moonshine-voice** | Mic, STT, trigger phrase, semantic command matching, TTS | Active | MIT | Worth a 20-minute trial |

Sources: RealtimeSTT [README](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/README.md), [releases](https://github.com/KoljaB/RealtimeSTT/releases), [wake-word docs](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/docs/wake-words.md); Pipecat [GitHub](https://github.com/pipecat-ai/pipecat); LiveKit Agents [GitHub](https://github.com/livekit/agents); speech_recognition [GitHub](https://github.com/Uberi/speech_recognition); sounddevice [docs](https://python-sounddevice.readthedocs.io/en/latest/installation.html); Moonshine [quickstart](https://moonshine-voice.readthedocs.io/en/stable/quickstart/).

RealtimeSTT specifics:

- Install `pip install "RealtimeSTT[faster-whisper]"`; minimal use is `AudioToTextRecorder()` then `recorder.text()`; needs PortAudio headers on Linux and `brew install portaudio` on macOS; CI covers Python 3.11 and 3.12; scripts need the `if __name__ == "__main__":` guard, "especially on Windows", because it uses multiprocessing. [README](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/README.md)
- Wake word: `wakeword_backend="oww"`, `openwakeword_model_paths="models/x.onnx"`, `wake_words_sensitivity` default 0.6 (the docs suggest starting near 0.35 for custom models), `wake_word_timeout` 5 s, callbacks such as `on_wakeword_detected`. [wake-word docs](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/docs/wake-words.md)
- Defaults that must be changed on a CPU laptop: `device="cuda"` and `model="tiny"`. [configuration.md](https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/docs/configuration.md)
- Known install issue: on Python 3.13+ openwakeword 0.6.0 cannot be installed because tflite-runtime has no wheels, and pip falls back to an older version with a different API. [RealtimeSTT issue 297](https://github.com/KoljaB/RealtimeSTT/issues/297). Stay on Python 3.12.

Simplest and still maintained: **RealtimeSTT** if wake word and automatic end-of-speech are wanted; **a plain sounddevice loop** if push-to-talk is accepted.

---

## 6. Transcript to structured command

### Rules or LLM

- The command space here is two slots with a handful of values each. Closed-set systems solve exactly this without an LLM: Home Assistant's `hassil` matches text against sentence templates with alternatives `(red | green | blue)` and optional words `[the]`. [hassil](https://github.com/home-assistant/hassil), [template syntax](https://developers.home-assistant.io/docs/voice/intent-recognition/template-sentence-syntax)
- The limit of templates: anything outside the listed forms is not recognised. [template syntax](https://developers.home-assistant.io/docs/voice/intent-recognition/template-sentence-syntax)
- Robot projects that use an LLM still fence it: the LLM may only choose from a fixed command list and never moves the arm directly [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton); a command is accepted only when an allow-list and the LLM classifier agree [SO101-Voice-Control](https://github.com/noittz/SO101-Voice-Control-Physical-AI); the voice agent exposes one tool and a local allow-list maps the task ID to a checkpoint [RoboticsHackCal](https://github.com/MaxBingham/RoboticsHackCal); the model is "forced to be a tool call", one action per spoken turn [earshot](https://github.com/MNS1007/earshot).
- An LLM earns its place when phrasing is open ("the thing for cutting wire" resolved to side cutters, with a photo of the bench). [Say-and-Grasp](https://github.com/adoodevv/say-and-grasp)

Recommendation **(inference)**: rules first. Lower-case the transcript, strip punctuation, look for slot synonyms in English and Spanish with fuzzy matching (edit distance), and only call an LLM when a slot is missing. Likely mis-hearings to put in the table: "striped" heard as "stripped" or "stripe"; "ball", "bowl" and "box" are acoustically close, so prefer destination words like "container", "basket", "hand", "me".

### Constraining an LLM to a closed set

- Ollama: pass a JSON schema in `format` (for example `Model.model_json_schema()` from Pydantic), validate with `model_validate_json`, set temperature 0, and also put the schema in the prompt. Ollama's cloud does not support structured outputs. [Ollama docs](https://docs.ollama.com/capabilities/structured-outputs)
- llama.cpp: `response_format` with a JSON schema is converted to a grammar, so tokens that break the schema get zero probability; a tiny GBNF rule such as `root ::= "a" | "b" | "c"` gives a guaranteed enum. This guarantees syntax, not that the value is right; only a subset of JSON Schema is supported. [Liquid AI docs](https://docs.liquid.ai/deployment/on-device/llama-cpp/structured-output), [GBNF guide](https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/grammars/README.md)
- Small models without constrained decoding often print the JSON as plain text in a code block instead of making the tool call. [dev.to test](https://dev.to/kuroko1t/what-happens-when-local-llms-fail-at-tool-calling-testing-7-models-with-a-rust-coding-agent-cep)

### Which small local models

- Docker's tool-calling evaluation (June 2025, 3,570 cases, MacBook Pro M4 Max): `qwen3:8b` F1 0.933 (best small local model), `llama3.1:8b` 0.835, `gemma3:4b` 0.733, `llama3.2:3b` 0.727. Quantised and full-precision variants behaved the same. [Docker blog](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/)
- In another test the Qwen3 versus Qwen2.5 generation boundary predicted tool-calling success better than parameter count. [dev.to test](https://dev.to/kuroko1t/what-happens-when-local-llms-fail-at-tool-calling-testing-7-models-with-a-rust-coding-agent-cep)
- A hobby SO-101 project uses `qwen3:1.7b` in Ollama as a command classifier next to an allow-list and got 15 of 20 trials right end to end (speech errors included). [SO101-Voice-Control](https://github.com/noittz/SO101-Voice-Control-Physical-AI)
- Latency: `llama3.2:3b` runs at about 55 tokens/s on an M3 Pro (18 GB) [localaimaster benchmarks](https://localaimaster.com/benchmarks); CPU-only figures are far lower, e.g. about 8.5 to 9 tokens/s for a 3 B model [It's FOSS](https://itsfoss.com/testing-local-llms-without-gpu) and 8 to 15 tokens/s quoted for Qwen3 8B on a 16 GB CPU-only machine [insiderllm](https://insiderllm.com/guides/cpu-only-llms-what-actually-works/).
- Where to run it: on the MacBook, not on the 4070 laptop (GPU memory conflict with LeRobot, see [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant)).

### Unclear commands

- Confirm before motion: one arm refuses the pick until the person confirms the exact object ("Is it the banana?"), and STOP counts as no. [hack_princeton](https://github.com/aditya-baniya-ai/hack_princeton)
- Separate stages for understanding, preparation, approval, execution, verification: "An agent connected to hardware needs more than a useful response." [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant)
- Moonshine's agent demo does trigger phrase, question, confirmation out of the box. [quickstart](https://moonshine-voice.readthedocs.io/en/stable/quickstart/)

Recommendation for a 5-minute slot **(inference)**: read the parsed command back ("Striped ball, to your hand") and start after a short pause unless a cancel key is pressed; ask one closed question only when a slot is missing ("Which ball?"); after two failed tries fall back to the keyboard.

---

## 7. Speaking back

| Option | Licence | Languages | Notes |
|---|---|---|---|
| **macOS `say`** | part of macOS | system voices; some need a one-time download | `say -v Name "text"`, `-r` rate, `-o file` writes AIFF/WAVE; `say -v ?` lists voices |
| **Piper** (`pip install piper-tts`) | GPL-3.0 (engine embeds espeak-ng); original MIT repo archived October 2025 | Spanish es_ES, es_MX, es_AR; Basque voices exist (antton, maider) | Each voice is an `.onnx` + `.onnx.json` pair with its own licence |
| **Kokoro-82M** | Apache 2.0 (model); `kokoro-onnx` wrapper MIT | 8 languages, 54 voices, Spanish included | 82 M params; `pip install kokoro>=0.9.2 soundfile` plus espeak-ng; `kokoro-onnx` supports Python 3.10 to 3.13 |
| **eSpeak NG** | GPL | many | Robotic sound; used as a fallback |
| **Moonshine TTS** | MIT | English, Spanish and others | Part of `moonshine-voice` |

Sources: [say man page](https://ss64.com/mac/say.html); Piper [GitHub](https://github.com/OHF-Voice/piper1-gpl), [voices list](https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/docs/VOICES.md), [review](https://www.promptquorum.com/power-local-llm/piper-tts-review), Basque voices [blog](https://xezpeleta.github.io/en/blog/euskarazko-ahots-sintesirako-ereduak) and [HiTZ model](https://huggingface.co/HiTZ/TTS-eu_antton/blob/main/README.md); Kokoro [model card](https://huggingface.co/hexgrad/Kokoro-82M), [kokoro-onnx](https://simple-repository.app.cern.ch/project/kokoro-onnx); eSpeak in use at [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant) and [Say-and-Grasp](https://github.com/adoodevv/say-and-grasp); Moonshine [GitHub](https://github.com/moonshine-ai/moonshine).

- Latency: Piper is listed at a real-time factor of 0.06 on an Apple M2 Max in a third-party table (secondary). [piper-plus](https://github.com/ayutaz/piper-plus/). `kokoro-onnx` claims near real-time on an M1. [kokoro-onnx](https://simple-repository.app.cern.ch/project/kokoro-onnx)
- A fully local Pipecat agent on an M-series Mac (Silero VAD, MLX Whisper, Gemma, Kokoro) reaches under 800 ms voice-to-voice. [macos-local-voice-agents](https://github.com/kwindla/macos-local-voice-agents)
- The official Piper voice list I opened shows Spanish but not Basque; the Basque Piper voices come from a community adaptation of HiTZ voices, listed as CC BY 4.0 on a third-party site. [voices list](https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/docs/VOICES.md), [blog](https://xezpeleta.github.io/en/blog/euskarazko-ahots-sintesirako-ereduak)

Recommendation **(inference)**: the robot only needs about a dozen sentences ("Which ball?", "Striped ball, to your hand", "Sorry, again please"). Render them once to WAV with `say` or Piper, play the files, and the demo has no TTS latency and no TTS failure mode. A loudspeaker is needed anyway; a laptop speaker will not be heard in the hall.

---

## 8. Hosted speech-to-text as a fallback

| Service | Free terms | Price after | Speed | Notes |
|---|---|---|---|---|
| **Groq** `whisper-large-v3-turbo` | Free plan exists; limits reported as 20 requests/min, 2,000/day, 7,200 audio-seconds/hour, 25 MB files | $0.04 per audio hour (large-v3: $0.111) | speed factor 216x | Minimum billed length 10 s; `language` parameter improves accuracy and latency; `prompt` up to 224 tokens |
| **Deepgram** Nova-3 | $200 credit, no expiry, no card | streaming $0.0048/min mono, $0.0058/min multilingual (promo) | streaming API | "60+ languages" |
| **ElevenLabs** Scribe | reported 5 hours/month free | reported $1 per audio hour | n/a | vendor blog |
| **AssemblyAI** | reported $50 credit or 100 min/month | n/a | n/a | sources disagree |

Sources: Groq [speech-to-text docs](https://console.groq.com/docs/speech-to-text), [rate limits](https://console.groq.com/docs/rate-limits), [third-party limits](https://spokenly.app/blog/free-speech-to-text-apis); Deepgram [pricing](https://deepgram.com/pricing); ElevenLabs [blog](https://elevenlabs.io/blog/assemblyai-alternatives); AssemblyAI [dev.to comparison](https://dev.to/build996/free-whisper-api-groq-deepgram-assemblyai-compared-562m).

- The Seeed SO-ARM tutorial runs entirely on Groq's free tier (Whisper turbo, Llama 3.1 8B, Orpheus TTS) and warns about rate limits: "wait between commands". [Seeed wiki](https://wiki.seeedstudio.com/respeaker_flex_soarm/)
- Hosted STT only helps if the network holds, and the microphone problem is the same. Use it as a second try behind the local model, with a 2 to 3 s timeout **(inference)**.

---

## 9. Prior art

**"AI Bartender", winner of the LeRobot Hackathon Bilbao 2025 (team Megas XLR).** Pipeline STT, then LLM (Gemini on free credits, with a joking bartender persona), then robot, then TTS; two SO-101 arms supplied by the organisers; ACT policies trained on 50 and 30 pouring demonstrations (20,000 steps, batch size 4) on cloud GPUs via Brev with NVIDIA credits; GR00T rejected because the demo needed local inference without internet latency. The write-up does not name the STT or TTS tools and does not explain how the LLM output selected the policy. What went wrong: the robot stopped working Saturday morning (suspected platform bug) and they reinstalled Ubuntu; electronics problems shortly before the demo; an extra-object experiment was dropped as laggy; in the final a crowd blocked the table so the cup could not be swapped and both drinks went into one cup, which the jury liked. [write-up](https://www.elchinaslanli.com/projects/how-we-won-the-lerobot-hackathon-2025-meet-the-ai-bartender), [university article](https://www.udg.edu/en/masters-en-tecnologia/mirs/petjada-mirs/aslanli)

**Voice-commanded tool-handing arm.** I found no write-up of such a project from Bilbao 2025; the event site lists only totals for the first edition. [event site](https://hackathonbilbao.eus/). The closest public project is **Say-and-Grasp** (Meta developer hackathon, October to November 2026): SO-101 "third hand" that fetches a named tool into the user's open hand; wake word "Wiggle", Enter push-to-talk and typed input; hosted STT and a hosted multimodal model that calls `fetch`/`put_away` tools; Piper for speech with espeak-ng fallback; 10 of 10 fetches into the hand, median 22 s from request to palm; depends entirely on a cloud API. [GitHub](https://github.com/adoodevv/say-and-grasp)

**Others on SO-100/SO-101 or similar:**

- **Seeed SO-ARM voice controller** (May 2026): openWakeWord "hey jarvis", fixed 3 s recording, Groq Whisper turbo, Groq Llama 3.1 8B mapping trigger phrases to named poses, Groq Orpheus TTS. Tips: lower the wake threshold to 0.3, stay within 1 m of the mic. [Seeed wiki](https://wiki.seeedstudio.com/respeaker_flex_soarm/)
- **RoboStrands** (SO-101, ACT): ReSpeaker Lite mic, Enter key, fixed 10 s window, local faster-whisper, Qwen through Ollama, eSpeak. Problems: short recordings missed part of the request; the LLM had to be unloaded from the GPU before the policy launched; mic is off during motion. [AWS Builder](https://builder.aws.com/content/3JI3aJOqqMS4ex561I2LbsudkEN/agents-for-humans-building-robostrands-a-voice-guided-robot-assistant)
- **RoboticsHackCal** (SO-101, ACT peanut hand-off to a person): the autonomous hand-off worked from the command line; "the ElevenLabs voice-to-robot path was not finished at the event". [GitHub](https://github.com/MaxBingham/RoboticsHackCal)
- **SO101-Voice-Control-Physical-AI**: Vosk on Windows writes a JSON file; WSL2 side checks an allow-list and a `qwen3:1.7b` classifier; 15 of 20 trials. [GitHub](https://github.com/noittz/SO101-Voice-Control-Physical-AI)
- **earshot** (SO-101, LeRobot 0.6.2): Pipecat, Deepgram nova-3, Silero VAD for end of turn, a small hosted LLM forced to one tool call per turn; voice process never moves the arm. Grasping, not voice, was the unreliable part. [GitHub](https://github.com/MNS1007/earshot)
- **Dum-E** (SO-100/101, GR00T): Pipecat with Deepgram, Anthropic and ElevenLabs. [GitHub](https://github.com/aaronsu11/dum-e)
- **hack_princeton** (OpenMANIPULATOR-X): cloud STT with offline Vosk 74-word fallback that always owns "stop"; confirmation before pick; mic muted during speech. [GitHub](https://github.com/aditya-baniya-ai/hack_princeton)

Pattern across all of them: **the voice part is the piece that slips or stays unfinished when time runs out, and the teams that showed it working had a key-press or typed path next to it.**

---

## Recommended stack for this team

### Minimal pipeline (build this first)

```
[headset / lavalier mic]
      |
[voice process, own venv, native Windows or the MacBook]
  1. Activation: press Enter to start, Enter to stop  (push-to-talk)
  2. Record 16 kHz mono with sounddevice, hard cap 8 s
  3. STT: onnx-asr + Parakeet TDT 0.6b v3 on CPU
  4. Parse: synonym table + fuzzy match  ->  {"target": "striped", "destination": "hand"}
     - require the wake phrase ("hey robot ...") in the transcript if the "wake-up call" should be visible
     - slot missing -> play "which ball?" clip and listen once more
  5. Play the matching pre-rendered confirmation clip
  6. Send one JSON line over a local TCP socket
      |
[state machine, robot environment]  <- also accepts keys 1..N as the same commands
```

Component choices:

| Piece | Choice | Why |
|---|---|---|
| Activation | Push-to-talk (Enter or a held key), wake phrase checked in the transcript | Only path shown to hold up in noise (section 1) |
| VAD | None with push-to-talk; Silero if hands-free is added | Key release ends the utterance |
| STT | Parakeet TDT 0.6b v3 via `onnx-asr`, CPU; fallback faster-whisper `small` int8 with language forced | Section 3 |
| Intent | Rules with fuzzy matching in English and Spanish; LLM only when rules fail | Section 6 |
| TTS | Pre-rendered clips made with `say` or Piper, played through a real loudspeaker | Section 7 |
| Transport | One JSON line over localhost TCP | Same split as [SO101-Voice-Control](https://github.com/noittz/SO101-Voice-Control-Physical-AI) |

Sketch of the voice process (API calls for `onnx_asr` as shown in its README; untested):

```python
import json, socket, wave, sounddevice as sd, onnx_asr
from rapidfuzz import fuzz

SR = 16000
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

TARGETS = {"striped": ["striped", "stripped", "stripe", "rayada", "rayas"],
           "solid":   ["solid", "plain", "lisa"]}
DESTS   = {"hand":      ["hand", "give it to me", "to me", "mano", "dámela"],
           "container": ["container", "basket", "box", "caja", "contenedor"]}

def slot(text, table, cutoff=80):
    best = max(((fuzz.partial_ratio(s, text), k) for k, syns in table.items() for s in syns))
    return best[1] if best[0] >= cutoff else None

while True:
    input("Enter = talk")
    rec = sd.rec(int(8 * SR), samplerate=SR, channels=1, dtype="int16")
    input("Enter = done"); sd.stop()
    with wave.open("cmd.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(rec.tobytes())
    text = model.recognize("cmd.wav").lower()
    cmd = {"target": slot(text, TARGETS), "destination": slot(text, DESTS), "transcript": text}
    print(cmd)
    if cmd["target"] or cmd["destination"]:
        socket.create_connection(("127.0.0.1", 5555)).sendall((json.dumps(cmd) + "\n").encode())
```

### Where each piece runs

- **Voice process**: native Windows Python 3.12 venv on the policy laptop (CPU only), or the MacBook Pro. Not inside WSL2 (mic access there is fragile, section 4) and not in the robot environment (separate venv, so torch 2.11 / CUDA 12.8 / opencv-headless stay untouched).
- **State machine and ACT policy**: unchanged, robot environment, RTX 4070.
- **Optional local LLM**: MacBook Pro (36 GB), through Ollama. Never on the 4070.
- **Link**: localhost if both run on the Windows laptop. If the voice process is on the Mac, connect the two laptops with a cable or a phone hotspot the team controls, not the venue Wi-Fi.
- **Loudspeaker**: any USB or Bluetooth speaker, placed away from the mic.

### Fallback ladder

1. Voice with wake phrase in the transcript, rules parser (the normal path).
2. Rules found no slot: one spoken re-ask, then optional LLM parse (Claude if online, Ollama on the Mac if not).
3. Local STT misbehaves: hosted STT (Groq Whisper turbo) with the same parser, 2 to 3 s timeout.
4. Voice unusable in the hall: keys 1..N on the keyboard trigger the same JSON commands. The presenter still says the sentence aloud; the demo continues.
5. Everything on the voice laptop is down: the state machine's own keyboard input on the robot laptop.

Always on: transcript and parsed command shown large on screen; mic muted while clips play and while the arm moves; a physical stop that does not depend on voice.

### Optional upgrades, in order of value

1. Hold-to-talk on a single key or a presenter clicker instead of two Enter presses.
2. Always-listening mode with a pre-trained "hey jarvis" model (openWakeWord through RealtimeSTT, or livekit-wakeword) and Silero end-of-speech, behind a switch, tested in the hall at its loudest.
3. LLM fallback with a JSON schema for the two slots.
4. One Basque sentence in the robot's replies.

### Build-time estimate (one person, estimate)

| Step | Time |
|---|---|
| venv, model download, mic test, push-to-talk record and transcribe | 45 to 60 min |
| Rules parser with synonyms, socket to state machine, keyboard fallback | 45 to 60 min |
| Pre-rendered clips and playback, mic muting | 30 min |
| Test in the hall with the real mic, extend the synonym table | 30 to 45 min |
| **Minimal pipeline total** | **about 2.5 to 3 h** |
| Wake word with a pre-trained model + VAD endpointing | +1 to 1.5 h |
| Custom wake phrase (training plus threshold tuning) | +1.5 to 2 h |
| LLM fallback | +1 h |
| **With all options** | **about 5 to 7 h** |

Do the hall test on Saturday while the room is loud, and freeze the voice code by Saturday evening.

---

## Unverified

Not confirmed from a primary source I opened, or my own reasoning:

- **Nothing was installed or run.** The code sketch, the claim that `onnx-asr` and faster-whisper work on this team's exact machines, and all build-time figures are estimates.
- **Latency for short commands.** I have no measured figure for a 3 to 5 s command on a laptop CPU for any model. Parakeet's RTFx 36 is from a desktop Ryzen 9800X3D; the faster-whisper CPU figure is for 13 minutes of audio on a desktop i7. Whisper models process fixed 30 s windows, which should make them slower than Parakeet or Moonshine on short clips; this is my understanding and only implied by the Moonshine paper abstract.
- **Parakeet v3 language control.** I did not confirm whether the language can be forced or is always auto-detected, nor the int8 model file size, nor that `onnx-asr` accepts in-memory arrays (the sketch therefore writes a WAV file).
- **Whisper and Basque.** That `eu` is among Whisper's 99 languages, and that the stock (not fine-tuned) models are poor on Basque, is from memory; the model card I opened does not list the languages.
- **Per-size Whisper accuracy on Spanish** (small versus turbo): no numbers found.
- **Picovoice Porcupine free tier.** The shutdown is reported by a forum user paraphrasing an email; Picovoice's pricing page could not be read; one third-party blog still lists a free plan.
- **Groq free-plan limits** (20 requests/min, 2,000/day, 7,200 audio-seconds/hour): third-party figures; the official page's table could not be confirmed as the free-plan table. ElevenLabs and AssemblyAI free terms are from vendor or third-party blogs and disagree between sources. No measured network latency for any hosted STT.
- **Leaderboard standings for late 2026** (Nemotron, Qwen3-ASR and similar at the top): from secondary blog posts, not the leaderboard.
- **RealtimeSTT release year.** The releases page shows "11 Nov" without a year; I read it as 2026 from the page footer and context. Its Windows behaviour was not checked beyond the README note.
- **livekit-wakeword on Windows** and the exact Python inference API were not confirmed. The accuracy comparison with openWakeWord is the vendor's own.
- **openWakeWord Colab notebook** current working state not checked; the latest release is from February 2024.
- **Close-talk versus far-field numbers** (2.5 to 4x, 52.0 % vs 29.6 %) and the **RSA-Bench denoising numbers** come from search summaries of the linked papers; I did not open the PDFs, and the attribution of each number to the specific linked paper is not checked. The 26 dB figure is my arithmetic.
- **faster-whisper `hotwords` and language-detection behaviour** come from a search summary, not the current source code.
- **Piper speed (RTF 0.06)** and **Basque Piper voice licence** are from third-party pages. Piper's latest version number is unclear (sources say 1.4.2 and 1.8.0).
- **Kokoro Spanish quality** is described as "partial support" in one NVIDIA card summary; not listened to.
- **Silero VAD pip package**: the README lists `torch>=1.12` as required; whether an ONNX-only install without torch works was not confirmed. The licence badge link text on GitHub says CC BY-NC 4.0 while the repo sidebar and file say MIT.
- **WSL2 networking direction** (whether Windows reaches a WSL2 server on localhost, or the reverse, without extra settings) was not researched; test it early.
- **usbipd-win with USB audio**: I did not find a source confirming that the stock WSL2 kernel includes the USB audio driver.
- **Small-LLM details**: no benchmark found for 4B-class models on single-shot slot filling, nor time-to-first-token on CPU; whether Qwen3's thinking mode must be disabled for speed was not checked.
- **Tool-handing arm from Bilbao 2025**: not found; only the AI Bartender is documented.
- **Design advice labelled (inference)** throughout: push-to-talk as primary, no software denoiser, pre-rendered clips, word choices for slots, confirmation style, hall-test timing.

---

## Addendum: using Claude for the command step

Added 2026-11-20 by Tian's agent. Facts are from the official docs as read by a documentation agent that evening; nothing here was run.

**A Claude subscription does not include API usage.**
- Per https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers.md : Max, Ultra and Team plans get monthly API credits, but only after a Console organization is linked. Free, Pro and Enterprise are not eligible. Everyone else buys Console credits on the Billing page.
- About 100 one-sentence calls cost a cent or two (estimate).

**Two ways to call Claude from the voice script.**

1. Claude Code in non-interactive mode, with the subscription login already on the laptop. Flags per https://code.claude.com/docs/en/cli-reference.md and https://code.claude.com/docs/en/headless.md :

   ```bash
   claude -p "<prompt with the transcript>" --output-format json --json-schema '<schema>' --max-turns 2 --model haiku --tools ""
   ```

   The result is in the `structured_output` field of the JSON. Start-up time per call is not documented, so measure it before relying on it. `--bare` is recommended for scripts but skips the subscription login and needs an API key.

2. The Python SDK with a Console API key (`ANTHROPIC_API_KEY`), in the voice process's own environment. Sketch, following the SDK's documented structured-output helper:

   ```python
   from typing import Literal, Optional
   import anthropic
   from pydantic import BaseModel

   class Command(BaseModel):
       target: Optional[Literal["striped"]]            # add colours here when they exist
       destination: Optional[Literal["hand", "container"]]

   client = anthropic.Anthropic(timeout=4.0, max_retries=0)

   def parse_command(transcript: str) -> Optional[Command]:
       try:
           response = client.messages.parse(
               model="claude-opus-5-5",
               max_tokens=1024,
               output_config={"effort": "low"},
               system="Turn the spoken robot command into the two slots. Use null for a slot the speaker did not mention.",
               messages=[{"role": "user", "content": transcript}],
               output_format=Command,
           )
           return response.parsed_output
       except anthropic.APIError:
           return None   # fall back to the synonym table
   ```

   `claude-opus-5-5` is the default model. If the reply is too slow for a live demo, `claude-haiku-5-5` is the small fast model and a one-line change; that choice is the team's.

**Keep the synonym table as the first parser and the fallback.** The vocabulary is two slots with a handful of values, the venue network may fail, and the rule that inference runs on our own laptops may or may not cover a cloud call for understanding a sentence. Ask a mentor before relying on it in the judged demo.
