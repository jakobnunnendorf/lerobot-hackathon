import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

import commands
import speech

# The texts still start with the dropped wake phrase, so that saved transcripts stay comparable; the parser now
# ignores it, and the two sentences without it are commands.
SENTENCES = [
    ("Hey bot, give it to me.", "hand"),
    ("Hey bot, give me the ball.", "hand"),
    ("Hey bot, hand it to me.", "hand"),
    ("Hey bot, put it in my hand.", "hand"),
    ("Hey bot, put it in the container.", "container"),
    ("Hey bot, drop it in the container.", "container"),
    ("Hey bot, put the ball in the container.", "container"),
    ("Hey bot, container, please.", "container"),
    ("Give it to me.", "hand"),
    ("Put it in the container.", "container"),
    ("Hey bot, what time is it?", None),
    ("Hey bot, open the window.", None),
    ("I think the robot is great.", None),
    ("Hey bot, give it to me or put it in the container.", None),
]
BABBLE_TEXT = (
    "We walked along the river to the museum this morning and had coffee near the old bridge. "
    "The train back is at six, so we should leave a little early and maybe buy some cheese on the way."
)
RATES = (0, 3)
CONDITIONS = (("clean", None), ("white", 10), ("white", 5), ("babble", 10), ("babble", 5))
PAD_SECONDS = 0.5
BABBLE_TALKERS = 4
# a fixed subset that covers every condition, so that a latency run on a quiet machine stays short
LATENCY_EVERY = 7
# without the prompt, to see whether the prompt makes Whisper write a wake phrase nobody said
LOADERS = speech.BACKENDS | {"whisper-base-no-prompt": lambda: speech.load_whisper("base.en", prompt=None)}

SYNTH_SCRIPT = r"""
param($job)
Add-Type -AssemblyName System.Speech
$j = Get-Content $job -Raw | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, 'Sixteen', 'Mono')
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($s.GetInstalledVoices() | Where-Object Enabled | ForEach-Object { $_.VoiceInfo.Name })
for ($v = 0; $v -lt $voices.Count; $v++) {
  $s.SelectVoice($voices[$v])
  foreach ($r in $j.rates) {
    $s.Rate = $r
    for ($i = 0; $i -lt $j.texts.Count; $i++) {
      $s.SetOutputToWaveFile((Join-Path $j.out "$v`_$r`_$i.wav"), $fmt)
      $s.Speak($j.texts[$i])
    }
  }
}
$s.SetOutputToNull()
ConvertTo-Json @($voices) | Set-Content (Join-Path $j.out 'voices.json')
"""


def synthesize(texts: list[str], out: Path) -> dict[tuple[str, int, int], np.ndarray]:
    out.mkdir(parents=True, exist_ok=True)
    job = json.dumps({"texts": texts, "rates": RATES, "out": str(out)})
    job_file, voices_file = out / "job.json", out / "voices.json"
    # reuse the audio of an earlier run with the same job, so that every backend is scored on identical clips
    if not (job_file.exists() and job_file.read_text() == job and voices_file.exists()):
        voices_file.unlink(missing_ok=True)
        job_file.write_text(job)
        run_synthesizer(out)
    names = json.loads(voices_file.read_text(encoding="utf-8-sig"))
    clips, seen = {}, set()
    for v, name in enumerate(names):
        voice_clips = {(name, r, i): speech.read_wav(out / f"{v}_{r}_{i}.wav") for r in RATES for i in range(len(texts))}
        # some Windows voices are registered twice under two names and sound identical
        digest = hashlib.sha1(b"".join(a.tobytes() for a in voice_clips.values())).hexdigest()
        if digest not in seen:
            seen.add(digest)
            clips |= voice_clips
    return clips


def run_synthesizer(out: Path) -> None:
    (out / "synth.ps1").write_text(SYNTH_SCRIPT)
    # PowerShell 7 also offers the OneCore voices; Windows PowerShell 5.1 only the three Desktop ones
    shell = shutil.which("pwsh") or "powershell"
    subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", out / "synth.ps1", out / "job.json"], check=True
    )


def babble(tracks: list[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    mix = np.zeros(length, np.float32)
    for index in rng.choice(len(tracks), BABBLE_TALKERS, replace=False):
        track = np.tile(tracks[index], length // len(tracks[index]) + 2)
        start = rng.integers(len(tracks[index]))
        mix += track[start : start + length] / np.sqrt(np.mean(tracks[index] ** 2))
    return mix


def noisy(clean: np.ndarray, kind: str, snr_db: float, tracks: list[np.ndarray], rng) -> np.ndarray:
    padded = np.pad(clean, int(PAD_SECONDS * speech.SAMPLE_RATE))
    if kind == "clean":
        return padded
    noise = rng.standard_normal(len(padded)).astype(np.float32) if kind == "white" else babble(tracks, len(padded), rng)
    noise *= np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    mixed = padded + noise
    return mixed / max(1.0, np.abs(mixed).max())


def build_clips(out: Path) -> list[dict]:
    texts = [text for text, _ in SENTENCES]
    speech_clips = synthesize(texts + [BABBLE_TEXT], out)
    babble_index = len(texts)
    tracks = [a for (_, r, i), a in speech_clips.items() if i == babble_index and r == 0]
    rng = np.random.default_rng(0)
    return [
        {"voice": voice, "rate": rate, "sentence": i, "condition": f"{kind} {snr or ''}".strip(),
         "audio": noisy(audio, kind, snr, tracks, rng)}
        for (voice, rate, i), audio in speech_clips.items() if i != babble_index
        for kind, snr in CONDITIONS
    ]


def measure(backend: str, clips: list[dict]) -> dict:
    started = time.perf_counter()
    transcribe = LOADERS[backend]()
    load_seconds = time.perf_counter() - started
    transcribe(clips[0]["audio"])  # warm-up: the first call allocates buffers and is not typical
    rows = []
    for clip in clips:
        started = time.perf_counter()
        transcript = transcribe(clip["audio"])
        seconds = time.perf_counter() - started
        rows.append({k: v for k, v in clip.items() if k != "audio"} | {"transcript": transcript, "seconds": seconds})
    return {"backend": backend, "load_seconds": load_seconds, "rows": rows}


def report_latency(result: dict) -> None:
    seconds = [r["seconds"] for r in result["rows"]]
    print(
        f"\n{result['backend']}: {len(seconds)} clips, load {result['load_seconds']:.1f} s, per clip median "
        f"{statistics.median(seconds):.2f} s, 90th percentile {statistics.quantiles(seconds, n=10)[-1]:.2f} s, "
        f"worst {max(seconds):.2f} s"
    )


def report_accuracy(rows: list[dict]) -> None:
    print(
        f"  {'condition':<11}{'accepted right':>16}{'rejects right':>15}{'false accepts':>15}"
        f"{'wrong action':>14}{'all right':>11}"
    )
    for condition in dict.fromkeys(r["condition"] for r in rows):
        print(f"  {condition:<11}" + score_line([r for r in rows if r["condition"] == condition]))
    print(f"  {'all':<11}" + score_line(rows))


def score_line(rows: list[dict]) -> str:
    accepts = [r for r in rows if SENTENCES[r["sentence"]][1]]
    rejects = [r for r in rows if not SENTENCES[r["sentence"]][1]]
    accepted_right = sum(heard(r) == expected(r) for r in accepts)
    rejected_right = sum(heard(r) is None for r in rejects)
    wrong_action = sum(heard(r) not in (None, expected(r)) for r in accepts)
    return (
        f"{accepted_right:>9}/{len(accepts):<6}{rejected_right:>8}/{len(rejects):<6}"
        f"{len(rejects) - rejected_right:>15}{wrong_action:>14}{(accepted_right + rejected_right) / len(rows):>11.0%}"
    )


def heard(row: dict) -> str | None:
    command = commands.parse(row["transcript"])
    return command and command.destination


def expected(row: dict) -> str | None:
    return SENTENCES[row["sentence"]][1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech-to-text plus parser on synthetic Windows TTS speech.")
    parser.add_argument("backends", nargs="*", default=list(LOADERS))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / ".venv" / "tmp" / "benchmark")
    parser.add_argument("--rescore", action="store_true", help="re-parse saved transcripts, no speech-to-text")
    parser.add_argument(
        "--latency", action="store_true", help=f"time every {LATENCY_EVERY}th clip only, saved in <out>/latency"
    )
    args = parser.parse_args()
    results = args.out / "latency" if args.latency else args.out
    if not args.rescore:
        clips = build_clips(args.out / "audio")
        if args.latency:
            clips = clips[::LATENCY_EVERY]
        print(f"{len(clips)} clips")
        results.mkdir(parents=True, exist_ok=True)
        for backend in args.backends:
            (results / f"{backend}.json").write_text(json.dumps(measure(backend, clips), indent=1))
    for backend in args.backends:
        saved = results / f"{backend}.json"
        if not saved.exists():
            print(f"\n{backend}: no saved result in {results}")
            continue
        result = json.loads(saved.read_text())
        report_latency(result)
        if not args.latency:
            report_accuracy(result["rows"])


if __name__ == "__main__":
    main()
