import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sounddevice as sd

import commands
import link
import speech

DEFAULT_BACKEND = "parakeet-en"
MAX_SECONDS = 8
MIN_SECONDS = 0.3
SILENCE_PEAK = 0.01
POLL_SECONDS = 0.2
BLOCK_SECONDS = 0.05
PRE_ROLL_SECONDS = 0.5
TAIL_SECONDS = 0.3
# the benchmark clips had this much silence at each end (benchmark.PAD_SECONDS); live recordings get the same
PAD_SECONDS = 0.5
RULE = "=" * 72
ACTIONS = {"hand": "pick it, give it to a hand", "container": "pick it, put it in container"}


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    transcribe = load_backend(args.backend)
    if args.wav:
        for path in args.wav:
            print(f"\n{path}")
            handle_audio(speech.read_wav(path), transcribe, args)
        return
    try:
        run_interactive(transcribe, args)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push-to-talk voice commands for the robot.")
    parser.add_argument("--backend", choices=speech.BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument("--device", type=device_id, help="input device number or part of its name")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--host", default=link.HOST)
    parser.add_argument("--port", type=int, default=link.PORT)
    parser.add_argument("--wav", type=Path, nargs="+", help="transcribe these files instead of the microphone")
    parser.add_argument("--dry-run", action="store_true", help="show the match scores and never send a command")
    return parser.parse_args()


def device_id(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def load_backend(name: str) -> speech.Transcribe | None:
    print(f"loading speech-to-text: {name} ...", flush=True)
    started = time.perf_counter()
    try:
        transcribe = speech.BACKENDS[name]()
    except Exception as error:  # whatever breaks here, the keyboard commands must keep working
        print(f"SPEECH-TO-TEXT FAILED TO LOAD: {error!r}\nKeys {', '.join(commands.KEYS)} still work.")
        return None
    print(f"ready in {time.perf_counter() - started:.1f} s")
    return transcribe


class Microphone:
    # Always open, keeping the last half second: a stream opened on Enter delivered its first audio 0.12-0.33 s
    # later on Tian's laptop, long enough to cut the first word of a command spoken right away.
    # See decisions/voice-commands.md.
    def __init__(self, device: int | str | None) -> None:
        self.lock = threading.Lock()
        self.recent: deque[np.ndarray] = deque(maxlen=round(PRE_ROLL_SECONDS / BLOCK_SECONDS))
        self.taken: list[np.ndarray] | None = None
        self.stream = sd.InputStream(
            samplerate=speech.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=round(BLOCK_SECONDS * speech.SAMPLE_RATE),
            callback=self.receive,
        )
        self.stream.start()

    def receive(self, data: np.ndarray, *_) -> None:
        with self.lock:
            self.recent.append(data[:, 0].copy())
            if self.taken is not None:
                self.taken.append(self.recent[-1])

    def start(self) -> None:
        with self.lock:
            self.taken = list(self.recent)

    def stop(self) -> np.ndarray:
        with self.lock:
            taken, self.taken = self.taken or [], None
        return np.concatenate(taken) if taken else np.zeros(0, np.float32)


def run_interactive(transcribe: speech.Transcribe | None, args: argparse.Namespace) -> None:
    lines = start_line_reader()
    microphone = open_microphone(args.device)
    while True:
        print_menu()
        line = next_line(lines)
        if line == "q":
            return
        if line == "":
            listen_once(lines, microphone, transcribe, args)
        elif (command := commands.from_key(line)) is not None:
            show(f"KEY {line}", command, "")
            deliver(command, args)
        discard_pending(lines)


def listen_once(
    lines: queue.Queue[str],
    microphone: Microphone | None,
    transcribe: speech.Transcribe | None,
    args: argparse.Namespace,
) -> None:
    if transcribe is None:
        print("No speech-to-text loaded. Use the number keys.")
        return
    if microphone is None:
        print("No microphone. Use the number keys, or pick a microphone with --list-devices and --device.")
        return
    handle_audio(record(lines, microphone), transcribe, args)


def start_line_reader() -> queue.Queue[str]:
    # a reader thread lets the recording stop on Enter or on the time limit, whichever comes first
    lines: queue.Queue[str] = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            lines.put(line.strip().lower())
        lines.put("q")

    threading.Thread(target=read, daemon=True).start()
    return lines


def next_line(lines: queue.Queue[str], timeout: float | None = None) -> str | None:
    # short waits keep Ctrl+C working: a blocking Queue.get cannot be interrupted on Windows
    deadline = None if timeout is None else time.monotonic() + timeout
    while deadline is None or time.monotonic() < deadline:
        try:
            return lines.get(timeout=POLL_SECONDS)
        except queue.Empty:
            pass
    return None


def discard_pending(lines: queue.Queue[str]) -> None:
    while not lines.empty():
        # a quit, typed or from closed input, must survive, or the program waits forever on a closed stdin
        if lines.get_nowait() == "q":
            lines.put("q")
            return


def open_microphone(device: int | str | None) -> Microphone | None:
    try:
        microphone = Microphone(device)
        print(f"microphone: {sd.query_devices(device, 'input')['name']}")
        return microphone
    except (sd.PortAudioError, ValueError) as error:
        print(f"MICROPHONE FAILED: {error}\nUse the number keys, or pick a microphone with --list-devices and --device.")
        return None


def record(lines: queue.Queue[str], microphone: Microphone) -> np.ndarray:
    microphone.start()
    print(f"RECORDING ... press Enter to stop (stops by itself after {MAX_SECONDS} s)", flush=True)
    if next_line(lines, MAX_SECONDS) is None:
        print(f"stopped after {MAX_SECONDS} s")
    # Enter is often pressed on the last syllable
    time.sleep(TAIL_SECONDS)
    audio = microphone.stop()
    return audio[: round((PRE_ROLL_SECONDS + MAX_SECONDS + TAIL_SECONDS) * speech.SAMPLE_RATE)]


def handle_audio(audio: np.ndarray, transcribe: speech.Transcribe | None, args: argparse.Namespace) -> None:
    if transcribe is None:
        return
    # Whisper invents sentences on silence, so silence never reaches the recogniser
    if len(audio) < MIN_SECONDS * speech.SAMPLE_RATE or np.abs(audio).max() < SILENCE_PEAK:
        print("ONLY SILENCE RECORDED. Wrong microphone? See --list-devices and --device.")
        return
    started = time.perf_counter()
    transcript = transcribe(np.pad(audio, round(PAD_SECONDS * speech.SAMPLE_RATE)))
    seconds = time.perf_counter() - started
    command = commands.parse(transcript)
    show(f"HEARD    {transcript!r}  ({seconds:.1f} s)", command, transcript)
    if args.dry_run:
        show_scores(transcript)
    if command is not None:
        deliver(command, args)


def show(heading: str, command: commands.Command | None, transcript: str) -> None:
    if command is None:
        result = f"NOT UNDERSTOOD: {commands.rejection_reason(transcript)}"
    else:
        result = f">>> {command.destination.upper()}  ({command.target} ball)"
    print(f"\n{RULE}\n  {heading}\n\n  {result}\n{RULE}", flush=True)


def show_scores(transcript: str) -> None:
    result = commands.scores(transcript)
    cutoff, margin = commands.MATCH_CUTOFF / 100, commands.MIN_MARGIN / 100
    print(f"  match scores, 0-1: text similarity, not probabilities. A command needs {cutoff:.2f} for one action")
    print(f"  and a lead of {margin:.2f} over the other action.")
    for destination, score in sorted(result.destinations.items(), key=lambda item: -item[1]):
        print(f"    {destination:<10} {ACTIONS.get(destination, ''):<30} {score:.2f}")
    print(f"    ball named: {', '.join(result.colours) or f'none, so {commands.DEFAULT_TARGET}'}")


def deliver(command: commands.Command, args: argparse.Namespace) -> None:
    message = asdict(command)
    if args.dry_run:
        print(f"dry run, not sent: {json.dumps(message)}")
        return
    if link.send(message, args.host, args.port):
        print(f"sent {json.dumps(message)}")
    else:
        print(
            f"NOT SENT: nothing is listening on {args.host}:{args.port}. "
            "Start the robot program, or python scripts/voice/link.py to test."
        )


def print_menu() -> None:
    keys = "   ".join(f"[{key}] {commands.DESTINATIONS[destination][0]}" for key, destination in commands.KEYS.items())
    print(f"\n[Enter] talk   {keys}   [q] quit")


if __name__ == "__main__":
    main()
