import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from commands import DESTINATIONS

SAMPLE_RATE = 16_000
# kept with the dropped wake phrase: the Whisper results in decisions/voice-commands.md were measured with it
WHISPER_PROMPT = " ".join(f"Hey bot, {phrases[0]}." for phrases in DESTINATIONS.values())

Transcribe = Callable[[np.ndarray], str]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def load_parakeet(model_name: str) -> Transcribe:
    # imported here so that only the chosen backend's package has to be installed
    import onnx_asr

    model = onnx_asr.load_model(model_name, quantization="int8")
    return lambda audio: model.recognize(audio, sample_rate=SAMPLE_RATE)


def load_whisper(model_name: str, prompt: str | None = WHISPER_PROMPT) -> Transcribe:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(
            audio, language="en", initial_prompt=prompt, without_timestamps=True, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments)

    return transcribe


BACKENDS: dict[str, Callable[[], Transcribe]] = {
    "parakeet": lambda: load_parakeet("nemo-parakeet-tdt-0.6b-v3"),
    "parakeet-en": lambda: load_parakeet("nemo-parakeet-tdt-0.6b-v2"),
    "whisper-base": lambda: load_whisper("base.en"),
    "whisper-small": lambda: load_whisper("small.en"),
}
