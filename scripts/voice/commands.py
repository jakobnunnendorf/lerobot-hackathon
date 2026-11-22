import re
from dataclasses import dataclass

from rapidfuzz import fuzz

DEFAULT_TARGET = "striped"
# Whole words, not fuzzy: a three-letter word such as "red" scores 100 inside "tired" or "hundred".
TARGETS = {
    "striped": {"striped", "stripy", "stripes", "stripe", "yellow", "green"},
    "red": {"red"},
    "blue": {"blue"},
}
DESTINATIONS = {
    "hand": (
        "give it to me", "give me the ball", "give me", "hand it to me", "hand me", "hand it over", "pass it to me",
        # "in the hand", not "put it in the hand": the longer phrase pulled "put it in the contain her" off container.
        "in my hand", "in the hand",
    ),
    "container": ("put it in the container", "container"),
}
# Whole words, like the colours: "dance" has no near-miss among the ball commands, and "dancing"/"boogie" cover
# "let's go dancing" and "boogie time".
DANCE_WORDS = {"dance", "dancing", "boogie"}
# Whole words for the same reason; "byte" is how Whisper tends to hear "bite".
BITE_WORDS = {"bite", "bites", "biting", "byte"}
KEYS = {"1": "hand", "2": "container"}
MATCH_CUTOFF = 80
MIN_MARGIN = 10


@dataclass(frozen=True)
class Command:
    target: str
    destination: str
    transcript: str
    source: str


# No wake phrase: Enter already marks the words as meant for the robot, and "hey bot" was misheard too often
# when tried live. See decisions/voice-commands.md.
def parse(transcript: str) -> Command | None:
    words = normalise(transcript).split()
    if (DANCE_WORDS | BITE_WORDS) & set(words):
        return None
    destination = best_match(" ".join(words), DESTINATIONS)
    if destination is None:
        return None
    target = named_target(words)
    if target is None:
        return None
    return Command(target, destination, transcript.strip(), "voice")


@dataclass(frozen=True)
class Scores:
    destinations: dict[str, float]
    colours: tuple[str, ...]


def scores(transcript: str) -> Scores:
    words = normalise(transcript).split()
    text = " ".join(words)
    destinations = {key: max(similarity(p, text) for p in phrases) / 100 for key, phrases in DESTINATIONS.items()}
    colours = tuple(target for target, colour_words in TARGETS.items() if colour_words & set(words))
    return Scores(destinations, colours)


def from_key(key: str) -> Command | None:
    destination = KEYS.get(key.strip())
    if destination is None:
        return None
    return Command(DEFAULT_TARGET, destination, "", "keyboard")


# A sentence with a dance word and a ball command is refused: which one was meant is a guess.
def wants_dance(transcript: str) -> bool:
    words = normalise(transcript).split()
    return bool(DANCE_WORDS & set(words)) and not BITE_WORDS & set(words) and best_match(" ".join(words), DESTINATIONS) is None


# Same rule for a bite: next to a dance word or a ball command, which one was meant is a guess.
def wants_bite(transcript: str) -> bool:
    words = normalise(transcript).split()
    return bool(BITE_WORDS & set(words)) and not DANCE_WORDS & set(words) and best_match(" ".join(words), DESTINATIONS) is None


def rejection_reason(transcript: str) -> str:
    words = normalise(transcript).split()
    if BITE_WORDS & set(words) and not wants_bite(transcript):
        return "a bite and another command in one sentence"
    if wants_dance(transcript):
        return "a dance: the dashboard starts it, not listen.py"
    if DANCE_WORDS & set(words):
        return "a dance and a ball command in one sentence"
    if best_match(" ".join(words), DESTINATIONS) is not None and named_target(words) is None:
        return "more than one ball colour named"
    return "no single command recognised"


# The striped ball is the demo's target, so a command that names no colour means it; two colours are refused,
# because guessing one would send the wrong ball.
def named_target(words: list[str]) -> str | None:
    named = [target for target, colour_words in TARGETS.items() if colour_words & set(words)]
    if not named:
        return DEFAULT_TARGET
    return named[0] if len(named) == 1 else None


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower().replace("'", "")).strip()


def best_match(text: str, table: dict[str, tuple[str, ...]]) -> str | None:
    scores = sorted(((max(similarity(p, text) for p in phrases), key) for key, phrases in table.items()), reverse=True)
    best, key = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0
    if best < MATCH_CUTOFF or best - runner_up < MIN_MARGIN:
        return None
    return key


def similarity(phrase: str, text: str) -> float:
    # partial_ratio alone would rate a fragment such as "me" 100 against "give it to me"
    if len(text) < len(phrase):
        return fuzz.ratio(phrase, text)
    return fuzz.partial_ratio(phrase, text)
