import re
from dataclasses import dataclass

from rapidfuzz import fuzz

WAKE_PHRASE = "hey bot"
# measured mis-hearings that fuzzy matching misses; ordinary sentence starts heard in its place ("people", "paper",
# "they bought") stay out, or any first word would pass. See decisions/voice-commands.md.
WAKE_VARIANTS = {
    "hey bot", "heybot", "hey robot",
    "hey bud", "hey bosch", "hey bach", "hey buck", "hey pop", "hey von", "hey vaughn", "hey vaugh", "hey vod",
    "hey fox", "hey folks", "hey books", "hey bye",
    "a bot", "k bot", "pay bot", "paybot", "paypot", "pay bought", "paybought", "paybook", "pay box", "paybox",
    "play bot", "playbot", "haybot", "haybox", "heybox", "baybot", "peabot",
}
WAKE_CUTOFF = 80

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
        "in my hand",
    ),
    "container": ("put it in the container", "container"),
}
KEYS = {"1": "hand", "2": "container"}
MATCH_CUTOFF = 80
MIN_MARGIN = 10


@dataclass(frozen=True)
class Command:
    target: str
    destination: str
    transcript: str
    source: str


def parse(transcript: str) -> Command | None:
    rest = strip_wake_phrase(normalise(transcript).split())
    if rest is None:
        return None
    text = " ".join(rest)
    destination = best_match(text, DESTINATIONS)
    if destination is None:
        return None
    target = named_target(rest)
    if target is None:
        return None
    return Command(target, destination, transcript.strip(), "voice")


def from_key(key: str) -> Command | None:
    destination = KEYS.get(key.strip())
    if destination is None:
        return None
    return Command(DEFAULT_TARGET, destination, "", "keyboard")


def rejection_reason(transcript: str) -> str:
    if strip_wake_phrase(normalise(transcript).split()) is None:
        return f'no "{WAKE_PHRASE}" at the start'
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


def strip_wake_phrase(words: list[str]) -> list[str] | None:
    score, length = max((wake_score(" ".join(words[:n])), n) for n in (1, 2, 3))
    return words[length:] if score >= WAKE_CUTOFF else None


def wake_score(head: str) -> float:
    return 100 if head in WAKE_VARIANTS else fuzz.ratio(head, WAKE_PHRASE)


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
