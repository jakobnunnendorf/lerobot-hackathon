import re
from dataclasses import dataclass

from rapidfuzz import fuzz

DEFAULT_TARGET = "striped"
# Whole words, not fuzzy: a three-letter word such as "red" scores 100 inside "tired" or "hundred".
TARGETS = {
    "striped": {"striped", "stripy", "stripes", "stripe", "yellow", "green", "target"},
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
KEYS = {"1": "hand", "2": "container"}
MATCH_CUTOFF = 80
MIN_MARGIN = 10


@dataclass(frozen=True)
class Command:
    target: str
    destination: str
    transcript: str
    source: str


# A spoken multi-step task: red/blue balls into their holes, then the striped ball (decisions/multistep-step-state.md).
@dataclass(frozen=True)
class Plan:
    steps: tuple[tuple[str, int], ...]
    destination: str | None
    transcript: str
    source: str

    def message(self) -> dict:
        return {"plan": [{"target": target, "count": count} for target, count in self.steps],
                "destination": self.destination, "transcript": self.transcript, "source": self.source}


COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
# "a"/"an" count as one inside a plan, but never make a sentence a plan: "give me a red ball" stays a single command.
ARTICLES = {"a", "an"}
# A count binds to the first colour within this many words: "3 of the blue balls".
COUNT_REACH = 4


# No wake phrase: Enter already marks the words as meant for the robot, and "hey bot" was misheard too often
# when tried live. See decisions/voice-commands.md.
def parse(transcript: str) -> Command | None:
    words = normalise(transcript).split()
    if DANCE_WORDS & set(words) or is_plan(words):
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
    return bool(DANCE_WORDS & set(words)) and best_match(" ".join(words), DESTINATIONS) is None


def rejection_reason(transcript: str) -> str:
    words = normalise(transcript).split()
    if is_plan(words) and not DANCE_WORDS & set(words):
        return plan_steps(words)[1]
    if wants_dance(transcript):
        return "a dance: the dashboard starts it, not listen.py"
    if DANCE_WORDS & set(words):
        return "a dance and a ball command in one sentence"
    if best_match(" ".join(words), DESTINATIONS) is not None and named_target(words) is None:
        return "more than one ball colour named"
    return "no single command recognised"


def parse_plan(transcript: str) -> Plan | None:
    words = normalise(transcript).split()
    if DANCE_WORDS & set(words) or not is_plan(words):
        return None
    steps, _reason = plan_steps(words)
    if steps is None:
        return None
    return Plan(steps, best_match(" ".join(words), DESTINATIONS), transcript.strip(), "voice")


# A sentence is a plan once a number is spoken: any count except "one", which also means "the striped one", unless it
# is followed by a colour.
def is_plan(words: list[str]) -> bool:
    return any(count_of(word) and (word != "one" or colour_near(words, i)) for i, word in enumerate(words))


def count_of(word: str) -> int | None:
    if word.isdigit():
        return int(word) or None
    return COUNTS.get(word)


def colour_of(word: str) -> str | None:
    return next((target for target, colour_words in TARGETS.items() if word in colour_words), None)


def colour_near(words: list[str], index: int) -> bool:
    return any(colour_of(word) for word in words[index + 1: index + 1 + COUNT_REACH])


# Steps in spoken order, the striped ball moved last: it ends the round. Returns (steps, "") or (None, reason).
def plan_steps(words: list[str]) -> tuple[tuple[tuple[str, int], ...] | None, str]:
    steps: list[tuple[str, int]] = []
    pending: tuple[int, str, int] | None = None
    for i, word in enumerate(words):
        if pending and i - pending[2] > COUNT_REACH:
            if must_bind(pending[1]):
                return None, f"a count without a known ball colour ({pending[1]!r})"
            pending = None
        count = count_of(word) or (1 if word in ARTICLES else None)
        if count is not None:
            if pending and must_bind(pending[1]) and word not in ARTICLES:
                return None, f"a count without a known ball colour ({pending[1]!r})"
            if not (pending and word in ARTICLES):
                pending = (count, word, i)
            continue
        colour = colour_of(word)
        if colour is None:
            continue
        if pending is None and colour == "striped" and steps and steps[-1][0] == "striped":
            continue
        if pending is None and colour != "striped":
            return None, f"no count for the {colour} balls"
        steps.append((colour, pending[0] if pending else 1))
        pending = None
    if pending and must_bind(pending[1]):
        return None, f"a count without a known ball colour ({pending[1]!r})"
    if sum(count for colour, count in steps if colour == "striped") > 1:
        return None, "there is only one striped ball"
    striped = [step for step in steps if step[0] == "striped"]
    return tuple([step for step in steps if step[0] != "striped"] + striped), ""


def must_bind(word: str) -> bool:
    return word not in ARTICLES and word != "one"


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
