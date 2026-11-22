import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "voice"))

from commands import (  # noqa: E402
    DEFAULT_TARGET, from_key, parse, parse_plan, rejection_reason, scores, wants_bite, wants_dance,
)

ACCEPTED = [
    ("Give it to me.", "hand"),
    ("give it to me", "hand"),
    ("GIVE IT TO ME!", "hand"),
    ("Give it to me, please.", "hand"),
    ("Can you give it to me?", "hand"),
    ("Give it to", "hand"),
    ("Give me the ball.", "hand"),
    ("Give me the striped ball.", "hand"),
    ("Hand it to me.", "hand"),
    ("Hand it over.", "hand"),
    ("Pass it to me.", "hand"),
    ("Put it in my hand.", "hand"),
    ("Put it in hand.", "hand"),
    ("Put it in the hand.", "hand"),
    ("Put the ball in my hand.", "hand"),
    ("Place it in my hand.", "hand"),
    ("Okay, give it to me.", "hand"),
    ("Robot, give it to me.", "hand"),
    ("Hey bot, give it to me.", "hand"),
    ("Hey Bob, give it to me.", "hand"),
    ("Paybot, give it to me.", "hand"),
    ("People, give it to me.", "hand"),
    ("Put it in the container.", "container"),
    ("put it in the container", "container"),
    ("Drop it in the container.", "container"),
    ("Put the ball in the container.", "container"),
    ("Put the striped ball in the container.", "container"),
    ("Container, please.", "container"),
    ("Container.", "container"),
    ("Into the container.", "container"),
    ("Put it in the containers.", "container"),
    ("Put it in the contain her.", "container"),
    ("Hey bot, put it in the container.", "container"),
    ("Paper, put it in the container.", "container"),
]

COLOURED = [
    ("Hand me the striped ball.", "hand", "striped"),
    ("Hand me the yellow and green ball.", "hand", "striped"),
    ("Hand me the red ball.", "hand", "red"),
    ("Give me the blue ball.", "hand", "blue"),
    ("Put the red ball in the container.", "container", "red"),
    ("I'm tired, give it to me.", "hand", "striped"),
    ("Give me a red ball.", "hand", "red"),
    ("Give me the striped one.", "hand", "striped"),
]

B3_R2_S = (("blue", 3), ("red", 2), ("striped", 1))
PLANS = [
    ("Pick up 3 blue balls and 2 red balls, then the target ball.", B3_R2_S, None),
    ("Take out three blue balls and two red balls, then give me the striped ball.", B3_R2_S, "hand"),
    ("Take out 3 blue and 2 red, then put the striped ball in the container.", B3_R2_S, "container"),
    ("Hey bot, take out three of the blue balls and two red ones, then hand me the yellow and green ball.", B3_R2_S, "hand"),
    ("Take out a red ball and two blue balls, then give me the striped one.",
     (("red", 1), ("blue", 2), ("striped", 1)), "hand"),
    ("Give me the striped ball after you take out one red ball.", (("red", 1), ("striped", 1)), "hand"),
    ("Take out four red balls.", (("red", 4),), None),
    ("Take out 1 blue, 1 red and 1 blue ball.", (("blue", 1), ("red", 1), ("blue", 1)), None),
]

PLAN_REJECTED = [
    ("Take out 3 purple balls, then give me the striped ball.", "a count without a known ball colour ('3')"),
    ("Take out blue balls and 2 red balls.", "no count for the blue balls"),
    ("Take out 2 red balls and the blue ones.", "no count for the blue balls"),
    ("Give me two striped balls after 1 red ball.", "there is only one striped ball"),
    ("Give it to me in 2 seconds.", "a count without a known ball colour ('2')"),
]

PLAN_MESSAGE = {"plan": [{"target": "blue", "count": 3}, {"target": "red", "count": 2}, {"target": "striped", "count": 1}],
                "destination": "hand", "transcript": PLANS[1][0], "source": "voice"}

REJECTED = [
    "",
    "Hey bot.",
    "Hello.",
    "I think the robot is great.",
    "What time is it?",
    "Open the window.",
    "Thank you.",
    "Stop.",
    "Give.",
    "Me.",
    "The ball.",
    "Hey bot, give.",
    "Hey bot, me.",
    "Hey bot, the ball.",
    "Give it to me or put it in the container.",
    "Put it in the container, no, give it to me.",
    "Hand me the red or the blue ball.",
    "Give me the red and the striped ball.",
]

REASONS = [
    ("What time is it?", "no single command recognised"),
    ("Give it to me or put it in the container.", "no single command recognised"),
    ("Hand me the red or the blue ball.", "more than one ball colour named"),
    ("Dance and give me the ball.", "a dance and a ball command in one sentence"),
    ("Let's dance.", "a dance: the dashboard starts it, not listen.py"),
    ("Bite and give it to me.", "a bite and another command in one sentence"),
    ("Dance and bite.", "a bite and another command in one sentence"),
]

DANCE = ["Dance.", "dance", "Let's dance!", "Show me a dance.", "Can you dance?", "Dance for us, robot.",
         "Do a little dance.", "Let's go dancing.", "Boogie time."]
NOT_DANCE = ["Give it to me.", "Put it in the container.", "Distance.", "Dance and give me the ball.",
             "Put it in the container and dance."]

BITE = ["Bite the finger.", "Bite my finger!", "bite", "byte the finger", "biting time"]
NOT_BITE = ["Give it to me.", "Put it in the container.", "Let's dance."]
MIXED_WITH_BITE = ["Bite and give it to me.", "Dance and bite."]

KEYS = [("1", "hand"), ("2", "container"), (" 1 ", "hand"), ("3", None), ("", None)]

failures = []
for transcript, expected in ACCEPTED:
    command = parse(transcript)
    if command is None or (command.destination, command.target, command.source) != (expected, DEFAULT_TARGET, "voice"):
        failures.append(f"expected {expected!r:12} got {command}  <- {transcript!r}")
for transcript, destination, target in COLOURED:
    command = parse(transcript)
    if command is None or (command.destination, command.target) != (destination, target):
        failures.append(f"expected {destination!r} {target!r} got {command}  <- {transcript!r}")
for transcript, steps, destination in PLANS:
    plan = parse_plan(transcript)
    if plan is None or (plan.steps, plan.destination) != (steps, destination) or parse(transcript) is not None:
        failures.append(f"plan: expected {steps} {destination!r}, got {plan}, parse {parse(transcript)}  <- {transcript!r}")
for transcript, reason in PLAN_REJECTED:
    if parse_plan(transcript) is not None or parse(transcript) is not None or rejection_reason(transcript) != reason:
        failures.append(f"plan: expected rejection {reason!r}, got {parse_plan(transcript)} / {rejection_reason(transcript)!r}"
                        f"  <- {transcript!r}")
for transcript in [t for t, _ in ACCEPTED] + [t for t, _, _ in COLOURED] + REJECTED + DANCE:
    if parse_plan(transcript) is not None:
        failures.append(f"plan from an old command  <- {transcript!r}")
if (message := parse_plan(PLANS[1][0]).message()) != PLAN_MESSAGE:
    failures.append(f"plan message: {message}")
for transcript in REJECTED:
    if (command := parse(transcript)) is not None:
        failures.append(f"expected rejection, got {command.destination!r}  <- {transcript!r}")
for transcript in DANCE:
    if not wants_dance(transcript) or parse(transcript) is not None:
        failures.append(f"expected a dance, got wants_dance={wants_dance(transcript)} parse={parse(transcript)}  <- {transcript!r}")
for transcript in NOT_DANCE:
    if wants_dance(transcript):
        failures.append(f"expected no dance  <- {transcript!r}")
for transcript in BITE:
    if not wants_bite(transcript) or parse(transcript) is not None or wants_dance(transcript):
        failures.append(f"expected a bite, got wants_bite={wants_bite(transcript)} parse={parse(transcript)}  <- {transcript!r}")
for transcript in NOT_BITE:
    if wants_bite(transcript):
        failures.append(f"expected no bite  <- {transcript!r}")
for transcript in MIXED_WITH_BITE:
    if wants_bite(transcript) or wants_dance(transcript) or parse(transcript) is not None:
        failures.append(f"expected a refusal, nothing guessed  <- {transcript!r}")
for key, expected in KEYS:
    command = from_key(key)
    if (command and command.destination) != expected or (command and command.source != "keyboard"):
        failures.append(f"key {key!r}: expected {expected!r}, got {command}")

for transcript, reason in REASONS:
    if (given := rejection_reason(transcript)) != reason:
        failures.append(f"reason: expected {reason!r}, got {given!r}  <- {transcript!r}")

shown = scores("Give me the red ball.")
if not (shown.destinations["hand"] == 1.0 > shown.destinations["container"] and shown.colours == ("red",)):
    failures.append(f"scores for 'Give me the red ball.': {shown}")

total = (len(ACCEPTED) + len(COLOURED) + len(REJECTED) + len(REASONS) + len(KEYS) + len(DANCE) + len(NOT_DANCE)
         + len(BITE) + len(NOT_BITE) + len(MIXED_WITH_BITE) + len(PLANS) + len(PLAN_REJECTED))
print(
    f"{total} cases: {len(ACCEPTED)} accepted, {len(COLOURED)} with a colour, {len(REJECTED)} rejected, "
    f"{len(REASONS)} reasons, {len(KEYS)} keys, {len(DANCE)} dances, {len(NOT_DANCE)} not dances, {len(BITE)} bites, "
    f"{len(NOT_BITE) + len(MIXED_WITH_BITE)} not bites, {len(PLANS)} plans, {len(PLAN_REJECTED)} plans rejected"
)
for failure in failures:
    print("FAIL", failure)
if failures:
    sys.exit(1)
print("PASS")
