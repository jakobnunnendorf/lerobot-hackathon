import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "voice"))

from commands import DEFAULT_TARGET, from_key, parse, scores  # noqa: E402

ACCEPTED = [
    ("Hey bot, give it to me.", "hand"),
    ("hey bot give it to me", "hand"),
    ("HEY BOT, GIVE IT TO ME!", "hand"),
    ("Hey bot! Give it to me, please.", "hand"),
    ("Hey bot, can you give it to me?", "hand"),
    ("Hey bot, give it to", "hand"),
    ("Hey bot, give me the ball.", "hand"),
    ("Hey bot, give me the striped ball.", "hand"),
    ("Hey bot, hand it to me.", "hand"),
    ("Hey bot, hand it over.", "hand"),
    ("Hey bot, pass it to me.", "hand"),
    ("Hey bot, put it in my hand.", "hand"),
    ("Hey-bot, give it to me.", "hand"),
    ("Heybot, give it to me.", "hand"),
    ("Hey robot, give it to me.", "hand"),
    ("Hey Bob, give it to me.", "hand"),
    ("Paybot, give it to me.", "hand"),
    ("A bot, put it in the container.", "container"),
    ("Hey bot, put it in the container.", "container"),
    ("hey bot put it in the container", "container"),
    ("Hey bot, drop it in the container.", "container"),
    ("Hey bot, put the ball in the container.", "container"),
    ("Hey bot, put the striped ball in the container.", "container"),
    ("Hey bot, container, please.", "container"),
    ("Hey bot, into the container.", "container"),
    ("Hey bot, put it in the containers.", "container"),
    ("Hey bot, put it in the contain her.", "container"),
]

COLOURED = [
    ("Hey bot, hand me the striped ball.", "hand", "striped"),
    ("Hey bot, hand me the yellow and green ball.", "hand", "striped"),
    ("Hey bot, hand me the red ball.", "hand", "red"),
    ("Hey bot, give me the blue ball.", "hand", "blue"),
    ("Hey bot, put the red ball in the container.", "container", "red"),
    ("Hey bot, I'm tired, give it to me.", "hand", "striped"),
]

REJECTED = [
    "",
    "Hey bot.",
    "Give it to me.",
    "Put it in the container.",
    "Hey, give it to me.",
    "Hey there, give it to me.",
    "Hey Siri, give it to me.",
    "Hey Google, put it in the container.",
    "Robot, give it to me.",
    "Okay, put it in the container.",
    "People, give it to me.",
    "Paper, put it in the container.",
    "I think the robot is great.",
    "Hey bot, what time is it?",
    "Hey bot, open the window.",
    "Hey bot, thank you.",
    "Hey bot, stop.",
    "Hey bot, give.",
    "Hey bot, me.",
    "Hey bot, the ball.",
    "Hey bot, give it to me or put it in the container.",
    "Hey bot, put it in the container, no, give it to me.",
    "Hey bot, hand me the red or the blue ball.",
    "Hey bot, give me the red and the striped ball.",
]

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
for transcript in REJECTED:
    if (command := parse(transcript)) is not None:
        failures.append(f"expected rejection, got {command.destination!r}  <- {transcript!r}")
for key, expected in KEYS:
    command = from_key(key)
    if (command and command.destination) != expected or (command and command.source != "keyboard"):
        failures.append(f"key {key!r}: expected {expected!r}, got {command}")

shown = scores("Hey bot, give me the red ball.")
if not (shown.wake == 1.0 and shown.destinations["hand"] == 1.0 > shown.destinations["container"] and shown.colours == ("red",)):
    failures.append(f"scores for 'Hey bot, give me the red ball.': {shown}")

total = len(ACCEPTED) + len(COLOURED) + len(REJECTED) + len(KEYS)
print(f"{total} cases: {len(ACCEPTED)} accepted, {len(COLOURED)} with a colour, {len(REJECTED)} rejected, {len(KEYS)} keys")
for failure in failures:
    print("FAIL", failure)
if failures:
    sys.exit(1)
print("PASS")
