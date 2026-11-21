# decisions/ - why things are the way they are

One file per question the team had to answer (which policy runs a skill, how skills chain, how the supervisor finds the ball). Each file says what was chosen, why, and every alternative that was considered or tried, with its result. The point is that nobody, person or agent, searches the same space twice.

The code says *what* it does. This folder says *why*, and what did not work.

- **Name the file after the question, not the answer or the date**: `skill-policies.md`, not `act.md` or `2026-11-21.md`. Then everything tried for one question stays in one file, and the file names work as the index.
- **Status** is one of `OPEN` (still exploring, nothing chosen yet), `DECIDED` or `SUPERSEDED by <file>`. Write a record while the question is still `OPEN`, as soon as something has been tried.
- **Keep the history.** When a decision changes, move the old choice into Alternatives with the reason it was dropped, then rewrite Decision.
- **Say how each claim is known**: tested on our hardware (who, when, run or dataset name), taken from a `research/` file, or assumed. Link to `kb/` and `research/` rather than copying from them.
- **Keep it short.** A record is a few bullets, not an essay.
- Which project idea the team builds is decided in `ideas/`, not here.

Template:

```markdown
# <the question, e.g. Which policies run the skills?>

- Status: DECIDED (2026-11-20) | OPEN | SUPERSEDED by <file>

## Decision

<one or two sentences>

## Why

- <reason, and how we know it>

## Alternatives

- **<alternative>** - tried <date, who, run or dataset>: <result>.
- **<alternative>** - considered, not tried: <why not>.

## Revisit if

- <what would reopen the question>
```
