# Where does the dance music play, and which music?

**Status:** DECIDED (2026-11-22, Tomi asked for it)

## Decision

When a round enters DANCE, the dashboard page plays a 10 s clip through the speakers of the computer that shows it (`scripts/dashboard/static/music.js`). It starts on the live `transition` event into DANCE, so the tile, voice, chat and `d` + Enter all trigger it, and it stops when the round leaves DANCE (ABORT included) or ends. The header has a **🔊 Dance music** toggle. By default it plays a chiptune generated in the browser (Web Audio, 144 bpm, no audio file). If `scripts/dashboard/static/music/dance.mp3` exists, that file plays instead; the folder is git-ignored.

## Why

- **In the browser, not in the robot program**: the robot program runs in WSL, which has no simple route to the Windows speakers, and audio playback there would sit next to the 30 Hz control loop. The dashboard already runs in a Windows browser on the robot laptop and sees every transition.
- **Only transitions less than 3 s old start it**: the first message after a page load replays the whole log, and old dances must stay silent.
- **Unlocked on the first click or key press on the page**: browsers mute audio until the page has had one, and a dance started by voice or from the terminal brings none. Tested in Chrome with the fake robot program (agent of Space0code, 2026-11-22): the tile starts the clip, old dances stay silent after a reload, the chiptune renders 10 s of sound with a peak of 0.29 at volume 0.25 (now 0.5), and a dropped-in file plays.

## Alternatives

- **A 10 s clip of Taylor Swift's "Shake It Off"** (what Tomi asked for first): not shipped. It is copyrighted, so the agent did not download it or put it in the repo. Anyone at the table can drop their own copy in as `static/music/dance.mp3`, which stays on their laptop.
- **Play the sound from `demo_chain.sh` or the supervisor**: ruled out, see Why.

## Revisit if

- Nobody at the table clicks the page before a voice-started dance, and the music stays silent: then add a "click to enable sound" banner.

Written by Claude
