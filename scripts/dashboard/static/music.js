// Written by Claude
"use strict";

// Plays in the dashboard's browser, not in the robot program: see decisions/dance-music.md.
const danceMusic = (() => {
  const CLIP_S = 10;
  const OWN_CLIP = "/static/music/dance.mp3";
  const BEAT_S = 60 / 144;
  const MELODY = [
    [76, 79, 84, 79, 76, 79, 81, 79],
    [76, 72, 69, 72, 76, 79, 76, 72],
    [77, 81, 84, 81, 77, 81, 79, 77],
    [79, 83, 86, 83, 79, 77, 76, 74],
    [76, 79, 84, 79, 86, 84, 79, 76],
    [74, 77, 79, 83, 84, null, 84, null],
  ];
  const BASS_ROOTS = [48, 45, 41, 43, 48, 43];

  let context = null;
  let master = null;
  let ownClip = null;
  let enabled = true;
  let stopTimer = null;

  const hz = (midi) => 440 * 2 ** ((midi - 69) / 12);

  function unlock() {
    if (!context) {
      context = new AudioContext();
      master = context.createGain();
      master.connect(context.destination);
    }
    if (context.state === "suspended") context.resume();
  }

  function tone(type, midi, start, length, volume) {
    const osc = context.createOscillator();
    const gain = context.createGain();
    osc.type = type;
    osc.frequency.value = hz(midi);
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(volume, start + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, start + length);
    osc.connect(gain).connect(master);
    osc.start(start);
    osc.stop(start + length + 0.02);
  }

  function noise(start, length, volume, highpass) {
    const buffer = context.createBuffer(1, Math.ceil(context.sampleRate * length), context.sampleRate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = Math.random() * 2 - 1;
    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const gain = context.createGain();
    source.buffer = buffer;
    filter.type = "highpass";
    filter.frequency.value = highpass;
    gain.gain.setValueAtTime(volume, start);
    gain.gain.exponentialRampToValueAtTime(0.001, start + length);
    source.connect(filter).connect(gain).connect(master);
    source.start(start);
  }

  function kick(start) {
    const osc = context.createOscillator();
    const gain = context.createGain();
    osc.frequency.setValueAtTime(150, start);
    osc.frequency.exponentialRampToValueAtTime(40, start + 0.15);
    gain.gain.setValueAtTime(0.8, start);
    gain.gain.exponentialRampToValueAtTime(0.001, start + 0.2);
    osc.connect(gain).connect(master);
    osc.start(start);
    osc.stop(start + 0.25);
  }

  function playChiptune() {
    const t0 = context.currentTime + 0.05;
    master.gain.cancelScheduledValues(t0);
    master.gain.setValueAtTime(0.5, t0);
    MELODY.forEach((bar, b) => {
      const barStart = t0 + b * 4 * BEAT_S;
      bar.forEach((note, i) => {
        if (note !== null) tone("square", note, barStart + i * BEAT_S / 2, BEAT_S * 0.45, 0.18);
      });
      for (let beat = 0; beat < 4; beat++) {
        const at = barStart + beat * BEAT_S;
        tone("triangle", BASS_ROOTS[b] + (beat % 2 ? 12 : 0), at, BEAT_S * 0.9, 0.5);
        if (beat % 2 === 0) kick(at); else noise(at, 0.12, 0.35, 1500);
        noise(at + BEAT_S / 2, 0.04, 0.12, 7000);
      }
    });
    const end = t0 + MELODY.length * 4 * BEAT_S - 2 * BEAT_S;
    for (const note of [60, 64, 67, 72]) tone("square", note, end, 2 * BEAT_S, 0.1);
  }

  async function findOwnClip() {
    try {
      if ((await fetch(OWN_CLIP, { method: "HEAD" })).ok) ownClip = new Audio(OWN_CLIP);
    } catch (error) { /* no own clip: the chiptune plays */ }
  }

  function play() {
    if (!enabled) return;
    stop();
    if (ownClip) {
      ownClip.currentTime = 0;
      ownClip.play().catch(() => {});
    } else {
      unlock();
      playChiptune();
    }
    stopTimer = setTimeout(stop, CLIP_S * 1000);
  }

  function stop() {
    clearTimeout(stopTimer);
    if (ownClip) ownClip.pause();
    if (context && master) {
      master.gain.cancelScheduledValues(context.currentTime);
      master.gain.setValueAtTime(0, context.currentTime);
    }
  }

  // Browsers keep audio muted until the page has had a click or a key press; a dance started by voice or the terminal has neither.
  for (const type of ["pointerdown", "keydown"]) document.addEventListener(type, unlock, { once: true, capture: true });
  findOwnClip();

  return {
    play, stop,
    setEnabled(on) { enabled = on; if (!on) stop(); },
    hasOwnClip: () => Boolean(ownClip),
  };
})();
