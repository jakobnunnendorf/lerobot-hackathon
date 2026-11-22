"use strict";

const TARGETS = [
  { id: "striped", label: "Striped", hint: "yellow and green" },
  { id: "red", label: "Red" },
  { id: "blue", label: "Blue" },
];
const DESTINATIONS = [
  { id: "hand", label: "Give to hand", icon: "✋" },
  { id: "container", label: "Put in container", icon: "📦" },
];
const FLOW = ["REST", "PICK", "VERIFY_HOLD", "GIVE", "PUT_DOWN", "DONE"];
const DANCE_FLOW = ["REST", "DANCE", "DONE"];
const BITE_FLOW = ["REST", "BITE", "DONE"];
const STATE_NAMES = { REST: "Rest", PICK: "Pick", VERIFY_HOLD: "Check ball", GIVE: "Hand over", PUT_DOWN: "Put down", DANCE: "Dance", BITE: "Bite", DONE: "Done", ABORT: "Aborted" };
const RESULT_BUTTONS = [
  ["success", "✓ success"], ["wrong_ball", "wrong ball"], ["empty", "empty"], ["dropped", "dropped"],
  ["still_digging", "digging"], ["other", "other"],
];
const HAND_LOST_S = 5;

const state = {
  status: null, statusAge: null, events: [], results: {}, log: null, speech: "loading",
  lastObserves: null, loopHz: null, holdSince: null, camerasLive: {}, mode: "chain", collect: null, datasets: null,
};

const $ = (id) => document.getElementById(id);

function eventTime(event) { return Date.parse(event.time); }

function rounds() {
  const byNumber = new Map();
  for (const e of state.events) {
    if (e.event === "round_start") byNumber.set(e.round, { number: e.round, start: e, transitions: [], checks: [], end: null });
    const r = byNumber.get(e.round);
    if (!r) continue;
    if (e.event === "transition") r.transitions.push(e);
    if (e.event === "grasp_check") r.checks.push(e);
    if (e.event === "round_end") r.end = e;
  }
  return [...byNumber.values()];
}

function running() {
  const all = rounds();
  const last = all[all.length - 1];
  return last && !last.end ? last : null;
}

function available() {
  const fresh = state.statusAge !== null && state.statusAge < 2;
  return fresh && state.status.available ? state.status.available : null;
}

function reasonFor(target, destination) {
  const allowed = available();
  if (!allowed) return "robot program not running";
  if (!allowed.targets.includes(target)) return `no ${target} pick skill yet (training data has only the striped ball)`;
  if (!allowed.destinations.includes(destination)) return "not set up in this robot program";
  if (running()) return "a round is running";
  return null;
}

function chainOf(start) {
  return start.dance ? `💃 ${start.dance}` : start.bite ? "🦷 bite" : `${start.target || "striped"} → ${start.destination || "-"}`;
}

// The robot program refuses a dance with a ball in the gripper as well; greyed out here so nobody has to try.
function danceReason() {
  const allowed = available();
  if (!allowed) return "robot program not running";
  if (!allowed.dance) return "no dance: start the robot program with --dance-episode";
  if (running()) return "a round is running";
  const band = state.status.hold_range || [6, 20];
  const gripper = state.status.gripper;
  if (typeof gripper === "number" && gripper >= band[0] && gripper <= band[1]) return "the gripper holds a ball";
  return null;
}

// Shown as planned, not hidden: the bite does not work on the arm yet (Tomi, 2026-11-22). Delete this line once it does.
const BITE_NOT_READY = "coming soon: not working on the arm yet";

function biteReason() {
  if (BITE_NOT_READY) return BITE_NOT_READY;
  const allowed = available();
  if (!allowed) return "robot program not running";
  if (!allowed.bite) return "no bite: start the robot program with --bite";
  if (running()) return "a round is running";
  const band = state.status.hold_range || [6, 20];
  const gripper = state.status.gripper;
  if (typeof gripper === "number" && gripper >= band[0] && gripper <= band[1]) return "the gripper holds a ball";
  return null;
}

function renderTiles() {
  const tiles = $("tiles");
  tiles.innerHTML = "";
  for (const t of TARGETS) {
    for (const d of DESTINATIONS) {
      const reason = reasonFor(t.id, d.id);
      const button = document.createElement("button");
      button.className = "tile" + (reason ? "" : " ready");
      button.disabled = Boolean(reason);
      button.title = reason || `Pick the ${t.label.toLowerCase()} ball, then ${d.label.toLowerCase()}`;
      button.innerHTML = `<span class="what"><span class="swatch ${t.id}"></span>${t.label} → ${d.icon}</span>` +
        `<span class="why">${reason || d.label}</span>`;
      button.onclick = () => startChain(t.id, d.id);
      tiles.appendChild(button);
    }
  }
  const reason = danceReason();
  const dance = document.createElement("button");
  dance.className = "tile" + (reason ? "" : " ready");
  dance.disabled = Boolean(reason);
  dance.title = reason || "Replay a recorded dance from rest, then back to rest";
  dance.innerHTML = `<span class="what">💃 Dance</span><span class="why">${reason || "from rest, gripper empty"}</span>`;
  dance.onclick = startDance;
  tiles.appendChild(dance);
  const biteBlocked = biteReason();
  const bite = document.createElement("button");
  bite.className = "tile" + (biteBlocked ? "" : " ready");
  bite.disabled = Boolean(biteBlocked);
  bite.title = biteBlocked || "Point at the gripper and it bites, from rest, gripper empty";
  bite.innerHTML = `<span class="what">🦷 Bite</span><span class="why">${biteBlocked || "point at the gripper"}</span>`;
  bite.onclick = startBite;
  tiles.appendChild(bite);
}

async function post(url, body, raw = false) {
  const response = await fetch(url, {
    method: "POST",
    headers: raw ? { "Content-Type": "application/octet-stream" } : { "Content-Type": "application/json" },
    body: raw ? body : JSON.stringify(body),
  });
  return response.json();
}

function say(text, kind) {
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.textContent = text;
  $("chat").appendChild(div);
  $("chat").scrollTop = $("chat").scrollHeight;
}

function describe(reply) {
  if (reply.ok) {
    const c = reply.command;
    if (c.dance) { say("▶ Starting: dance", "ok"); return; }
    if (c.bite) { say("▶ Starting: bite", "ok"); return; }
    say(`▶ Starting: ${c.target} ball → ${c.destination === "hand" ? "your hand" : "the container"}`, "ok");
    return;
  }
  let text = `✗ ${reply.reason}`;
  if (reply.scores) {
    const parts = Object.entries(reply.scores.destinations).map(([k, v]) => `${k} ${v.toFixed(2)}`);
    text += ` (${parts.join(", ")}; needs 0.80)`;
  }
  say(text, "bad");
}

async function startChain(target, destination) {
  say(`${target} → ${destination}`, "me");
  describe(await post("/api/command", { target, destination }));
}

async function startDance() {
  say("dance", "me");
  describe(await post("/api/dance", {}));
}

async function startBite() {
  say("bite", "me");
  describe(await post("/api/bite", {}));
}

$("chat-form").onsubmit = async (event) => {
  event.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text) return;
  $("chat-input").value = "";
  say(text, "me");
  describe(await post("/api/text", { text }));
};

$("abort").onclick = async () => {
  const reply = await post("/api/abort", {});
  say(reply.ok ? "■ ABORT sent: the arm stops and holds. Power switch is the real stop." : "■ ABORT not delivered: robot program not listening. Use the power switch.", "bad");
};

// Counts from the value just asked for, so a quick double click before the status catches up still adds two.
async function changeMaxPicks(change) {
  const now = state.picksAsked ?? (state.status || {}).max_pick_attempts;
  if (typeof now !== "number") return say("Max picks: the robot program is not running, or too old to change it.", "bad");
  const reply = await post("/api/max_picks", { value: now + change });
  if (reply.ok) state.picksAsked = reply.value;
  else say(`Max picks not changed: ${reply.reason}`, "bad");
}
$("picks-down").onclick = () => changeMaxPicks(-1);
$("picks-up").onclick = () => changeMaxPicks(1);

const recorder = { context: null, stream: null, chunks: [], timer: null, started: 0 };

async function startRecording() {
  if (recorder.context) return;
  try {
    recorder.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } });
  } catch (error) {
    say(`🎤 microphone blocked: ${error.message}`, "bad");
    return;
  }
  recorder.context = new AudioContext({ sampleRate: 16000 });
  const source = recorder.context.createMediaStreamSource(recorder.stream);
  const processor = recorder.context.createScriptProcessor(4096, 1, 1);
  recorder.chunks = [];
  processor.onaudioprocess = (e) => recorder.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(processor);
  processor.connect(recorder.context.destination);
  recorder.started = performance.now();
  $("mic").classList.add("recording");
  $("mic-state").textContent = "Listening… click 🎤 again (or release Space) to send. Stops by itself after 8 s.";
  recorder.timer = setTimeout(stopRecording, 8000);
}

async function stopRecording() {
  if (!recorder.context) return;
  clearTimeout(recorder.timer);
  recorder.stream.getTracks().forEach((track) => track.stop());
  await recorder.context.close();
  recorder.context = null;
  $("mic").classList.remove("recording");
  const length = recorder.chunks.reduce((n, c) => n + c.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of recorder.chunks) { audio.set(chunk, offset); offset += chunk.length; }
  $("mic-state").textContent = "Recognising…";
  const reply = await post("/api/voice", audio.buffer, true);
  $("mic-state").textContent = "Click 🎤 or hold Space to talk.";
  if (reply.text) say(`🎤 “${reply.text}”${reply.seconds ? ` (${reply.seconds} s)` : ""}`, "me");
  describe(reply);
}

$("mic").onclick = () => (recorder.context ? stopRecording() : startRecording());
const typing = () => ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
const COLLECT_KEYS = { Space: "next", ArrowLeft: "discard", Escape: "stop" };
document.addEventListener("keydown", (e) => {
  if (typing() || e.repeat) return;
  if (state.mode === "collect" && COLLECT_KEYS[e.code]) { e.preventDefault(); collectControl(COLLECT_KEYS[e.code]); return; }
  if (state.mode === "chain" && e.code === "Space") { e.preventDefault(); startRecording(); }
});
document.addEventListener("keyup", (e) => {
  if (state.mode === "chain" && e.code === "Space" && !typing()) { e.preventDefault(); stopRecording(); }
});

function setFeeds(on) {
  for (const name of ["top", "wrist"]) {
    const feed = $(`feed-${name}`);
    const img = feed.querySelector("img");
    if (on && !feed.classList.contains("on")) { img.src = `/stream/${name}?t=${Date.now()}`; feed.classList.add("on"); }
    if (!on && feed.classList.contains("on")) { img.removeAttribute("src"); feed.classList.remove("on"); }
  }
}

function renderRound() {
  const current = running();
  const all = rounds();
  const shown = current || all[all.length - 1];
  if (state.mode === "collect") return;
  setFeeds(Boolean(current) || $("toggle-cameras").checked);
  if (!shown) {
    $("round-title").textContent = "Waiting for a command";
    $("round-chain").textContent = "";
    renderStates(null, []);
    setBudget(0, "");
    return;
  }
  const chain = shown.start.dance || shown.start.bite ? chainOf(shown.start) : `${shown.start.target || "striped"} ball → ${shown.start.destination || "no destination"}`;
  const last = shown.transitions[shown.transitions.length - 1];
  const now = current ? (last ? last.target : "REST") : (shown.end.outcome === "aborted" ? "ABORT" : "DONE");
  const visited = ["REST", ...shown.transitions.map((t) => t.target)];
  $("round-chain").textContent = chain;
  if (current) {
    $("round-title").textContent = `Round ${shown.number}: ${STATE_NAMES[now] || now}`;
    const since = last ? eventTime(last) : eventTime(shown.start);
    const elapsed = (Date.now() - since) / 1000;
    const budget = state.status && state.status.budgets_s ? state.status.budgets_s[now] : null;
    setBudget(budget ? Math.min(elapsed / budget, 1) : 0,
      budget ? `${elapsed.toFixed(0)} s of ${budget.toFixed(0)} s` : `${elapsed.toFixed(0)} s`);
  } else {
    $("round-title").textContent = `Round ${shown.number}: ${shown.end.outcome.replace("_", " ")} in ${shown.end.duration_s} s`;
    setBudget(0, "");
  }
  renderStates(now, visited, shown.start.dance ? DANCE_FLOW : shown.start.bite ? BITE_FLOW : null);
}

function renderStates(now, visited, fixedFlow = null) {
  const div = $("states");
  div.innerHTML = "";
  const flow = fixedFlow ? fixedFlow
    : visited.includes("PUT_DOWN") || !visited.includes("GIVE") ? FLOW : FLOW.filter((s) => s !== "PUT_DOWN");
  for (const s of now === "ABORT" ? [...flow, "ABORT"] : flow) {
    const chip = document.createElement("div");
    chip.className = "state" + (s === now ? (s === "ABORT" ? " abort" : " now") : visited.includes(s) ? " done" : "");
    chip.textContent = STATE_NAMES[s];
    div.appendChild(chip);
  }
}

function setBudget(fraction, text) {
  $("budget-bar").style.width = `${(fraction * 100).toFixed(1)}%`;
  $("budget-bar").style.background = fraction > 0.85 ? "var(--bad)" : fraction > 0.6 ? "var(--warn)" : "var(--accent)";
  $("budget-text").textContent = text;
}

function renderLive() {
  const s = state.status || {};
  const fresh = state.statusAge !== null && state.statusAge < 2;
  const badge = $("badge-robot");
  badge.textContent = fresh ? (s.fake ? "FAKE robot program" : "Robot program: running") : "Robot program: not running";
  badge.className = `badge ${fresh ? (s.fake ? "fake" : "on") : "off"}`;
  $("badge-speech").textContent = `Speech: ${state.speech}`;
  $("badge-speech").className = `badge debug ${state.speech === "ready" ? "on" : state.speech.startsWith("failed") ? "off" : ""}`;
  $("badge-log").textContent = `Log: ${state.log || "none"}`;
  $("badge-rec").textContent = state.recording ? `● REC ${state.recording}` : "Video: between rounds";
  $("badge-rec").className = `badge debug ${state.recording ? "rec" : ""}`;

  const band = s.hold_range || [6, 20];
  $("hold-band").style.left = `${band[0]}%`;
  $("hold-band").style.width = `${band[1] - band[0]}%`;
  if (typeof s.gripper === "number") {
    $("gripper-needle").style.left = `calc(${Math.min(Math.max(s.gripper, 0), 100)}% - 2px)`;
    const holding = s.gripper >= band[0] && s.gripper <= band[1];
    $("gripper-value").textContent = `${s.gripper.toFixed(1)} ${holding ? "(holding band)" : ""}`;
  }
  const mode = s.give_mode;
  if (mode === "hold") state.holdSince = state.holdSince || Date.now();
  else state.holdSince = null;
  const chip = $("give-mode");
  chip.className = `chip ${mode || ""}`;
  chip.textContent = !mode ? "-" : mode === "hold"
    ? `hand lost: home in ${Math.max(HAND_LOST_S - (Date.now() - state.holdSince) / 1000, 0).toFixed(1)} s`
    : { follow: "following the hand", home: "back at present pose, waiting", drop: "dropping", done: "celebrating" }[mode] || mode;
  $("palms").textContent = s.palms === undefined ? "-" : `${s.palms}${s.hands && s.hands.length ? ` (x ${s.hands[0][0].toFixed(2)}, y ${s.hands[0][1].toFixed(2)} m)` : ""}`;
  $("tip").textContent = s.tip ? `${(s.tip[2] * 100).toFixed(0)} cm high` : "-";
  $("loop-hz").textContent = state.loopHz ? `${state.loopHz.toFixed(0)} Hz ${state.loopHz < 25 ? "(slow!)" : ""}` : "-";
  $("policy").textContent = s.pick_policy ? s.pick_policy.split("/").slice(-4).join("/") : "-";
  if (s.max_pick_attempts === state.picksAsked) state.picksAsked = null;
  $("max-picks").textContent = state.picksAsked ?? s.max_pick_attempts ?? "-";

  const all = rounds();
  const checks = all.flatMap((r) => r.checks);
  const lastCheck = checks[checks.length - 1];
  $("grasp").textContent = lastCheck ? `${lastCheck.verdict} (reading ${lastCheck.reading ?? "-"})` : "-";
  const holds = state.events.filter((e) => e.event === "transition" && e.held_areas_px);
  const lastHold = holds[holds.length - 1];
  $("held").textContent = lastHold
    ? `${lastHold.held_check}: ${Object.entries(lastHold.held_areas_px).map(([k, v]) => `${k} ${v}`).join(", ")} px (need ${lastHold.held_threshold_px ?? s.min_held_target_area_px ?? "-"})`
    : "-";
}

function renderHistory() {
  const body = $("rounds");
  body.innerHTML = "";
  const finished = rounds().filter((r) => r.end).reverse();
  for (const r of finished) {
    const tr = document.createElement("tr");
    const marked = state.results[String(r.number)] || {};
    tr.innerHTML = `<td>${r.number}</td><td>${chainOf(r.start)}</td>` +
      `<td class="outcome-${r.end.outcome}">${r.end.outcome}</td><td>${r.end.duration_s}</td>` +
      `<td>${r.end.pick_attempts ?? "-"}</td><td>${r.end.wrong_balls ?? "-"}</td><td></td><td></td>`;
    const buttons = document.createElement("div");
    buttons.className = "result-buttons";
    const note = document.createElement("input");
    note.value = marked.note || "";
    note.placeholder = "what happened?";
    for (const [id, label] of RESULT_BUTTONS) {
      const b = document.createElement("button");
      b.textContent = label;
      if (marked.result === id) b.classList.add("picked");
      b.onclick = async () => {
        await post("/api/result", { round: r.number, result: id, note: note.value });
        state.results[String(r.number)] = { result: id, note: note.value };
        renderHistory();
      };
      buttons.appendChild(b);
    }
    tr.children[6].appendChild(buttons);
    tr.children[7].appendChild(note);
    body.appendChild(tr);
  }
  const marked = Object.values(state.results);
  const success = marked.filter((m) => m.result === "success").length;
  const outcomes = finished.reduce((acc, r) => ({ ...acc, [r.end.outcome]: (acc[r.end.outcome] || 0) + 1 }), {});
  $("tally").textContent = `${finished.length} rounds · ` +
    Object.entries(outcomes).map(([k, v]) => `${k} ${v}`).join(" · ") +
    (marked.length ? ` · marked success ${success}/${marked.length}` : "");
}

function renderEvents() {
  const lines = state.events.slice(-30).reverse().map((e) => {
    const t = e.time ? e.time.slice(11, 19) : "";
    if (e.event === "transition") return `${t}  ${e.source} → ${e.target}: ${e.reason}`;
    if (e.event === "grasp_check") return `${t}  grasp check: ${e.verdict} (reading ${e.reading ?? "-"})`;
    if (e.event === "round_start") return `${t}  round ${e.round} started: ${chainOf(e)}`;
    if (e.event === "round_end") return `${t}  round ${e.round} ended: ${e.outcome} in ${e.duration_s} s`;
    return `${t}  ${e.event}`;
  });
  $("events").textContent = lines.join("\n");
  $("events").style.whiteSpace = "pre";
}

// Phases of scripts/collect/record_session.py, in lerobot-record's order: an episode is saved only when the pause
// after it ends, so "next" in the pause keeps it and "discard" there throws it away.
const COLLECT_PHASES = {
  starting: ["", "STARTING", "Connecting to both arms and cameras…", null],
  waiting: ["wait", "WAITING · not recording", "Put the arm at rest and reset the scene, then start the first episode.", "▶ Start episode"],
  recording: ["rec", "RECORDING episode %n", "Do the task. Finish when done, or throw it away if it went wrong.", "■ Finish episode"],
  reset: ["wait", "PAUSED · episode %n not saved yet", "Reset the scene. Next keeps it and starts the next episode; ✗ throws it away.", "▶ Keep it · next episode"],
  saving: ["info", "SAVING…", "Wait a moment.", null],
  discarded: ["info", "THROWN AWAY", "It is recorded again after the pause.", null],
  finishing: ["info", "FINISHING · saving everything", "Wait. Do not close the dashboard or the terminal.", null],
  done: ["done", "SESSION ENDED", "", null],
  uploading: ["info", "UPLOADING to the Hub", "", null],
  uploaded: ["done", "UPLOADED · on the Hub", "", null],
  upload_failed: ["rec", "UPLOAD FAILED · the dataset is safe on this laptop", "", null],
};
const COLLECT_ENDED = ["done", "uploaded", "upload_failed"];

function collectStatus() {
  const s = state.status;
  return s && s.mode === "collect" ? s : null;
}

function collectRunning() {
  const fresh = state.statusAge !== null && state.statusAge < 2;
  const s = collectStatus();
  return Boolean((state.collect && state.collect.running) || (fresh && s && !COLLECT_ENDED.includes(s.phase)));
}

function collectHint(s, phase, hint) {
  if (!s) return hint;
  const name = s.dataset;
  if (phase === "done") {
    const upload = s.upload && s.saved ? "Uploading next…" : s.saved ? "Not uploaded: use ⇪ Upload to the Hub." : "";
    return `${s.saved} saved this session · ${name} now has ${s.dataset_total} episodes. ${upload}`;
  }
  if (phase === "uploading") {
    return `${name}: ${s.upload_mb} MB folder; only new or changed files are sent. The next session can start when this is done.`;
  }
  if (phase === "uploaded") {
    return `${name} has ${s.dataset_total} episodes on the Hub after ${s.upload_s} s. A training run started now sees them.`;
  }
  if (phase === "upload_failed") return `${s.upload_error || "unknown error"} · use Retry upload.`;
  return hint;
}

async function collectControl(control) {
  if (!collectRunning()) return;
  await post("/api/collect/control", { control });
}

function setMode(mode) {
  state.mode = mode;
  try { localStorage.setItem("mode", mode); } catch (error) { /* per-viewer convenience only */ }
  document.body.classList.toggle("mode-collect", mode === "collect");
  $("mode-chain").classList.toggle("on", mode === "chain");
  $("mode-collect").classList.toggle("on", mode === "collect");
  if (mode === "collect") loadDatasets();
  render();
}

async function loadDatasets() {
  const data = await (await fetch("/api/collect/datasets")).json();
  state.datasets = data;
  const select = $("collect-dataset");
  const chosen = select.value;
  select.innerHTML = "";
  for (const d of data.datasets) {
    const option = new Option(`${d.name} (${d.episodes} episodes)`, d.name);
    select.appendChild(option);
  }
  select.appendChild(new Option("New dataset…", ""));
  if ([...select.options].some((o) => o.value === chosen)) select.value = chosen;
  else if (!data.datasets.length) select.value = "";
  pickDataset();
}

function pickDataset() {
  const name = $("collect-dataset").value;
  $("collect-new-label").hidden = Boolean(name);
  const tasks = (state.datasets && state.datasets.tasks) || {};
  if (name && tasks[name]) $("collect-task").value = tasks[name];
  $("collect-task-note").textContent = name
    ? `Adding to ${state.datasets.user}/${name}: use the task sentence it was recorded with.`
    : "";
}

$("collect-dataset").onchange = pickDataset;
$("mode-chain").onclick = () => setMode("chain");
$("mode-collect").onclick = () => setMode("collect");
$("collect-next").onclick = () => collectControl("next");
$("collect-discard").onclick = () => collectControl("discard");
$("collect-stop").onclick = () => collectControl("stop");
$("collect-force").onclick = async () => {
  if (confirm("Stop the recording like Ctrl+C? The episodes saved so far are kept.")) await post("/api/collect/force_stop", {});
};
$("collect-upload-now").onclick = async () => {
  const s = collectStatus();
  if (!s || !s.dataset) return;
  const reply = await post("/api/collect/upload", { name: s.dataset.split("/").pop() });
  $("collect-start-note").textContent = reply.ok ? "Uploading…" : `✗ ${reply.reason}`;
};
$("collect-form").onsubmit = async (event) => {
  event.preventDefault();
  const name = $("collect-dataset").value || $("collect-name").value.trim();
  const form = { name, task: $("collect-task").value.trim(), episodes: Number($("collect-episodes").value),
    episode_s: Number($("collect-limit").value), upload: $("collect-upload").checked };
  const fake = state.datasets && state.datasets.fake ? " (FAKE recorder: no arm, no dataset)" : "";
  const upload = form.upload ? "\nWhen it ends, the dataset is uploaded to the Hugging Face Hub." : "";
  if (!confirm(`Start recording ${form.episodes} episodes into ${name}${fake}?${upload}\n\nThe follower arm will mirror the leader arm. Keep hands clear of the follower.`)) return;
  const reply = await post("/api/collect/start", form);
  $("collect-start-note").textContent = reply.ok ? "Starting…" : `✗ ${reply.reason}`;
};

function renderCollect() {
  if (state.mode !== "collect") return;
  const s = collectStatus();
  const busy = collectRunning();
  const phase = s ? s.phase : null;
  const [kind, title, hint, nextLabel] = COLLECT_PHASES[phase] || ["", "No session", "Fill in the form and start a session.", null];
  const episode = s && s.episode !== null && s.episode !== undefined ? s.episode : "";
  $("collect-banner").className = `collect-banner collect-only ${kind}`;
  $("collect-title").textContent = title.replace("%n", episode);
  $("collect-hint").textContent = collectHint(s, phase, hint);
  setFeeds(busy || $("toggle-cameras").checked);

  const age = state.statusAge || 0;
  const elapsed = s && s.phase_started ? s.time - s.phase_started + age : 0;
  const limit = s && phase === "recording" ? s.episode_limit_s : null;
  $("collect-bar").style.width = limit ? `${Math.min(elapsed / limit, 1) * 100}%` : "0";
  $("collect-bar").style.background = limit && elapsed / limit > 0.85 ? "var(--bad)" : "var(--accent)";
  $("collect-time").textContent = s && busy ? (limit ? `${elapsed.toFixed(0)} s of ${limit} s` : `${elapsed.toFixed(0)} s`) : "";

  $("collect-counts").innerHTML = s
    ? `<span>Saved <b>${s.saved}</b> / ${s.target}</span><span>Thrown away <b>${s.discarded}</b></span>` +
      `<span>${s.dataset}: <b>${s.dataset_total}</b> episodes</span>` +
      `<span>Loop <b>${state.loopHz ? state.loopHz.toFixed(0) : "-"}</b> Hz</span>`
    : "";
  $("collect-next").textContent = nextLabel || "▶ Start episode";
  $("collect-next").disabled = !busy || !nextLabel;
  $("collect-discard").disabled = !busy || !["recording", "reset"].includes(phase);
  $("collect-discard").textContent = phase === "reset" ? "✗ Throw away last" : "✗ Throw away";
  $("collect-stop").disabled = !busy || ["finishing", "uploading", ...COLLECT_ENDED].includes(phase);
  $("collect-force").hidden = !(state.collect && state.collect.force_stop);
  const offerUpload = !busy && s && s.dataset && (phase === "upload_failed" || (phase === "done" && s.saved > 0));
  $("collect-upload-now").hidden = !offerUpload;
  $("collect-upload-now").textContent = phase === "upload_failed" ? "⇪ Retry upload" : "⇪ Upload to the Hub";
  for (const field of $("collect-form").elements) field.disabled = busy;
  const output = state.collect ? state.collect.output.join("\n") : "";
  if ($("collect-output").textContent !== output) {
    $("collect-output").textContent = output;
    $("collect-output").scrollTop = $("collect-output").scrollHeight;
  }
}

function render() {
  renderTiles();
  renderRound();
  renderLive();
  renderHistory();
  renderEvents();
  renderCollect();
}

async function loadResults() {
  const rows = await (await fetch("/api/results")).json();
  state.results = Object.fromEntries(rows.map((r) => [String(r.round), r]));
}

function connect() {
  const source = new EventSource("/events");
  let renderedAt = 0;
  source.onmessage = (message) => {
    const data = JSON.parse(message.data);
    if (data.reset) { state.events = []; loadResults(); }
    state.events.push(...data.events);
    if (data.status && state.status && typeof data.status.observes === "number" && state.lastObserves) {
      const [count, at] = state.lastObserves;
      const seconds = (data.status.time - at);
      if (seconds > 0.5) {
        state.loopHz = (data.status.observes - count) / seconds;
        state.lastObserves = [data.status.observes, data.status.time];
      }
    } else if (data.status && typeof data.status.observes === "number") {
      state.lastObserves = [data.status.observes, data.status.time];
    }
    state.status = data.status;
    state.statusAge = data.status_age;
    state.log = data.log;
    state.speech = data.speech;
    state.recording = data.recording;
    const wasRunning = state.collect && state.collect.running;
    state.collect = data.collect;
    if (collectStatus() && state.mode !== "collect" && collectRunning()) setMode("collect");
    if (wasRunning && !data.collect.running && state.mode === "collect") loadDatasets();
    if (Date.now() - renderedAt > 200) { render(); renderedAt = Date.now(); }
  };
  source.onerror = () => { $("badge-robot").textContent = "Dashboard server unreachable"; $("badge-robot").className = "badge off"; };
}

$("toggle-demo").onchange = (e) => document.body.classList.toggle("demo", e.target.checked);
$("toggle-cameras").onchange = () => renderRound();
$("summary").onclick = async () => {
  const text = await (await fetch("/api/summary")).text();
  $("summary-text").value = text;
  try { await navigator.clipboard.writeText(text); } catch (error) { /* the dialog still shows it */ }
  $("summary-dialog").showModal();
};

say("Pick a tile, type a command, or click 🎤 and say “give me the ball” (or “let's dance”).", "info");
let savedMode = "chain";
try { savedMode = localStorage.getItem("mode") || "chain"; } catch (error) { /* per-viewer convenience only */ }
setMode(savedMode === "collect" ? "collect" : "chain");
loadResults().then(connect);
setInterval(() => { renderRound(); renderCollect(); }, 500);
