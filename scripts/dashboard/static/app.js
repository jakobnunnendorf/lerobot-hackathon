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
const STATE_NAMES = { REST: "Rest", PICK: "Pick", VERIFY_HOLD: "Check ball", GIVE: "Hand over", PUT_DOWN: "Put down", DONE: "Done", ABORT: "Aborted" };
const RESULT_BUTTONS = [
  ["success", "✓ success"], ["wrong_ball", "wrong ball"], ["empty", "empty"], ["dropped", "dropped"],
  ["still_digging", "digging"], ["other", "other"],
];
const HAND_LOST_S = 5;

const state = {
  status: null, statusAge: null, events: [], results: {}, log: null, speech: "loading",
  lastObserves: null, loopHz: null, holdSince: null, camerasLive: {},
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
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement.tagName !== "INPUT") { e.preventDefault(); startRecording(); }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && document.activeElement.tagName !== "INPUT") { e.preventDefault(); stopRecording(); }
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
  setFeeds(Boolean(current) || $("toggle-cameras").checked);
  if (!shown) {
    $("round-title").textContent = "Waiting for a command";
    $("round-chain").textContent = "";
    renderStates(null, []);
    setBudget(0, "");
    return;
  }
  const chain = `${shown.start.target || "striped"} ball → ${shown.start.destination || "no destination"}`;
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
  renderStates(now, visited);
}

function renderStates(now, visited) {
  const div = $("states");
  div.innerHTML = "";
  const flow = visited.includes("PUT_DOWN") || !visited.includes("GIVE") ? FLOW : FLOW.filter((s) => s !== "PUT_DOWN");
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
    tr.innerHTML = `<td>${r.number}</td><td>${r.start.target || "striped"} → ${r.start.destination || "-"}</td>` +
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
    if (e.event === "round_start") return `${t}  round ${e.round} started: ${e.target || "striped"} → ${e.destination || "-"}`;
    if (e.event === "round_end") return `${t}  round ${e.round} ended: ${e.outcome} in ${e.duration_s} s`;
    return `${t}  ${e.event}`;
  });
  $("events").textContent = lines.join("\n");
  $("events").style.whiteSpace = "pre";
}

function render() {
  renderTiles();
  renderRound();
  renderLive();
  renderHistory();
  renderEvents();
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

say("Pick a tile, type a command, or click 🎤 and say “give me the ball”.", "info");
loadResults().then(connect);
setInterval(() => renderRound(), 500);
