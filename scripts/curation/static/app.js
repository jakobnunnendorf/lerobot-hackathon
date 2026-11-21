const $ = s => document.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids.filter(k => k != null));
  return node;
};
const fmtNum = v => v == null || v === "" ? "" : (Number.isInteger(v) ? String(v) : Number(v).toFixed(1));
const short = cam => cam.replace("observation.images.", "");
const store = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };

const S = {
  meta: null, rows: [], view: [], sort: { key: null, dir: 1 }, filters: {},
  cur: null, ep: null, traces: null, t: 0, playing: false, last: 0,
  players: {}, useFrames: false, art: {},
};

const COLS = [
  { key: "dataset", label: "dataset", type: "cat", text: r => r.dataset.split("/").pop().replace(/^so101_/, "") },
  { key: "episode", label: "ep", type: "num" },
  { key: "seconds", label: "sec", type: "num" },
  { key: "hit_limit", label: "120s", type: "cat", text: r => r.hit_limit ? "yes" : "no" },
  { key: "verdict", label: "verdict", type: "cat" },
  { key: "scene_class", label: "scene", type: "cat" },
  { key: "clears", label: "clears", type: "num" },
  { key: "open_max", label: "widest", type: "num", title: "widest gripper reading in the episode" },
  { key: "open_at_close", label: "open@1st", type: "num", title: "gripper reading where the first close starts" },
  { key: "final_grip", label: "final", type: "num", title: "gripper reading at the last frame (ball ~11-16, empty ~0-2)" },
  { key: "closes", label: "closes", type: "num", title: "close events in the gripper state (moves over 6 units)" },
  { key: "regrasps", label: "regr", type: "num", title: "regrasps, from the auto table" },
  { key: "source", label: "source", type: "cat", text: r => r.source === "auto" && r.auto_source ? `auto:${r.auto_source}` : (r.source || "none") },
  { key: "note", label: "note / reason", type: "text" },
];
const cellText = (c, r) => c.text ? c.text(r) : (c.type === "num" ? fmtNum(r[c.key]) : (r[c.key] ?? ""));

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function numPredicate(s) {
  s = s.trim();
  if (!s) return null;
  if (s === "empty") return x => x == null || x === "";
  let m = s.match(/^(>=|<=|>|<|=)?\s*(-?\d+(?:\.\d+)?)$/);
  if (m) {
    const v = +m[2], op = m[1] || "=";
    return x => x != null && x !== "" && { ">": x > v, "<": x < v, ">=": x >= v, "<=": x <= v, "=": x == v }[op];
  }
  m = s.match(/^(-?\d+(?:\.\d+)?)\s*(?:-|\.\.)\s*(-?\d+(?:\.\d+)?)$/);
  if (m) return x => x != null && x !== "" && x >= +m[1] && x <= +m[2];
  return null;
}

function applyView() {
  const preds = COLS.map(c => {
    const f = S.filters[c.key];
    if (!f) return null;
    if (c.type === "num") { const p = numPredicate(f); return p && (r => p(r[c.key] == null ? null : +r[c.key])); }
    if (c.type === "cat") return r => cellText(c, r) === f;
    return r => String(cellText(c, r)).toLowerCase().includes(f.toLowerCase());
  }).filter(Boolean);
  S.view = S.rows.filter(r => preds.every(p => p(r)));
  const { key, dir } = S.sort;
  if (key) {
    const c = COLS.find(c => c.key === key);
    const val = r => c.type === "num" ? (r[key] == null || r[key] === "" ? -Infinity : +r[key]) : String(cellText(c, r));
    S.view.sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * dir);
  }
  renderBody();
}

function renderHead() {
  const head = $("#eps thead");
  head.innerHTML = "";
  const labels = el("tr"), filters = el("tr");
  for (const c of COLS) {
    const th = el("th", { textContent: c.label + (S.sort.key === c.key ? (S.sort.dir > 0 ? " ▲" : " ▼") : ""), title: c.title || "" });
    th.onclick = () => { S.sort = { key: c.key, dir: S.sort.key === c.key ? -S.sort.dir : 1 }; renderHead(); applyView(); };
    labels.append(th);
    let input;
    if (c.type === "cat") {
      const values = [...new Set(S.rows.map(r => cellText(c, r)))].sort();
      input = el("select", {}, el("option", { value: "", textContent: "all" }), ...values.map(v => el("option", { value: v, textContent: v || "(empty)" })));
    } else {
      input = el("input", { placeholder: c.type === "num" ? ">20, 1-3" : "text" });
    }
    input.value = S.filters[c.key] || "";
    input.oninput = () => { S.filters[c.key] = input.value; store.set("filters", JSON.stringify(S.filters)); applyView(); };
    filters.append(el("th", {}, input));
  }
  head.append(labels, filters);
}

function rowCells(tr, r) {
  tr.innerHTML = "";
  for (const c of COLS) {
    const td = el("td", { textContent: cellText(c, r) });
    if (c.type === "num") td.className = "num";
    if (c.key === "note") { td.className = "note"; td.title = r.note || ""; }
    if (c.key === "verdict" && r.verdict) td.className = `v-${r.verdict}` + (r.person?.verdict ? " person" : "");
    if (c.key === "scene_class" && r.person?.scene_class) td.className = "person";
    tr.append(td);
  }
  tr.classList.toggle("sel", S.cur?.id === r.id);
}

function renderBody() {
  const body = $("#eps tbody");
  body.innerHTML = "";
  for (const r of S.view) {
    const tr = el("tr");
    tr.dataset.id = r.id;
    rowCells(tr, r);
    tr.onclick = () => select(r);
    body.append(tr);
  }
  const minutes = S.view.reduce((a, r) => a + r.seconds, 0) / 60;
  $("#count").textContent = `${S.view.length} of ${S.rows.length} episodes shown, ${minutes.toFixed(1)} min`;
}

function move(step) {
  if (!S.view.length) return;
  const i = S.view.findIndex(r => r.id === S.cur?.id);
  select(S.view[Math.max(0, Math.min(S.view.length - 1, i < 0 ? 0 : i + step))]);
}

async function select(r) {
  pause();
  S.cur = r;
  document.querySelectorAll("#eps tbody tr").forEach(tr => tr.classList.toggle("sel", tr.dataset.id === r.id));
  document.querySelector(`#eps tbody tr[data-id="${r.id}"]`)?.scrollIntoView({ block: "nearest" });
  showLabel();
  $("#title").textContent = `${r.dataset} · episode ${r.episode} · ${r.seconds} s`;
  const [ep, traces] = await Promise.all([api(`/api/episode/${r.ds}/${r.episode}`), api(`/api/traces/${r.ds}/${r.episode}`).catch(() => null)]);
  if (S.cur?.id !== r.id) return;
  S.ep = ep;
  S.traces = traces;
  S.t = 0;
  setupPlayers();
  drawTraces();
  showAuto();
  seek(0);
}

const duration = () => S.ep ? S.ep.seconds : 0;

function setupPlayers() {
  const box = $("#videos");
  for (const [cam, info] of Object.entries(S.ep.videos)) {
    let p = S.players[cam];
    if (!p) {
      p = S.players[cam] = { video: el("video", { muted: true, playsInline: true, preload: "auto" }), img: el("img", { hidden: true }), cap: el("figcaption") };
      p.fig = el("figure", {}, p.video, p.img, p.cap);
      p.video.addEventListener("error", () => videoFailed(cam));
      p.img.onload = p.img.onerror = () => { p.loading = false; if (p.pending != null) loadFrame(cam, p.pending); };
      box.append(p.fig);
    }
    p.info = info;
    p.cap.textContent = short(cam) + (info.present ? "" : " (video file not downloaded)");
    p.fig.hidden = false;
    p.video.hidden = S.useFrames;
    p.img.hidden = !S.useFrames;
    if (!S.useFrames && info.present && p.video.getAttribute("src") !== info.url) p.video.src = info.url;
    p.video.playbackRate = +$("#speed").value;
  }
  for (const cam of Object.keys(S.players)) if (!(cam in S.ep.videos)) S.players[cam].fig.hidden = true;
}

const activePlayers = () => S.ep ? Object.keys(S.ep.videos).map(c => [c, S.players[c]]).filter(([, p]) => p && p.info.present) : [];

function videoFailed(cam) {
  if (S.useFrames) return;
  $("#mode").textContent = `browser cannot play ${short(cam)} (${S.players[cam].video.error?.message || "error"}): using server frames`;
  setFrames(true);
}

function setFrames(on) {
  S.useFrames = on;
  $("#frames").checked = on;
  if (S.ep) { setupPlayers(); seek(S.t); }
  if (!$("#mode").textContent.startsWith("browser")) $("#mode").textContent = on ? "JPEG frames decoded on the server" : "mp4 in the browser";
}

function loadFrame(cam, t) {
  const p = S.players[cam];
  if (p.loading) { p.pending = t; return; }
  p.pending = null;
  p.loading = true;
  p.img.src = `/api/frame/${S.cur.ds}/${S.cur.episode}/${cam}?t=${t.toFixed(3)}`;
}

function seek(t) {
  S.t = Math.max(0, Math.min(duration(), t));
  for (const [cam, p] of activePlayers()) {
    if (S.useFrames) loadFrame(cam, S.t);
    else p.video.currentTime = p.info.from + S.t;
  }
}

function play() {
  if (!S.ep) return;
  if (S.t >= duration() - 0.05) seek(0);
  S.playing = true;
  $("#play").textContent = "pause";
  if (!S.useFrames) for (const [, p] of activePlayers()) p.video.play().catch(() => {});
}

function pause() {
  S.playing = false;
  $("#play").textContent = "play";
  for (const p of Object.values(S.players)) p.video.pause();
}

function tick(now) {
  const dt = (now - S.last) / 1000;
  S.last = now;
  if (S.playing && S.ep) {
    const players = activePlayers();
    if (S.useFrames || !players.length) {
      S.t += dt * +$("#speed").value;
      if (S.useFrames) players.forEach(([cam]) => loadFrame(cam, S.t));
    } else {
      const [, master] = players[0];
      S.t = master.video.currentTime - master.info.from;
      for (const [, p] of players.slice(1)) {
        if (Math.abs(p.video.currentTime - p.info.from - S.t) > 0.12) p.video.currentTime = p.info.from + S.t;
      }
    }
    if (S.t >= duration()) { pause(); seek(duration()); }
  }
  drawCursor();
  requestAnimationFrame(tick);
}

const PAD = { l: 34, r: 8, t: 14, b: 14 };
const JOINT_COLORS = ["#2f6fde", "#e0663b", "#2aa876", "#a64fd1", "#c9a227"];
const xOf = (W, t) => PAD.l + (t / Math.max(duration(), 1e-6)) * (W - PAD.l - PAD.r);
const tOf = (W, x) => (x - PAD.l) / (W - PAD.l - PAD.r) * duration();

function plot(canvas, series, { yMin, yMax, events = [], grid = [] }) {
  const W = canvas.width = canvas.clientWidth, H = canvas.height;
  const ctx = canvas.getContext("2d");
  const fg = getComputedStyle(document.body).color;
  const y = v => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  ctx.clearRect(0, 0, W, H);
  ctx.font = "10px system-ui";
  ctx.fillStyle = fg;
  ctx.strokeStyle = "rgba(128,128,128,0.25)";
  for (const g of grid) {
    ctx.beginPath(); ctx.moveTo(PAD.l, y(g)); ctx.lineTo(W - PAD.r, y(g)); ctx.stroke();
    ctx.fillText(String(g), 2, y(g) + 3);
  }
  const fps = S.traces.fps;
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.3;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    s.values.forEach((v, i) => { const X = xOf(W, i / fps), Y = y(v); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const e of events) {
    const X = xOf(W, e.frame / fps);
    ctx.strokeStyle = ctx.fillStyle = e.kind === "close" ? "#c0392b" : "#1f8a4c";
    ctx.beginPath(); ctx.moveTo(X, PAD.t); ctx.lineTo(X, H - PAD.b); ctx.stroke();
    ctx.fillText(`${e.kind === "close" ? "C" : "O"} ${e.value}`, X + 2, PAD.t - 3);
  }
  let lx = PAD.l + 4;
  for (const s of series.filter(s => s.label)) {
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, lx, H - 3);
    lx += ctx.measureText(s.label).width + 12;
  }
  S.art[canvas.id] = ctx.getImageData(0, 0, W, H);
}

function drawTraces() {
  for (const id of ["grip", "joints"]) { const c = $("#" + id); c.getContext("2d").clearRect(0, 0, c.width, c.height); S.art[id] = null; }
  if (!S.traces) return;
  const { state, action, events, names } = S.traces;
  const g = 5;
  const gripVals = state[g].concat(action[g]);
  plot($("#grip"), [
    { values: state[g], color: "#2f6fde", width: 1.6, label: "gripper state" },
    { values: action[g], color: "#e0663b", dash: [4, 3], label: "gripper action" },
  ], { yMin: Math.min(-2, ...gripVals), yMax: Math.max(65, ...gripVals) + 2, events, grid: [0, 15, 30, 45, 60] });
  const joints = state.slice(0, 5);
  const all = joints.flat();
  const lo = Math.floor(Math.min(...all) / 20) * 20, hi = Math.ceil(Math.max(...all) / 20) * 20;
  const grid = [];
  for (let v = lo; v <= hi; v += (hi - lo > 160 ? 50 : 20)) grid.push(v);
  plot($("#joints"), joints.map((values, i) => ({ values, color: JOINT_COLORS[i], label: (names[i] || `joint ${i}`).replace(".pos", "") })),
    { yMin: lo, yMax: hi, events: events.filter(e => e.kind === "close"), grid });
}

function drawCursor() {
  $("#time").textContent = S.ep ? `${S.t.toFixed(2)} / ${duration().toFixed(1)} s · f${Math.round(S.t * S.ep.fps)}` : "";
  for (const id of ["grip", "joints"]) {
    const art = S.art[id];
    if (!art) continue;
    const c = $("#" + id), ctx = c.getContext("2d");
    ctx.putImageData(art, 0, 0);
    const X = xOf(c.width, S.t);
    ctx.strokeStyle = getComputedStyle(document.body).color;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, c.height); ctx.stroke();
  }
}

for (const id of ["grip", "joints"]) {
  const c = $("#" + id);
  const scrub = e => { if (S.ep) { pause(); seek(tOf(c.width, e.offsetX * c.width / c.clientWidth)); } };
  c.addEventListener("mousedown", e => { scrub(e); c.onmousemove = ev => ev.buttons && scrub(ev); });
}

function showLabel() {
  const r = S.cur;
  $("#verdicts").innerHTML = "";
  S.meta.verdicts.forEach((v, i) => {
    const b = el("button", { className: `chip ${v}` + (r.verdict === v ? " on" : ""), textContent: `${i + 1} ${v}` });
    b.onclick = () => saveLabel({ verdict: v });
    $("#verdicts").append(b);
  });
  $("#scenes").innerHTML = "";
  S.meta.scene_classes.forEach((s, i) => {
    const b = el("button", { className: "chip" + (r.scene_class === s ? " on" : ""), textContent: `${i + 4} ${s}` });
    b.onclick = () => saveLabel({ scene_class: s });
    $("#scenes").append(b);
  });
  $("#task").value = r.person?.task || "";
  $("#task").placeholder = r.task || r.orig_task || "";
  $("#note").value = r.person?.note || "";
  $("#note").placeholder = r.note || "";
  $("#src").textContent = r.person ? `person label by ${r.person.by || "?"} at ${(r.person.time || "").slice(11, 19)}`
    : r.source === "auto" ? `auto label (${r.auto_source || "table"}, confidence ${r.confidence || "?"})` : "no label yet";
}

function showAuto() {
  const dl = $("#auto");
  dl.innerHTML = "";
  const add = (k, v) => { if (v != null && v !== "") dl.append(el("dt", { textContent: k }), el("dd", { textContent: v })); };
  add("dataset task", S.cur.orig_task);
  for (const [k, v] of Object.entries(S.cur.auto || {})) add(`table ${k}`, v);
  add("closes (here)", S.cur.closes);
  add("first close starts at", S.cur.open_at_close);
}

async function saveLabel(patch) {
  const by = $("#by").value.trim();
  if (!by) { $("#saved").textContent = "type your name at the top first"; $("#by").focus(); return; }
  const r = S.cur, p = r.person || {};
  const body = { dataset: r.dataset, episode: r.episode, verdict: p.verdict ?? null, scene_class: p.scene_class ?? null,
    task: p.task ?? null, note: p.note ?? null, by, ...patch };
  try {
    const updated = await api("/api/label", body);
    Object.assign(r, updated);
    const tr = document.querySelector(`#eps tbody tr[data-id="${r.id}"]`);
    if (tr) rowCells(tr, r);
    if (S.cur?.id === r.id) showLabel();
    $("#saved").textContent = `saved ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    $("#saved").textContent = `NOT saved: ${err.message}`;
  }
}

let typingTimer;
for (const id of ["task", "note"]) {
  const input = $("#" + id);
  const commit = () => { clearTimeout(typingTimer); if (S.cur && input.value !== (S.cur.person?.[id] || "")) saveLabel({ [id]: input.value }); };
  input.addEventListener("input", () => { clearTimeout(typingTimer); typingTimer = setTimeout(commit, 1200); });
  input.addEventListener("change", commit);
}

document.addEventListener("keydown", e => {
  if (e.target.matches("input, textarea, select")) { if (e.key === "Escape") e.target.blur(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const step = e.shiftKey ? 1 / (S.ep?.fps || 30) : 1;
  const actions = {
    j: () => move(1), k: () => move(-1),
    " ": () => S.playing ? pause() : play(),
    ArrowLeft: () => { pause(); seek(S.t - step); }, ArrowRight: () => { pause(); seek(S.t + step); },
    Home: () => seek(0),
  };
  S.meta.verdicts.forEach((v, i) => actions[String(i + 1)] = () => S.cur && saveLabel({ verdict: v }));
  S.meta.scene_classes.forEach((s, i) => actions[String(i + 4)] = () => S.cur && saveLabel({ scene_class: s }));
  if (actions[e.key]) { e.preventDefault(); actions[e.key](); }
});

$("#play").onclick = () => S.playing ? pause() : play();
$("#speed").onchange = () => Object.values(S.players).forEach(p => p.video.playbackRate = +$("#speed").value);
$("#frames").onchange = () => { $("#mode").textContent = ""; setFrames($("#frames").checked); };
$("#by").value = store.get("by") || "";
$("#by").oninput = () => store.set("by", $("#by").value.trim());
window.addEventListener("resize", () => { if (S.traces) drawTraces(); });

async function init() {
  S.meta = await api("/api/datasets");
  S.rows = await api("/api/episodes");
  try { S.filters = JSON.parse(store.get("filters") || "{}"); } catch { S.filters = {}; }
  const canAv1 = document.createElement("video").canPlayType('video/mp4; codecs="av01.0.05M.08"');
  setFrames(!canAv1 || new URLSearchParams(location.search).has("frames"));
  if (!canAv1) $("#mode").textContent = "browser reports no AV1 support: using server frames";
  $("#status").textContent = `${S.meta.datasets.length} datasets · ${S.rows.length} episodes · table rows ${S.meta.table_rows} · labels → ${S.meta.labels_path}`;
  renderHead();
  applyView();
  requestAnimationFrame(tick);
  const wanted = new URLSearchParams(location.search).get("ep");
  const first = S.view.find(r => r.id === wanted) || S.view[0];
  if (first) select(first);
}
init().catch(err => { $("#status").textContent = `failed to load: ${err.message}`; });
