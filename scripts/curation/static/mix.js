const $ = (selector) => document.querySelector(selector);
const RANGE_FIELDS = ["seconds", "open_max", "open_at_close", "final_grip", "closes"];
const VERDICT_CHOICES = ["keep", "drop", "look", "none"];
const OUTPUT_PREFIX = "molnrb/so101_pick_mix_";
const include = new Set();
const exclude = new Set();
let rows = [];
let classes = [];
let selected = new Set();
let savedMixes = [];
let outputEdited = false;
let timer = null;
let seq = 0;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const key = (r) => `${r.dataset}:${r.episode}`;
const shortName = (repo) => repo.split("/").pop().replace(/^so101_pick_/, "");
const pct = (n, total) => (total ? Math.round((100 * n) / total) : 0);
const checked = (name) => [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((i) => i.value);
const setChecked = (name, values) => document.querySelectorAll(`input[name="${name}"]`).forEach((i) => (i.checked = values.includes(i.value)));
const bound = (input) => (input.value.trim() === "" ? null : Number(input.value));

function setStatus(text, error = false) {
  $("#status").textContent = text;
  $("#status").classList.toggle("err", error);
}

async function api(path, body) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `${res.status} ${res.statusText}`);
  return data;
}

function mostCommon(values) {
  const counts = new Map();
  for (const v of values) if (v) counts.set(v, (counts.get(v) || 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1])[0]?.[0] || "";
}

function defaultTask(sceneClass) {
  const ofClass = rows.filter((r) => r.scene_class === sceneClass);
  return mostCommon(ofClass.map((r) => r.task)) || mostCommon(ofClass.map((r) => r.orig_task));
}

function extent(field) {
  const values = rows.map((r) => r[field]).filter((v) => typeof v === "number");
  return values.length ? [Math.min(...values), Math.max(...values)] : ["", ""];
}

function checkboxes(name, values, on) {
  return values.map((v) => `<label class="cb"><input type="checkbox" name="${name}" value="${esc(v)}"${on.includes(v) ? " checked" : ""}> ${esc(v)}</label>`).join("");
}

function buildForm(info) {
  classes = info.scene_classes;
  $("#sources").innerHTML = info.datasets.map((d) => {
    const ofDataset = rows.filter((r) => r.dataset === d.repo_id);
    const kept = ofDataset.filter((r) => r.verdict === "keep").length;
    return `<label class="cb"><input type="checkbox" name="src" value="${esc(d.repo_id)}" checked> ${esc(d.repo_id)}
      <span class="muted">${ofDataset.length} episodes, ${kept} keep</span></label>`;
  }).join("");
  $("#verdicts").innerHTML = checkboxes("verdict", VERDICT_CHOICES, ["keep"]);
  $("#scenes").innerHTML = checkboxes("scene", [...classes, "none"], []);
  $("#ranges tbody").innerHTML = RANGE_FIELDS.map((f) => {
    const [lo, hi] = extent(f);
    return `<tr><td>${f}</td><td><input type="number" step="any" id="lo-${f}" placeholder="${lo}"></td><td><input type="number" step="any" id="hi-${f}" placeholder="${hi}"></td></tr>`;
  }).join("");
  $("#tasks").innerHTML = classes.map((c) => `<div class="row"><span class="k">${esc(c)}</span><input id="task-${c}" value="${esc(defaultTask(c))}"></div>`).join("");
}

function spec() {
  const ranges = {};
  for (const f of RANGE_FIELDS) {
    const bounds = [bound($(`#lo-${f}`)), bound($(`#hi-${f}`))];
    if (bounds.some((b) => b !== null)) ranges[f] = bounds;
  }
  const taskByClass = Object.fromEntries(classes.map((c) => [c, $(`#task-${c}`).value.trim()]).filter(([, task]) => task));
  return {
    name: $("#name").value.trim(),
    sources: checked("src"),
    filter: { verdict: checked("verdict"), scene_class: checked("scene"), exclude_hit_limit: $("#nolimit").checked, ranges },
    include: [...include],
    exclude: [...exclude],
    task_by_class: taskByClass,
    output_repo_id: $("#output").value.trim(),
  };
}

function replaceSet(set, values) {
  set.clear();
  for (const v of values || []) set.add(v);
}

function loadSpec(mix) {
  const filter = mix.filter || { verdict: ["keep"] };
  $("#name").value = mix.name || "";
  $("#output").value = mix.output_repo_id || "";
  outputEdited = Boolean(mix.output_repo_id);
  setChecked("src", mix.sources || []);
  setChecked("verdict", filter.verdict || []);
  setChecked("scene", filter.scene_class || []);
  $("#nolimit").checked = Boolean(filter.exclude_hit_limit);
  for (const f of RANGE_FIELDS) {
    const [lo, hi] = filter.ranges?.[f] || [null, null];
    $(`#lo-${f}`).value = lo ?? "";
    $(`#hi-${f}`).value = hi ?? "";
  }
  for (const c of classes) $(`#task-${c}`).value = mix.task_by_class?.[c] || "";
  replaceSet(include, mix.include);
  replaceSet(exclude, mix.exclude);
  const loaded = new Set([...document.querySelectorAll('input[name="src"]')].map((i) => i.value));
  const missing = (mix.sources || []).filter((s) => !loaded.has(s));
  refresh().then(() => setStatus(missing.length ? `loaded ${mix.name}; sources not on this server: ${missing.join(", ")}` : `loaded ${mix.name}`, missing.length > 0));
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(refresh, 300);
}

async function refresh() {
  clearTimeout(timer);
  const mine = ++seq;
  setStatus("counting...");
  try {
    const [counts, freshRows] = await Promise.all([api("/api/mix/preview", spec()), api("/api/episodes")]);
    if (mine !== seq) return;
    rows = freshRows;
    selected = new Set(counts.selected);
    renderStats(counts);
    renderList();
    setStatus("");
  } catch (error) {
    if (mine === seq) setStatus(`preview failed: ${error.message}`, true);
  }
}

function tallyTable(title, tally, total) {
  const max = Math.max(1, ...Object.values(tally).map((t) => t.episodes));
  const body = Object.entries(tally).map(([name, t]) =>
    `<tr><td>${esc(name)}</td><td class="num">${t.episodes}</td><td class="num">${t.minutes}</td><td class="num">${pct(t.episodes, total)}%</td>
     <td><span class="bar" style="width:${Math.round((t.episodes / max) * 140)}px"></span></td></tr>`).join("");
  return `<table><thead><tr><th>${title}</th><th>episodes</th><th>min</th><th>share</th><th></th></tr></thead><tbody>${body}</tbody></table>`;
}

function styleBar(styles, total) {
  const parts = [["direct", "direct"], ["clear_first", "clear first"], ["unknown", "unknown"]];
  const stack = parts.map(([k]) => `<span class="s-${k}" style="flex:${styles[k] || 0} 1 0"></span>`).join("");
  const legend = parts.map(([k, label]) => `<span><i class="s-${k}"></i>${label} ${styles[k] || 0} (${pct(styles[k] || 0, total)}%)</span>`).join("");
  return `<div class="stack">${stack}</div><div class="legend">${legend}</div>`;
}

function histogram(title, hist) {
  const max = Math.max(1, ...hist.counts);
  const last = hist.bins.length - 1;
  const columns = hist.counts.map((n, i) => {
    const range = i === last ? `${hist.bins[i]}+` : `${hist.bins[i]}-${hist.bins[i + 1]}`;
    return `<div class="hc" title="${range}: ${n}"><span>${n || ""}</span><i style="height:${Math.round((n / max) * 76)}px"></i><span class="bin">${hist.bins[i]}</span></div>`;
  }).join("");
  return `<div class="ht">${title}</div><div class="hist">${columns}</div>`;
}

function renderStats(counts) {
  const warnings = [...(counts.episodes ? [] : ["the mix selects no episodes"]), ...counts.warnings];
  $("#warnings").innerHTML = warnings.map((w) => `<div>${esc(w)}</div>`).join("");
  $("#warnings").hidden = !warnings.length;
  $("#total").innerHTML = `<b>${counts.episodes}</b> episodes &middot; <b>${counts.minutes}</b> min`;
  $("#byclass").innerHTML = tallyTable("scene class", counts.by_class, counts.episodes);
  $("#bydataset").innerHTML = tallyTable("dataset", counts.by_dataset, counts.episodes);
  $("#style").innerHTML = styleBar(counts.pick_style, counts.episodes);
  $("#hist-open").innerHTML = histogram("widest opening (open_max)", counts.open_max_hist);
  $("#hist-close").innerHTML = histogram("opening at first close (open_at_close)", counts.open_at_close_hist);
  const tasks = Object.entries(counts.tasks).sort((a, b) => b[1] - a[1]);
  $("#tasklist").innerHTML = `<table><tbody>${tasks.map(([task, n]) => `<tr><td>${esc(task)}</td><td class="num">${n}</td></tr>`).join("")}</tbody></table>`;
}

function renderList() {
  const showAll = $("#showall").checked;
  const sources = new Set(checked("src"));
  const shown = rows.filter((r) => {
    const k = key(r);
    return selected.has(k) || include.has(k) || exclude.has(k) || (showAll && sources.has(r.dataset));
  });
  $("#listcount").textContent = `${selected.size} selected, ${shown.length} shown; by hand: ${include.size} included, ${exclude.size} excluded`;
  $("#list tbody").innerHTML = shown.map((r) => {
    const k = key(r);
    const on = selected.has(k);
    const mark = include.has(k) ? "+" : exclude.has(k) ? "-" : "";
    const cls = [on ? "" : "off", exclude.has(k) ? "excluded" : ""].join(" ");
    return `<tr class="${cls}"><td><input type="checkbox" data-key="${esc(k)}"${on ? " checked" : ""}></td>
      <td title="${esc(r.dataset)}">${esc(shortName(r.dataset))}</td><td class="num">${r.episode}</td><td class="v-${esc(r.verdict)}">${esc(r.verdict || "-")}</td>
      <td>${esc(r.scene_class || "-")}</td><td class="num">${r.seconds}${r.hit_limit ? "!" : ""}</td><td class="num">${r.open_max ?? ""}</td>
      <td class="num">${r.open_at_close ?? ""}</td><td class="mark">${mark}</td></tr>`;
  }).join("");
}

async function toggle(k, on) {
  const [undo, force] = on ? [exclude, include] : [include, exclude];
  if (undo.delete(k)) {
    await refresh();
    if (selected.has(k) === on) return;
  }
  force.add(k);
  await refresh();
}

async function save() {
  $("#saveout").hidden = true;
  setStatus("saving...");
  try {
    const result = await api("/api/mix/save", spec());
    $("#paths").innerHTML = `<div>${esc(result.json)}</div><div>${esc(result.sh)}</div>`;
    $("#script").textContent = result.script;
    $("#saveout").hidden = false;
    setStatus(`saved ${result.counts.episodes} episodes, ${result.counts.minutes} min`);
    loadSaved();
  } catch (error) {
    setStatus(`save failed: ${error.message}`, true);
  }
}

async function copyScript() {
  try {
    await navigator.clipboard.writeText($("#script").textContent);
  } catch {
    // navigator.clipboard only exists on https or localhost; the server is often opened by LAN address.
    const range = document.createRange();
    range.selectNodeContents($("#script"));
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    document.execCommand("copy");
  }
  setStatus("script copied");
}

async function loadSaved() {
  try {
    savedMixes = await api("/api/mixes");
  } catch (error) {
    $("#mixes").innerHTML = `<span class="warn">could not list saved mixes: ${esc(error.message)}</span>`;
    return;
  }
  $("#mixes").innerHTML = savedMixes.length
    ? savedMixes.map((m, i) => `<div class="mix"><button data-load="${i}">load</button> <b>${esc(m.name)}</b>
        ${m.counts?.episodes ?? "?"} episodes, ${m.counts?.minutes ?? "?"} min &rarr; ${esc(m.output_repo_id)}<div class="muted">${esc(m.file)}</div></div>`).join("")
    : `<span class="muted">none yet</span>`;
}

function onFormInput(event) {
  const id = event.target.id;
  if (id === "output") outputEdited = event.target.value.trim() !== "";
  if ((id === "name" || id === "output") && !outputEdited) $("#output").value = $("#name").value.trim() ? OUTPUT_PREFIX + $("#name").value.trim() : "";
  if (id !== "name" && id !== "output") schedule();
}

async function init() {
  const [info, firstRows] = await Promise.all([api("/api/datasets"), api("/api/episodes")]);
  rows = firstRows;
  buildForm(info);
  $("#form").addEventListener("input", onFormInput);
  $("#save").addEventListener("click", save);
  $("#copy").addEventListener("click", copyScript);
  $("#mixes").addEventListener("click", (e) => e.target.dataset.load && loadSpec(savedMixes[Number(e.target.dataset.load)]));
  $("#list").addEventListener("change", (e) => e.target.dataset.key && toggle(e.target.dataset.key, e.target.checked));
  $("#showall").addEventListener("change", renderList);
  $("#clearhand").addEventListener("click", () => { include.clear(); exclude.clear(); refresh(); });
  await Promise.all([refresh(), loadSaved()]);
}

init().catch((error) => setStatus(`could not load: ${error.message}`, true));
