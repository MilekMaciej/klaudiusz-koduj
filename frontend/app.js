/* =========================================================================
   Tysiąc · Prompt Battles – logika interfejsu (czysty JavaScript, bez frameworków)

   Przepływ:
     1. Konfiguracja: tryb (zwykły / sandbox), kto gra na miejscach A i B,
        model i strategia (preset z pliku albo własny prompt).
     2. POST /api/matches -> serwer tworzy mecz (i token, jeśli gra człowiek).
     3. "Następny ruch" / "Auto" -> POST /api/matches/{id}/step (jeden ruch bota).
        Ruch człowieka -> POST /api/matches/{id}/move (z tokenem w nagłówku).
   Przeglądarka nigdy nie dostaje kart bota, gdy gra człowiek – serwer zwraca
   wtedy wyłącznie obserwację człowieka.
   ========================================================================= */
"use strict";

const SUIT_SYMBOL = { hearts: "♥", diamonds: "♦", clubs: "♣", spades: "♠" };
const SUIT_NAME = { hearts: "kier", diamonds: "karo", clubs: "trefl", spades: "pik" };
const SHORT_SUIT = { h: "hearts", d: "diamonds", c: "clubs", s: "spades", "♥": "hearts", "♦": "diamonds", "♣": "clubs", "♠": "spades" };
const RANKS = ["9", "J", "Q", "K", "10", "A"];
const PHASE_PL = {
  bidding: "licytacja", choose_musik: "wybór musiku", discard: "odkładanie kart",
  declare: "deklaracja", play: "rozgrywka", game_over: "koniec gry",
};
const KIND_PL = { llm: "Bot LLM", heuristic: "Bot regułowy", human: "Człowiek" };
const MELD_POINTS = { hearts: 100, diamonds: 80, clubs: 60, spades: 40 };

const app = {
  presets: [], models: [], defaultModel: null, mode: "normal",
  matchId: null, token: null, human: null, viewer: "spectator",
  view: null, auto: false, busy: false, peek: "A",
  solver: null, pollTimer: null, logLen: 0,
};

/* ---------------------------------------------------------------- narzędzia */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, { method = "GET", body, token } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["X-Player-Token"] = token;
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  let data = null;
  try { data = await res.json(); } catch (_) { /* brak treści */ }
  if (!res.ok) {
    const detail = data && data.detail;
    const message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`;
    const error = new Error(message);
    error.status = res.status;
    throw error;
  }
  return data;
}

/* ---------------------------------------------------------------- karty */

/** "Ah" | "10h" | "Th" | "Q♠" | "A_hearts" (opcjonalnie z "*") -> {code, meld} albo null */
function parseCard(token) {
  let t = String(token).trim();
  const meld = t.endsWith("*");
  if (meld) t = t.slice(0, -1);
  if (t.includes("_")) {
    const [rank, suit] = t.split("_");
    return RANKS.includes(rank) && SUIT_SYMBOL[suit] ? { code: t, meld } : null;
  }
  if (t.length < 2) return null;
  let rank = t.slice(0, -1).toUpperCase();
  const suit = SHORT_SUIT[t.slice(-1).toLowerCase()];
  if (rank === "T") rank = "10";
  return RANKS.includes(rank) && suit ? { code: `${rank}_${suit}`, meld } : null;
}

function cardParts(code) {
  const [rank, suit] = code.split("_");
  return { rank, suit, symbol: SUIT_SYMBOL[suit], red: suit === "hearts" || suit === "diamonds" };
}

function cardLabel(code) {
  if (!code) return "–";
  const { rank, symbol } = cardParts(code);
  return `${rank}${symbol}`;
}

function cardEl(code, { small = false, legal = false, dim = false, meld = false, onClick = null, title = null } = {}) {
  if (!code) return el("div", { class: `card back${small ? " small" : ""}`, "aria-label": "zakryta karta" });
  const { rank, symbol, red, suit } = cardParts(code);
  const node = el("div", {
    class: ["card", red && "red", small && "small", legal && "legal", dim && "dim"].filter(Boolean).join(" "),
    title: title || `${rank} ${SUIT_NAME[suit]}`,
    role: onClick ? "button" : null, tabindex: onClick ? "0" : null,
  }, el("span", { class: "rank", text: rank }), el("span", { class: "suit", text: symbol }));
  if (meld) node.append(el("span", { class: "meld-dot", text: "meld" }));
  if (onClick) {
    node.addEventListener("click", onClick);
    node.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } });
  }
  return node;
}

function miniCards(codes, cls = "") {
  return el("div", { class: `mini-cards ${cls}` }, (codes || []).map((c) => cardEl(c, { small: true })));
}

/* ================================================================ KONFIGURACJA */

async function initSetup() {
  try {
    const [presets, models] = await Promise.all([api("/api/presets"), api("/api/models")]);
    app.presets = presets;
    app.models = models.models;
    app.defaultModel = models.default;
    const test = models.provider !== "meta" ? " · tryb testowy" : "";
    const ready = app.models.filter((m) => m.ready).length;
    $("#llm-info").innerHTML = "";
    $("#llm-info").append("Domyślny model: ", el("b", { text: models.default || "brak" }),
      ` (${models.provider}${test}) · modeli z kluczem: ${ready}`);
  } catch (e) {
    $("#llm-info").textContent = "Nie udało się pobrać konfiguracji serwera.";
    showSetupError(`Serwer nie odpowiada: ${e.message}`);
  }
  try {
    app.solver = await api("/api/solver");
  } catch (_) { app.solver = { available: false, reason: "brak odpowiedzi serwera" }; }
  const analysisBox = $("#analysis");
  analysisBox.disabled = !app.solver.available;
  analysisBox.checked = app.solver.available;
  $("#solver-status").textContent = app.solver.available
    ? `Q i strata każdego ruchu w rozgrywce (double-dummy, ${app.solver.samples} światów)`
    : `niedostępna: ${app.solver.reason}`;
  buildSeat("A", { preset: "agresywna" });
  buildSeat("B", { preset: "defensywna" });

  $$(".mode-btn").forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));
  $$("[data-example]").forEach((btn) => btn.addEventListener("click", () => fillExample(btn.dataset.example)));
  $("#sb-phase").addEventListener("change", updatePhaseFields);
  $$(".cards-input").forEach((input) => input.addEventListener("input", () => updateChips(input)));
  $("#start").addEventListener("click", startMatch);
  updatePhaseFields();
}

function setMode(mode) {
  app.mode = mode;
  $$(".mode-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
    b.setAttribute("aria-selected", String(b.dataset.mode === mode));
  });
  $("#sandbox-panel").hidden = mode !== "sandbox";
  updateVisibilityBox();
}

/** Wybór widoczności kart ma sens tylko w sandboxie z człowiekiem. */
function updateVisibilityBox() {
  const anyHuman = $$(".seat-kind").some((s) => s.value === "human");
  $("#visibility-box").hidden = !(app.mode === "sandbox" && anyHuman);
}

function buildSeat(player, { preset }) {
  const node = $("#seat-template").content.firstElementChild.cloneNode(true);
  node.dataset.player = player;
  $(".seat-title", node).append(el("span", { class: "seat-badge", text: player }), `Gracz ${player}`);

  // model
  const modelSel = $(".seat-model", node);
  for (const m of app.models) {
    modelSel.append(el("option", { value: m.model, disabled: !m.ready },
      `${m.label} — ${m.model}${m.note ? ` (${m.note})` : ""}${m.ready ? "" : " · brak klucza"}`));
  }
  const firstReady = app.models.find((m) => m.ready && m.model === app.defaultModel) || app.models.find((m) => m.ready);
  if (firstReady) modelSel.value = firstReady.model;

  // strategia
  const presetSel = $(".seat-preset", node);
  for (const p of app.presets) presetSel.append(el("option", { value: p.name }, `${p.name} — ${p.title}`));
  if (app.presets.some((p) => p.name === preset)) presetSel.value = preset;
  const preview = () => {
    const p = app.presets.find((x) => x.name === presetSel.value);
    $(".preset-preview", node).textContent = p ? p.text : "(brak presetów w katalogu strategies/)";
  };
  presetSel.addEventListener("change", preview);
  preview();

  const radios = $$(".strategy-source input", node);
  radios.forEach((r) => { r.name = `src-${player}`; r.addEventListener("change", () => syncSeat(node)); });
  const textarea = $(".seat-strategy", node);
  textarea.addEventListener("input", () => { $(".counter", node).textContent = `${textarea.value.length} / 8000`; });
  $(".copy-preset", node).addEventListener("click", () => {
    const p = app.presets.find((x) => x.name === presetSel.value);
    textarea.value = p ? p.text : "";
    textarea.dispatchEvent(new Event("input"));
    radios.find((r) => r.value === "custom").checked = true;
    syncSeat(node);
    textarea.focus();
  });
  $(".seat-kind", node).addEventListener("change", () => syncSeat(node));
  $("#seats").append(node);
  syncSeat(node);
}

function syncSeat(node) {
  updateVisibilityBox();
  const kind = $(".seat-kind", node).value;
  const source = $(".strategy-source input:checked", node).value;
  $(".llm-only", node).hidden = kind !== "llm";
  $(".heuristic-only", node).hidden = kind !== "heuristic";
  $(".human-only", node).hidden = kind !== "human";
  $(".preset-box", node).hidden = source !== "preset";
  $(".custom-box", node).hidden = source !== "custom";
}

function collectSeat(player) {
  const node = $(`.seat[data-player="${player}"]`);
  return {
    kind: $(".seat-kind", node).value,
    strategy_source: $(".strategy-source input:checked", node).value,
    preset: $(".seat-preset", node).value || null,
    strategy: $(".seat-strategy", node).value,
    model: $(".seat-model", node).value || null,
  };
}

/* ---------- sandbox ---------- */

function updatePhaseFields() {
  const phase = $("#sb-phase").value;
  $$(".only-phase").forEach((node) => { node.hidden = !node.dataset.phases.split(",").includes(phase); });
}

function updateChips(input) {
  const box = $(`.chips[data-for="${input.id}"]`);
  const count = $(`.count[data-for="${input.id}"]`);
  const tokens = input.value.replace(/,/g, " ").split(/\s+/).filter(Boolean);
  box.replaceChildren(...tokens.map((t) => {
    const parsed = parseCard(t);
    if (!parsed) return el("span", { class: "chip bad", title: "nierozpoznana karta" }, t);
    const { red } = cardParts(parsed.code);
    return el("span", { class: `chip${red ? " red" : ""}` }, cardLabel(parsed.code) + (parsed.meld ? "*" : ""));
  }));
  if (count) count.textContent = tokens.length ? `(${tokens.length})` : "";
}

const EXAMPLES = {
  meld: { phase: "play", to_move: "A", hand_a: "Ah 10h Kh Qh Kc 9s", hand_b: "", bidder: "A", contract: 140,
    tricks: "Ad 9d\n10d Jd\nAc Jc\n10s Qs", score_a: 420, score_b: 510 },
  defend: { phase: "play", to_move: "B", hand_a: "Kh As 10d", hand_b: "Ah 9h Js 9d", bidder: "A", contract: 120,
    table: "Kc", tricks: "Ac 9c\nA: Qh* Jh\n10s Ks", score_a: 300, score_b: 380 },
  bid: { phase: "bidding", to_move: "A", hand_a: "Ah 10h Kh Qh Ad 10d Kc Jc 9s 9h", hand_b: "", bid: 120 },
  clear: { phase: "play" },
};

function fillExample(name) {
  const ex = EXAMPLES[name];
  const set = (id, v) => { const n = $(`#${id}`); n.value = v ?? ""; if (n.classList.contains("cards-input")) updateChips(n); };
  set("sb-phase", ex.phase); set("sb-to-move", ex.to_move || "A");
  set("sb-hand-a", ex.hand_a); set("sb-hand-b", ex.hand_b);
  set("sb-bidder", ex.bidder || ""); set("sb-contract", ex.contract); set("sb-trump", "");
  set("sb-table", ex.table); set("sb-tricks", ex.tricks); set("sb-bid", ex.bid); set("sb-bid-won", "");
  set("sb-revealed", ""); set("sb-musik-1", ""); set("sb-musik-2", ""); set("sb-dealer", "");
  set("sb-melds-a", ""); set("sb-melds-b", ""); set("sb-history", "");
  set("sb-score-a", ex.score_a ?? 0); set("sb-score-b", ex.score_b ?? 0);
  updatePhaseFields();
}

function collectSandbox() {
  const v = (id) => $(`#${id}`).value.trim();
  const num = (id) => (v(id) === "" ? null : Number(v(id)));
  const phase = v("sb-phase");
  const suits = (id) => v(id).split(/[\s,]+/).filter(Boolean);
  const spec = {
    phase, to_move: v("sb-to-move"),
    hands: { A: v("sb-hand-a"), B: v("sb-hand-b") },
    scores: { A: num("sb-score-a") || 0, B: num("sb-score-b") || 0 },
  };
  if (phase === "bidding" || phase === "choose_musik") spec.musik = [v("sb-musik-1"), v("sb-musik-2")];
  if (phase === "bidding") { spec.bid = num("sb-bid"); spec.dealer = v("sb-dealer") || null; }
  if (["choose_musik", "discard", "declare"].includes(phase)) spec.bid = num("sb-bid-won");
  if (["discard", "declare", "play"].includes(phase)) spec.musik = v("sb-revealed");
  if (phase === "play") {
    Object.assign(spec, {
      bidder: v("sb-bidder") || null, contract: num("sb-contract"), trump: v("sb-trump") || null,
      table: v("sb-table"), tricks: $("#sb-tricks").value,
      melds: { A: suits("sb-melds-a"), B: suits("sb-melds-b") },
    });
  }
  const history = v("sb-history");
  if (history) {
    try { spec.history = JSON.parse(history); } catch (e) { throw new Error(`Historia (JSON): ${e.message}`); }
    if (!Array.isArray(spec.history)) throw new Error("Historia (JSON) musi być listą zdarzeń.");
  }
  return spec;
}

/* ---------- start ---------- */

function showSetupError(message) {
  const box = $("#setup-error");
  box.hidden = !message;
  box.textContent = message || "";
  if (message) box.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function startMatch() {
  showSetupError(null);
  const payload = {
    mode: app.mode,
    seats: { A: collectSeat("A"), B: collectSeat("B") },
    seed: $("#seed").value === "" ? null : Number($("#seed").value),
    max_rounds: Number($("#max-rounds").value) || 10,
    analysis: $("#analysis").checked && !$("#analysis").disabled,
    open_cards: !$("#visibility-box").hidden && $("input[name=visibility]:checked").value === "open",
  };
  try {
    if (app.mode === "sandbox") payload.sandbox = collectSandbox();
  } catch (e) { showSetupError(e.message); return; }

  const button = $("#start");
  button.disabled = true;
  try {
    const res = await api("/api/matches", { method: "POST", body: payload });
    Object.assign(app, { matchId: res.match_id, token: res.human_token, human: res.human,
                         viewer: res.human || "spectator", auto: false, peek: "A" });
    app.view = await api(`/api/matches/${app.matchId}?viewer=${app.viewer}`, { token: app.token });
    $("#setup").hidden = true;
    $("#game").hidden = false;
    renderGame();
    window.scrollTo({ top: 0 });
  } catch (e) {
    showSetupError(e.message);
  } finally {
    button.disabled = false;
  }
}

/* ================================================================ MECZ */

function seatName(view, p) {
  const seat = view.seats[p];
  return `${p} · ${seat.label}`;
}

async function doStep() {
  if (app.busy || !app.view) return;
  app.busy = true;
  renderStatus();
  let error = null;
  try {
    app.view = await api(`/api/matches/${app.matchId}/step?viewer=${app.viewer}`, { method: "POST", token: app.token });
  } catch (e) { error = e; }
  app.busy = false;                      // dopiero teraz – karty człowieka mają być klikalne
  if (error) {
    stopAuto();
    showGameError({ kind: "http", message: error.message });
  } else {
    renderGame();
  }
}

async function humanMove(move) {
  if (app.busy) return;
  app.busy = true;
  renderStatus();
  let error = null;
  try {
    app.view = await api(`/api/matches/${app.matchId}/move`, { method: "POST", token: app.token, body: { move } });
  } catch (e) { error = e; }
  app.busy = false;
  if (error) {
    renderStatus();
    showGameError({ kind: "human", message: error.message });
    return;
  }
  renderGame();
  if (app.auto) runAuto();
}

async function runAuto() {
  while (app.auto) {
    const status = app.view && app.view.status;
    if (status !== "running") {
      if (status !== "waiting_human") stopAuto();   // przy ruchu człowieka "Auto" czeka
      break;
    }
    await doStep();
    await sleep(Number($("#speed").value));
  }
}

function stopAuto() {
  app.auto = false;
  renderStatus();
}

function showGameError(err) {
  const box = $("#game-error");
  if (!err) { box.hidden = true; box.replaceChildren(); return; }
  const hints = {
    model_move: "Model nie podał poprawnego ruchu mimo ponowień. Dopracuj strategię albo ponów próbę.",
    llm_unavailable: "Dostawca chwilowo niedostępny (np. limit zapytań). Odczekaj chwilę i ponów.",
    llm_config: "Błąd konfiguracji (klucz/model). Popraw .env i zrestartuj serwer.",
    anti_cheat: "Anti-cheat zablokował zapytanie – obserwacja zawierała ukryte informacje. To błąd w kodzie.",
    illegal_move: "Silnik odrzucił ruch modelu.",
  };
  box.replaceChildren(
    el("strong", { text: err.player ? `Ruch gracza ${err.player} nie został wykonany. ` : "Błąd: " }),
    err.message,
    hints[err.kind] ? el("div", { class: "muted", text: hints[err.kind] }) : null,
    ["model_move", "llm_unavailable", "illegal_move", "http"].includes(err.kind)
      ? el("div", { class: "actions" }, el("button", { type: "button", class: "primary", onclick: () => { showGameError(null); doStep(); } }, "Ponów ruch"))
      : null,
  );
  box.hidden = false;
}

/* ---------------------------------------------------------------- render */

function renderGame() {
  const v = app.view;
  if (!v) return;
  const judge = v.judge || null;
  const obs = v.observation || null;

  // pasek meczu
  $("#game-meta").replaceChildren(
    el("span", { class: "tag accent", text: v.mode === "sandbox" ? "Sandbox" : "Zwykła rozgrywka" }),
    el("span", { class: "tag", text: `rozdanie ${v.round}${v.max_rounds ? ` / ${v.max_rounds}` : ""}` }),
    el("span", { class: "tag", text: PHASE_PL[v.phase] || v.phase }),
    v.human ? el("span", { class: "tag", text: `grasz jako ${v.human}` }) : el("span", { class: "tag", text: "widok sędziego" }),
    v.human ? el("span", { class: `tag${v.open_cards ? " accent" : ""}`, text: v.open_cards ? "karty jawne" : "karty ukryte" }) : null,
  );
  const toggle = $("#toggle-open");
  toggle.hidden = !(v.mode === "sandbox" && v.human);
  toggle.classList.toggle("on", Boolean(v.open_cards));
  toggle.textContent = v.open_cards ? "👁 Karty jawne · ukryj" : "👁 Pokaż karty przeciwnika";
  toggle.title = "Sandbox: przełącz widoczność kart (Jawna / Klasyczna). Bot zawsze widzi tylko swoje karty.";
  const lead = Math.max(...Object.values(v.scores));
  $("#scores").replaceChildren(...["A", "B"].map((p) => el("div", { class: `score${v.scores[p] === lead && lead > 0 ? " leading" : ""}` },
    el("span", { text: seatName(v, p) }), el("b", { text: v.scores[p] }))));

  // błędy i ostrzeżenia
  showGameError(v.last_error);
  const warn = $("#game-warnings");
  warn.hidden = !v.warnings || !v.warnings.length;
  if (!warn.hidden) warn.replaceChildren(el("strong", { text: "Uwagi do scenariusza:" }), el("ul", {}, v.warnings.map((w) => el("li", { text: w }))));

  // stół
  const bottom = v.human || "A";
  const top = bottom === "A" ? "B" : "A";
  $("#seat-top").replaceChildren(...renderSeat(v, top, judge, obs, "top"));
  $("#seat-bottom").replaceChildren(...renderSeat(v, bottom, judge, obs, "bottom"));
  $("#center").replaceChildren(...renderCenter(v, judge, obs));

  renderHumanControls(v, obs);
  renderPeek(v, judge);
  renderQuality(v);
  renderLog(v);
  renderStatus();
  schedulePoll();
}

/** Wyniki solvera liczą się w tle – odświeżaj widok, dopóki jakaś analiza trwa. */
function schedulePoll() {
  clearTimeout(app.pollTimer);
  const v = app.view;
  if (!v || !v.log.some((e) => e.analysis && e.analysis.status === "pending")) return;
  app.pollTimer = setTimeout(async () => {
    const typing = document.activeElement && $("#human-controls").contains(document.activeElement);
    if (app.busy || !app.matchId || typing) { schedulePoll(); return; }
    try {
      app.view = await api(`/api/matches/${app.matchId}?viewer=${app.viewer}`, { token: app.token });
      renderGame();
    } catch (_) { /* kolejna próba przy następnym ruchu */ }
  }, 1000);
}

function renderQuality(v) {
  const box = $("#quality");
  if (!v.analysis_enabled) {
    box.hidden = !(v.solver === null && app.solver && !app.solver.available);
    if (!box.hidden) box.replaceChildren(el("div", { class: "note", text: `Analiza solverem niedostępna: ${app.solver.reason}` }));
    return;
  }
  box.hidden = false;
  const rows = ["A", "B"].map((p) => {
    const s = v.analysis_summary[p];
    const text = s.analysed
      ? `śr. strata ${s.avg_loss.toFixed(1)} · optymalne ${s.optimal}/${s.analysed}`
      : "brak ocenionych ruchów";
    return el("div", { class: "row" }, el("span", { text: seatName(v, p) }),
      el("b", { text: text + (s.pending ? ` · liczę ${s.pending}` : "") }));
  });
  box.replaceChildren(el("div", { class: "note", text: "Jakość ruchów w rozgrywce wg solvera C++ (double-dummy):" }), ...rows);
}

const sameMove = (a, b) => a && b && a.card === b.card && a.action === b.action;
const moveLabel = (m) => cardLabel(m.card) + (m.action === "meld" ? " +meld" : "");
const fmtQ = (n) => (n > 0 ? "+" : "") + n.toFixed(1);
const Q_HELP = "Q = średnia wartość rozdania (wynik rozgrywającego − obrońcy, z perspektywy gracza) przy " +
  "optymalnej grze double-dummy, w losowych układach kart zgodnych z tym, co gracz wiedział. " +
  "Strata = max Q − Q zagranego ruchu.";

function renderAnalysis(a) {
  if (!a) return null;
  if (a.status === "pending") return el("div", { class: "analysis" }, el("div", { class: "line" }, el("span", { class: "badge wait", text: "⏳ solver liczy…" })));
  if (a.status === "hidden") return el("div", { class: "analysis" }, el("div", { class: "line" }, el("span", { class: "badge wait", text: "analiza ukryta do końca rozdania" })));
  if (a.status !== "done") return el("div", { class: "analysis muted", text: `analiza niedostępna: ${a.message || a.status}` });
  const loss = a.loss ?? 0;
  const cls = loss < 0.5 ? "ok" : loss < 10 ? "mid" : "bad";
  const line = el("div", { class: "line", title: Q_HELP },
    el("span", { class: `badge ${cls}`, text: loss < 0.5 ? "✓ optymalny" : `strata ${loss.toFixed(1)}` }),
    el("span", { text: `Q ${fmtQ(a.q)}` }),
    loss >= 0.5 ? el("span", { text: `· najlepszy ${moveLabel(a.best)} (Q ${fmtQ(a.max_q)})` }) : null);
  const table = el("table", { class: "qtable" }, a.moves.map((m) => el("tr", {
    class: [sameMove(m, a.model) && "played", sameMove(m, a.best) && "best"].filter(Boolean).join(" ") || null,
  }, el("td", { class: cardParts(m.card).red ? "red" : null, text: moveLabel(m) }),
     el("td", { text: `Q ${fmtQ(m.q)}` }), el("td", { class: "muted", text: `±${m.se.toFixed(1)}` }),
     el("td", { text: a.max_q - m.q < 0.05 ? "–" : `−${(a.max_q - m.q).toFixed(1)}` }))));
  return el("div", { class: "analysis" }, line,
    el("details", {}, el("summary", { text: `Q wszystkich ruchów (${a.worlds} światów, ${Math.round(a.solver_ms)} ms)` }), table));
}

function renderSeat(v, p, judge, obs, position) {
  const seat = v.seats[p];
  const pub = v.public;
  const isTurn = v.current_player === p && !["finished", "round_limit"].includes(v.status);
  const label = el("div", { class: "seat-label" },
    el("span", { class: "who", text: seatName(v, p) }),
    seat.kind === "llm" && seat.model ? el("span", { class: "pill", text: seat.model })
      : seat.kind === "heuristic" ? el("span", { class: "pill", text: "bez API" }) : null,
    seat.preset ? el("span", { class: "pill", text: `preset: ${seat.preset}` }) : null,
    pub.bid_winner === p && pub.contract ? el("span", { class: "pill", text: `rozgrywający · ${pub.contract}` }) : null,
    isTurn ? el("span", { class: "pill turn", text: "ma ruch" }) : null,
    judge && seat.kind !== "human"
      ? el("span", { class: "judge", text: v.human ? "karty jawne (sandbox) – bot o tym nie wie" : "karty widoczne tylko dla sędziego" })
      : null,
  );

  let cards;
  if (obs && p === v.human) {
    cards = renderHumanHand(v, obs);
  } else if (judge) {
    cards = judge.hands[p].map((c) => cardEl(c));
  } else {
    cards = Array.from({ length: pub.hand_sizes[p] }, () => cardEl(null));
  }
  const hand = el("div", { class: "hand" }, cards);
  return position === "top" ? [label, hand] : [hand, label];
}

function renderHumanHand(v, obs) {
  const myTurn = v.your_turn && !app.busy;
  const legal = obs.legal_moves || [];
  const playable = new Set(legal.filter((m) => m.action === "play_card" || m.action === "discard").map((m) => m.card));
  const meldable = new Set(legal.filter((m) => m.action === "meld").map((m) => m.card));
  const action = obs.phase === "discard" ? "discard" : "play_card";
  const clickable = myTurn && (obs.phase === "play" || obs.phase === "discard");
  return obs.hand.map((c) => cardEl(c, {
    legal: clickable && playable.has(c),
    dim: clickable && !playable.has(c),
    meld: clickable && meldable.has(c),
    onClick: clickable && playable.has(c) ? () => humanMove({ action, card: c }) : null,
    title: clickable && playable.has(c) ? (action === "discard" ? "Odłóż tę kartę" : "Zagraj tę kartę") : null,
  }));
}

function renderCenter(v, judge, obs) {
  const pub = v.public;
  const trump = pub.trump ? `${SUIT_SYMBOL[pub.trump]} ${SUIT_NAME[pub.trump]}` : "brak";
  const melds = ["A", "B"].map((p) => (pub.melds[p] || []).map((s) => `${p}: ${SUIT_SYMBOL[s]} ${MELD_POINTS[s]}`)).flat();
  const left = el("div", { class: "info" },
    el("div", {}, "Licytacja: ", el("b", { text: pub.bid ? `${pub.bid} (${pub.bid_winner || "–"})` : "–" })),
    el("div", {}, "Kontrakt: ", el("b", { text: pub.contract ? `${pub.contract} · ${pub.bid_winner}` : "–" })),
    el("div", {}, "Atut: ", el("b", { text: trump })),
    el("div", {}, "Meldunki: ", el("b", { text: melds.length ? melds.join(", ") : "–" })),
  );

  const trick = el("div", { class: "trick" });
  if (pub.table.length) {
    for (const t of pub.table) trick.append(el("div", { class: "trick-slot" }, cardEl(t.card), el("span", { text: t.player })));
  } else {
    trick.append(el("div", { class: "trick-empty", text: v.phase === "play" ? "nowa lewa" : PHASE_PL[v.phase] || "" }));
  }

  const right = el("div", { class: "info right" },
    el("div", {}, `Lewy: ${pub.tricks_played} / 10 · punkty A ${pub.trick_points.A} · B ${pub.trick_points.B}`),
    pub.revealed_musik ? el("div", {}, "Odkryty musik:") : null,
    pub.revealed_musik ? miniCards(pub.revealed_musik, "right") : null,
  );
  if (judge) {
    if (judge.chosen_musik === null) {
      right.append(el("div", {}, "Musiki (zakryte):"), miniCards(judge.musiki.flat(), "right"));
    } else {
      right.append(el("div", {}, "Niewybrany musik:"), miniCards(judge.musiki[1 - judge.chosen_musik], "right"));
    }
    if (judge.discards.length) right.append(el("div", {}, "Odłożone (rozgrywający):"), miniCards(judge.discards, "right"));
  } else if (obs && obs.your_discards) {
    right.append(el("div", {}, "Twoje odłożone:"), miniCards(obs.your_discards, "right"));
  }
  return [left, trick, right];
}

function renderHumanControls(v, obs) {
  const box = $("#human-controls");
  const show = obs && v.your_turn && v.status === "waiting_human";
  box.hidden = !show;
  if (!show) { box.replaceChildren(); return; }
  const legal = obs.legal_moves;
  const find = (action) => legal.find((m) => m.action === action);
  const parts = [];

  if (obs.phase === "bidding") {
    const bid = find("bid");
    parts.push(el("span", { class: "prompt", text: `Licytacja (stawka: ${obs.current_bid ?? "brak"}, Twój max: ${obs.your_max_bid})` }));
    if (bid) {
      const input = el("input", { type: "number", min: bid.min, max: bid.max, step: bid.step, value: bid.min, "aria-label": "Kwota" });
      parts.push(input, el("button", { type: "button", class: "primary", onclick: () => humanMove({ action: "bid", value: Number(input.value) }) }, "Licytuj"));
    }
    if (find("pass")) parts.push(el("button", { type: "button", onclick: () => humanMove({ action: "pass" }) }, "Pas"));
  } else if (obs.phase === "choose_musik") {
    parts.push(el("span", { class: "prompt", text: "Wygrałeś licytację – wybierz musik:" }));
    for (const m of legal) parts.push(el("button", { type: "button", class: "primary", onclick: () => humanMove({ action: "choose_musik", value: m.value }) }, `Musik ${m.value + 1}`));
  } else if (obs.phase === "discard") {
    parts.push(el("span", { class: "prompt", text: `Kliknij kartę do odłożenia (masz ${obs.hand.length}, zostaw 10).` }));
  } else if (obs.phase === "declare") {
    const d = find("declare");
    const input = el("input", { type: "number", min: d.min, max: d.max, step: d.step, value: d.min, "aria-label": "Kontrakt" });
    parts.push(el("span", { class: "prompt", text: `Deklaracja kontraktu (${d.min}–${d.max})` }), input,
      el("button", { type: "button", class: "primary", onclick: () => humanMove({ action: "declare", value: Number(input.value) }) }, "Deklaruj"));
  } else if (obs.phase === "play") {
    parts.push(el("span", { class: "prompt", text: obs.table.length ? "Dołóż kartę (podświetlone są dozwolone)." : "Wyjdź do lewy – kliknij kartę." }));
    for (const m of legal.filter((x) => x.action === "meld")) {
      const suit = m.card.split("_")[1];
      parts.push(el("button", { type: "button", class: "primary", onclick: () => humanMove(m) },
        `Melduj ${SUIT_SYMBOL[suit]} (+${MELD_POINTS[suit]}) i zagraj ${cardLabel(m.card)}`));
    }
  }
  box.replaceChildren(...parts);
}

function renderPeek(v, judge) {
  const peek = $("#peek");
  const observations = judge && judge.bot_observations;
  const bots = observations ? Object.keys(observations).filter((p) => observations[p]) : [];
  peek.hidden = !bots.length;
  if (!bots.length) return;
  if (!bots.includes(app.peek)) app.peek = bots[0];
  $("#peek-tabs").replaceChildren(...bots.map((p) => el("button", {
    type: "button", class: p === app.peek ? "active" : "",
    onclick: () => { app.peek = p; renderPeek(app.view, app.view.judge); },
  }, `Bot ${p} · kontekst ${v.seats[p].context_id || "-"}`)));
  $("#peek-json").textContent = JSON.stringify(observations[app.peek], null, 2);
}

function renderLog(v) {
  const list = $("#log");
  const items = [];
  const results = Object.fromEntries((v.round_results || []).map((r) => [r.round, r]));
  let lastRound = null;
  const pushResult = (round) => {
    const r = results[round];
    if (!r) return;
    const made = r.contract_made ? "ugrał" : "nie ugrał";
    items.push(el("li", { class: "round-end" },
      el("div", { class: "who", text: `Koniec rozdania ${round}` }),
      el("div", { text: `${r.bidder} ${made} ${r.contract} · zmiana A ${signed(r.score_change.A)}, B ${signed(r.score_change.B)}` }),
      el("div", { class: "meta", text: `wynik: A ${r.scores.A} · B ${r.scores.B}` })));
  };
  for (const e of v.log) {
    if (lastRound !== null && e.round !== lastRound) pushResult(lastRound);
    lastRound = e.round;
    const meta = e.kind === "llm"
      ? `${e.model}${e.fallback_used ? " (zapasowy)" : ""} · ${e.tokens} tok · ${e.ms} ms${e.attempts > 1 ? ` · ${e.attempts} próby` : ""}`
      : KIND_PL[e.kind];
    items.push(el("li", { class: `p-${e.player}` },
      el("div", { class: "where", text: `R${e.round} · ${PHASE_PL[e.phase] || e.phase}` }),
      el("div", {}, el("span", { class: "who", text: `${seatName(v, e.player)}: ` }), e.text),
      e.reasoning ? el("div", { class: "why", text: e.reasoning.startsWith("(") ? e.reasoning : `„${e.reasoning}”` }) : null,
      renderAnalysis(e.analysis),
      el("div", { class: "meta", text: meta })));
  }
  if (lastRound !== null && lastRound !== v.round) pushResult(lastRound);
  if (["finished", "round_limit"].includes(v.status) && lastRound === v.round) pushResult(lastRound);
  if (!items.length) items.push(el("li", { class: "empty", text: "Jeszcze nikt nie wykonał ruchu." }));
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
  list.replaceChildren(...items);
  if (nearBottom || v.log.length !== app.logLen) list.scrollTop = list.scrollHeight;   // nie przewijaj, gdy czytasz starsze wpisy
  app.logLen = v.log.length;
}

function signed(n) { return n > 0 ? `+${n}` : String(n); }

function renderStatus() {
  const v = app.view;
  const status = $("#status");
  const stepBtn = $("#step");
  const autoBtn = $("#auto");
  if (!v) return;
  const running = v.status === "running";
  stepBtn.disabled = app.busy || !running;
  autoBtn.disabled = !app.auto && !running && v.status !== "waiting_human";
  autoBtn.textContent = app.auto ? "⏸ Pauza" : "▶ Auto";
  status.classList.toggle("busy", app.busy);
  const current = v.current_player ? seatName(v, v.current_player) : "";
  const texts = {
    running: app.busy ? `${current} myśli…` : `Ruch: ${current}`,
    waiting_human: "Twój ruch – wybierz poniżej.",
    error: "Ruch wstrzymany – zobacz komunikat powyżej.",
    finished: `Koniec partii – wygrywa ${v.winner ? seatName(v, v.winner) : "–"}!`,
    round_limit: `Osiągnięto limit ${v.max_rounds} rozdań. Wynik: A ${v.scores.A} · B ${v.scores.B}.`,
  };
  status.textContent = app.busy && v.status !== "running" ? "Wysyłam ruch…" : texts[v.status] || v.status;
  $("#controls .bot-controls").hidden = false;
  if (v.human && v.status === "waiting_human" && !app.busy) renderHumanControls(v, v.observation);
}

/* ---------------------------------------------------------------- start */

$("#step").addEventListener("click", doStep);
$("#auto").addEventListener("click", () => {
  app.auto = !app.auto;
  renderStatus();
  if (app.auto) runAuto();
});
$("#toggle-open").addEventListener("click", async () => {
  if (!app.view) return;
  try {
    app.view = await api(`/api/matches/${app.matchId}/visibility`, { method: "POST", token: app.token, body: { open: !app.view.open_cards } });
    renderGame();
  } catch (e) { showGameError({ kind: "http", message: e.message }); }
});
$("#back").addEventListener("click", () => {
  stopAuto();
  clearTimeout(app.pollTimer);
  if (app.matchId) api(`/api/matches/${app.matchId}`, { method: "DELETE" }).catch(() => {});
  Object.assign(app, { matchId: null, token: null, human: null, view: null });
  $("#game").hidden = true;
  $("#setup").hidden = false;
});

initSetup();
