import "./style.css";
import { DataSet, Network } from "vis-network/standalone";
import {
  ensureGlobe,
  ensureMap,
  linksFromTriples,
  resizeGlobe,
  resizeMap,
  setGeoCallbacks,
  setGeoLinks,
  setLinksVisible,
  setPlaceMode,
  syncMarkers,
} from "./geo.js";
import { applyI18n, LANG_META, t } from "./i18n.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";

let documentId = null;
let historyId = null;
let documentText = "";
let lastTriples = [];
let lastEngine = null;
let lastSource = null;
let allNodes = [];
let graphNodes = [];
let graphEdges = [];
let geoMarkers = [];
let geoLinks = [];
let currentView = "graph";
let disabled = new Set();
let mutedNodes = new Set(); // transparent nodes
let mutedEdgeIds = new Set(); // transparent (off) edges
let network = null;
let highlightNodes = new Set();
let highlightEdges = new Set();
let lastDebug = {
  prompt_user: "",
  raw_response: "",
  parsed_json: [],
  error: "",
};
let networkBound = false;
let historyItems = [];
let ctxEntityName = null;
let entityName = null;
/** Hide node/edge text labels (useful in fullscreen). */
let hideGraphLabels = false;

function engineLabel(engine) {
  const e = (engine || "").toLowerCase();
  if (e === "microgpt") return "MicroGPT";
  if (e === "deepseek-v4") return "DeepSeek V4 Pro";
  if (e === "deepseek") return "DeepSeek 3.2";
  if (e === "gigachat") return "GigaChat";
  if (e === "auto") return "Авто";
  return "Gemma";
}

function isDeepseekEngine(engine) {
  const e = (engine || "").toLowerCase();
  return e === "deepseek" || e === "deepseek-v4";
}

function isBridgeEngine(engine) {
  const e = (engine || "").toLowerCase();
  return isDeepseekEngine(e) || e === "gigachat";
}

function resolveBridgeEngine(engine) {
  const e = (engine || "").toLowerCase();
  if (isBridgeEngine(e)) return e;
  return "deepseek-v4";
}

/** Color matrix: origin × kind (explicit / hidden / false). */
function edgeStyle(kind, origin) {
  const k = (kind || "explicit").toLowerCase();
  const o = (origin || "base").toLowerCase();
  const styles = {
    base: {
      explicit: { color: "#4e6676", font: "#8a9aa6", dashes: false, width: 1 },
      hidden: { color: "#2ec4b6", font: "#5fd9cd", dashes: [7, 5], width: 2.4 },
      false: { color: "#ff5c8a", font: "#ff8aad", dashes: [4, 4], width: 2.6 },
    },
    append: {
      explicit: { color: "#e9c46a", font: "#f0d78c", dashes: false, width: 2.3 },
      hidden: { color: "#7eb563", font: "#a3d48a", dashes: [7, 5], width: 2.5 },
      false: { color: "#f4a261", font: "#f7b980", dashes: [4, 4], width: 2.6 },
    },
    bridge: {
      explicit: { color: "#a78bfa", font: "#c4b5fd", dashes: false, width: 2.5 },
      hidden: { color: "#60a5fa", font: "#93c5fd", dashes: [7, 5], width: 2.5 },
      false: { color: "#f472b6", font: "#f9a8d4", dashes: [4, 4], width: 2.6 },
    },
  };
  const pack = styles[o] || styles.base;
  return pack[k] || pack.explicit;
}

function isNetworkTextEngine(engine) {
  const e = (engine || "").toLowerCase();
  return isDeepseekEngine(e) || e === "gigachat";
}

const LANGS = LANG_META;

function selectedLanguage() {
  const v = (el.language && el.language.value) || "ru";
  return LANGS[v] ? v : "ru";
}

function setLanguage(code, { persist = true } = {}) {
  const lang = LANGS[code] ? code : "ru";
  if (el.language) el.language.value = lang;
  const meta = LANGS[lang];
  if (el.langBtn && meta) {
    const flag = el.langBtn.querySelector(".lang-flag");
    const name = el.langBtn.querySelector(".lang-name");
    if (flag) {
      if (flag.tagName === "IMG") flag.src = meta.flag;
      else flag.textContent = meta.flag;
    }
    if (name) name.textContent = meta.name;
  }
  if (el.langMenu) {
    el.langMenu.querySelectorAll('[role="option"]').forEach((opt) => {
      opt.setAttribute(
        "aria-selected",
        opt.getAttribute("data-lang") === lang ? "true" : "false"
      );
    });
  }
  applyI18n(lang);
  syncGenerateButtonLabel();
  syncGraphLabelsBtn();
  if (el.analyze) {
    const label = el.analyze.querySelector(".btn-label");
    if (label && !el.analyze.classList.contains("is-loading")) {
      label.textContent = t("analyze");
    }
  }
  if (el.graphFs && !document.fullscreenElement) {
    el.graphFs.textContent = t("fullscreen");
  }
  if (persist) {
    try {
      localStorage.setItem("grafix_language", lang);
    } catch (_) {
      /* ignore */
    }
  }
  closeLangMenu();
}

function openLangMenu() {
  if (!el.langMenu || !el.langBtn || !el.langPick) return;
  el.langMenu.hidden = false;
  el.langBtn.setAttribute("aria-expanded", "true");
  el.langPick.classList.add("is-open");
}

function closeLangMenu() {
  if (!el.langMenu || !el.langBtn || !el.langPick) return;
  el.langMenu.hidden = true;
  el.langBtn.setAttribute("aria-expanded", "false");
  el.langPick.classList.remove("is-open");
}

function initLanguagePicker() {
  let saved = "ru";
  try {
    saved = localStorage.getItem("grafix_language") || "ru";
  } catch (_) {
    saved = "ru";
  }
  setLanguage(saved, { persist: false });
  if (el.langBtn) {
    el.langBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (el.langMenu && el.langMenu.hidden) openLangMenu();
      else closeLangMenu();
    });
  }
  if (el.langMenu) {
    el.langMenu.addEventListener("click", (e) => {
      const opt = e.target.closest('[role="option"]');
      if (!opt) return;
      setLanguage(opt.getAttribute("data-lang"));
    });
  }
  document.addEventListener("click", (e) => {
    if (el.langPick && !el.langPick.contains(e.target)) closeLangMenu();
  });
  initWorkspaceLangVisibility();
}

function showWorkspaceLang(on) {
  if (!el.workspaceLang) return;
  el.workspaceLang.hidden = !on;
  if (!on) closeLangMenu();
}

function initWorkspaceLangVisibility() {
  const workspace = el.workspace || document.getElementById("workspace");
  if (!workspace || !el.workspaceLang) return;

  const reveal = () => showWorkspaceLang(true);

  // Only after leaving the first screen (hero) — when workspace enters view
  if (typeof IntersectionObserver === "function") {
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && entry.intersectionRatio > 0.12) {
            reveal();
          } else if (!entry.isIntersecting && window.scrollY < 80) {
            showWorkspaceLang(false);
          }
        }
      },
      { root: null, threshold: [0.12, 0.25] }
    );
    io.observe(workspace);
  }

  // Click «Open workspace» / hash navigation
  document.querySelectorAll('a[href="#workspace"]').forEach((a) => {
    a.addEventListener("click", () => {
      // show after scroll settles into workspace
      setTimeout(reveal, 350);
    });
  });
  if (location.hash === "#workspace") reveal();
  window.addEventListener("hashchange", () => {
    if (location.hash === "#workspace") reveal();
  });
}

function selectedTextEngine() {
  const sel = el.qaEngine || el.entityQaEngine;
  const e = (sel ? sel.value : "gemma") || "gemma";
  return isNetworkTextEngine(e) ? e : "gemma";
}

function setQaEngine(value, source) {
  const next = isNetworkTextEngine(value) ? value : "gemma";
  if (el.qaEngine && source !== el.qaEngine) el.qaEngine.value = next;
  if (el.entityQaEngine && source !== el.entityQaEngine) el.entityQaEngine.value = next;
  syncGenerateButtonLabel();
}

function syncGenerateButtonLabel() {
  const btn = el.entityGemma;
  if (!btn || btn.classList.contains("is-loading")) return;
  const label = btn.querySelector(".btn-label");
  if (label) {
    label.textContent = `${t("generate")} (${engineLabel(selectedTextEngine())})`;
  }
}

const el = {
  text: document.getElementById("text-input"),
  analyze: document.getElementById("btn-analyze"),
  appendText: document.getElementById("append-text"),
  appendBtn: document.getElementById("btn-append"),
  bridgeBtn: document.getElementById("btn-bridge"),
  appendMeta: document.getElementById("append-meta"),
  meta: document.getElementById("extract-meta"),
  engine: document.getElementById("engine-select"),
  language: document.getElementById("language-select"),
  langBtn: document.getElementById("lang-btn"),
  langMenu: document.getElementById("lang-menu"),
  langPick: document.getElementById("lang-pick"),
  workspaceLang: document.getElementById("workspace-lang"),
  workspace: document.getElementById("workspace"),
  qaEngine: document.getElementById("qa-engine-select"),
  entityQaEngine: document.getElementById("entity-qa-engine-select"),
  reasoningToggle: document.getElementById("reasoning-toggle"),
  question: document.getElementById("question-input"),
  ask: document.getElementById("btn-ask"),
  answer: document.getElementById("answer"),
  graph: document.getElementById("graph"),
  units: document.getElementById("unit-list"),
  lost: document.getElementById("lost-list"),
  docId: document.getElementById("doc-id"),
  add: document.getElementById("btn-add"),
  editS: document.getElementById("edit-s"),
  editR: document.getElementById("edit-r"),
  editO: document.getElementById("edit-o"),
  editForm: document.getElementById("edit-form"),
  editFormHome: document.getElementById("edit-form-home"),
  editMeta: document.getElementById("edit-meta"),
  fsEditDock: document.getElementById("fs-edit-dock"),
  fsEditSlot: document.getElementById("fs-edit-slot"),
  debugBtn: document.getElementById("btn-debug"),
  debugClose: document.getElementById("btn-debug-close"),
  debugDrawer: document.getElementById("debug-drawer"),
  debugPrompt: document.getElementById("debug-prompt"),
  debugRaw: document.getElementById("debug-raw"),
  debugParsed: document.getElementById("debug-parsed"),
  debugError: document.getElementById("debug-error"),
  saveGraph: document.getElementById("btn-save-graph"),
  graphFs: document.getElementById("btn-graph-fs"),
  graphFsExit: document.getElementById("btn-graph-fs-exit"),
  graphLabelsBtn: document.getElementById("btn-graph-labels"),
  graphStage: document.getElementById("graph-stage"),
  historyTabs: document.getElementById("history-tabs"),
  historyCount: document.getElementById("history-count"),
  historyEmpty: document.getElementById("history-empty"),
  historySelect: document.getElementById("history-select"),
  historyPanel: document.getElementById("history-panel"),
  ctxMenu: document.getElementById("ctx-menu"),
  ctxAbout: document.getElementById("ctx-about"),
  ctxDelete: document.getElementById("ctx-delete"),
  entityDrawer: document.getElementById("entity-drawer"),
  entityTitle: document.getElementById("entity-title"),
  entityClose: document.getElementById("btn-entity-close"),
  entityRelations: document.getElementById("entity-relations"),
  entityComment: document.getElementById("entity-comment"),
  entitySave: document.getElementById("btn-entity-save"),
  entityGemma: document.getElementById("btn-entity-gemma"),
  entityMeta: document.getElementById("entity-meta"),
  fileInput: document.getElementById("file-input"),
  fileMeta: document.getElementById("file-meta"),
  fileAppend: document.getElementById("file-append"),
  dropZone: document.getElementById("drop-zone"),
  askWebSearch: document.getElementById("ask-web-search"),
  askWebSources: document.getElementById("ask-web-sources"),
  entityWebSearch: document.getElementById("entity-web-search"),
  entityWebSources: document.getElementById("entity-web-sources"),
  edgeLegend: document.getElementById("edge-legend"),
  viewGraph: document.getElementById("view-graph"),
  viewMap: document.getElementById("view-map"),
  viewGlobe: document.getElementById("view-globe"),
  mapEntity: document.getElementById("map-entity-select"),
  mapSearch: document.getElementById("map-search-input"),
  mapPlaceMode: document.getElementById("map-place-mode"),
  mapUseDeepseek: document.getElementById("map-use-deepseek"),
  mapGeocode: document.getElementById("btn-map-geocode"),
  mapGeoGraph: document.getElementById("btn-map-geo-graph"),
  mapGeoAuto: document.getElementById("map-geo-auto"),
  mapShowLinks: document.getElementById("map-show-links"),
  globeShowLinks: document.getElementById("globe-show-links"),
  mapGeoEngine: document.getElementById("map-geo-engine"),
  mapMeta: document.getElementById("map-meta"),
  globeMeta: document.getElementById("globe-meta"),
  tabGraph: document.getElementById("tab-view-graph"),
  tabMap: document.getElementById("tab-view-map"),
  tabGlobe: document.getElementById("tab-view-globe"),
};

async function post(url, body) {
  const res = await fetch(`${API_BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail || data.error || res.statusText;
    throw new Error(msg);
  }
  return data;
}

async function api(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail || data.error || res.statusText;
    throw new Error(msg);
  }
  return data;
}

function setDebug(debug, fallbackParsed) {
  const d = debug || {};
  lastDebug = {
    prompt_user: d.prompt_user || "",
    raw_response: d.raw_response || "",
    parsed_json: d.parsed_json || fallbackParsed || [],
    error: d.error || "",
    comments_raw: d.comments_raw || "",
    comments_error: d.comments_error || "",
    comments_missing: d.comments_missing || [],
  };
  if (el.debugPrompt) el.debugPrompt.textContent = lastDebug.prompt_user || "—";
  let raw = lastDebug.raw_response || "—";
  if (d.reasoning) {
    raw = "=== Reasoning ===\n" + d.reasoning + "\n\n=== Ответ ===\n" + raw;
  }
  if (Array.isArray(d.contradictions) && d.contradictions.length) {
    raw +=
      "\n\n=== Противоречия ===\n" + JSON.stringify(d.contradictions, null, 2);
  }
  if (Array.isArray(d.entities) && d.entities.length) {
    raw += "\n\n=== Entities ===\n" + JSON.stringify(d.entities, null, 2);
  }
  if (lastDebug.comments_raw) {
    raw += "\n\n=== Комментарии к сущностям ===\n" + lastDebug.comments_raw;
  }
  if (d.geo_raw) {
    raw += "\n\n=== Географ (места/события) ===\n" + d.geo_raw;
  }
  if (el.debugRaw) el.debugRaw.textContent = raw;
  if (el.debugParsed) {
    el.debugParsed.textContent = lastDebug.parsed_json.length
      ? JSON.stringify(lastDebug.parsed_json, null, 2)
      : "—";
  }
  const errs = [lastDebug.error, lastDebug.comments_error, d.geo_error].filter(Boolean);
  if (lastDebug.comments_missing && lastDebug.comments_missing.length) {
    errs.push("без комментария: " + lastDebug.comments_missing.join(", "));
  }
  if (el.debugError) el.debugError.textContent = errs.join(" · ");
}

function openDebug() {
  if (el.debugDrawer) el.debugDrawer.hidden = false;
}

function closeDebug() {
  if (el.debugDrawer) el.debugDrawer.hidden = true;
}

function renderWebSources(container, results, error) {
  if (!container) return;
  container.innerHTML = "";
  const items = results || [];
  if (!items.length && !error) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  if (error && !items.length) {
    const li = document.createElement("li");
    li.textContent = `Поиск: ${error}`;
    container.appendChild(li);
    return;
  }
  for (const r of items) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = r.url || "#";
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = r.title || r.url || "Источник";
    li.appendChild(a);
    if (r.snippet) {
      const snip = document.createElement("span");
      snip.className = "web-snip";
      snip.textContent = r.snippet;
      li.appendChild(snip);
    }
    container.appendChild(li);
  }
  if (error) {
    const li = document.createElement("li");
    li.textContent = `Поиск: ${error}`;
    container.appendChild(li);
  }
}

function hideCtxMenu() {
  if (el.ctxMenu) el.ctxMenu.hidden = true;
  ctxEntityName = null;
}

function showCtxMenu(x, y, name) {
  if (!el.ctxMenu) return;
  ctxEntityName = name;
  el.ctxMenu.hidden = false;
  const pad = 8;
  const stage = el.graphStage;
  const fs = isGraphFullscreen() && stage;
  const bounds = fs ? stage.getBoundingClientRect() : {
    left: 0,
    top: 0,
    right: window.innerWidth,
    bottom: window.innerHeight,
    width: window.innerWidth,
    height: window.innerHeight,
  };
  // Measure after show
  const rect = el.ctxMenu.getBoundingClientRect();
  const left = Math.min(Math.max(bounds.left + pad, x), bounds.right - rect.width - pad);
  const top = Math.min(Math.max(bounds.top + pad, y), bounds.bottom - rect.height - pad);
  el.ctxMenu.style.left = `${left}px`;
  el.ctxMenu.style.top = `${top}px`;
}

function formatRelation(rel, entity) {
  const neighbor = rel.neighbor || "?";
  const relation = rel.relation || "?";
  const kind = (rel.kind || "explicit").toLowerCase();
  const tag =
    kind === "hidden" ? " · скрытая" : kind === "false" ? " · ложная" : "";
  const conf =
    rel.confidence != null && rel.confidence !== ""
      ? ` · ${Number(rel.confidence).toFixed(2)}`
      : "";
  let line;
  if (rel.direction === "in") {
    line = `${neighbor} —[${relation}]→ ${entity}${tag}${conf}`;
  } else {
    line = `${entity} —[${relation}]→ ${neighbor}${tag}${conf}`;
  }
  if (rel.evidence) line += `\n  ↳ ${rel.evidence}`;
  return line;
}

function fillEntityPanel(data) {
  entityName = data.name || entityName;
  if (el.entityTitle) el.entityTitle.textContent = entityName || "О сущности";
  if (el.entityComment) el.entityComment.value = data.comment || "";
  if (el.entityRelations) {
    el.entityRelations.innerHTML = "";
    for (const rel of data.relations || []) {
      const li = document.createElement("li");
      const kind = (rel.kind || "explicit").toLowerCase();
      if (kind === "hidden") li.classList.add("rel-hidden");
      if (kind === "false") li.classList.add("rel-false");
      const origin = (rel.origin || "base").toLowerCase();
      if (origin === "append") li.classList.add("rel-append");
      if (origin === "bridge") li.classList.add("rel-bridge");
      if (kind === "hidden" && origin === "append") li.classList.add("rel-append-hidden");
      if (kind === "false" && origin === "append") li.classList.add("rel-append-false");
      if (kind === "hidden" && origin === "bridge") li.classList.add("rel-bridge-hidden");
      if (kind === "false" && origin === "bridge") li.classList.add("rel-bridge-false");
      li.textContent = formatRelation(rel, entityName);
      el.entityRelations.appendChild(li);
    }
  }
  if (data.highlight_nodes) highlightNodes = new Set(data.highlight_nodes);
  if (data.highlight_edges) highlightEdges = new Set(data.highlight_edges);
  drawGraph(graphNodes, graphEdges);
}

async function openEntityPanel(name) {
  hideCtxMenu();
  if (!documentId) {
    el.meta.textContent = "Сначала проанализируйте текст";
    return;
  }
  entityName = name;
  if (el.entityMeta) el.entityMeta.textContent = "Загрузка…";
  if (el.entityDrawer) el.entityDrawer.hidden = false;
  try {
    const q = new URLSearchParams({
      name,
      document_id: documentId,
    });
    const data = await api(`/api/entity?${q.toString()}`);
    fillEntityPanel(data);
    if (el.entityMeta) {
      el.entityMeta.textContent = `связей: ${(data.relations || []).length}`;
    }
    renderWebSources(el.entityWebSources, [], null);
  } catch (err) {
    if (el.entityMeta) el.entityMeta.textContent = String(err.message || err);
  }
}

function closeEntityPanel() {
  if (el.entityDrawer) el.entityDrawer.hidden = true;
}

async function saveEntityComment() {
  if (!entityName || !documentId) return;
  try {
    if (el.entityMeta) el.entityMeta.textContent = "Сохранение…";
    const data = await post("/api/entity/comment", {
      name: entityName,
      comment: el.entityComment ? el.entityComment.value : "",
      document_id: documentId,
      history_id: historyId,
    });
    fillEntityPanel(data);
    if (el.entityMeta) el.entityMeta.textContent = "Комментарий сохранён";
  } catch (err) {
    if (el.entityMeta) el.entityMeta.textContent = String(err.message || err);
  }
}

async function generateEntityComment() {
  if (!entityName || !documentId) return;
  const btn = el.entityGemma;
  const label = btn ? btn.querySelector(".btn-label") : null;
  const textEngine = selectedTextEngine();
  const modelName = engineLabel(textEngine);
  if (btn) {
    btn.disabled = true;
    btn.classList.add("is-loading");
  }
  if (label) label.textContent = "Генерация…";
  if (el.entityMeta) el.entityMeta.textContent = `${modelName} пишет комментарий…`;
  try {
    const useWeb = !!(el.entityWebSearch && el.entityWebSearch.checked);
    if (useWeb && el.entityMeta) {
      el.entityMeta.textContent = `Поиск + ${modelName}…`;
    }
    const data = await post("/api/entity/comment/generate", {
      name: entityName,
      document_id: documentId,
      document_text: documentText || el.text.value,
      history_id: historyId,
      // Только вставить в поле — сохранение через кнопку «Сохранить»
      save: false,
      web_search: useWeb,
      engine: textEngine,
      reasoning: selectedReasoning(),
    });
    const comment = (data.comment || "").trim();
    fillEntityPanel({ ...data, comment });
    if (el.entityComment) {
      el.entityComment.value = comment;
      el.entityComment.classList.remove("is-filled");
      void el.entityComment.offsetWidth;
      if (comment) el.entityComment.classList.add("is-filled");
      el.entityComment.focus();
      el.entityComment.scrollTop = 0;
    }
    renderWebSources(el.entityWebSources, data.web_results, data.web_error);
    if (data.debug) setDebug(data.debug, [{ comment }]);
    let meta;
    if (comment) {
      meta = "Комментарий вставлен в поле — нажми «Сохранить»";
    } else {
      meta = data.debug?.error || `${modelName} вернула пустой комментарий — открой Debug-лог`;
    }
    if (useWeb) {
      const n = (data.web_results || []).length;
      meta += data.web_error ? " · поиск: ошибка" : ` · источников: ${n}`;
    }
    if (el.entityMeta) el.entityMeta.textContent = meta;
  } catch (err) {
    if (el.entityMeta) el.entityMeta.textContent = String(err.message || err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("is-loading");
    }
    syncGenerateButtonLabel();
  }
}

function formatHistoryTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function applyPayload(data) {
  if (data.document_id) {
    documentId = data.document_id;
    el.docId.textContent = documentId.slice(0, 8) + "…";
  }
  if (data.history_id) historyId = data.history_id;
  if (typeof data.text === "string") {
    documentText = data.text;
    if (el.text && data.text !== el.text.value) el.text.value = data.text;
  }
  if (Array.isArray(data.triples)) lastTriples = data.triples;
  if (data.engine) lastEngine = data.engine;
  if (data.source) lastSource = data.source;
  if (data.nodes) allNodes = data.nodes;
  if (Array.isArray(data.markers)) {
    geoMarkers = data.markers;
  }
  if (Array.isArray(data.geo_links)) {
    geoLinks = data.geo_links;
  } else if (Array.isArray(data.markers) && Array.isArray(lastTriples)) {
    geoLinks = linksFromTriples(geoMarkers, lastTriples);
  }
  if (Array.isArray(data.markers) || Array.isArray(data.geo_links)) {
    refreshGeoViews();
  }
  if (data.highlight_nodes) highlightNodes = new Set(data.highlight_nodes);
  else highlightNodes = new Set();
  if (data.highlight_edges) highlightEdges = new Set(data.highlight_edges);
  else highlightEdges = new Set();
  renderUnits();
  fillMapEntitySelect();
  refreshVisibility();
  renderHistoryTabs();
}

function renderUnits() {
  el.units.innerHTML = "";
  for (const n of allNodes) {
    const li = document.createElement("li");
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !disabled.has(n.id);
    cb.addEventListener("change", () => {
      if (cb.checked) disabled.delete(n.id);
      else disabled.add(n.id);
      refreshVisibility();
    });
    label.appendChild(cb);
    label.appendChild(document.createTextNode(n.label || n.id));
    li.appendChild(label);
    el.units.appendChild(li);
  }
}

async function refreshVisibility() {
  if (!documentId) {
    drawGraph([], []);
    return;
  }
  const data = await post("/api/toggle", {
    document_id: documentId,
    disabled: [...disabled],
  });
  el.lost.innerHTML = "";
  for (const e of data.lost_edges || []) {
    const li = document.createElement("li");
    li.textContent = `${e.source} —[${e.relation}]→ ${e.target}`;
    el.lost.appendChild(li);
  }
  drawGraph(data.nodes || [], data.edges || []);
}

function renderHistoryTabs() {
  if (el.historyCount) {
    el.historyCount.textContent = historyItems.length ? `${historyItems.length}` : "";
  }
  if (el.historyEmpty) el.historyEmpty.hidden = historyItems.length > 0;

  if (el.historyTabs) {
    el.historyTabs.innerHTML = "";
    for (const item of historyItems) {
      const tab = document.createElement("div");
      tab.className = "history-tab" + (item.id === historyId ? " is-active" : "");
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", item.id === historyId ? "true" : "false");

      const main = document.createElement("button");
      main.type = "button";
      main.className = "history-tab-main";
      main.title = item.preview || item.title || "";
      main.innerHTML =
        `<span class="history-tab-title"></span>` +
        `<span class="history-tab-meta"></span>`;
      main.querySelector(".history-tab-title").textContent = item.title || "Граф";
      main.querySelector(".history-tab-meta").textContent =
        `${formatHistoryTime(item.updated_at)} · ${item.triple_count ?? 0} св.`;
      main.addEventListener("click", () => loadHistoryItem(item.id));

      const del = document.createElement("button");
      del.type = "button";
      del.className = "history-tab-del";
      del.title = "Удалить";
      del.setAttribute("aria-label", "Удалить из истории");
      del.textContent = "×";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteHistoryItem(item.id);
      });

      tab.appendChild(main);
      tab.appendChild(del);
      el.historyTabs.appendChild(tab);
    }
  }

  if (el.historySelect) {
    const prev = el.historySelect.value;
    el.historySelect.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = historyItems.length ? "— выбрать граф —" : "— история пуста —";
    el.historySelect.appendChild(opt0);
    for (const item of historyItems) {
      const opt = document.createElement("option");
      opt.value = item.id;
      const when = formatHistoryTime(item.updated_at);
      opt.textContent = `${item.title || "Граф"} · ${when} · ${item.triple_count ?? 0} св.`;
      el.historySelect.appendChild(opt);
    }
    const keep = historyId || prev;
    if (keep && [...el.historySelect.options].some((o) => o.value === keep)) {
      el.historySelect.value = keep;
    }
  }
}

async function refreshHistoryList() {
  try {
    const data = await api("/api/graphs");
    historyItems = data.items || [];
    renderHistoryTabs();
  } catch (err) {
    if (el.meta) el.meta.textContent = String(err.message || err);
  }
}

async function loadHistoryItem(id) {
  try {
    el.meta.textContent = "Загрузка истории…";
    disabled = new Set();
    mutedNodes = new Set();
    mutedEdgeIds = new Set();
    const data = await api(`/api/graphs/${encodeURIComponent(id)}?restore=true`);
    historyId = data.history_id || id;
    setDebug({}, data.triples || []);
    applyPayload(data);
    el.meta.textContent =
      `история · троек: ${(data.triples || []).length}` +
      (data.source ? ` · источник: ${data.source}` : "") +
      (data.engine ? ` · ${data.engine}` : "");
    await refreshHistoryList();
  } catch (err) {
    el.meta.textContent = String(err.message || err);
  }
}

async function deleteHistoryItem(id) {
  try {
    await api(`/api/graphs/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (historyId === id) historyId = null;
    await refreshHistoryList();
  } catch (err) {
    el.meta.textContent = String(err.message || err);
  }
}

async function saveCurrentGraph() {
  if (!documentId && !(el.text.value || "").trim()) {
    el.meta.textContent = "Нечего сохранять — сначала проанализируйте текст";
    return;
  }
  try {
    const data = await post("/api/graphs", {
      document_id: documentId,
      text: documentText || el.text.value,
      triples: lastTriples,
      nodes: allNodes,
      edges: graphEdges.length ? graphEdges : undefined,
      markers: geoMarkers,
      engine: lastEngine || (el.engine ? el.engine.value : null),
      source: lastSource,
      history_id: historyId,
    });
    historyId = data.history_id || (data.history && data.history.id) || historyId;
    if (data.document_id) documentId = data.document_id;
    el.meta.textContent = "Сохранено в историю";
    await refreshHistoryList();
  } catch (err) {
    el.meta.textContent = String(err.message || err);
  }
}

function isGraphFullscreen() {
  const stage = el.graphStage;
  if (!stage) return false;
  return document.fullscreenElement === stage || document.webkitFullscreenElement === stage;
}

function syncGraphFsUi() {
  const on = isGraphFullscreen();
  if (el.graphFs) {
    el.graphFs.hidden = on;
    el.graphFs.textContent = t("fullscreen");
    el.graphFs.title = t("fullscreen_title");
  }
  if (el.graphFsExit) el.graphFsExit.hidden = !on;
  if (el.graphLabelsBtn) el.graphLabelsBtn.hidden = !on;
  if (el.graphStage) {
    el.graphStage.classList.toggle("is-fullscreen", on);
    el.graphStage.classList.toggle("labels-hidden", on && hideGraphLabels);
  }
  syncGraphLabelsBtn();
  if (on && el.historyPanel && !el.historyPanel.open) {
    el.historyPanel.open = true;
  }
  syncFsEditDock(on);
  syncFsOverlays(on);
  resizeGraphNetwork();
  if (graphNodes.length) drawGraph(graphNodes, graphEdges);
}

function labelsCurrentlyHidden() {
  return hideGraphLabels && isGraphFullscreen();
}

function syncGraphLabelsBtn() {
  const btn = el.graphLabelsBtn;
  if (!btn) return;
  const hidden = hideGraphLabels;
  btn.setAttribute("aria-pressed", hidden ? "true" : "false");
  btn.classList.toggle("is-active", hidden);
  btn.textContent = hidden ? t("show_labels") : t("hide_labels");
  btn.title = hidden ? t("show_labels_title") : t("hide_labels_title");
}

function toggleGraphLabels() {
  hideGraphLabels = !hideGraphLabels;
  if (el.graphStage) {
    el.graphStage.classList.toggle(
      "labels-hidden",
      hideGraphLabels && isGraphFullscreen()
    );
  }
  syncGraphLabelsBtn();
  drawGraph(graphNodes, graphEdges);
}

/** Move edit form into fullscreen stage so правки доступны на весь экран. */
function syncFsEditDock(on) {
  const form = el.editForm;
  if (!form) return;
  if (on) {
    if (el.fsEditSlot && form.parentElement !== el.fsEditSlot) {
      el.fsEditSlot.appendChild(form);
    }
    form.classList.add("is-fs-dock");
    if (el.fsEditDock) el.fsEditDock.hidden = false;
  } else {
    form.classList.remove("is-fs-dock");
    if (el.editFormHome && form.parentElement !== el.editFormHome) {
      el.editFormHome.appendChild(form);
    }
    if (el.fsEditDock) el.fsEditDock.hidden = true;
  }
}

/**
 * Entity drawer / ctx menu: on body in normal mode (panel has overflow+backdrop-filter),
 * inside graph-stage in fullscreen (otherwise they vanish).
 */
function syncFsOverlays(on) {
  const stage = el.graphStage;
  const host = document.body;
  if (!stage) return;
  const target = on ? stage : host;
  for (const node of [el.ctxMenu, el.entityDrawer]) {
    if (node && node.parentElement !== target) target.appendChild(node);
  }
}

function resizeGraphNetwork() {
  if (network) {
    requestAnimationFrame(() => {
      try {
        network.redraw();
        network.fit({ animation: { duration: 280, easingFunction: "easeInOutQuad" } });
      } catch (_) {
        /* ignore */
      }
    });
  }
  resizeMap();
  resizeGlobe();
}

function fillMapEntitySelect() {
  if (!el.mapEntity) return;
  const prev = el.mapEntity.value;
  el.mapEntity.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = allNodes.length ? "— выбери сущность —" : "Сначала проанализируй текст";
  el.mapEntity.appendChild(opt0);
  for (const n of allNodes) {
    const opt = document.createElement("option");
    opt.value = n.id;
    opt.textContent = n.label || n.id;
    el.mapEntity.appendChild(opt);
  }
  if (prev && [...el.mapEntity.options].some((o) => o.value === prev)) {
    el.mapEntity.value = prev;
  }
}

function refreshGeoViews() {
  setGeoLinks(geoLinks);
  syncMarkers(geoMarkers, { onRemove: removeGeoMarker, links: geoLinks });
  if (el.mapMeta) {
    el.mapMeta.textContent = geoMarkers.length
      ? `меток: ${geoMarkers.length}` +
        (geoLinks.length ? ` · связей: ${geoLinks.length}` : "")
      : "клик / поиск / «Собрать географ»";
  }
  if (el.globeMeta) {
    el.globeMeta.textContent = geoMarkers.length
      ? `точек: ${geoMarkers.length}` +
        (geoLinks.length ? ` · дуг: ${geoLinks.length}` : "")
      : "Метки и связи графа появляются на глобусе";
  }
}

async function persistMarkers() {
  if (!documentId) return;
  try {
    await api("/api/markers", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: documentId,
        history_id: historyId,
        markers: geoMarkers,
      }),
    });
  } catch (err) {
    if (el.mapMeta) el.mapMeta.textContent = String(err.message || err);
  }
}

function addGeoMarker(entity, lat, lng, note = "", kind = "place") {
  const name = (entity || "").trim() || "Точка";
  const id = `${name}:${lat.toFixed(5)}:${lng.toFixed(5)}:${Date.now()}`;
  geoMarkers = [
    ...geoMarkers,
    {
      id,
      entity: name,
      lat,
      lng,
      note: String(note || "").trim(),
      kind: kind === "event" ? "event" : "place",
      auto: false,
    },
  ];
  geoLinks = linksFromTriples(geoMarkers, lastTriples);
  refreshGeoViews();
  persistMarkers();
}

function removeGeoMarker(id) {
  geoMarkers = geoMarkers.filter((m) => m.id !== id);
  geoLinks = linksFromTriples(geoMarkers, lastTriples);
  refreshGeoViews();
  persistMarkers();
}

function selectedGeoEngine() {
  const v = el.mapGeoEngine ? el.mapGeoEngine.value : "deepseek-v4";
  return v === "deepseek" ? "deepseek" : "deepseek-v4";
}

function selectedReasoning() {
  return !!(el.reasoningToggle && el.reasoningToggle.checked);
}

function syncLinkToggles(checked) {
  const on = !!checked;
  if (el.mapShowLinks) el.mapShowLinks.checked = on;
  if (el.globeShowLinks) el.globeShowLinks.checked = on;
  setLinksVisible(on);
}

async function buildGeoGraphFromApi() {
  if (!documentId && !(documentText || (el.text && el.text.value))) {
    if (el.mapMeta) el.mapMeta.textContent = "Сначала проанализируй текст";
    return;
  }
  const eng = selectedGeoEngine();
  if (el.mapMeta) el.mapMeta.textContent = `${engineLabel(eng)} собирает географ…`;
  try {
    const data = await post("/api/geo/graph", {
      document_id: documentId,
      document_text: documentText || (el.text ? el.text.value : ""),
      triples: lastTriples,
      replace: true,
      engine: eng,
      reasoning: selectedReasoning(),
    });
    if (Array.isArray(data.markers)) geoMarkers = data.markers;
    if (Array.isArray(data.geo_links)) geoLinks = data.geo_links;
    else geoLinks = linksFromTriples(geoMarkers, lastTriples);
    refreshGeoViews();
    ensureMap();
    if (data.debug) setDebug({ ...(lastDebug || {}), ...data.debug }, lastTriples);
    if (el.mapMeta) {
      el.mapMeta.textContent = `гео: ${geoMarkers.length} точек, ${geoLinks.length} связей · ${engineLabel(data.engine || eng)}`;
    }
  } catch (err) {
    if (el.mapMeta) el.mapMeta.textContent = String(err.message || err);
  }
}

function switchView(view) {
  currentView = view === "map" || view === "globe" ? view : "graph";
  const tabs = [
    [el.tabGraph, "graph"],
    [el.tabMap, "map"],
    [el.tabGlobe, "globe"],
  ];
  for (const [btn, name] of tabs) {
    if (btn) btn.classList.toggle("is-active", name === currentView);
  }
  if (el.viewGraph) {
    el.viewGraph.hidden = currentView !== "graph";
    el.viewGraph.classList.toggle("is-active", currentView === "graph");
  }
  if (el.viewMap) {
    el.viewMap.hidden = currentView !== "map";
    el.viewMap.classList.toggle("is-active", currentView === "map");
  }
  if (el.viewGlobe) {
    el.viewGlobe.hidden = currentView !== "globe";
    el.viewGlobe.classList.toggle("is-active", currentView === "globe");
  }
  if (currentView === "map") {
    ensureMap();
    refreshGeoViews();
    resizeMap();
  } else if (currentView === "globe") {
    ensureGlobe();
    refreshGeoViews();
    resizeGlobe();
  } else if (network) {
    resizeGraphNetwork();
  }
}

async function geocodeSelectedEntity() {
  const entity = el.mapEntity ? el.mapEntity.value.trim() : "";
  const query = el.mapSearch ? el.mapSearch.value.trim() : "";
  if (!entity && !query) {
    if (el.mapMeta) el.mapMeta.textContent = "Выбери сущность или введи поиск";
    return;
  }
  const useDeepseek = !!(el.mapUseDeepseek && el.mapUseDeepseek.checked);
  const label = query || entity;
  if (el.mapMeta) {
    el.mapMeta.textContent = useDeepseek
      ? `DeepSeek + карта: ${label}…`
      : `Карта OSM: ${label}…`;
  }
  try {
    const data = await post("/api/geocode/locate", {
      entity,
      query,
      document_id: documentId,
      document_text: documentText || (el.text ? el.text.value : ""),
      use_deepseek: useDeepseek,
      limit: 5,
    });
    const hit = (data.results || [])[0];
    if (!hit) {
      const err =
        (data.deepseek && data.deepseek.error) ||
        "Ничего не найдено — уточни запрос или кликни по карте";
      if (el.mapMeta) el.mapMeta.textContent = err;
      return;
    }
    const markerName = entity || query || hit.display_name || "Точка";
    const noteParts = [];
    if (hit.display_name) noteParts.push(hit.display_name);
    if (hit.source) noteParts.push(hit.source);
    if (data.deepseek && data.deepseek.note) noteParts.push(data.deepseek.note);
    addGeoMarker(markerName, hit.lat, hit.lng, noteParts.join(" · "));
    ensureMap();
    const src = hit.source === "deepseek" ? "DeepSeek" : "OSM";
    const name = hit.display_name || "";
    if (el.mapMeta) {
      el.mapMeta.textContent = name
        ? `${src}: ${name.length > 56 ? name.slice(0, 56) + "…" : name}`
        : `метка добавлена (${src})`;
    }
  } catch (err) {
    if (el.mapMeta) el.mapMeta.textContent = String(err.message || err);
  }
}

async function toggleGraphFullscreen() {
  const stage = el.graphStage;
  if (!stage) return;
  try {
    if (isGraphFullscreen()) {
      if (document.exitFullscreen) await document.exitFullscreen();
      else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
    } else {
      if (stage.requestFullscreen) await stage.requestFullscreen();
      else if (stage.webkitRequestFullscreen) stage.webkitRequestFullscreen();
    }
  } catch (err) {
    if (el.meta) el.meta.textContent = String(err.message || err);
  }
}

function drawGraph(nodes, edges) {
  graphNodes = nodes || [];
  graphEdges = edges || [];

  // Nodes touched by a muted (off) edge become transparent "blocks"
  const transparentNodes = new Set(mutedNodes);
  for (const e of graphEdges) {
    if (mutedEdgeIds.has(e.id)) {
      transparentNodes.add(e.source);
      transparentNodes.add(e.target);
    }
  }

  const visNodes = new DataSet(
    graphNodes.map((n) => {
      const nodeMuted = transparentNodes.has(n.id);
      const hi = highlightNodes.has(n.id);
      const noLabel = labelsCurrentlyHidden();
      return {
        id: n.id,
        label: noLabel ? " " : n.label || n.id,
        opacity: nodeMuted ? 0.2 : 1,
        color: {
          background: nodeMuted
            ? "rgba(28, 36, 44, 0.15)"
            : hi
              ? "#4e6676"
              : "#1c242c",
          border: nodeMuted
            ? "rgba(106, 127, 140, 0.2)"
            : hi
              ? "#dadee1"
              : "#6a7f8c",
          highlight: { background: "#4e6676", border: "#f4f6f7" },
        },
        font: {
          color: nodeMuted ? "rgba(244, 246, 247, 0.25)" : "#f4f6f7",
          face: "Resist Sans Text",
          size: noLabel ? 0 : 14,
        },
      };
    })
  );

  const visEdges = new DataSet(
    graphEdges.map((e) => {
      const off = mutedEdgeIds.has(e.id);
      const linked =
        transparentNodes.has(e.source) || transparentNodes.has(e.target);
      const showRed = !off && linked;
      const hi = highlightEdges.has(e.id);
      const kind = (e.kind || "explicit").toLowerCase();
      const origin = (e.origin || "base").toLowerCase();
      const isHidden = kind === "hidden";
      const isFalse = kind === "false";
      const style = edgeStyle(kind, origin);
      let baseColor = style.color;
      let fontColor = style.font;
      let dashes = style.dashes;
      let width = style.width;
      let label = e.label || e.relation || "";
      if (isHidden && label && !/скрыт/i.test(label)) label = `${label} · скрытая`;
      if (isFalse && label && !/ложн/i.test(label)) label = `${label} · ложная`;
      const noLabel = labelsCurrentlyHidden();
      return {
        id: e.id,
        from: e.source,
        to: e.target,
        label: noLabel ? undefined : label,
        title: [
          e.evidence ? `evidence: ${e.evidence}` : "",
          e.confidence != null && e.confidence !== ""
            ? `confidence: ${e.confidence}`
            : "",
          kind !== "explicit" ? `kind: ${kind}` : "",
          origin !== "base" ? `origin: ${origin}` : "",
        ]
          .filter(Boolean)
          .join("\n") || undefined,
        arrows: "to",
        dashes: off ? false : dashes,
        width: showRed ? 2.6 : width,
        color: {
          color: off
            ? "rgba(78, 102, 118, 0.1)"
            : showRed
              ? "#c45c5c"
              : hi
                ? "#dadee1"
                : baseColor,
          highlight: showRed ? "#e07070" : style.font,
          opacity: off ? 0.1 : 1,
        },
        font: {
          color: off
            ? "rgba(138, 154, 166, 0.12)"
            : showRed
              ? "#c97878"
              : fontColor,
          size: noLabel ? 0 : 11,
          face: "Resist Sans Text",
          strokeWidth: 0,
        },
      };
    })
  );

  const options = {
    physics: {
      barnesHut: { gravitationalConstant: -2800, springLength: 120 },
    },
    interaction: { hover: true, multiselect: false },
    edges: { smooth: { type: "dynamic" } },
  };

  if (network) {
    network.setData({ nodes: visNodes, edges: visEdges });
  } else {
    network = new Network(el.graph, { nodes: visNodes, edges: visEdges }, options);
  }

  if (el.edgeLegend) {
    const edges = graphEdges || [];
    el.edgeLegend.hidden = !edges.length;
    const has = (origin, kind) =>
      edges.some(
        (e) =>
          (e.origin || "base") === origin &&
          (e.kind || "explicit").toLowerCase() === kind
      );
    const show = (sel, on) => {
      const node = el.edgeLegend.querySelector(sel);
      if (node) node.hidden = !on;
    };
    show(".edge-leg-append", has("append", "explicit"));
    show(".edge-leg-append-hidden", has("append", "hidden"));
    show(".edge-leg-append-false", has("append", "false"));
    show(".edge-leg-bridge", has("bridge", "explicit"));
    show(".edge-leg-bridge-hidden", has("bridge", "hidden"));
    show(".edge-leg-bridge-false", has("bridge", "false"));
  }

  if (network && !networkBound) {
    networkBound = true;
    if (el.graph) {
      el.graph.addEventListener("contextmenu", (e) => e.preventDefault());
    }
    network.on("oncontext", (params) => {
      params.event.preventDefault();
      const nodeId =
        params.nodes && params.nodes.length
          ? params.nodes[0]
          : network.getNodeAt(params.pointer.DOM);
      if (!nodeId) {
        hideCtxMenu();
        return;
      }
      const evt = params.event.srcEvent || params.event;
      showCtxMenu(evt.clientX, evt.clientY, nodeId);
    });
    network.on("click", () => hideCtxMenu());
    network.on("dragStart", () => hideCtxMenu());
    network.on("doubleClick", (params) => {
      hideCtxMenu();
      // Double-click connection: turn off → endpoints transparent, other links red
      if (params.edges && params.edges.length) {
        const edgeId = params.edges[0];
        if (mutedEdgeIds.has(edgeId)) mutedEdgeIds.delete(edgeId);
        else mutedEdgeIds.add(edgeId);
        drawGraph(graphNodes, graphEdges);
        return;
      }
      // Double-click node: same mode for the block
      if (params.nodes && params.nodes.length) {
        const id = params.nodes[0];
        if (mutedNodes.has(id)) mutedNodes.delete(id);
        else mutedNodes.add(id);
        drawGraph(graphNodes, graphEdges);
      }
    });
  }
}

el.analyze.addEventListener("click", async () => {
  el.meta.textContent = "Анализ…";
  disabled = new Set();
  mutedNodes = new Set();
  mutedEdgeIds = new Set();
  // New analysis = new history iteration (keep document_id continuity)
  historyId = null;
  el.analyze.disabled = true;
  el.analyze.classList.add("is-loading");
  const label = el.analyze.querySelector(".btn-label");
  if (label) label.textContent = t("analyzing");
  try {
    documentText = el.text.value;
    const data = await post("/api/analyze", {
      text: el.text.value,
      document_id: documentId,
      engine: el.engine ? el.engine.value : "gemma",
      geo_graph: el.mapGeoAuto ? el.mapGeoAuto.checked : true,
      geo_engine: selectedGeoEngine(),
      reasoning: selectedReasoning(),
      language: selectedLanguage(),
    });
    const eng = engineLabel(data.engine);
    el.meta.textContent =
      `модель: ${eng}` +
      (selectedReasoning() && (isDeepseekEngine(data.engine) || data.engine === "gigachat")
        ? " · reasoning"
        : "") +
      ` · ${t("site_language").toLowerCase()}: ${LANGS[selectedLanguage()]?.name || selectedLanguage()}` +
      ` · источник: ${data.source} · троек: ${data.triples.length}` +
      (data.comments && Object.keys(data.comments).length
        ? ` · комментариев: ${Object.keys(data.comments).length}`
        : "") +
      (data.markers && data.markers.length ? ` · гео: ${data.markers.length}` : "") +
      (data.geo_links && data.geo_links.length
        ? ` · гео-связей: ${data.geo_links.length}`
        : "") +
      (data.lm_ready
        ? ` · Gemma: ${data.lm_model || "ok"}`
        : data.engine === "gemma" || data.engine === "auto"
          ? " · Gemma выкл"
          : "") +
      (data.openrouter_ready
        ? ` · DeepSeek: ${data.openrouter_model || "ok"}`
        : data.engine === "deepseek" || data.engine === "deepseek-v4"
          ? " · DeepSeek: нет ключа"
          : "") +
      (data.gigachat_ready
        ? ` · GigaChat: ${data.gigachat_model || "ok"}`
        : data.engine === "gigachat"
          ? " · GigaChat: нет ключа"
          : "") +
      (data.model_ready ? " · MicroGPT ок" : "");
    if (data.hint) {
      el.meta.textContent += " — " + data.hint;
    }
    setDebug(data.debug, data.triples);
    applyPayload(data);
    await refreshHistoryList();
  } catch (err) {
    el.meta.textContent = String(err.message || err);
  } finally {
    el.analyze.disabled = false;
    el.analyze.classList.remove("is-loading");
    if (label) label.textContent = t("analyze");
  }
});

async function runAppendOrBridge(mode) {
  const text = (el.appendText && el.appendText.value) || "";
  if (!text.trim()) {
    if (el.appendMeta) el.appendMeta.textContent = t("append_need_text");
    return;
  }
  if (!documentId && !(graphEdges && graphEdges.length)) {
    if (el.appendMeta) {
      el.appendMeta.textContent = t("append_need_graph");
    }
    return;
  }
  const btn = mode === "bridge" ? el.bridgeBtn : el.appendBtn;
  const label = btn && btn.querySelector(".btn-label");
  const idle = mode === "bridge" ? t("append_bridge") : t("append_add");
  if (el.appendMeta) {
    el.appendMeta.textContent =
      mode === "bridge" ? t("append_bridging_meta") : t("append_adding_meta");
  }
  if (btn) {
    btn.disabled = true;
    btn.classList.add("is-loading");
  }
  if (label) label.textContent = mode === "bridge" ? t("append_bridging") : t("append_adding");
  try {
    const endpoint =
      mode === "bridge" ? "/api/analyze/bridge" : "/api/analyze/append";
    const payload = {
      text: text.trim(),
      document_id: documentId,
      history_id: historyId,
      reasoning: selectedReasoning(),
      language: selectedLanguage(),
    };
    if (mode === "bridge") {
      payload.engine = resolveBridgeEngine(el.engine && el.engine.value);
      payload.document_text = documentText || (el.text && el.text.value) || "";
    } else {
      payload.engine = el.engine ? el.engine.value : "gemma";
    }
    const data = await post(endpoint, payload);
    if (el.text && data.text) el.text.value = data.text;
    documentText = data.text || documentText;
    setDebug(data.debug, data.triples);
    applyPayload(data);
    await refreshHistoryList();
    if (el.appendMeta) {
      el.appendMeta.textContent =
        data.hint ||
        (mode === "bridge"
          ? `склейка: +${data.new_count || 0} новых, +${data.bridge_count || 0} мостов`
          : `добавлено рёбер: ${data.added_count ?? 0}`);
    }
    if (el.meta && data.hint) {
      el.meta.textContent = data.hint;
    }
  } catch (err) {
    if (el.appendMeta) el.appendMeta.textContent = String(err.message || err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("is-loading");
    }
    if (label) label.textContent = idle;
  }
}

if (el.appendBtn) {
  el.appendBtn.addEventListener("click", () => runAppendOrBridge("append"));
}
if (el.bridgeBtn) {
  el.bridgeBtn.addEventListener("click", () => runAppendOrBridge("bridge"));
}

el.ask.addEventListener("click", async () => {
  try {
    const useWeb = !!(el.askWebSearch && el.askWebSearch.checked);
    const textEngine = selectedTextEngine();
    const modelName = engineLabel(textEngine);
    el.answer.textContent = useWeb ? `Поиск + ${modelName}…` : "Ответ…";
    const data = await post("/api/ask", {
      question: el.question.value,
      document_id: documentId,
      document_text: documentText || el.text.value,
      web_search: useWeb,
      engine: textEngine,
      reasoning: selectedReasoning(),
    });
    const modeLabel =
      data.qa_mode === "text_deepseek_v4"
        ? useWeb
          ? "[DeepSeek V4 Pro + веб]\n"
          : "[DeepSeek V4 Pro по тексту]\n"
        : data.qa_mode === "text_deepseek"
          ? useWeb
            ? "[DeepSeek 3.2 + веб]\n"
            : "[DeepSeek 3.2 по тексту]\n"
          : data.qa_mode === "text_gigachat"
            ? useWeb
              ? "[GigaChat + веб]\n"
              : "[GigaChat по тексту]\n"
            : data.qa_mode === "text_gemma"
              ? useWeb
                ? "[Gemma + веб]\n"
                : "[Gemma по тексту]\n"
              : "[Граф]\n";
    el.answer.textContent = modeLabel + (data.answer || "");
    renderWebSources(el.askWebSources, data.web_results, data.web_error);
    if (data.debug) {
      setDebug(
        {
          prompt_user: data.debug.prompt_user || "Вопрос: " + el.question.value,
          raw_response: data.debug.raw_response || data.answer || "",
          parsed_json: [
            {
              mode: data.qa_mode,
              answer: data.answer,
              web_results: data.web_results || [],
            },
          ],
          error: data.debug.error || "",
        },
        []
      );
    }
    applyPayload(data);
  } catch (err) {
    el.answer.textContent = String(err.message || err);
    renderWebSources(el.askWebSources, [], String(err.message || err));
  }
});

el.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter") el.ask.click();
});

el.add.addEventListener("click", async () => {
  try {
    if (el.editMeta) el.editMeta.textContent = "Добавляю…";
    const data = await post("/api/edit", {
      action: "add_triple",
      document_id: documentId,
      subject: el.editS.value.trim(),
      relation: el.editR.value.trim(),
      object: el.editO.value.trim(),
      history_id: historyId,
    });
    if (Array.isArray(data.triples)) lastTriples = data.triples;
    applyPayload(data);
    await refreshHistoryList();
    if (el.editMeta) el.editMeta.textContent = "Связь добавлена";
  } catch (err) {
    const msg = String(err.message || err);
    if (el.editMeta) el.editMeta.textContent = msg;
    else el.answer.textContent = msg;
  }
});

if (el.saveGraph) el.saveGraph.addEventListener("click", saveCurrentGraph);
if (el.historySelect) {
  el.historySelect.addEventListener("change", () => {
    const id = el.historySelect.value;
    if (id) loadHistoryItem(id);
  });
}
if (el.graphFs) el.graphFs.addEventListener("click", toggleGraphFullscreen);
if (el.graphFsExit) el.graphFsExit.addEventListener("click", toggleGraphFullscreen);
if (el.graphLabelsBtn) el.graphLabelsBtn.addEventListener("click", toggleGraphLabels);
document.addEventListener("fullscreenchange", () => {
  syncGraphFsUi();
});
document.addEventListener("webkitfullscreenchange", () => {
  syncGraphFsUi();
});
syncGraphFsUi();

setGeoCallbacks({
  activate: (m) => {
    if (m && m.entity) openEntityPanel(m.entity);
  },
  place: (lat, lng) => {
    const entity = el.mapEntity ? el.mapEntity.value.trim() : "";
    if (!entity) {
      if (el.mapMeta) el.mapMeta.textContent = "Сначала выбери сущность";
      return;
    }
    if (!documentId) {
      if (el.mapMeta) el.mapMeta.textContent = "Сначала проанализируй текст";
      return;
    }
    addGeoMarker(entity, lat, lng);
  },
});
if (el.mapPlaceMode) {
  setPlaceMode(el.mapPlaceMode.checked);
  el.mapPlaceMode.addEventListener("change", () => setPlaceMode(el.mapPlaceMode.checked));
}
if (el.mapShowLinks) {
  el.mapShowLinks.addEventListener("change", () => syncLinkToggles(el.mapShowLinks.checked));
}
if (el.globeShowLinks) {
  el.globeShowLinks.addEventListener("change", () => syncLinkToggles(el.globeShowLinks.checked));
}
syncLinkToggles(el.mapShowLinks ? el.mapShowLinks.checked : true);
if (el.mapGeocode) el.mapGeocode.addEventListener("click", geocodeSelectedEntity);
if (el.mapGeoGraph) el.mapGeoGraph.addEventListener("click", buildGeoGraphFromApi);
if (el.mapSearch) {
  el.mapSearch.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      geocodeSelectedEntity();
    }
  });
}
if (el.tabGraph) el.tabGraph.addEventListener("click", () => switchView("graph"));
if (el.tabMap) el.tabMap.addEventListener("click", () => switchView("map"));
if (el.tabGlobe) el.tabGlobe.addEventListener("click", () => switchView("globe"));

if (el.debugBtn) el.debugBtn.addEventListener("click", openDebug);
if (el.debugClose) el.debugClose.addEventListener("click", closeDebug);
if (el.debugDrawer) {
  el.debugDrawer.addEventListener("click", (e) => {
    if (e.target === el.debugDrawer) closeDebug();
  });
}
if (el.ctxAbout) {
  el.ctxAbout.addEventListener("click", () => {
    if (ctxEntityName) openEntityPanel(ctxEntityName);
  });
}
if (el.ctxDelete) {
  el.ctxDelete.addEventListener("click", async () => {
    const name = ctxEntityName;
    hideCtxMenu();
    if (!name || !documentId) return;
    if (!window.confirm(`Удалить узел «${name}» и все его связи?`)) return;
    try {
      if (el.editMeta) el.editMeta.textContent = "Удаляю…";
      const data = await post("/api/edit", {
        action: "delete_node",
        document_id: documentId,
        name,
        history_id: historyId,
      });
      if (Array.isArray(data.triples)) lastTriples = data.triples;
      if (entityName === name) closeEntityPanel();
      applyPayload(data);
      await refreshHistoryList();
      if (el.editMeta) el.editMeta.textContent = `Узел «${name}» удалён`;
    } catch (err) {
      const msg = String(err.message || err);
      if (el.editMeta) el.editMeta.textContent = msg;
      else el.answer.textContent = msg;
    }
  });
}
if (el.entityClose) el.entityClose.addEventListener("click", closeEntityPanel);
if (el.entityDrawer) {
  el.entityDrawer.addEventListener("click", (e) => {
    if (e.target === el.entityDrawer) closeEntityPanel();
  });
}
if (el.entitySave) el.entitySave.addEventListener("click", saveEntityComment);
if (el.entityGemma) el.entityGemma.addEventListener("click", generateEntityComment);
if (el.qaEngine) {
  el.qaEngine.addEventListener("change", () => setQaEngine(el.qaEngine.value, el.qaEngine));
}
if (el.entityQaEngine) {
  el.entityQaEngine.addEventListener("change", () =>
    setQaEngine(el.entityQaEngine.value, el.entityQaEngine)
  );
}
syncGenerateButtonLabel();
document.addEventListener("click", (e) => {
  if (!el.ctxMenu || el.ctxMenu.hidden) return;
  if (el.ctxMenu.contains(e.target)) return;
  hideCtxMenu();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    hideCtxMenu();
    closeEntityPanel();
  }
});

async function uploadDocumentFile(file) {
  if (!file) return;
  if (el.fileMeta) el.fileMeta.textContent = `Читаю ${file.name}…`;
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : detail || data.error || res.statusText;
      throw new Error(msg);
    }
    const incoming = (data.text || "").trim();
    if (!incoming) throw new Error("Пустой текст после разбора файла");
    const append = !!(el.fileAppend && el.fileAppend.checked);
    const current = (el.text.value || "").trim();
    if (append && current) {
      el.text.value = `${current}\n\n${incoming}`;
    } else {
      el.text.value = incoming;
    }
    documentText = el.text.value;
    if (el.fileMeta) {
      const mode = append && current ? "добавлено" : "заменено";
      el.fileMeta.textContent = `${data.filename} · ${data.chars} символов · ${mode}`;
    }
  } catch (err) {
    if (el.fileMeta) el.fileMeta.textContent = String(err.message || err);
  } finally {
    if (el.fileInput) el.fileInput.value = "";
  }
}

if (el.fileInput) {
  el.fileInput.addEventListener("change", () => {
    const file = el.fileInput.files && el.fileInput.files[0];
    if (file) uploadDocumentFile(file);
  });
}

if (el.dropZone) {
  const setDrag = (on) => el.dropZone.classList.toggle("is-dragover", on);
  ["dragenter", "dragover"].forEach((ev) => {
    el.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDrag(true);
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    el.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (ev === "dragleave") setDrag(false);
    });
  });
  el.dropZone.addEventListener("drop", (e) => {
    setDrag(false);
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) uploadDocumentFile(file);
  });
}

el.text.value =
  "Эльвира Ковач защищалась в Лаборатории К-17. Донат Ившич основал компанию Северный меридиан. " +
  "Тимур Асланов — сотрудник Гельвеций-Прайм и соавтор патента Северного меридиана. " +
  "Освальд Бэр — бенефициар Фонда «Тихая гавань».";

initLanguagePicker();
refreshHistoryList();
