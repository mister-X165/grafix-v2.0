import L from "leaflet";
import "leaflet/dist/leaflet.css";
import Globe from "globe.gl";

/** @typedef {{ id: string, entity: string, lat: number, lng: number, note?: string, kind?: string }} GeoMarker */
/** @typedef {{ source: string, target: string, source_id?: string, target_id?: string, relation?: string, kind?: string }} GeoLink */

let map = null;
/** @type {Map<string, L.Marker>} */
const leafletLayers = new Map();
/** @type {L.LayerGroup | null} */
let leafletLinkLayer = null;
let globe = null;
let globeReady = false;
let placeMode = true;
let linksVisible = true;
/** @type {GeoMarker[]} */
let lastMarkers = [];
/** @type {GeoLink[]} */
let currentLinks = [];
/** @type {(m: GeoMarker) => void | null} */
let onMarkerActivate = null;
/** @type {(lat: number, lng: number) => void | null} */
let onMapClickPlace = null;

const pinPlace = L.divIcon({
  className: "map-pin map-pin-place",
  html: "<span class=\"map-pin-dot\"></span>",
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

const pinEvent = L.divIcon({
  className: "map-pin map-pin-event",
  html: "<span class=\"map-pin-dot\"></span>",
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

function iconFor(m) {
  return (m.kind || "").toLowerCase() === "event" ? pinEvent : pinPlace;
}

function linkColor(kind) {
  const k = (kind || "explicit").toLowerCase();
  if (k === "hidden") return "#2ec4b6";
  if (k === "false") return "#ff5c8a";
  return "#8a9aa6";
}

function bearingDeg(lat1, lng1, lat2, lng2) {
  const toRad = Math.PI / 180;
  const φ1 = lat1 * toRad;
  const φ2 = lat2 * toRad;
  const Δλ = (lng2 - lng1) * toRad;
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

function alongPoint(a, b, t) {
  return {
    lat: a.lat + (b.lat - a.lat) * t,
    lng: a.lng + (b.lng - a.lng) * t,
  };
}

function resolveEnds(markers, link) {
  const byId = new Map(markers.map((m) => [m.id, m]));
  const byName = new Map(markers.map((m) => [String(m.entity || "").toLowerCase(), m]));
  const a =
    (link.source_id && byId.get(link.source_id)) ||
    byName.get(String(link.source || "").toLowerCase());
  const b =
    (link.target_id && byId.get(link.target_id)) ||
    byName.get(String(link.target || "").toLowerCase());
  if (!a || !b || a.id === b.id) return null;
  return { a, b };
}

export function setGeoCallbacks({ activate, place }) {
  onMarkerActivate = activate || null;
  onMapClickPlace = place || null;
}

export function setPlaceMode(on) {
  placeMode = !!on;
  const el = document.getElementById("leaflet-map");
  if (el) el.classList.toggle("is-placing", placeMode);
}

export function setLinksVisible(on) {
  linksVisible = !!on;
  syncLeafletLinks(lastMarkers, currentLinks);
  syncGlobeArcs(lastMarkers, currentLinks);
}

export function getLinksVisible() {
  return linksVisible;
}

export function ensureMap() {
  const root = document.getElementById("leaflet-map");
  if (!root || map) {
    resizeMap();
    return map;
  }
  map = L.map(root, {
    zoomControl: true,
    attributionControl: true,
  }).setView([55.75, 37.62], 4);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
  }).addTo(map);

  leafletLinkLayer = L.layerGroup().addTo(map);

  map.on("click", (e) => {
    if (!placeMode || !onMapClickPlace) return;
    onMapClickPlace(e.latlng.lat, e.latlng.lng);
  });

  setTimeout(() => map.invalidateSize(), 80);
  return map;
}

export function ensureGlobe() {
  const root = document.getElementById("globe-root");
  if (!root) return null;
  if (globe) {
    resizeGlobe();
    return globe;
  }
  const w = root.clientWidth || 640;
  const h = root.clientHeight || 440;
  globe = Globe()(root)
    .width(w)
    .height(h)
    .backgroundColor("rgba(0,0,0,0)")
    .globeImageUrl("https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg")
    .bumpImageUrl("https://unpkg.com/three-globe/example/img/earth-topology.png")
    .showAtmosphere(true)
    .atmosphereColor("#4e6676")
    .atmosphereAltitude(0.18)
    .pointsMerge(false)
    .pointAltitude(0.012)
    .pointRadius(0.45)
    .pointColor((d) => ((d.kind || "").toLowerCase() === "event" ? "#ff5c8a" : "#2ec4b6"))
    .pointLabel((d) => {
      const tag = (d.kind || "").toLowerCase() === "event" ? "событие" : "место";
      return `<b>${d.entity}</b> · ${tag}${d.note ? `<br/>${d.note}` : ""}`;
    })
    .onPointClick((d) => {
      if (d && onMarkerActivate) onMarkerActivate(d);
    })
    .arcsData([])
    .arcColor((d) => linkColor(d.kind))
    .arcAltitude(0.18)
    .arcStroke(0.7)
    .arcDashLength(0.45)
    .arcDashGap(0.18)
    .arcDashAnimateTime(2800)
    .arcLabel((d) => {
      const rel = d.relation || "?";
      return `${d.source || ""} —[${rel}]→ ${d.target || ""}`;
    })
    // WebGL typeface (helvetiker) has no Cyrillic → use HTML chips instead
    .labelsData([])
    .htmlElementsData([])
    .htmlLat("lat")
    .htmlLng("lng")
    .htmlAltitude((d) => (d.altitude != null ? d.altitude : 0.03))
    .htmlElement((d) => makeGlobeHtmlLabel(d))
    .htmlTransitionDuration(0);

  globe.controls().autoRotate = true;
  globe.controls().autoRotateSpeed = 0.35;
  globeReady = true;
  return globe;
}

function makeGlobeHtmlLabel(d) {
  const node = document.createElement("div");
  const kind = (d.kind || "explicit").toLowerCase();
  node.className = `globe-html-label globe-html-label-${kind}`;
  const color = linkColor(kind);
  node.style.color = color;
  node.style.borderColor = color;
  node.textContent = d.text || "";
  if (d.title) node.title = d.title;
  return node;
}

/**
 * @param {GeoMarker[]} markers
 * @param {{ onRemove?: (id: string) => void, links?: GeoLink[] }} [opts]
 */
export function syncMarkers(markers, opts = {}) {
  const list = Array.isArray(markers) ? markers : [];
  lastMarkers = list;
  currentLinks = Array.isArray(opts.links) ? opts.links : currentLinks;
  syncLeaflet(list);
  syncLeafletLinks(list, currentLinks);
  syncGlobe(list);
  syncGlobeArcs(list, currentLinks);
  renderMarkerList(list, opts.onRemove);
}

/**
 * @param {GeoLink[]} links
 */
export function setGeoLinks(links) {
  currentLinks = Array.isArray(links) ? links : [];
}

function syncLeaflet(markers) {
  if (!map) return;
  const ids = new Set(markers.map((m) => m.id));
  for (const [id, layer] of leafletLayers) {
    if (!ids.has(id)) {
      map.removeLayer(layer);
      leafletLayers.delete(id);
    }
  }
  for (const m of markers) {
    const existing = leafletLayers.get(m.id);
    if (existing) {
      existing.setLatLng([m.lat, m.lng]);
      existing.setIcon(iconFor(m));
      existing.setPopupContent(popupHtml(m));
      continue;
    }
    const marker = L.marker([m.lat, m.lng], { icon: iconFor(m) })
      .addTo(map)
      .bindPopup(popupHtml(m));
    marker.on("click", () => {
      if (onMarkerActivate) onMarkerActivate(m);
    });
    leafletLayers.set(m.id, marker);
  }
  if (markers.length === 1) {
    map.setView([markers[0].lat, markers[0].lng], Math.max(map.getZoom(), 5));
  } else if (markers.length > 1) {
    const bounds = L.latLngBounds(markers.map((m) => [m.lat, m.lng]));
    map.fitBounds(bounds.pad(0.25));
  }
}

function syncLeafletLinks(markers, links) {
  if (!map) return;
  if (!leafletLinkLayer) leafletLinkLayer = L.layerGroup().addTo(map);
  leafletLinkLayer.clearLayers();
  if (!linksVisible) return;

  for (const link of links || []) {
    const ends = resolveEnds(markers, link);
    if (!ends) continue;
    const { a, b } = ends;
    const color = linkColor(link.kind);
    const rel = String(link.relation || "?").trim() || "?";
    const dash = (link.kind || "") === "explicit" ? null : "6 6";

    const line = L.polyline(
      [
        [a.lat, a.lng],
        [b.lat, b.lng],
      ],
      {
        color,
        weight: 2.4,
        opacity: 0.9,
        dashArray: dash,
      }
    );
    line.bindPopup(
      `${escapeHtml(a.entity)} —[${escapeHtml(rel)}]→ ${escapeHtml(b.entity)}`
    );
    leafletLinkLayer.addLayer(line);

    const mid = alongPoint(a, b, 0.5);
    const labelIcon = L.divIcon({
      className: "map-link-label-wrap",
      html: `<span class="map-link-label" style="border-color:${color};color:${color}">${escapeHtml(rel)}</span>`,
      iconSize: [1, 1],
      iconAnchor: [0, 0],
    });
    const labelMarker = L.marker([mid.lat, mid.lng], {
      icon: labelIcon,
      interactive: true,
      keyboard: false,
      zIndexOffset: 400,
    });
    labelMarker.bindPopup(
      `${escapeHtml(a.entity)} —[${escapeHtml(rel)}]→ ${escapeHtml(b.entity)}`
    );
    leafletLinkLayer.addLayer(labelMarker);

    const tip = alongPoint(a, b, 0.88);
    const deg = bearingDeg(a.lat, a.lng, b.lat, b.lng);
    const arrowIcon = L.divIcon({
      className: "map-link-arrow-wrap",
      html: `<span class="map-link-arrow" style="color:${color};transform:rotate(${deg}deg)">➤</span>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });
    leafletLinkLayer.addLayer(
      L.marker([tip.lat, tip.lng], {
        icon: arrowIcon,
        interactive: false,
        keyboard: false,
        zIndexOffset: 350,
      })
    );
  }
}

function syncGlobe(markers) {
  if (!globe || !globeReady) return;
  globe.pointsData(markers);
  refreshGlobeHtmlLabels(markers, currentLinks);
}

function syncGlobeArcs(markers, links) {
  if (!globe || !globeReady) return;
  if (!linksVisible) {
    globe.arcsData([]);
  } else {
    const arcs = [];
    for (const link of links || []) {
      const ends = resolveEnds(markers, link);
      if (!ends) continue;
      const { a, b } = ends;
      const rel = String(link.relation || "?").trim() || "?";
      arcs.push({
        startLat: a.lat,
        startLng: a.lng,
        endLat: b.lat,
        endLng: b.lng,
        relation: rel,
        kind: link.kind || "explicit",
        source: a.entity,
        target: b.entity,
      });
    }
    globe.arcsData(arcs);
  }
  globe.labelsData([]);
  refreshGlobeHtmlLabels(markers, links);
}

function refreshGlobeHtmlLabels(markers, links) {
  if (!globe || !globeReady) return;
  const htmlLabels = [];
  for (const m of markers || []) {
    if (!(m.entity || "").trim()) continue;
    htmlLabels.push({
      lat: m.lat,
      lng: m.lng,
      altitude: 0.02,
      text: m.entity,
      kind: (m.kind || "").toLowerCase() === "event" ? "event" : "place",
      title: m.note || m.entity || "",
    });
  }
  if (linksVisible) {
    for (const link of links || []) {
      const ends = resolveEnds(markers, link);
      if (!ends) continue;
      const { a, b } = ends;
      const rel = String(link.relation || "?").trim() || "?";
      const mid = alongPoint(a, b, 0.5);
      htmlLabels.push({
        lat: mid.lat,
        lng: mid.lng,
        altitude: 0.035,
        text: rel,
        kind: link.kind || "explicit",
        title: `${a.entity} —[${rel}]→ ${b.entity}`,
      });
    }
  }
  globe.htmlElementsData(htmlLabels);
}

function popupHtml(m) {
  const kind = (m.kind || "place").toLowerCase() === "event" ? "событие" : "место";
  const note = m.note ? `<div class="map-pop-note">${escapeHtml(m.note)}</div>` : "";
  return `<strong>${escapeHtml(m.entity)}</strong> <em>${kind}</em>${note}<div class="map-pop-coords">${m.lat.toFixed(4)}, ${m.lng.toFixed(4)}</div>`;
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * @param {GeoMarker[]} markers
 * @param {(id: string) => void} [onRemove]
 */
export function renderMarkerList(markers, onRemove) {
  const ul = document.getElementById("marker-list");
  if (!ul) return;
  ul.innerHTML = "";
  if (!markers.length) {
    const li = document.createElement("li");
    li.className = "marker-empty";
    li.textContent = "Меток нет — «Собрать географ», поиск или клик по карте";
    ul.appendChild(li);
    return;
  }
  for (const m of markers) {
    const li = document.createElement("li");
    const kind = (m.kind || "place").toLowerCase() === "event" ? "событие" : "место";
    if ((m.kind || "").toLowerCase() === "event") li.classList.add("is-event");
    const main = document.createElement("button");
    main.type = "button";
    main.className = "marker-item-main";
    main.textContent = `${m.entity} · ${kind} · ${m.lat.toFixed(3)}, ${m.lng.toFixed(3)}`;
    main.title = m.note || m.entity;
    main.addEventListener("click", () => {
      if (map) {
        map.setView([m.lat, m.lng], 8);
        const layer = leafletLayers.get(m.id);
        if (layer) layer.openPopup();
      }
      if (globe) {
        globe.pointOfView({ lat: m.lat, lng: m.lng, altitude: 1.6 }, 800);
        globe.controls().autoRotate = false;
      }
      if (onMarkerActivate) onMarkerActivate(m);
    });
    const del = document.createElement("button");
    del.type = "button";
    del.className = "marker-item-del";
    del.title = "Удалить метку";
    del.textContent = "×";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      if (onRemove) onRemove(m.id);
    });
    li.appendChild(main);
    li.appendChild(del);
    ul.appendChild(li);
  }
}

export function resizeMap() {
  if (map) setTimeout(() => map.invalidateSize(), 60);
}

export function resizeGlobe() {
  if (!globe) return;
  const root = document.getElementById("globe-root");
  if (!root) return;
  const w = root.clientWidth || 640;
  const h = root.clientHeight || 440;
  globe.width(w);
  globe.height(h);
}

export function focusMarkerOnGlobe(m) {
  if (!globe || !m) return;
  globe.controls().autoRotate = false;
  globe.pointOfView({ lat: m.lat, lng: m.lng, altitude: 1.55 }, 900);
}

/** Build geo links from triples where both ends have markers. */
export function linksFromTriples(markers, triples) {
  const byName = new Map(
    (markers || []).map((m) => [String(m.entity || "").toLowerCase(), m])
  );
  const out = [];
  const seen = new Set();
  for (const t of triples || []) {
    const a = byName.get(String(t.subject || "").toLowerCase());
    const b = byName.get(String(t.object || "").toLowerCase());
    if (!a || !b || a.id === b.id) continue;
    const rel = t.relation || "связан_с";
    const key = `${a.id}|${b.id}|${rel}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      source: a.entity,
      target: b.entity,
      source_id: a.id,
      target_id: b.id,
      relation: rel,
      kind: t.kind || "explicit",
    });
  }
  return out;
}
