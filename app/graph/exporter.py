"""Экспорт графа связей в автономный HTML (интерактивный, офлайн).

HTML использует собственную лёгкую JS-компоновку: координаты уже посчитаны
детерминированным Python-layout (app.graph.layout), JS добавляет перетаскивание
узлов, зум и подписи — без CDN и внешних библиотек (работа офлайн, §49).
"""

from __future__ import annotations

import html
import json
from typing import Any

_Tcolors = {
    "Article": "#2f6fed", "Claim": "#7c3aed", "Evidence": "#0891b2",
    "Source": "#15803d", "Verdict": "#b45309", "Entity": "#64748b",
}
_EDGE_COLORS = {"supports": "#15803d", "contradicts": "#b91c1c",
                "derived_from": "#b45309", "cites": "#6366f1"}


def _esc(s: str) -> str:
    return html.escape(str(s or ""), quote=True)


def export_graph_html(graph: dict[str, Any], title: str = "Граф связей") -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    W = max(900.0, max((n.get("x", 0) for n in nodes), default=800) + 120)
    H = max(620.0, max((n.get("y", 0) for n in nodes), default=560) + 120)
    return f"""<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{_esc(title)}</title><style>
body{{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0}}
header{{padding:10px 16px;display:flex;gap:14px;align-items:center;background:#1e293b}}
h1{{font-size:16px;margin:0}} .legend{{display:flex;gap:10px;font-size:12px;flex-wrap:wrap}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
svg{{display:block;width:100%;height:calc(100vh - 56px);cursor:grab}}
.edge{{stroke-width:1.6}} .elabel{{font-size:10px;fill:#94a3b8;pointer-events:none}}
.node circle{{stroke:#0f172a;stroke-width:2;cursor:pointer}}
.node text{{font-size:11px;fill:#e2e8f0;pointer-events:none}}
</style></head><body>
<header><h1>{_esc(title)} · узлов: {len(nodes)}, рёбер: {len(edges)}</h1>
<div class="legend">{''.join(f"<span><i class='dot' style='background:{c}'></i>{t}</span>" for t,c in _Tcolors.items())}</div>
</header>
<svg id='g' viewBox='0 0 {W:.0f} {H:.0f}'></svg>
<script>
const DATA = {payload};
const COLORS = {_esc(json.dumps(_Tcolors, ensure_ascii=False))};
const ECOLORS = {_esc(json.dumps(_EDGE_COLORS, ensure_ascii=False))};
const svg = document.getElementById('g');
const NS = 'http://www.w3.org/2000/svg';
const byId = {{}};
DATA.nodes.forEach(n => byId[n.id] = n);
let scale = 1, tx = 0, ty = 0;
const root = document.createElementNS(NS,'g'); svg.appendChild(root);
function el(tag, attrs) {{ const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }}
const edgeEls = [], nodeEls = [];
function redraw() {{
  root.innerHTML = '';
  DATA.edges.forEach(e => {{
    const a = byId[e.source], b = byId[e.target];
    if (!a || !b) return;
    const col = ECOLORS[e.relation] || '#475569';
    edgeEls.push(root.appendChild(el('line', {{x1:a.x,y1:a.y,x2:b.x,y2:b.y,
      stroke:col,class:'edge','stroke-dasharray': e.kind==='hidden'?'4 4':''}})));
    const mx=(a.x+b.x)/2, my=(a.y+b.y)/2;
    root.appendChild(el('text', {{x:mx,y:my-3,class:'elabel','text-anchor':'middle'}})).textContent = e.relation;
  }});
  DATA.nodes.forEach(n => {{
    const g = el('g', {{class:'node', transform:`translate(${{n.x}},${{n.y}})`}});
    g.appendChild(el('circle', {{r: n.type==='Article'?14:(n.type==='Verdict'?11:8),
      fill: COLORS[n.type]||'#64748b'}}));
    const t = el('text', {{y:-14,'text-anchor':'middle'}});
    let lab = n.label||n.id; if (lab.length>46) lab = lab.slice(0,44)+'…';
    t.textContent = lab; g.appendChild(t);
    const ti = el('title', {{}}); ti.textContent = (n.type?n.type+': ':'')+(n.label||'');
    g.appendChild(ti); nodeEls.push(g); root.appendChild(g);
    let drag=null;
    g.addEventListener('pointerdown', ev=>{{drag={{x:ev.clientX,y:ev.clientY,nx:n.x,ny:n.y}};
      g.setPointerCapture(ev.pointerId);}});
    g.addEventListener('pointermove', ev=>{{if(!drag)return;
      const r=svg.getBoundingClientRect(); const vb=svg.viewBox.baseVal;
      const k=vb.width/r.width;
      n.x=drag.nx+(ev.clientX-drag.x)*k; n.y=drag.ny+(ev.clientY-drag.y)*k; redraw();}});
    g.addEventListener('pointerup', ()=>drag=null);
  }});
}}
svg.addEventListener('wheel', ev=>{{
  ev.preventDefault();
  const vb=svg.viewBox.baseVal; const f = ev.deltaY>0?1.1:0.9;
  vb.width*=f; vb.height*=f; svg.setAttribute('viewBox',`${{vb.x}} ${{vb.y}} ${{vb.width}} ${{vb.height}}`);
}}, {{passive:false}});
redraw();
</script></body></html>"""


def save_graph_html(graph: dict[str, Any], path, title: str = "Граф связей") -> str:
    """Сохранить граф как HTML-файл; вернуть путь."""
    content = export_graph_html(graph, title=title)
    p = str(path)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p
