"""Экспорт отчёта в Markdown / HTML (§ расширение функционала: «Сохранить отчёт»).

Требования стабильности и приватности:
- экспорт полностью детерминирован, не обращается к сети и LLM;
- URL в отчёте — только из реально полученных источников (уже гарантировано
  quality gate'ом в generator.py); сюда попадают данные готового Report;
- HTML-файл автономен (inline-CSS), без внешних ресурсов — его можно открыть
  офлайн и передать другому человеку.
"""

from __future__ import annotations

import html
from datetime import datetime

from app.core.models import EvidenceDirection, Report, SourceMeta, VerdictLabel
from app.report.generator import DISCLAIMER

LABEL_RU = {
    VerdictLabel.VERIFIED: "Подтверждено",
    VerdictLabel.LIKELY_TRUE: "Скорее всего верно",
    VerdictLabel.PARTIALLY_TRUE: "Частично верно",
    VerdictLabel.MISLEADING: "Вводит в заблуждение",
    VerdictLabel.LIKELY_FALSE: "Скорее всего неверно",
    VerdictLabel.FALSE: "Недостоверно",
    VerdictLabel.INSUFFICIENT_EVIDENCE: "Недостаточно доказательств",
}

_TONE = {
    VerdictLabel.VERIFIED: "#15803d", VerdictLabel.LIKELY_TRUE: "#16a34a",
    VerdictLabel.PARTIALLY_TRUE: "#b45309", VerdictLabel.MISLEADING: "#b45309",
    VerdictLabel.LIKELY_FALSE: "#dc2626", VerdictLabel.FALSE: "#b91c1c",
    VerdictLabel.INSUFFICIENT_EVIDENCE: "#6b7280",
}


def _score(report: Report, source_id: str) -> float | None:
    for sc in report.scores:
        if sc.source_id == source_id:
            return sc.overall
    return None


def export_markdown(report: Report) -> str:
    """Отчёт в Markdown (для документов, wiki, мессенджеров)."""
    by_id_src = {s.id: s for s in report.sources}
    by_id_claims = {c.id: c for c in report.claims}
    lines: list[str] = []
    lines.append(f"# Отчёт проверки достоверности: {report.input_url}")
    lines.append("")
    lines.append(f"- **Общий вердикт:** {LABEL_RU.get(report.overall_verdict, report.overall_verdict.value)}"
                 f" (уверенность {report.overall_confidence:.0%})")
    lines.append(f"- **Режим:** {report.mode.value}")
    lines.append(f"- **Сгенерирован:** {report.generated_at}")
    if report.model_info.model:
        lines.append(f"- **Модель:** {report.model_info.backend} / {report.model_info.model}")
    lines.append("")
    lines.append("## Резюме")
    lines.append("")
    lines.append(report.summary or "—")
    lines.append("")
    lines.append("## Утверждения и вердикты")
    for v in report.verdicts:
        cl = by_id_claims.get(v.claim_id)
        title = cl.text if cl else v.claim_id
        lines.append("")
        lines.append(f"### {LABEL_RU.get(v.label, v.label.value)}"
                     f" · уверенность {v.confidence:.0%}")
        lines.append("")
        lines.append(f"> {title}")
        lines.append("")
        if v.explanation:
            lines.append(v.explanation)
            lines.append("")
        if v.insufficient_reason:
            lines.append(f"_Причина недостаточности:_ {v.insufficient_reason}")
            lines.append("")
        evs = [e for e in report.evidence if e.claim_id == v.claim_id]
        for direction, head in ((EvidenceDirection.SUPPORTS, "Доказательства ЗА"),
                                (EvidenceDirection.CONTRADICTS, "Доказательства ПРОТИВ")):
            de = [e for e in evs if e.direction == direction]
            if not de:
                continue
            lines.append(f"**{head}:**")
            for e in de:
                src = by_id_src.get(e.source_id)
                mark = " ✓цитата проверена" if e.quote_verified else ""
                link = f" — [{src.url}]({src.url})" if src else ""
                lines.append(f"- «{e.text}» (сила {e.strength:.2f}{mark}){link}")
            lines.append("")
    if report.limitations:
        lines.append("## Ограничения проверки")
        lines.extend(f"- {x}" for x in report.limitations)
        lines.append("")
    used_ids = {e.source_id for e in report.evidence}
    used = [s for s in report.sources if s.id in used_ids and s.fetched]
    if used:
        lines.append("## Источники")
        for s in used:
            q = _score(report, s.id)
            qs = f", качество {q:.2f}" if q is not None else ""
            lines.append(f"- [T{s.tier}{qs}] [{s.title or s.url}]({s.url})")
        lines.append("")
    lines.append("## Методология")
    lines.append("")
    lines.append(report.methodology or "—")
    lines.append("")
    lines.append(f"---\n*{DISCLAIMER}*")
    return "\n".join(lines)


_CSS = """
body{font-family:'Segoe UI',system-ui,Arial,sans-serif;background:#f4f6fb;color:#1f2430;
     margin:0;padding:24px;}
.page{max-width:860px;margin:0 auto;background:#fff;border-radius:14px;
      box-shadow:0 2px 14px rgba(31,41,55,.08);padding:28px 34px;}
h1{font-size:22px;margin:0 0 4px;} h2{font-size:17px;margin:26px 0 8px;color:#374151;}
.meta{color:#6b7280;font-size:13px;margin-bottom:14px;}
.badge{display:inline-block;color:#fff;font-weight:700;border-radius:8px;
       padding:8px 16px;font-size:17px;margin:6px 0;}
.summary{background:#f7f9fc;border:1px solid #e6eaf2;border-radius:10px;
         padding:14px 16px;line-height:1.5;}
.claim{border:1px solid #e6eaf2;border-left:5px solid #ccc;border-radius:10px;
       padding:12px 16px;margin:12px 0;background:#fbfcfe;}
.claim .text{font-weight:600;margin:2px 0 6px;}
.ev{margin:6px 0;padding:8px 10px;border-radius:8px;font-size:14px;line-height:1.45;}
.ev.for{background:#ecfdf5;border:1px solid #bbf7d0;}
.ev.against{background:#fef2f2;border:1px solid #fecaca;}
.src{font-size:13px;margin:4px 0;color:#374151;word-break:break-all;}
.lim{color:#b45309;font-size:14px;margin:4px 0;}
a{color:#2f6fed;text-decoration:none;} a:hover{text-decoration:underline;}
.small{font-size:12px;color:#6b7280;}
.footer{margin-top:26px;border-top:1px solid #e6eaf2;padding-top:12px;
        font-size:12px;color:#6b7280;}
"""


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def export_html(report: Report) -> str:
    """Автономный HTML-отчёт (сохраняется как файл, открывается офлайн)."""
    by_id_src = {s.id: s for s in report.sources}
    by_id_claims = {c.id: c for c in report.claims}
    tone = _TONE.get(report.overall_verdict, "#6b7280")
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
                 "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                 f"<title>Отчёт проверки — {_esc(report.input_url)[:80]}</title>"
                 f"<style>{_CSS}</style></head><body><div class='page'>")
    parts.append(f"<h1>Отчёт проверки достоверности</h1>"
                 f"<div class='meta'>{_esc(report.input_url)}</div>")
    parts.append(f"<div class='badge' style='background:{tone}'>"
                 f"{LABEL_RU.get(report.overall_verdict, report.overall_verdict.value)}"
                 f" · {report.overall_confidence:.0%}</div>")
    gen = report.generated_at[:10]
    parts.append(f"<div class='meta'>Режим: {_esc(report.mode.value)} · "
                 f"Дата: {_esc(gen)} · Проверено утверждений: {len(report.verdicts)}</div>")
    parts.append("<h2>Резюме</h2>"
                 f"<div class='summary'>{_esc(report.summary).replace(chr(10), '<br>')}</div>")

    parts.append("<h2>Утверждения и вердикты</h2>")
    for v in report.verdicts:
        vtone = _TONE.get(v.label, "#6b7280")
        cl = by_id_claims.get(v.claim_id)
        text = _esc(cl.text if cl else v.claim_id)
        parts.append(f"<div class='claim' style='border-left-color:{vtone}'>"
                     f"<b style='color:{vtone}'>{LABEL_RU.get(v.label, v.label.value)}"
                     f" · {v.confidence:.0%}</b>"
                     f"<div class='text'>{text}</div>")
        if v.explanation:
            parts.append(f"<div>{_esc(v.explanation)}</div>")
        if v.insufficient_reason:
            parts.append(f"<div class='lim'>Причина недостаточности: "
                         f"{_esc(v.insufficient_reason)}</div>")
        evs = [e for e in report.evidence if e.claim_id == v.claim_id]
        for e in evs:
            cls = "for" if e.direction == EvidenceDirection.SUPPORTS else \
                  "against" if e.direction == EvidenceDirection.CONTRADICTS else None
            if cls is None:
                continue
            src = by_id_src.get(e.source_id)
            check = " · цитата проверена" if e.quote_verified else ""
            link = (f"<div class='src'><a href='{_esc(src.url)}'>{_esc(src.url)}</a></div>"
                    if src else "")
            parts.append(f"<div class='ev {cls}'>«{_esc(e.text)}»"
                         f"<div class='small'>сила {e.strength:.2f} · "
                         f"релевантность {e.relevance:.2f}{check}</div>{link}</div>")
        parts.append("</div>")

    used_ids = {e.source_id for e in report.evidence}
    used = [s for s in report.sources if s.id in used_ids and s.fetched]
    if used:
        parts.append("<h2>Источники</h2>")
        for s in used:
            q = _score(report, s.id)
            qs = f" · качество {q:.2f}" if q is not None else ""
            parts.append(f"<div class='src'>[T{s.tier}{qs}] "
                         f"<a href='{_esc(s.url)}'>{_esc(s.title or s.url)}</a></div>")

    if report.limitations:
        parts.append("<h2>Ограничения</h2>")
        for x in report.limitations:
            parts.append(f"<div class='lim'>— {_esc(x)}</div>")

    parts.append(f"<h2>Методология</h2><div class='small'>{_esc(report.methodology)}"
                 .replace("\n", "<br>") + "</div>")
    parts.append(f"<div class='footer'>{_esc(DISCLAIMER)}</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


def default_filename(report: Report, ext: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    host = report.input_url.split("//")[-1].split("/")[0] or "report"
    safe = "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in host)[:40]
    return f"factcheck-{safe}-{stamp}.{ext}"
