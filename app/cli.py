"""CLI-режим (ТЗ §1, §51: приложение запускаемо и через GUI, и через консоль).

Использование:
    python -m app.cli "https://example.com/article" [--mode research|controlled]
        [--json output.json] [--quiet]

Прогресс по этапам (§11) печатается в stdout; итог — человекочитаемое резюме.
Коды возврата: 0 — успех, 2 — фатальная ошибка анализа, 3 — проблема ввода.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_config
from app.core.models import AnalysisMode, PipelineStage, Report, VerdictLabel
from app.dependencies import ensure_or_exit
from app.logging_setup import setup_logging
from app.pipeline import FactCheckPipeline, PipelineOutcome
from app.report.generator import DISCLAIMER

LABEL_RU = {
    VerdictLabel.VERIFIED: "Подтверждено",
    VerdictLabel.LIKELY_TRUE: "Скорее всего верно",
    VerdictLabel.PARTIALLY_TRUE: "Частично верно",
    VerdictLabel.MISLEADING: "Вводит в заблуждение",
    VerdictLabel.LIKELY_FALSE: "Скорее всего неверно",
    VerdictLabel.FALSE: "Неверно",
    VerdictLabel.INSUFFICIENT_EVIDENCE: "Недостаточно доказательств",
}


def _print_progress(stage: PipelineStage, msg: str) -> None:
    suffix = f" — {msg}" if msg else ""
    print(f"[{stage.value}]{suffix}", flush=True)


def render_text(outcome: PipelineOutcome) -> str:
    r = outcome.report
    if r is None:
        return f"ОШИБКА: {outcome.error}"
    lines = [
        "=" * 72,
        f"ОТЧЁТ ПРОВЕРКИ: {r.input_url}",
        f"Режим: {r.mode.value} | Общий вердикт: "
        f"{LABEL_RU.get(r.overall_verdict, r.overall_verdict.value)} "
        f"(confidence {r.overall_confidence:.2f})",
        f"Время: {outcome.duration_s:.1f} с | Проект: {r.project_id}",
        "=" * 72,
        "",
        f"Резюме: {r.summary}",
        "",
        "Утверждения:",
    ]
    by_id = {c.id: c for c in r.claims}
    for v in r.verdicts:
        c = by_id.get(v.claim_id)
        t = c.text[:140] if c else v.claim_id
        lines.append(f"  • [{LABEL_RU.get(v.label, v.label.value)} "
                     f"{v.confidence:.2f}] {t}")
        if v.insufficient_reason:
            lines.append(f"      причина: {v.insufficient_reason}")
    lines.append("")
    lines.append(f"Источников привлечено: {len(r.sources)} "
                 f"(получено: {sum(1 for s in r.sources if s.fetched)})")
    for s in r.sources[:12]:
        q = next((x.overall for x in r.scores if x.source_id == s.id), None)
        qs = f" качество={q:.2f}" if q is not None else ""
        status = "OK" if s.fetched else (s.fetch_error or "не получен")
        lines.append(f"  - [{status}] {s.url[:100]}{qs}")
    if r.limitations:
        lines.append("")
        lines.append("Ограничения:")
        lines.extend(f"  ! {x}" for x in r.limitations)
    lines += ["", "Методология:", r.methodology, "", DISCLAIMER]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="factcheck",
                                description="Локальная проверка медицинских утверждений")
    p.add_argument("url", help="URL материала для проверки")
    p.add_argument("--mode", choices=["research", "controlled"],
                   default=get_config().research.mode_default,
                   help="режим исследования (по умолчанию из config.yaml)")
    p.add_argument("--json", metavar="PATH", help="сохранить отчёт JSON в файл")
    p.add_argument("--quiet", action="store_true", help="не печатать прогресс этапов")
    args = p.parse_args(argv)

    ensure_or_exit()
    cfg = get_config()
    setup_logging(cfg)

    url = args.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        print("URL должен начинаться с http:// или https://", file=sys.stderr)
        return 3

    mode = AnalysisMode.CONTROLLED if args.mode == "controlled" else AnalysisMode.RESEARCH
    progress = None if args.quiet else _print_progress
    pipe = FactCheckPipeline(cfg)
    outcome = pipe.run(url, mode, progress=progress)

    print()
    print(render_text(outcome))

    if outcome.report is not None and args.json:
        Path(args.json).write_text(outcome.report.model_dump_json(indent=2),
                                   encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.json}")

    if outcome.error:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
