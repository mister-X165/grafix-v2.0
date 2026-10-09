"""ResearchPlanner (ТЗ §16): для каждого claim — план запросов.

LLM формулирует запросы; если LLM недоступна/ошиблась — детерминированный
шаблонный план (термины claim + фиксированные цели). Это НЕ фиктивные данные:
запросы просто упрощены и помечаются в audit.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from app.config import AppConfig, get_config
from app.core.models import Claim, QueryPurpose, SearchQuery
from app.llm.base import LLMProvider, StructuredOutputError
from app.prompts import build_messages, get_system_prompt

log = logging.getLogger(__name__)


class _PlannedQuery(BaseModel):
    query: str
    purpose: QueryPurpose = QueryPurpose.NEUTRAL


class _Plan(BaseModel):
    queries: list[_PlannedQuery] = Field(default_factory=list)


_STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "with", "is", "are",
         "that", "this", "it", "as", "by", "to", "с", "и", "в", "на", "для", "как"}


def key_terms(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-]{3,}", text.lower())
    out: list[str] = []
    for w in words:
        if w not in _STOP and w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


class ResearchPlanner:
    def __init__(self, llm: Optional[LLMProvider], cfg: Optional[AppConfig] = None) -> None:
        self.llm = llm
        self.cfg = cfg or get_config()

    def plan(self, claim: Claim, project_id: str) -> list[SearchQuery]:
        """Синхронная версия (CLI/тесты)."""
        if self.llm is not None:
            try:
                plan = self.llm.structured_generate(self._messages(claim), _Plan)
                return self._to_queries(plan, claim, project_id)
            except Exception as e:  # graceful degradation (§50): любая ошибка LLM — fallback
                log.warning("planner LLM failed for %s: %s — template plan", claim.id, e)
        return self._template_plan(claim, project_id)

    async def plan_async(self, claim: Claim, project_id: str) -> list[SearchQuery]:
        """Асинхронная версия для pipeline: LLM вне event loop."""
        if self.llm is not None:
            try:
                plan = await self.llm.structured_generate_async(self._messages(claim), _Plan)
                res = self._to_queries(plan, claim, project_id)
                if res:
                    return res
            except Exception as e:  # graceful degradation (§50): любая ошибка LLM — fallback
                log.warning("planner LLM failed for %s: %s — template plan", claim.id, e)
        return self._template_plan(claim, project_id)

    def _messages(self, claim: Claim) -> list[dict]:
        system = get_system_prompt("research_planner", self.cfg.prompts_dir)
        return build_messages(
            system,
            f"Составь исследовательский план для claim: {claim.normalized_text}",
            {"claim": claim.text + ("\nКонтекст: " + claim.context if claim.context else "")},
        )

    def _to_queries(self, plan: _Plan, claim: Claim, project_id: str) -> list[SearchQuery]:
        items = [q for q in plan.queries if q.query.strip()]
        if not items:
            return []
        seen_purposes = {q.purpose for q in items}
        # гарантируем наличие contradiction-запроса (§17B, §18)
        if QueryPurpose.CONTRADICTION not in seen_purposes:
            items.append(_PlannedQuery(query=self._fallback(claim, QueryPurpose.CONTRADICTION),
                                       purpose=QueryPurpose.CONTRADICTION))
        return [SearchQuery(project_id=project_id, claim_id=claim.id,
                            query=q.query.strip()[:200],
                            purpose=q.purpose) for q in items[:8]]

    def _template_plan(self, claim: Claim, project_id: str) -> list[SearchQuery]:
        """Публичное имя (используется pipeline как fallback)."""
        return self.template_plan(claim, project_id)

    def template_plan(self, claim: Claim, project_id: str) -> list[SearchQuery]:
        # детерминированный шаблонный план
        purposes = [QueryPurpose.NEUTRAL, QueryPurpose.SCIENTIFIC, QueryPurpose.PRIMARY,
                    QueryPurpose.REVIEW, QueryPurpose.GUIDELINE, QueryPurpose.CONTRADICTION]
        return [SearchQuery(project_id=project_id, claim_id=claim.id,
                            query=self._fallback(claim, p), purpose=p) for p in purposes]

    def _fallback(self, claim: Claim, purpose: QueryPurpose) -> str:
        core = " ".join(key_terms(claim.normalized_text))
        suffix = {
            QueryPurpose.NEUTRAL: "",
            QueryPurpose.SCIENTIFIC: "clinical evidence",
            QueryPurpose.PRIMARY: "randomized controlled trial",
            QueryPurpose.REVIEW: "systematic review meta-analysis",
            QueryPurpose.GUIDELINE: "guideline WHO CDC",
            QueryPurpose.CONTRADICTION: "no benefit adverse effects limitations",
        }[purpose]
        return f"{core} {suffix}".strip()[:200]
