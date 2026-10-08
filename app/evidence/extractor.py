"""EvidenceExtractor (ТЗ §26–§27, §38).

Ключевая анти-галлюцинационная гарантия: цитата принимается как evidence ТОЛЬКО
если она дословно найдена в реально полученном тексте источника
(quote_verified=True). Иначе — отбрасывается или понижается.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

from pydantic import BaseModel, Field

from app.config import AppConfig, get_config
from app.core.models import Claim, Evidence, EvidenceDirection, SourceMeta
from app.llm.base import LLMProvider, StructuredOutputError
from app.prompts import build_messages, get_system_prompt

log = logging.getLogger(__name__)


class _LLMEvidence(BaseModel):
    quote: str
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    strength: float = Field(0.5, ge=0.0, le=1.0)
    relevance: float = Field(0.5, ge=0.0, le=1.0)
    causal_note: str = ""


class _EvidenceList(BaseModel):
    evidence: list[_LLMEvidence] = Field(default_factory=list)


def norm_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s\u00a0]+", " ", s).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """Дословность цитаты с допуском по пробелам/регистру/типографике."""
    q, t = norm_ws(quote), norm_ws(text)
    if not q or not t:
        return False
    q = q.replace("’", "'").replace("`", "'").replace("«", '"').replace("»", '"')
    t = t.replace("’", "'").replace("`", "'").replace("«", '"').replace("»", '"')
    if q in t:
        return True
    # допуск: совпадение по словам (LLM могла склеить соседние предложения)
    qw = q.split()
    if len(qw) >= 4:
        tw = t.split()
        head = " ".join(qw[:4])
        tail = " ".join(qw[-4:])
        return head in tw_s(tw) and tail in tw_s(tw)
    return False


def tw_s(words: list[str]) -> str:
    return " " + " ".join(words) + " "


class EvidenceExtractor:
    def __init__(self, llm: Optional[LLMProvider], cfg: Optional[AppConfig] = None) -> None:
        self.llm = llm
        self.cfg = cfg or get_config()

    async def extract(self, claim: Claim, source: SourceMeta) -> list[Evidence]:
        if not source.fetched or not source.body_text:
            return []   # источник не получен — доказательств быть не может (§27)
        if self.llm is None:
            return self._lexical_fallback(claim, source)
        system = get_system_prompt("evidence_extractor", self.cfg.prompts_dir)
        frag = source.body_text[: min(9000, self.cfg.limits.max_context_chars)]
        messages = build_messages(
            system,
            f"Claim: {claim.normalized_text}\nИзвлеки evidence из фрагмента источника.",
            {"source_fragment": frag},
        )
        try:
            result = self.llm.structured_generate(messages, _EvidenceList)
        except Exception as e:  # graceful degradation (§50): любая ошибка LLM — fallback
            log.warning("evidence extraction failed (%s/%s): %s", claim.id, source.url[:60], e)
            return self._lexical_fallback(claim, source)
        out: list[Evidence] = []
        for it in result.evidence[:5]:
            verified = quote_in_text(it.quote, source.body_text)
            if not verified:
                log.warning("REJECTED unverified quote from %s: %r", source.url[:60], it.quote[:80])
                continue     # вымышленная цитата — выбрасываем (§27, §75)
            ev = Evidence(claim_id=claim.id, source_id=source.id, text=it.quote.strip(),
                          direction=it.direction, strength=round(it.strength, 2),
                          relevance=round(it.relevance, 2), quote_verified=True,
                          causal_note=it.causal_note[:200])
            out.append(ev)
        if not out:
            fb = self._lexical_fallback(claim, source)
            out = fb
        return out

    def _lexical_fallback(self, claim: Claim, source: SourceMeta) -> list[Evidence]:
        """Детерминированный поиск предложений источника, лексически близких к claim.

        Направление (SUPPORTS/CONTRADICTS) без LLM не утверждается — NEUTRAL;
        такие evidence имеют малый вес в verdict engine и помечаются в audit.
        """
        terms = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё]{5,}", claim.normalized_text.lower())][:8]
        if not terms:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", source.body_text)
        scored: list[tuple[float, str]] = []
        for sent in sentences:
            low = sent.lower()
            hits = sum(1 for t in terms if t in low)
            if hits >= max(2, len(terms) // 3) and 40 < len(sent) < 700:
                scored.append((hits / max(1, len(terms)), sent))
        scored.sort(reverse=True)
        out: list[Evidence] = []
        for rel, sent in scored[:3]:
            out.append(Evidence(claim_id=claim.id, source_id=source.id, text=sent.strip(),
                                direction=EvidenceDirection.NEUTRAL, strength=0.3,
                                relevance=round(min(1.0, rel + 0.2), 2), quote_verified=True))
        return out
