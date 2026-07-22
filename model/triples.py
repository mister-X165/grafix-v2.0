"""Serialize / parse Grafix triples for MicroGPT sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEP = "<SEP>"
TRIPLE_SEP = " ; "
FIELD_SEP = "|"


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    kind: str = "explicit"  # explicit | hidden | false
    evidence: str = ""
    confidence: float | None = None

    def encode(self) -> str:
        return f"{self.subject.strip()}{FIELD_SEP}{self.relation.strip()}{FIELD_SEP}{self.object.strip()}"

    @classmethod
    def parse(cls, raw: str) -> Triple | None:
        parts = [p.strip() for p in raw.strip().split(FIELD_SEP)]
        if len(parts) != 3 or not all(parts):
            return None
        return cls(subject=parts[0], relation=parts[1], object=parts[2])

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "kind": self.kind or "explicit",
        }
        if self.evidence:
            out["evidence"] = self.evidence
        if self.confidence is not None:
            out["confidence"] = self.confidence
        return out


def normalize_edge_kind(value: Any) -> str:
    """Map model output to explicit | hidden | false."""
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"hidden", "latent", "implied", "implicit", "скрытая", "скрытый", "скрытое", "латентная", "подразумеваемая"}:
        return "hidden"
    if raw in {
        "false",
        "fake",
        "spurious",
        "invalid",
        "denied",
        "refuted",
        "ложная",
        "ложный",
        "ложное",
        "мнимая",
        "опровергнутая",
        "ошибочная",
    }:
        return "false"
    return "explicit"


def encode_example(text: str, triples: list[Triple]) -> str:
    body = TRIPLE_SEP.join(t.encode() for t in triples)
    return f"{text.strip()}{SEP}{body}"


def encode_prompt(text: str) -> str:
    """Prefix used at inference: model continues after SEP."""
    return f"{text.strip()}{SEP}"


def parse_triples_suffix(suffix: str) -> list[Triple]:
    """Parse generated text after SEP into triples; skip malformed chunks."""
    triples: list[Triple] = []
    for chunk in suffix.split(TRIPLE_SEP):
        chunk = chunk.strip().rstrip(";")
        if not chunk:
            continue
        # Model may emit trailing junk after last triple
        if FIELD_SEP not in chunk:
            continue
        # Take only first three fields if extra separators appear
        parts = chunk.split(FIELD_SEP)
        if len(parts) >= 3:
            candidate = FIELD_SEP.join(parts[:3])
            t = Triple.parse(candidate)
            if t is not None:
                triples.append(t)
    return triples


def parse_full_sequence(sequence: str) -> tuple[str, list[Triple]]:
    if SEP not in sequence:
        return sequence.strip(), []
    text, suffix = sequence.split(SEP, 1)
    return text.strip(), parse_triples_suffix(suffix)
