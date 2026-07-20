"""Import all STELLAR Alexey dossiers into Grafix JSONL + Gemma SFT export.

Pairing convention:
  graph_text.md   + graph_text_2.md  -> dossier 1 (structured + prose)
  graph_text_N.md + graph_text_{N+1}.md for odd N>=3 -> prose + annotation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "STELLAR-Complete-Clarity-main" / "STELLAR-Complete-Clarity-main" / "texts"
OUT_FULL = ROOT / "data" / "stellar_alexey.jsonl"
OUT_CHUNKS = ROOT / "data" / "stellar_alexey_chunks.jsonl"
OUT_SFT = ROOT / "data" / "gemma_sft.jsonl"
RELATIONS = ROOT / "data" / "relations.txt"

SYSTEM_SFT = (
    "Ты извлекаешь граф знаний из текста. "
    'Верни только JSON: {"triples":[{"subject":"...","relation":"...","object":"..."}]} '
    "Без markdown и пояснений. Только явные факты из текста."
)


def _canon(name: str) -> str:
    name = name.strip()
    if (name.startswith("«") and name.endswith("»")) or (
        name.startswith('"') and name.endswith('"')
    ):
        name = name[1:-1].strip()
    name = re.sub(r"\s+", " ", name)
    replacements = {"Лidия Френкель": "Лидия Френкель"}
    return replacements.get(name, name)


def _norm_rel(rel: str) -> str:
    rel = rel.strip().lower().replace(" ", "_")
    rel = re.sub(r"[^\wа-яё_+-]+", "", rel, flags=re.I)
    return rel or "связан_с"


def _dedupe(triples: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for t in triples:
        key = (t["subject"], t["relation"], t["object"])
        if key in seen or not all(key):
            continue
        seen.add(key)
        out.append(t)
    return out


def parse_explicit_triples_arrows(md: str) -> list[dict]:
    """graph_text.md style: - A — rel → B"""
    triples: list[dict] = []
    in_section = False
    for line in md.splitlines():
        if re.search(r"явные\s+связи", line, re.I):
            in_section = True
            continue
        if in_section and line.startswith("##"):
            break
        if not in_section:
            continue
        raw = line.strip()
        if not raw.startswith("-") and not raw.startswith("*"):
            continue
        m = re.match(r"^[-*]\s*(.+?)\s+[—–-]\s*(\S+)\s+[→>]\s*(.+)$", raw)
        if not m:
            continue
        subj, rel, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        chunks = [c.strip() for c in rest.split(";") if c.strip()]
        first = chunks[0]
        if first.startswith("(") and first.endswith(")"):
            first = first[1:-1].strip()
        else:
            first = re.sub(r"\s*\([^)]*\)\s*$", "", first).strip()
        if not first:
            continue
        triples.append({"subject": _canon(subj), "relation": _norm_rel(rel), "object": _canon(first)})
        for chunk in chunks[1:]:
            m2 = re.match(r"(\S+)\s+[→>]\s*(.+)$", chunk)
            if m2:
                triples.append(
                    {
                        "subject": _canon(subj),
                        "relation": _norm_rel(m2.group(1)),
                        "object": _canon(m2.group(2)),
                    }
                )
    return _dedupe(triples)


def parse_explicit_triples_midash(md: str) -> list[dict]:
    """Annotation style: A — rel — B (one per line or ·-separated)."""
    triples: list[dict] = []
    body = md
    msec = re.search(r"явные\s+триплет", md, re.I)
    if msec:
        body = md[msec.start() :]
        cut = re.search(r"выводимые|ловушки\s+для", body, re.I)
        if cut:
            body = body[: cut.start()]

    parts = re.split(r"[·\n]", body)
    pat = re.compile(r"^\s*(.+?)\s+[—–-]\s*(.+?)\s+[—–-]\s*(.+?)\s*$")
    for part in parts:
        part = part.strip().strip(".")
        if not part or len(part) < 5:
            continue
        if re.match(r"^(сущност|организац|явные|люди|места|продукт|идеолог|документ)", part, re.I):
            continue
        m = pat.match(part)
        if not m:
            continue
        s, r, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        o = re.sub(r"\s*\([^)]*\)\s*$", "", o).strip()
        if len(s) < 2 or len(o) < 2 or len(r) < 2:
            continue
        if len(s) > 120 or len(o) > 160:
            continue
        triples.append({"subject": _canon(s), "relation": _norm_rel(r), "object": _canon(o)})
    return _dedupe(triples)


def prose_without_meta(md: str) -> str:
    cut = re.search(r"##\s*5\.\s*Явные", md)
    body = md[: cut.start()] if cut else md
    lines = []
    for line in body.splitlines():
        if line.startswith("|") or line.startswith("---"):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = line.strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def clean_prose(text: str) -> str:
    text = text.strip()
    # Drop instructional preambles like "Тоже самое для"
    text = re.sub(r"^тоже\s+самое\s+для\s*\n?", "", text, flags=re.I)
    return text.strip()


def to_sft_row(row_id: str, text: str, triples: list[dict]) -> dict:
    assistant = json.dumps({"triples": triples}, ensure_ascii=False)
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": SYSTEM_SFT},
            {"role": "user", "content": f"Текст:\n{text.strip()}\n\nJSON:"},
            {"role": "assistant", "content": assistant},
        ],
    }


def chunks_from_triples(dossier_id: str, triples: list[dict], limit: int = 8) -> list[dict]:
    """Make short SFT-friendly examples from gold triples."""
    rows = []
    for i, t in enumerate(triples[:limit], start=1):
        s, r, o = t["subject"], t["relation"], t["object"]
        text = f"{s} — {r.replace('_', ' ')} — {o}."
        rows.append(
            {
                "id": f"{dossier_id}-chunk-{i:02d}",
                "text": text,
                "triples": [t],
            }
        )
    return rows


SHORT_TEMPLATES = [
    ("Эльвира Ковач защищалась в Лаборатории К-17.", "Эльвира Ковач", "защищалась_в", "Лаборатория К-17"),
    ("Донат Ившич основал компанию Северный меридиан.", "Донат Ившич", "основал", "Северный меридиан"),
    ("Виктор Алешин основал компанию Гелиотрон.", "Виктор Алешин", "основал", "Гелиотрон"),
    ("АК-12 создан в рамках программы Ратник.", "АК-12", "создан_в_рамках", "Ратник"),
    ("Хуан Перон избран президентом Аргентины.", "Хуан Перон", "избран_президентом", "Аргентина"),
]


def merge_relations(new_rels: set[str]) -> None:
    existing = set()
    if RELATIONS.exists():
        existing = {l.strip() for l in RELATIONS.read_text(encoding="utf-8").splitlines() if l.strip()}
    merged = sorted(existing | {r for r in new_rels if r})
    RELATIONS.write_text("\n".join(merged) + "\n", encoding="utf-8")


def discover_pairs(alexey: Path) -> list[tuple[str, Path, Path | None, str]]:
    """Return list of (dossier_id, prose_path, annotation_path|None, kind)."""
    pairs: list[tuple[str, Path, Path | None, str]] = []

    d1_struct = alexey / "graph_text.md"
    d1_prose = alexey / "graph_text_2.md"
    if d1_struct.exists():
        pairs.append(("d1", d1_prose if d1_prose.exists() else d1_struct, d1_struct, "arrows"))

    # Odd N >= 3: prose N + annotation N+1
    nums = []
    for p in alexey.glob("graph_text_*.md"):
        m = re.search(r"graph_text_(\d+)\.md$", p.name)
        if m:
            nums.append(int(m.group(1)))
    for n in sorted(set(nums)):
        if n < 3 or n % 2 == 0:
            continue
        prose = alexey / f"graph_text_{n}.md"
        ann = alexey / f"graph_text_{n + 1}.md"
        if prose.exists() and ann.exists():
            pairs.append((f"d{(n + 1) // 2}", prose, ann, "midash"))
        elif prose.exists():
            pairs.append((f"d{(n + 1) // 2}", prose, None, "midash"))

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    args = parser.parse_args()
    src = Path(args.src)
    alexey = src / "Alexey"
    if not alexey.exists():
        print(f"Missing {alexey}", file=sys.stderr)
        return 1

    pairs = discover_pairs(alexey)
    if not pairs:
        print("No dossier pairs found", file=sys.stderr)
        return 1

    full_rows: list[dict] = []
    chunk_rows: list[dict] = []
    report: list[str] = []

    for dossier_id, prose_path, ann_path, kind in pairs:
        prose_raw = prose_path.read_text(encoding="utf-8") if prose_path.exists() else ""
        if dossier_id == "d1" and kind == "arrows" and ann_path:
            # ann_path is structured md with arrows; prose may be graph_text_2
            structured = ann_path.read_text(encoding="utf-8")
            triples = parse_explicit_triples_arrows(structured)
            prose_text = clean_prose(prose_raw) if prose_path.name != "graph_text.md" else prose_without_meta(structured)
            # Also keep structured summary text
            full_rows.append(
                {
                    "id": f"stellar-{dossier_id}-structured",
                    "text": prose_without_meta(structured)[:6000],
                    "triples": triples,
                    "source": str(ann_path.relative_to(ROOT)),
                }
            )
        else:
            triples = parse_explicit_triples_midash(ann_path.read_text(encoding="utf-8")) if ann_path else []
            prose_text = clean_prose(prose_raw)

        if not triples:
            report.append(f"{dossier_id}: WARN 0 triples from {ann_path.name if ann_path else '?'}")
            continue

        full_rows.append(
            {
                "id": f"stellar-{dossier_id}-prose",
                "text": prose_text[:7000],
                "triples": triples,
                "source": str(prose_path.relative_to(ROOT)),
            }
        )
        if ann_path and dossier_id != "d1":
            full_rows.append(
                {
                    "id": f"stellar-{dossier_id}-annotated",
                    "text": prose_text[:7000],
                    "triples": triples,
                    "source": str(ann_path.relative_to(ROOT)),
                }
            )

        chunk_rows.extend(chunks_from_triples(f"stellar-{dossier_id}", triples, limit=6))
        report.append(
            f"{dossier_id}: {len(triples)} triples | prose={prose_path.name} | ann={ann_path.name if ann_path else '-'}"
        )

    for i, (text, s, r, o) in enumerate(SHORT_TEMPLATES, start=1):
        chunk_rows.append(
            {
                "id": f"stellar-hand-{i:02d}",
                "text": text,
                "triples": [{"subject": s, "relation": r, "object": o}],
            }
        )

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FULL.open("w", encoding="utf-8") as f:
        for row in full_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with OUT_CHUNKS.open("w", encoding="utf-8") as f:
        for row in chunk_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sft_rows = []
    for row in full_rows + chunk_rows:
        if not row.get("triples"):
            continue
        # Cap very large triple lists for a single SFT example (keep quality)
        triples = row["triples"]
        if len(triples) > 40:
            triples = triples[:40]
        sft_rows.append(to_sft_row(row["id"], row["text"], triples))
    with OUT_SFT.open("w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_rels = {t["relation"] for row in full_rows for t in row["triples"]}
    all_rels |= {t["relation"] for row in chunk_rows for t in row["triples"]}
    merge_relations(all_rels)

    print("=== STELLAR import report ===")
    for line in report:
        print(line)
    print(f"Wrote {OUT_FULL} ({len(full_rows)} docs)")
    print(f"Wrote {OUT_CHUNKS} ({len(chunk_rows)} short examples)")
    print(f"Wrote {OUT_SFT} ({len(sft_rows)} chat SFT examples for Gemma LoRA)")
    print(f"Updated {RELATIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
