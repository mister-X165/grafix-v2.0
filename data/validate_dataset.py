"""Validate Grafix JSONL dataset against the contract in SCHEMA.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_TRIPLE_KEYS = ("subject", "relation", "object")


def load_relations(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def validate_file(path: Path, relations: set[str] | None) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"L{lineno}: invalid JSON ({e})")
                continue

            if not isinstance(obj, dict):
                errors.append(f"L{lineno}: expected object")
                continue

            for key in ("id", "text", "triples"):
                if key not in obj:
                    errors.append(f"L{lineno}: missing field '{key}'")

            eid = obj.get("id")
            if isinstance(eid, str):
                if eid in seen_ids:
                    errors.append(f"L{lineno}: duplicate id '{eid}'")
                seen_ids.add(eid)
            else:
                errors.append(f"L{lineno}: 'id' must be a string")

            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"L{lineno}: 'text' must be a non-empty string")

            triples = obj.get("triples")
            if not isinstance(triples, list):
                errors.append(f"L{lineno}: 'triples' must be a list")
                continue

            for ti, triple in enumerate(triples):
                if not isinstance(triple, dict):
                    errors.append(f"L{lineno}: triples[{ti}] must be an object")
                    continue
                for key in REQUIRED_TRIPLE_KEYS:
                    val = triple.get(key)
                    if not isinstance(val, str) or not val.strip():
                        errors.append(f"L{lineno}: triples[{ti}].{key} must be a non-empty string")
                rel = triple.get("relation")
                if relations is not None and isinstance(rel, str) and rel not in relations:
                    errors.append(f"L{lineno}: unknown relation '{rel}' (not in relations.txt)")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Grafix JSONL dataset")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("sample.jsonl")),
        help="Path to JSONL file",
    )
    parser.add_argument(
        "--relations",
        default=str(Path(__file__).with_name("relations.txt")),
        help="Closed relation vocabulary (optional file)",
    )
    parser.add_argument(
        "--allow-open-relations",
        action="store_true",
        help="Do not require relations to be in relations.txt",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    relations = None if args.allow_open_relations else load_relations(Path(args.relations))
    errors = validate_file(path, relations)
    if errors:
        print(f"FAIL: {len(errors)} error(s) in {path}")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
