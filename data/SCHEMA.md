# Grafix dataset contract

One example = one text + a list of triples. File format: **JSONL** (one JSON object per line).

## Schema

```json
{
  "id": "ex001",
  "text": "Маша работает в Яндексе и живёт в Москве.",
  "triples": [
    {"subject": "Маша", "relation": "работает_в", "object": "Яндекс"},
    {"subject": "Маша", "relation": "живёт_в", "object": "Москва"}
  ]
}
```

| Field | Type | Rules |
|-------|------|--------|
| `id` | string | Unique within the file |
| `text` | string | Source sentence(s); non-empty |
| `triples` | array | May be empty; each item has `subject`, `relation`, `object` |
| `subject` / `object` | string | Canonical form (nominative), not inflected surface forms |
| `relation` | string | Prefer values from `relations.txt` (closed vocab) |

## Guidelines

- Language: primarily Russian.
- Only explicit facts from the text — no inference.
- Prefer closed `relation` list for tiny MicroGPT.
- Canonical entity names: «Яндекс», not «Яндексе».

## Training serialization

Text and triples are joined into one target sequence:

```
{text}<SEP>{subject}|{relation}|{object} ; ...
```

Special markers are defined in code (`model/triples.py`).
