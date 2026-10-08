# Модель данных (Pydantic + SQLite)

## Перечисления

- `ClaimType`: MEDICAL_FACT, STATISTICAL_CLAIM, CAUSAL_CLAIM, SAFETY_CLAIM, EFFICACY_CLAIM, DIAGNOSTIC_CLAIM, TREATMENT_CLAIM, PREVENTION_CLAIM, SCIENTIFIC_CLAIM, ADVERTISEMENT, OPINION, PREDICTION, QUOTE, ANALOGY
- `EvidenceDirection`: SUPPORTS, CONTRADICTS, NEUTRAL
- `VerdictLabel`: VERIFIED, LIKELY_TRUE, PARTIALLY_TRUE, MISLEADING, LIKELY_FALSE, FALSE, INSUFFICIENT_EVIDENCE
- `AnalysisMode`: CONTROLLED, RESEARCH
- `SourceTier`: 1..8 (иерархия доказательств §21)
- `ContentNature`: INFORMATION, ADVERTISEMENT, NATIVE_ADVERTISEMENT, SPONSORED_CONTENT, AFFILIATE_CONTENT

## Основные модели (`app/core/models.py`)

```
Claim(id, document_id, text, normalized_text, claim_type, importance 0..1,
      check_required, context, statistical_flags[], hedging_observed)
SourceMeta(id, url, title, author, organization, publication_date, retrieved_at,
      source_type, doi, pmid, tier, study_type, funding, conflicts_of_interest,
      commercial_interest, content_nature, original_source_url, is_derivative,
      independent_cluster_key, fetched: bool, fetch_error)
SourceScores(authority, methodology, recency, relevance, independence, transparency,
      overall — все 0..1; хранятся и исходные параметры, и итог (§22))
Evidence(id, claim_id, source_id, text(дословная цитата), direction, strength 0..1,
      relevance 0..1, quote_verified: bool)
SearchQuery(id, project_id, claim_id, provider, query, purpose[neutral|scientific|
      primary|review|guideline|contradiction], result_count, error)
Verdict(claim_id, label, confidence 0..1, explanation, supporting_evidence_ids,
      contradicting_evidence_ids, insufficient_reason, correlation_vs_causation_note)
Report(project_id, overall_verdict, overall_confidence, summary, claims, verdicts,
      sources, limitations[], methodology, generated_at, model_info)
AuditEvent(ts, project_id, stage, event, detail_json)
```

## SQLite (`app/storage/db.py`)

Таблицы: `projects, documents, sources, claims, evidence, search_queries, verdicts, reports, audit_events, doc_cache(url, content_hash, body BLOB, meta_json, retrieved_at)`.
Сложные поля сериализуются JSON-колонками (паттерн `graph/history.py` из Grafix). Кэш проверяется по паре (url, content_hash) — повторно материал не скачивается (§48).
