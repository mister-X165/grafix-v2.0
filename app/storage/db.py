"""SQLite-хранилище: история анализов, кэш документов, audit trail (ТЗ §46–§48, §105).

Паттерн заимствован из graph/history.py Grafix: JSON-колонки, PRAGMA table_info
для миграций, check_same_thread=False (GUI-потоки). Один экземпляр на приложение.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import AppConfig, get_config
from app.core.models import (AnalysisMode, AuditEvent, Document, Evidence, ModelInfo,
                             Project, Report, SearchQuery, SourceMeta, SourceScores,
                             Verdict)

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects(
    id TEXT PRIMARY KEY, input_url TEXT, mode TEXT, created_at TEXT,
    status TEXT, model_json TEXT);
CREATE TABLE IF NOT EXISTS documents(
    id TEXT PRIMARY KEY, project_id TEXT, url TEXT, title TEXT, author TEXT,
    published TEXT, text TEXT, content_hash TEXT, retrieved_at TEXT, meta_json TEXT);
CREATE TABLE IF NOT EXISTS sources(
    id TEXT PRIMARY KEY, project_id TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS claims(
    id TEXT PRIMARY KEY, project_id TEXT, document_id TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS evidence(
    id TEXT PRIMARY KEY, project_id TEXT, claim_id TEXT, source_id TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS source_scores(
    source_id TEXT PRIMARY KEY, project_id TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS search_queries(
    id TEXT PRIMARY KEY, project_id TEXT, claim_id TEXT, provider TEXT, query TEXT,
    purpose TEXT, result_count INTEGER, error TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS verdicts(
    rowid INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, claim_id TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS reports(
    project_id TEXT PRIMARY KEY, generated_at TEXT, overall_verdict TEXT,
    overall_confidence REAL, data_json TEXT);
CREATE TABLE IF NOT EXISTS audit_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, project_id TEXT, stage TEXT,
    event TEXT, detail_json TEXT);
CREATE TABLE IF NOT EXISTS doc_cache(
    url TEXT PRIMARY KEY, content_hash TEXT, content_type TEXT, body BLOB,
    retrieved_at TEXT);
CREATE INDEX IF NOT EXISTS idx_claims_proj ON claims(project_id);
CREATE INDEX IF NOT EXISTS idx_evidence_proj ON evidence(project_id);
CREATE INDEX IF NOT EXISTS idx_audit_proj ON audit_events(project_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    """Тонкая обёртка над SQLite с типизированными save/load методами."""

    _instances: dict[str, "Database"] = {}
    _lock = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tx = threading.RLock()
        with self._tx:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """PRAGMA table_info-миграции (паттерн graph/history.py): добавляем
        недостающие колонки поверх существующих БД без потери данных."""
        migrations = {
            "projects": {"title": "TEXT DEFAULT ''", "duration_s": "REAL DEFAULT 0"},
        }
        for table, cols in migrations.items():
            have = {r["name"] for r in
                    self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, decl in cols.items():
                if col not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    @classmethod
    def instance(cls, cfg: Optional[AppConfig] = None) -> "Database":
        cfg = cfg or get_config()
        key = str(cfg.db_path)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(Path(key))
            return cls._instances[key]

    def close(self) -> None:
        with self._tx:
            self._conn.close()

    # ------------------------------------------------------------ cache ----
    def cache_lookup(self, url: str) -> Optional[dict[str, Any]]:
        with self._tx:
            row = self._conn.execute(
                "SELECT url, content_hash, content_type, body FROM doc_cache WHERE url=?",
                (url,)).fetchone()
        if not row:
            return None
        return {"url": row["url"], "content_hash": row["content_hash"],
                "content_type": row["content_type"] or "", "body": bytes(row["body"])}

    def cache_store(self, url: str, content_hash: str, body: bytes,
                    content_type: str = "") -> None:
        with self._tx:
            self._conn.execute(
                "INSERT OR REPLACE INTO doc_cache(url, content_hash, content_type, body, retrieved_at)"
                " VALUES(?,?,?,?,?)", (url, content_hash, content_type, body, _utcnow()))
            self._conn.commit()

    # ---------------------------------------------------------- entities ---
    def save_project(self, p: Project) -> None:
        self._upsert("projects", {"id": p.id, "input_url": p.input_url, "mode": p.mode.value,
                                  "created_at": p.created_at, "status": p.status,
                                  "model_json": p.model_info.model_dump_json(),
                                  "title": p.title, "duration_s": p.duration_s})

    def update_project_status(self, project_id: str, status: str) -> None:
        with self._tx:
            self._conn.execute("UPDATE projects SET status=? WHERE id=?", (status, project_id))
            self._conn.commit()

    def finish_project(self, project_id: str, *, status: str, title: str = "",
                       duration_s: float = 0.0) -> None:
        """Итоговая запись проекта: статус + метаданные для списка истории."""
        with self._tx:
            self._conn.execute(
                "UPDATE projects SET status=?, title=COALESCE(NULLIF(?,''),title),"
                " duration_s=? WHERE id=?",
                (status, title, duration_s, project_id))
            self._conn.commit()

    def save_document(self, project_id: str, d: Document) -> None:
        self._upsert("documents", {"id": d.id, "project_id": project_id, "url": d.url,
                                   "title": d.title, "author": d.author, "published": d.published,
                                   "text": d.text, "content_hash": d.content_hash,
                                   "retrieved_at": d.retrieved_at,
                                   "meta_json": json.dumps(d.metadata, ensure_ascii=False)})

    def save_source(self, project_id: str, s: SourceMeta) -> None:
        self._upsert("sources", {"id": s.id, "project_id": project_id,
                                 "data_json": s.model_dump_json()})

    def save_claim(self, project_id: str, c) -> None:
        self._upsert("claims", {"id": c.id, "project_id": project_id,
                                "document_id": c.document_id, "data_json": c.model_dump_json()})

    def save_evidence(self, project_id: str, e: Evidence) -> None:
        self._upsert("evidence", {"id": e.id, "project_id": project_id,
                                  "claim_id": e.claim_id, "source_id": e.source_id,
                                  "data_json": e.model_dump_json()})

    def save_source_score(self, project_id: str, sc: SourceScores) -> None:
        self._upsert("source_scores", {"source_id": sc.source_id, "project_id": project_id,
                                       "data_json": sc.model_dump_json()})

    def save_search_query(self, q: SearchQuery) -> None:
        self._upsert("search_queries", {"id": q.id, "project_id": q.project_id,
                                        "claim_id": q.claim_id, "provider": q.provider,
                                        "query": q.query, "purpose": q.purpose.value,
                                        "result_count": q.result_count, "error": q.error,
                                        "created_at": _utcnow()})

    def save_verdict(self, project_id: str, v: Verdict) -> None:
        with self._tx:
            self._conn.execute(
                "DELETE FROM verdicts WHERE project_id=? AND claim_id=?",
                (project_id, v.claim_id))
            self._conn.execute(
                "INSERT INTO verdicts(project_id, claim_id, data_json) VALUES(?,?,?)",
                (project_id, v.claim_id, v.model_dump_json()))
            self._conn.commit()

    def save_report(self, r: Report) -> Path:
        """Сохранить отчёт в БД и JSON-файл в reports/. Возвращает путь к файлу."""
        self._upsert("reports", {"project_id": r.project_id, "generated_at": r.generated_at,
                                 "overall_verdict": r.overall_verdict.value,
                                 "overall_confidence": r.overall_confidence,
                                 "data_json": r.model_dump_json()})
        path = self.cfg_reports_dir / f"{r.project_id}.json"
        try:
            path.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        except OSError:
            import logging as _lg
            _lg.getLogger(__name__).warning("не удалось записать файл отчёта %s", path)
        return path

    @property
    def cfg_reports_dir(self) -> Path:
        p = self.path.parent.parent / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def audit(self, ev: AuditEvent) -> None:
        with self._tx:
            self._conn.execute(
                "INSERT INTO audit_events(ts, project_id, stage, event, detail_json) VALUES(?,?,?,?,?)",
                (ev.ts, ev.project_id, ev.stage, ev.event,
                 json.dumps(ev.detail, ensure_ascii=False)))
            self._conn.commit()

    # ------------------------------------------------------------ reads ----
    def list_projects(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._tx:
            rows = self._conn.execute(
                """SELECT p.*, r.overall_verdict, r.overall_confidence
                   FROM projects p LEFT JOIN reports r ON r.project_id=p.id
                   ORDER BY p.created_at DESC LIMIT ?""", (limit,)).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            try:
                d["model_info"] = ModelInfo(**json.loads(d.get("model_json") or "{}"))
            except Exception:
                d["model_info"] = ModelInfo()
            out.append(d)
        return out

    def load_report(self, project_id: str) -> Optional[Report]:
        with self._tx:
            row = self._conn.execute(
                "SELECT data_json FROM reports WHERE project_id=?", (project_id,)).fetchone()
        if not row:
            return None
        return Report.model_validate_json(row["data_json"])

    def report_path(self, project_id: str) -> Path:
        """Путь к JSON-файлу отчёта (пишется в save_report)."""
        return self.cfg_reports_dir / f"{project_id}.json"

    def delete_project(self, project_id: str) -> bool:
        """Удалить анализ и все его данные из истории (§ история: удаление)."""
        tables = ("documents", "sources", "claims", "evidence", "source_scores",
                  "search_queries", "verdicts", "reports", "audit_events", "projects")
        with self._tx:
            for t in tables:
                key = "id" if t == "projects" else "project_id"
                self._conn.execute(f"DELETE FROM {t} WHERE {key}=?", (project_id,))
            cur = self._conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            self._conn.commit()
        try:
            p = self.report_path(project_id)
            if p.exists():
                p.unlink()
        except OSError:
            pass
        return cur.rowcount > 0

    def clear_cache(self) -> int:
        with self._tx:
            cur = self._conn.execute("DELETE FROM doc_cache")
            self._conn.commit()
        return cur.rowcount

    def audit_trail(self, project_id: str) -> list[AuditEvent]:
        with self._tx:
            rows = self._conn.execute(
                "SELECT ts, project_id, stage, event, detail_json FROM audit_events"
                " WHERE project_id=? ORDER BY id", (project_id,)).fetchall()
        return [AuditEvent(ts=r["ts"], project_id=r["project_id"], stage=r["stage"],
                           event=r["event"], detail=json.loads(r["detail_json"] or "{}"))
                for r in rows]

    def insufficient_evidence_stats(self) -> dict[str, float]:
        """Метрика insufficient_evidence_rate (§34, §74) по всем вердиктам."""
        with self._tx:
            total = self._conn.execute("SELECT COUNT(*) c FROM verdicts").fetchone()["c"]
            insuff = self._conn.execute(
                "SELECT COUNT(*) c FROM verdicts WHERE data_json LIKE '%INSUFFICIENT_EVIDENCE%'"
            ).fetchone()["c"]
        return {"total_verdicts": total,
                "insufficient": insuff,
                "insufficient_evidence_rate": (insuff / total) if total else 0.0}

    # ---------------------------------------------------------------- util -
    def _upsert(self, table: str, values: dict[str, Any]) -> None:
        cols = ",".join(values)
        marks = ",".join("?" * len(values))
        pk = next(iter(values))
        upd = ",".join(f"{c}=excluded.{c}" for c in values if c != pk)
        sql = (f"INSERT INTO {table}({cols}) VALUES({marks})"
               f" ON CONFLICT({pk}) DO UPDATE SET {upd}" if upd else
               f"INSERT OR REPLACE INTO {table}({cols}) VALUES({marks})")
        with self._tx:
            self._conn.execute(sql, tuple(values.values()))
            self._conn.commit()
