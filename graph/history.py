"""Local SQLite history for saved graph snapshots (no auth)."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "grafix_history.db"


def resolve_history_db_path(db_path: Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    env = (os.environ.get("GRAFIX_HISTORY_DB") or "").strip()
    return Path(env) if env else DEFAULT_DB


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _preview(text: str, limit: int = 48) -> str:
    one = " ".join((text or "").split())
    if not one:
        return "Пустой граф"
    if len(one) <= limit:
        return one
    return one[: limit - 1].rstrip() + "…"


def _title_from_text(text: str) -> str:
    return _preview(text, 40)


class GraphHistory:
    """Persist analyze/edit snapshots for the history UI."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = resolve_history_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graphs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    document_id TEXT,
                    text TEXT NOT NULL,
                    engine TEXT,
                    source TEXT,
                    triples_json TEXT NOT NULL,
                    nodes_json TEXT NOT NULL,
                    edges_json TEXT NOT NULL,
                    comments_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(graphs)").fetchall()}
            if "comments_json" not in cols:
                conn.execute(
                    "ALTER TABLE graphs ADD COLUMN comments_json TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graphs_updated ON graphs(updated_at DESC)"
            )
            conn.commit()

    def list_graphs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at, document_id, text, engine, source,
                       triples_json, nodes_json, edges_json, comments_json
                FROM graphs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_summary(r) for r in rows]

    def get(self, graph_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM graphs WHERE id = ?",
                (graph_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_full(row)

    def save(
        self,
        *,
        text: str,
        triples: list[dict],
        nodes: list[dict],
        edges: list[dict],
        document_id: str | None = None,
        engine: str | None = None,
        source: str | None = None,
        title: str | None = None,
        graph_id: str | None = None,
        comments: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        gid = graph_id or str(uuid.uuid4())
        resolved_title = (title or "").strip() or _title_from_text(text)
        comments_payload = comments if comments is not None else {}

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at, comments_json FROM graphs WHERE id = ?",
                (gid,),
            ).fetchone()
            created = existing["created_at"] if existing else now
            if comments is None and existing is not None:
                try:
                    comments_payload = json.loads(existing["comments_json"] or "{}")
                except json.JSONDecodeError:
                    comments_payload = {}
            conn.execute(
                """
                INSERT INTO graphs (
                    id, title, created_at, updated_at, document_id, text, engine, source,
                    triples_json, nodes_json, edges_json, comments_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at,
                    document_id = excluded.document_id,
                    text = excluded.text,
                    engine = excluded.engine,
                    source = excluded.source,
                    triples_json = excluded.triples_json,
                    nodes_json = excluded.nodes_json,
                    edges_json = excluded.edges_json,
                    comments_json = excluded.comments_json
                """,
                (
                    gid,
                    resolved_title,
                    created,
                    now,
                    document_id,
                    text or "",
                    engine,
                    source,
                    json.dumps(triples, ensure_ascii=False),
                    json.dumps(nodes, ensure_ascii=False),
                    json.dumps(edges, ensure_ascii=False),
                    json.dumps(comments_payload, ensure_ascii=False),
                ),
            )
            conn.commit()
        item = self.get(gid)
        assert item is not None
        return item

    def rename(self, graph_id: str, title: str) -> dict[str, Any] | None:
        title = (title or "").strip()
        if not title:
            return self.get(graph_id)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE graphs SET title = ?, updated_at = ? WHERE id = ?",
                (title, _utc_now(), graph_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(graph_id)

    def delete(self, graph_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM graphs WHERE id = ?", (graph_id,))
            conn.commit()
            return cur.rowcount > 0

    def _row_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        triples = json.loads(row["triples_json"] or "[]")
        nodes = json.loads(row["nodes_json"] or "[]")
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "document_id": row["document_id"],
            "engine": row["engine"],
            "source": row["source"],
            "preview": _preview(row["text"]),
            "triple_count": len(triples),
            "node_count": len(nodes),
        }

    def _row_full(self, row: sqlite3.Row) -> dict[str, Any]:
        summary = self._row_summary(row)
        try:
            comments = json.loads(row["comments_json"] or "{}")
        except (KeyError, json.JSONDecodeError, TypeError):
            comments = {}
        if not isinstance(comments, dict):
            comments = {}
        summary.update(
            {
                "text": row["text"],
                "triples": json.loads(row["triples_json"] or "[]"),
                "nodes": json.loads(row["nodes_json"] or "[]"),
                "edges": json.loads(row["edges_json"] or "[]"),
                "comments": comments,
            }
        )
        return summary
