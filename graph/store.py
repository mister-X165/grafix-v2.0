"""Graph store: Neo4j with in-memory fallback for local MVP."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphSnapshot:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    document_id: str | None = None


class GraphStore:
    """Abstract interface used by Flask and QA."""

    def clear_document(self, document_id: str) -> None:
        raise NotImplementedError

    def upsert_triples(self, document_id: str, text: str, triples: list[dict]) -> GraphSnapshot:
        raise NotImplementedError

    def get_graph(self, document_id: str | None = None) -> GraphSnapshot:
        raise NotImplementedError

    def add_triple(self, document_id: str, subject: str, relation: str, obj: str) -> GraphSnapshot:
        raise NotImplementedError

    def delete_node(self, document_id: str, name: str) -> GraphSnapshot:
        raise NotImplementedError

    def neighbors_of(self, name: str, document_id: str | None = None) -> list[dict]:
        raise NotImplementedError

    def relations_of(self, name: str, document_id: str | None = None) -> list[dict]:
        raise NotImplementedError

    def lost_edges_if_disabled(self, disabled_names: list[str], document_id: str | None = None) -> list[dict]:
        """Edges that disappear from the visible graph when given nodes are off."""
        snap = self.get_graph(document_id)
        disabled = {n.strip() for n in disabled_names}
        lost = []
        for e in snap.edges:
            if e["source"] in disabled or e["target"] in disabled:
                lost.append(e)
        return lost

    def close(self) -> None:
        pass


@dataclass
class InMemoryGraphStore(GraphStore):
    """Simple document-scoped graph for when Neo4j is unavailable."""

    documents: dict[str, dict] = field(default_factory=dict)

    def clear_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)

    def upsert_triples(self, document_id: str, text: str, triples: list[dict]) -> GraphSnapshot:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for t in triples:
            s, r, o = t["subject"].strip(), t["relation"].strip(), t["object"].strip()
            if not s or not o:
                continue
            from model.triples import normalize_edge_kind, normalize_origin

            kind = normalize_edge_kind(t.get("kind"))
            origin = normalize_origin(t.get("origin"))
            nodes[s] = {"id": s, "label": s, "type": "Entity"}
            nodes[o] = {"id": o, "label": o, "type": "Entity"}
            edge: dict[str, Any] = {
                "id": f"{s}|{r}|{o}|{kind}|{origin}",
                "source": s,
                "target": o,
                "label": r,
                "relation": r,
                "kind": kind,
                "origin": origin,
            }
            evidence = str(t.get("evidence") or "").strip()
            if evidence:
                edge["evidence"] = evidence
            conf = t.get("confidence")
            if conf is not None and conf != "":
                try:
                    edge["confidence"] = float(conf)
                except (TypeError, ValueError):
                    pass
            edges.append(edge)
        self.documents[document_id] = {"text": text, "nodes": nodes, "edges": edges}
        return self.get_graph(document_id)

    def get_graph(self, document_id: str | None = None) -> GraphSnapshot:
        if document_id and document_id in self.documents:
            doc = self.documents[document_id]
            return GraphSnapshot(
                nodes=list(doc["nodes"].values()),
                edges=list(doc["edges"]),
                document_id=document_id,
            )
        # Merge all docs if no id
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for doc in self.documents.values():
            nodes.update(doc["nodes"])
            edges.extend(doc["edges"])
        return GraphSnapshot(nodes=list(nodes.values()), edges=edges, document_id=document_id)

    def add_triple(self, document_id: str, subject: str, relation: str, obj: str) -> GraphSnapshot:
        doc = self.documents.setdefault(document_id, {"text": "", "nodes": {}, "edges": []})
        doc["nodes"][subject] = {"id": subject, "label": subject, "type": "Entity"}
        doc["nodes"][obj] = {"id": obj, "label": obj, "type": "Entity"}
        edge = {
            "id": f"{subject}|{relation}|{obj}|explicit",
            "source": subject,
            "target": obj,
            "label": relation,
            "relation": relation,
            "kind": "explicit",
        }
        if edge not in doc["edges"] and not any(e["id"] == edge["id"] for e in doc["edges"]):
            doc["edges"].append(edge)
        return self.get_graph(document_id)

    def delete_node(self, document_id: str, name: str) -> GraphSnapshot:
        doc = self.documents.get(document_id)
        if not doc:
            return GraphSnapshot(nodes=[], edges=[], document_id=document_id)
        doc["nodes"].pop(name, None)
        doc["edges"] = [e for e in doc["edges"] if e["source"] != name and e["target"] != name]
        return self.get_graph(document_id)

    def neighbors_of(self, name: str, document_id: str | None = None) -> list[dict]:
        snap = self.get_graph(document_id)
        out = []
        for e in snap.edges:
            kind = e.get("kind") or "explicit"
            item: dict[str, Any] = {
                "direction": "out" if e["source"] == name else "in",
                "relation": e["relation"],
                "neighbor": e["target"] if e["source"] == name else e["source"],
                "kind": kind,
                "origin": e.get("origin") or "base",
            }
            if e["source"] != name and e["target"] != name:
                continue
            if e.get("evidence"):
                item["evidence"] = e["evidence"]
            if e.get("confidence") is not None:
                item["confidence"] = e["confidence"]
            out.append(item)
        return out

    def relations_of(self, name: str, document_id: str | None = None) -> list[dict]:
        return self.neighbors_of(name, document_id)


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT document_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.id IS UNIQUE"
            )

    def close(self) -> None:
        self._driver.close()

    def clear_document(self, document_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc})-[r:HAS_ENTITY]->(e:Entity)
                OPTIONAL MATCH (e)-[rel]->(other:Entity)
                WHERE exists((:Document {id: $doc})-[:HAS_ENTITY]->(other))
                DELETE rel
                WITH d, e
                OPTIONAL MATCH (d)-[h:HAS_ENTITY]->(e)
                DELETE h
                WITH d
                OPTIONAL MATCH (d)
                DELETE d
                """,
                doc=document_id,
            )
            # Clean orphan entities with no document links
            session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (:Document)-[:HAS_ENTITY]->(e)
                DETACH DELETE e
                """
            )

    def upsert_triples(self, document_id: str, text: str, triples: list[dict]) -> GraphSnapshot:
        self.clear_document(document_id)
        with self._driver.session() as session:
            session.run(
                "MERGE (d:Document {id: $doc}) SET d.text = $text",
                doc=document_id,
                text=text,
            )
            for t in triples:
                s, r, o = t["subject"].strip(), t["relation"].strip(), t["object"].strip()
                if not s or not o:
                    continue
                from model.triples import normalize_edge_kind

                kind = normalize_edge_kind(t.get("kind"))
                from model.triples import normalize_origin

                origin = normalize_origin(t.get("origin"))
                # Dynamic relationship type: sanitize to Neo4j identifier
                rel_type = _rel_type(r)
                session.run(
                    f"""
                    MATCH (d:Document {{id: $doc}})
                    MERGE (a:Entity {{name: $s}})
                    MERGE (b:Entity {{name: $o}})
                    MERGE (d)-[:HAS_ENTITY]->(a)
                    MERGE (d)-[:HAS_ENTITY]->(b)
                    MERGE (a)-[rel:{rel_type}]->(b)
                    SET rel.label = $r, rel.document_id = $doc, rel.kind = $kind,
                        rel.origin = $origin,
                        rel.evidence = $evidence, rel.confidence = $confidence
                    """,
                    doc=document_id,
                    s=s,
                    o=o,
                    r=r,
                    kind=kind,
                    origin=origin,
                    evidence=str(t.get("evidence") or "").strip() or None,
                    confidence=(
                        float(t["confidence"])
                        if t.get("confidence") is not None and t.get("confidence") != ""
                        else None
                    ),
                )
        return self.get_graph(document_id)

    def get_graph(self, document_id: str | None = None) -> GraphSnapshot:
        with self._driver.session() as session:
            if document_id:
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc})-[:HAS_ENTITY]->(a:Entity)
                    OPTIONAL MATCH (a)-[r]->(b:Entity)
                    WHERE exists((:Document {id: $doc})-[:HAS_ENTITY]->(b))
                      AND r.document_id = $doc
                    RETURN a.name AS source, type(r) AS rel_type, r.label AS label,
                           r.kind AS kind, r.origin AS origin, r.evidence AS evidence,
                           r.confidence AS confidence,
                           b.name AS target
                    """,
                    doc=document_id,
                )
            else:
                result = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    RETURN a.name AS source, type(r) AS rel_type, r.label AS label,
                           r.kind AS kind, r.origin AS origin, r.evidence AS evidence,
                           r.confidence AS confidence,
                           b.name AS target
                    """
                )
            nodes: dict[str, dict] = {}
            edges: list[dict] = []
            for record in result:
                src = record["source"]
                if src:
                    nodes[src] = {"id": src, "label": src, "type": "Entity"}
                tgt = record["target"]
                if tgt:
                    nodes[tgt] = {"id": tgt, "label": tgt, "type": "Entity"}
                    label = record["label"] or record["rel_type"] or "RELATED"
                    from model.triples import normalize_edge_kind, normalize_origin

                    kind = normalize_edge_kind(record["kind"])
                    origin = normalize_origin(record.get("origin"))
                    edge = {
                        "id": f"{src}|{label}|{tgt}|{kind}|{origin}",
                        "source": src,
                        "target": tgt,
                        "label": label,
                        "relation": label,
                        "kind": kind,
                        "origin": origin,
                    }
                    if record.get("evidence"):
                        edge["evidence"] = record["evidence"]
                    if record.get("confidence") is not None:
                        edge["confidence"] = record["confidence"]
                    edges.append(edge)
            # Also include isolated entities linked to document
            if document_id:
                ents = session.run(
                    """
                    MATCH (d:Document {id: $doc})-[:HAS_ENTITY]->(e:Entity)
                    RETURN e.name AS name
                    """,
                    doc=document_id,
                )
                for rec in ents:
                    name = rec["name"]
                    nodes[name] = {"id": name, "label": name, "type": "Entity"}
        return GraphSnapshot(nodes=list(nodes.values()), edges=edges, document_id=document_id)

    def add_triple(self, document_id: str, subject: str, relation: str, obj: str) -> GraphSnapshot:
        rel_type = _rel_type(relation)
        with self._driver.session() as session:
            session.run(
                "MERGE (d:Document {id: $doc})",
                doc=document_id,
            )
            session.run(
                f"""
                MATCH (d:Document {{id: $doc}})
                MERGE (a:Entity {{name: $s}})
                MERGE (b:Entity {{name: $o}})
                MERGE (d)-[:HAS_ENTITY]->(a)
                MERGE (d)-[:HAS_ENTITY]->(b)
                MERGE (a)-[rel:{rel_type}]->(b)
                SET rel.label = $r, rel.document_id = $doc, rel.kind = 'explicit'
                """,
                doc=document_id,
                s=subject,
                o=obj,
                r=relation,
            )
        return self.get_graph(document_id)

    def delete_node(self, document_id: str, name: str) -> GraphSnapshot:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc})-[h:HAS_ENTITY]->(e:Entity {name: $name})
                OPTIONAL MATCH (e)-[r]-()
                WHERE r.document_id = $doc OR r IS NULL
                DELETE r, h
                WITH e
                WHERE NOT (:Document)-[:HAS_ENTITY]->(e)
                DETACH DELETE e
                """,
                doc=document_id,
                name=name,
            )
        return self.get_graph(document_id)

    def neighbors_of(self, name: str, document_id: str | None = None) -> list[dict]:
        with self._driver.session() as session:
            if document_id:
                result = session.run(
                    """
                    MATCH (a:Entity {name: $name})-[r]-(b:Entity)
                    WHERE r.document_id = $doc OR r.document_id IS NULL
                    RETURN a.name AS a, b.name AS b, type(r) AS t, r.label AS label,
                           r.kind AS kind, r.origin AS origin, r.evidence AS evidence, r.confidence AS confidence,
                           startNode(r) = a AS outgoing
                    """,
                    name=name,
                    doc=document_id,
                )
            else:
                result = session.run(
                    """
                    MATCH (a:Entity {name: $name})-[r]-(b:Entity)
                    RETURN a.name AS a, b.name AS b, type(r) AS t, r.label AS label,
                           r.kind AS kind, r.origin AS origin, r.evidence AS evidence, r.confidence AS confidence,
                           startNode(r) = a AS outgoing
                    """,
                    name=name,
                )
            out = []
            from model.triples import normalize_edge_kind, normalize_origin

            for rec in result:
                rel = rec["label"] or rec["t"]
                kind = normalize_edge_kind(rec["kind"])
                item = {
                    "direction": "out" if rec["outgoing"] else "in",
                    "relation": rel,
                    "neighbor": rec["b"],
                    "kind": kind,
                    "origin": normalize_origin(rec.get("origin")),
                }
                if rec.get("evidence"):
                    item["evidence"] = rec["evidence"]
                if rec.get("confidence") is not None:
                    item["confidence"] = rec["confidence"]
                out.append(item)
            return out

    def relations_of(self, name: str, document_id: str | None = None) -> list[dict]:
        return self.neighbors_of(name, document_id)


def _rel_type(relation: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in relation.upper())
    if not cleaned or cleaned[0].isdigit():
        cleaned = "REL_" + cleaned
    return cleaned or "RELATED"


def connect_graph_store() -> GraphStore:
    """Try Neo4j; fall back to in-memory."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "grafix")
    try:
        store = Neo4jGraphStore(uri, user, password)
        # Ping
        with store._driver.session() as session:
            session.run("RETURN 1")
        print(f"Connected to Neo4j at {uri}")
        return store
    except Exception:
        print("Neo4j not running on localhost:7687 — using in-memory graph (OK for local demo)")
        return InMemoryGraphStore()


def new_document_id() -> str:
    return str(uuid.uuid4())
