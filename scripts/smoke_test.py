"""Smoke tests for Grafix core + FastAPI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph.qa import answer_question
from graph.store import InMemoryGraphStore
from model.infer import Extractor
from model.triples import Triple, encode_example, parse_full_sequence


def test_triples_roundtrip():
    t = [Triple("Маша", "работает_в", "Яндекс")]
    seq = encode_example("Маша работает в Яндексе.", t)
    text, parsed = parse_full_sequence(seq)
    assert text.startswith("Маша")
    assert parsed[0].object == "Яндекс"


def test_heuristic_extract():
    r = Extractor().extract("Маша работает в Яндексе и живёт в Москве.")
    assert len(r["triples"]) >= 1
    rels = {t["relation"] for t in r["triples"]}
    assert "работает_в" in rels or "живёт_в" in rels


def test_graph_and_lost_edges():
    s = InMemoryGraphStore()
    s.upsert_triples(
        "d1",
        "t",
        [
            {"subject": "Маша", "relation": "работает_в", "object": "Яндекс"},
            {"subject": "Маша", "relation": "живёт_в", "object": "Москва"},
        ],
    )
    lost = s.lost_edges_if_disabled(["Маша"], "d1")
    assert len(lost) == 2
    ans = answer_question(s, "кто связан с Маша?", "d1")
    assert "Яндекс" in ans["answer"] or "Москва" in ans["answer"]


def test_fastapi_analyze():
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    res = client.post(
        "/api/analyze",
        json={
            "text": "Москва находится в России.",
            "document_id": "test-doc",
            "engine": "microgpt",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["triples"]
    assert data["nodes"]
    assert data.get("history_id")

    listed = client.get("/api/graphs")
    assert listed.status_code == 200
    assert listed.json()["count"] >= 1

    hid = data["history_id"]
    loaded = client.get(f"/api/graphs/{hid}")
    assert loaded.status_code == 200
    assert loaded.json()["history_id"] == hid
    assert loaded.json()["text"]

    deleted = client.delete(f"/api/graphs/{hid}")
    assert deleted.status_code == 200

    res2 = client.post(
        "/api/ask",
        json={"question": "какие связи у Москва?", "document_id": "test-doc"},
    )
    assert res2.status_code == 200
    assert "Россия" in res2.json()["answer"] or "связ" in res2.json()["answer"].lower()

    res3 = client.post(
        "/api/toggle",
        json={"document_id": "test-doc", "disabled": ["Москва"]},
    )
    assert res3.status_code == 200
    assert len(res3.json()["lost_edges"]) >= 1


if __name__ == "__main__":
    test_triples_roundtrip()
    test_heuristic_extract()
    test_graph_and_lost_edges()
    test_fastapi_analyze()
    print("all smoke tests OK")
