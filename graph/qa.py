"""QA over graph templates + Gemma for free-text questions about the document."""

from __future__ import annotations

import re
from typing import Any

from graph.store import GraphStore


GRAPH_HINT = (
    "Пока по графу понимаю шаблоны:\n"
    "• кто связан с X?\n"
    "• какие связи у X?\n"
    "• покажи граф\n"
    "• добавь связь A | отношение | B\n"
    "• удали узел X\n\n"
    "Вопросы по смыслу текста идут в Gemma (нужен LM Studio)."
)


def is_graph_question(question: str) -> bool:
    q = question.strip().lower()
    patterns = (
        r"^кто\s+связан\s+с\b",
        r"^какие\s+связи\s+у\b",
        r"^покажи\s+граф",
        r"^что\s+в\s+графе",
        r"^перечисли",
        r"^добавь\s+связь\b",
        r"^удали\s+узел\b",
    )
    return any(re.search(p, q, re.I) for p in patterns)


def answer_question(
    store: GraphStore,
    question: str,
    document_id: str | None = None,
    document_text: str | None = None,
    prefer_gemma_for_text: bool = True,
    web_search: bool = False,
    engine: str = "gemma",
    reasoning: bool = False,
) -> dict[str, Any]:
    q = question.strip()
    engine = (engine or "gemma").strip().lower()
    if engine not in {"gemma", "deepseek", "deepseek-v4", "gigachat", "microgpt", "auto"}:
        engine = "gemma"
    text_engine = (
        engine
        if engine in {"deepseek", "deepseek-v4", "gigachat"}
        else "gemma"
    )

    if is_graph_question(q) and not web_search:
        result = _answer_graph(store, q, document_id)
        result["qa_mode"] = "graph"
        result.setdefault("debug", {})
        result["web_results"] = []
        return result

    # Free-text / about document → Gemma or DeepSeek (optionally with web)
    if prefer_gemma_for_text:
        web_payload: dict[str, Any] = {"results": [], "error": None, "query": q}
        if web_search:
            from graph.websearch import search_web

            web_payload = search_web(q, max_results=5)

        has_doc = bool((document_text or "").strip())
        has_web = bool(web_payload.get("results"))
        qa_mode = (
            "text_deepseek_v4"
            if text_engine == "deepseek-v4"
            else "text_deepseek"
            if text_engine.startswith("deepseek")
            else "text_gigachat"
            if text_engine == "gigachat"
            else "text_gemma"
        )

        if not has_doc and not has_web:
            return {
                "answer": (
                    "Нет текста документа"
                    + (" и поиск ничего не нашёл." if web_search else ". Сначала нажми «Анализировать».")
                ),
                "highlight_nodes": [],
                "highlight_edges": [],
                "action": None,
                "qa_mode": qa_mode,
                "web_results": web_payload.get("results") or [],
                "web_error": web_payload.get("error"),
                "debug": {
                    "raw_response": "",
                    "error": web_payload.get("error") or "no document text",
                },
            }

        if text_engine in {"deepseek", "deepseek-v4"}:
            from model.openrouter import openrouter_for_engine

            client = openrouter_for_engine(text_engine)
            if not client.ready:
                return {
                    "answer": (
                        "Выбран DeepSeek, но нет OPENROUTER_API_KEY. "
                        "Задай ключ в окружении или переключи модель на Gemma.\n\n" + GRAPH_HINT
                    ),
                    "highlight_nodes": [],
                    "highlight_edges": [],
                    "action": None,
                    "qa_mode": qa_mode,
                    "web_results": web_payload.get("results") or [],
                    "web_error": web_payload.get("error"),
                    "debug": {"raw_response": "", "error": "OPENROUTER_API_KEY missing"},
                }
        elif text_engine == "gigachat":
            from model.gigachat import GigaChatExtractor

            client = GigaChatExtractor()
            if not client.ready:
                return {
                    "answer": (
                        "Выбран GigaChat, но нет GIGACHAT_CREDENTIALS. "
                        "Задай Auth key в .env или переключи модель.\n\n" + GRAPH_HINT
                    ),
                    "highlight_nodes": [],
                    "highlight_edges": [],
                    "action": None,
                    "qa_mode": qa_mode,
                    "web_results": web_payload.get("results") or [],
                    "web_error": web_payload.get("error"),
                    "debug": {"raw_response": "", "error": "GIGACHAT_CREDENTIALS missing"},
                }
        else:
            from model.lmstudio import LMStudioExtractor

            client = LMStudioExtractor()
            if not client.ping():
                return {
                    "answer": (
                        "Вопрос похож на вопрос по тексту, но LM Studio выключен. "
                        "Запусти Gemma-сервер, переключись на DeepSeek / GigaChat "
                        "или спроси по шаблону графа.\n\n" + GRAPH_HINT
                    ),
                    "highlight_nodes": [],
                    "highlight_edges": [],
                    "action": None,
                    "qa_mode": qa_mode,
                    "web_results": web_payload.get("results") or [],
                    "web_error": web_payload.get("error"),
                    "debug": {"raw_response": "", "error": "LM Studio offline"},
                }

        gem = client.answer_about_text(
            document_text or "",
            q,
            web_results=web_payload.get("results") if has_web else None,
            **(
                {"reasoning": bool(reasoning)}
                if text_engine in {"deepseek", "deepseek-v4", "gigachat"}
                else {}
            ),
        )
        debug_err = gem.get("error")
        if web_payload.get("error"):
            debug_err = (debug_err + "; " if debug_err else "") + f"web: {web_payload['error']}"
        return {
            "answer": gem["answer"],
            "highlight_nodes": [],
            "highlight_edges": [],
            "action": None,
            "qa_mode": qa_mode,
            "source": gem.get("source"),
            "lm_model": gem.get("model"),
            "web_results": web_payload.get("results") or [],
            "web_error": web_payload.get("error"),
            "debug": {
                "raw_response": gem.get("raw_response") or "",
                "error": debug_err,
                "model": gem.get("model"),
                "prompt_user": gem.get("prompt_user") or "",
            },
        }

    return {
        "answer": GRAPH_HINT,
        "highlight_nodes": [],
        "highlight_edges": [],
        "action": None,
        "qa_mode": "graph",
        "web_results": [],
        "debug": {},
    }


def _answer_graph(store: GraphStore, q: str, document_id: str | None) -> dict[str, Any]:
    highlight_nodes: list[str] = []
    highlight_edges: list[str] = []

    m = re.search(r"кто\s+связан\s+с\s+[«\"']?(.+?)[»\"']?\s*\??$", q, re.I)
    if m:
        name = m.group(1).strip()
        neigh = store.neighbors_of(name, document_id)
        highlight_nodes = [name] + [n["neighbor"] for n in neigh]
        if not neigh:
            return {
                "answer": f"Для «{name}» связей не найдено.",
                "highlight_nodes": [name],
                "highlight_edges": [],
                "action": None,
            }
        parts = []
        for n in neigh:
            arrow = "→" if n["direction"] == "out" else "←"
            parts.append(f"{name} {arrow}[{n['relation']}] {n['neighbor']}")
            highlight_edges.append(
                f"{name}|{n['relation']}|{n['neighbor']}"
                if n["direction"] == "out"
                else f"{n['neighbor']}|{n['relation']}|{name}"
            )
        return {
            "answer": "Связи:\n" + "\n".join(parts),
            "highlight_nodes": highlight_nodes,
            "highlight_edges": highlight_edges,
            "action": None,
        }

    m = re.search(r"какие\s+связи\s+у\s+[«\"']?(.+?)[»\"']?\s*\??$", q, re.I)
    if m:
        name = m.group(1).strip()
        neigh = store.relations_of(name, document_id)
        highlight_nodes = [name] + [n["neighbor"] for n in neigh]
        if not neigh:
            return {
                "answer": f"У «{name}» нет связей в графе.",
                "highlight_nodes": [name],
                "highlight_edges": [],
                "action": None,
            }
        lines = [f"• {n['relation']} ({n['direction']}): {n['neighbor']}" for n in neigh]
        return {
            "answer": f"Связи у «{name}»:\n" + "\n".join(lines),
            "highlight_nodes": highlight_nodes,
            "highlight_edges": [],
            "action": None,
        }

    m = re.search(
        r"добавь\s+связь\s+(.+?)\s*[|/]\s*(.+?)\s*[|/]\s*(.+)$",
        q,
        re.I,
    )
    if m and document_id:
        s, r, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        store.add_triple(document_id, s, r, o)
        return {
            "answer": f"Добавлена связь: {s} —[{r}]→ {o}",
            "highlight_nodes": [s, o],
            "highlight_edges": [f"{s}|{r}|{o}"],
            "action": "graph_updated",
        }

    m = re.search(r"добавь\s+связь\s+(\S+)\s+(\S+)\s+(\S+)\s*$", q, re.I)
    if m and document_id:
        s, r, o = m.group(1), m.group(2), m.group(3)
        store.add_triple(document_id, s, r, o)
        return {
            "answer": f"Добавлена связь: {s} —[{r}]→ {o}",
            "highlight_nodes": [s, o],
            "highlight_edges": [f"{s}|{r}|{o}"],
            "action": "graph_updated",
        }

    m = re.search(r"удали\s+узел\s+[«\"']?(.+?)[»\"']?\s*$", q, re.I)
    if m and document_id:
        name = m.group(1).strip()
        store.delete_node(document_id, name)
        return {
            "answer": f"Узел «{name}» удалён из графа документа.",
            "highlight_nodes": [],
            "highlight_edges": [],
            "action": "graph_updated",
        }

    if re.search(r"покажи\s+граф|что\s+в\s+графе|перечисли", q, re.I):
        snap = store.get_graph(document_id)
        if not snap.edges:
            names = ", ".join(n["label"] for n in snap.nodes) or "(пусто)"
            return {
                "answer": f"Узлы: {names}. Рёбер нет.",
                "highlight_nodes": [n["id"] for n in snap.nodes],
                "highlight_edges": [],
                "action": None,
            }
        lines = [f"{e['source']} —[{e['relation']}]→ {e['target']}" for e in snap.edges]
        return {
            "answer": "Граф:\n" + "\n".join(lines),
            "highlight_nodes": [n["id"] for n in snap.nodes],
            "highlight_edges": [e["id"] for e in snap.edges],
            "action": None,
        }

    return {
        "answer": GRAPH_HINT,
        "highlight_nodes": [],
        "highlight_edges": [],
        "action": None,
    }
