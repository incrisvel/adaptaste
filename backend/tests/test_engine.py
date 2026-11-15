import asyncio
from collections import Counter

from recommendation import candidate_retriever, engine


def fake_retrieve_all(encoded):
    return {
        "review_bert": {
            "mechanism": "review_bert",
            "documents": [
                {"doc_id": "r1", "place_id": "p1", "position": 1, "score": 0.9},
                {"doc_id": "r2", "place_id": "p2", "position": 2, "score": 0.8},
                {"doc_id": "r3", "place_id": "p1", "position": 3, "score": 0.7},
            ],
            "place_ids_available": {"p1", "p2", "p3"},
            "skipped": None,
        },
    }


def prepare(monkeypatch):
    monkeypatch.setattr(
        candidate_retriever,
        "encode_query",
        lambda query: {"raw": query, "lemmatized": ["x"], "bert": None},
    )
    monkeypatch.setattr(candidate_retriever, "retrieve_all", fake_retrieve_all)
    monkeypatch.setattr(engine, "place_names", lambda ids: _async({"p1": "A"}))
    monkeypatch.setattr(engine, "place_analyses", lambda ids: _async({}))
    monkeypatch.setattr(engine, "_reviews_indexed", lambda: Counter({"p1": 1, "p2": 2}))


async def _async(value):
    return value


def test_recommend_devolve_no_maximo_k(monkeypatch):
    prepare(monkeypatch)

    result = asyncio.run(engine.recommend("q", k=1))

    assert len(result) == 1


def test_recommend_devolve_uma_linha_por_estabelecimento(monkeypatch):
    prepare(monkeypatch)

    result = asyncio.run(engine.recommend("q", k=10))

    assert [candidate["place_id"] for candidate in result] == ["p1", "p2"]
    assert [candidate["position"] for candidate in result] == [1, 2]


def test_recommend_traz_as_evidencias(monkeypatch):
    prepare(monkeypatch)

    result = asyncio.run(engine.recommend("q", k=1))

    assert result[0]["name"] == "A"
    assert result[0]["reviews_indexed"] == 1
    assert result[0]["components"]["relevance"] == 1.0
    assert set(result[0]["mechanisms"]) == {"review_bert"}
