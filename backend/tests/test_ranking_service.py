import pytest

from recommendation.ranking_service import (
    ANALYSIS_WEIGHTS,
    RELEVANCE_WEIGHT,
    fuse,
    rerank,
)


def response(places, available, skipped=None):
    return {
        "places": [
            {"place_id": place_id, "position": position, "score": 0.0, "documents": []}
            for place_id, position in places
        ],
        "place_ids_available": set(available),
        "skipped": skipped,
    }


def test_fuse_normaliza_por_cobertura_de_dados():
    so_reviews = {
        "review_bert": response([("p", 1)], {"p"}),
        "review_tfidf": response([("p", 1)], {"p"}),
    }
    com_descricao_sem_aparecer = {
        "description_bert": response([], {"p"}),
        "description_tfidf": response([], {"p"}),
        **so_reviews,
    }

    assert fuse(so_reviews)[0]["score"] > fuse(com_descricao_sem_aparecer)[0]["score"]


def test_fuse_lugar_sem_descricao_empata_com_lugar_de_descricao_fraca():
    results = {
        "description_bert": response([("p2", 1)], {"p2"}),
        "description_tfidf": response([("p2", 1)], {"p2"}),
        "review_bert": response([("p1", 1), ("p2", 2)], {"p1", "p2"}),
        "review_tfidf": response([("p1", 1), ("p2", 2)], {"p1", "p2"}),
    }

    scores = {place["place_id"]: place["score"] for place in fuse(results)}

    assert scores["p1"] == pytest.approx(scores["p2"], rel=0.05)


def test_fuse_ignora_mecanismo_skipped():
    normal = {
        "review_tfidf": response([("p", 1)], {"p"}),
        "description_bert": response([], {"p"}),
    }
    skipped = {
        "review_tfidf": response([("p", 1)], {"p"}),
        "description_bert": response([], {"p"}, skipped="consulta sem termos"),
    }

    assert fuse(skipped)[0]["score"] == pytest.approx(2 * fuse(normal)[0]["score"])


def test_fuse_nao_repete_estabelecimento_e_numera_posicoes():
    results = {
        "review_bert": response([("p1", 1), ("p2", 2)], {"p1", "p2"}),
        "review_tfidf": response([("p2", 1), ("p1", 3)], {"p1", "p2"}),
    }

    fused = fuse(results)

    assert len({place["place_id"] for place in fused}) == 2
    assert [place["position"] for place in fused] == [1, 2]


def test_fuse_nao_devolve_lugar_fora_de_todos_os_rankings():
    results = {
        "review_bert": response([("p1", 1)], {"p1", "p2"}),
        "review_tfidf": response([("p1", 1)], {"p1", "p2"}),
    }

    assert [place["place_id"] for place in fuse(results)] == ["p1"]


def test_fuse_guarda_o_resultado_de_cada_mecanismo():
    results = {"review_bert": response([("p", 1)], {"p"})}

    fused = fuse(results)

    assert set(fused[0]["mechanisms"]) == {"review_bert"}
    assert fused[0]["mechanisms"]["review_bert"]["position"] == 1


def test_rerank_sem_analises_normaliza_a_relevancia_no_topo():
    candidates = [
        {"place_id": "p1", "score": 0.5, "mechanisms": {}},
        {"place_id": "p2", "score": 0.25, "mechanisms": {}},
    ]

    ranked = rerank(candidates, analyses={}, names={"p1": "A"})

    assert ranked[0]["place_id"] == "p1"
    assert ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[1]["score"] == pytest.approx(0.5)
    assert ranked[0]["name"] == "A"
    assert ranked[0]["components"] == {
        "relevance": 1.0,
        "dietary": None,
        "aspect": None,
        "quality": None,
    }


def test_rerank_usa_analise_disponivel_e_renormaliza_os_pesos():
    candidates = [
        {"place_id": "p1", "score": 1.0, "mechanisms": {}},
        {"place_id": "p2", "score": 1.0, "mechanisms": {}},
    ]
    analyses = {"p1": {"quality": 0.0}, "p2": {"quality": 1.0}}

    ranked = rerank(candidates, analyses, names={})

    disponivel = RELEVANCE_WEIGHT + ANALYSIS_WEIGHTS["quality"]
    assert ranked[0]["place_id"] == "p2"
    assert ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[1]["score"] == pytest.approx(RELEVANCE_WEIGHT / disponivel)


def test_rerank_nao_usa_analise_de_componente_ausente():
    candidates = [{"place_id": "p1", "score": 1.0, "mechanisms": {}}]

    ranked = rerank(
        candidates, analyses={"p1": {"dietary": 0.0, "aspect": None}}, names={}
    )

    disponivel = RELEVANCE_WEIGHT + ANALYSIS_WEIGHTS["dietary"]
    assert ranked[0]["components"]["dietary"] == 0.0
    assert ranked[0]["components"]["aspect"] is None
    assert ranked[0]["score"] == pytest.approx(RELEVANCE_WEIGHT / disponivel)
