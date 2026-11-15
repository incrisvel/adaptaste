import pytest

from recommendation.relevance_scorer import RRF_K, aggregate


def test_aggregate_soma_reciprocos_das_tres_melhores():
    documents = [
        {"place_id": "p1", "position": 1, "score": 0.9},
        {"place_id": "p2", "position": 2, "score": 0.8},
        {"place_id": "p2", "position": 3, "score": 0.7},
        {"place_id": "p2", "position": 4, "score": 0.6},
    ]

    places = aggregate(documents)

    assert [place["place_id"] for place in places] == ["p2", "p1"]
    assert places[0]["score"] == pytest.approx(
        1 / (RRF_K + 2) + 1 / (RRF_K + 3) + 1 / (RRF_K + 4)
    )
    assert places[1]["score"] == pytest.approx(1 / (RRF_K + 1))


def test_aggregate_ignora_o_quarto_documento_do_mesmo_estabelecimento():
    documents = [
        {"place_id": "p1", "position": position, "score": 0.0}
        for position in (1, 2, 3, 4, 5)
    ]

    places = aggregate(documents)

    assert len(places) == 1
    assert places[0]["score"] == pytest.approx(
        1 / (RRF_K + 1) + 1 / (RRF_K + 2) + 1 / (RRF_K + 3)
    )
    assert len(places[0]["documents"]) == 3


def test_aggregate_numera_posicoes_sem_repetir_estabelecimento():
    documents = [
        {"place_id": "p1", "position": 1, "score": 0.9},
        {"place_id": "p1", "position": 2, "score": 0.8},
        {"place_id": "p2", "position": 3, "score": 0.7},
    ]

    places = aggregate(documents)

    assert len({place["place_id"] for place in places}) == 2
    assert [place["position"] for place in places] == [1, 2]


def test_aggregate_ordena_por_posicao_do_documento_nao_por_score():
    documents = [
        {"place_id": "p1", "position": 5, "score": 0.99},
        {"place_id": "p2", "position": 1, "score": 0.10},
    ]

    assert aggregate(documents)[0]["place_id"] == "p2"
