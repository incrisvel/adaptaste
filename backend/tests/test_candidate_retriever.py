import pytest

from recommendation.candidate_retriever import (
    MAX_DOCUMENTS_PER_PLACE,
    check_alignment,
    documents,
    encode_query,
)

REVIEW_INDEX = [
    {"id": "r0", "place_id": "p1"},
    {"id": "r1", "place_id": "p1"},
    {"id": "r2", "place_id": "p2"},
    {"id": "r3", "place_id": "p1"},
    {"id": "r4", "place_id": "p1"},
    {"id": "r5", "place_id": "p2"},
]


def test_documents_limita_a_tres_por_estabelecimento():
    selected = documents([0, 1, 2, 3, 4, 5], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], REVIEW_INDEX)

    assert [document["doc_id"] for document in selected] == ["r0", "r1", "r2", "r3", "r5"]
    assert [document["position"] for document in selected] == [1, 2, 3, 4, 6]


def test_documents_mantem_a_posicao_global_como_referencia():
    selected = documents([0, 1, 2, 3, 4, 5], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], REVIEW_INDEX)

    assert selected[0]["score"] == pytest.approx(0.9)
    assert selected[-1]["position"] == 6


def test_documents_para_quando_todos_os_lugares_atingem_o_teto():
    index = [
        {"id": f"r{row}", "place_id": f"p{row % 2}"}
        for row in range(10)
    ]

    selected = documents(list(range(10)), [0.0] * 10, index)

    assert len(selected) == 2 * MAX_DOCUMENTS_PER_PLACE


def test_documents_de_descricao_usa_place_id_como_doc_id():
    index = [{"place_id": "p1", "name": "A"}]

    assert documents([0], [1.0], index)[0]["doc_id"] == "p1"


def test_encode_query_rejeita_consulta_vazia():
    with pytest.raises(ValueError, match="consulta vazia"):
        encode_query("   ")


def test_check_alignment_aceita_contagens_iguais():
    check_alignment("descrições", [{"place_id": "p1"}], 1)


def test_check_alignment_falha_com_artefatos_de_geracoes_diferentes():
    with pytest.raises(ValueError, match="desalinhados"):
        check_alignment("descrições", [{"place_id": "p1"}], 2)
