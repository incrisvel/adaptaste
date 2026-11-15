import pytest

from generate_places_artifacts import build, index_for


def test_index_for_guarda_place_id_e_name():
    places = [{"id": "0x1", "name": "A", "description_lemmatized": "comida boa"}]

    assert index_for(places) == [{"place_id": "0x1", "name": "A"}]


def test_build_falha_quando_algum_lugar_nao_foi_preprocessado():
    places = [
        {
            "id": "0x1",
            "name": "A",
            "gemini_summary": "comida boa",
            "description_lemmatized": None,
        }
    ]

    with pytest.raises(ValueError, match="make preprocess"):
        build(places)
