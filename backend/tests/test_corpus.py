from nlp.corpus import description_text


def test_description_text_junta_na_ordem_dos_campos():
    place = {
        "gemini_summary": "resumo",
        "tripadvisor_description": "tripadvisor",
        "google_maps_description": "google",
    }

    assert description_text(place) == "google\ntripadvisor\nresumo"


def test_description_text_ignora_vazias():
    place = {
        "google_maps_description": "   ",
        "tripadvisor_description": None,
        "gemini_summary": "só o resumo",
    }

    assert description_text(place) == "só o resumo"


def test_description_text_sem_nenhuma_fonte_e_vazia():
    assert description_text({}) == ""
