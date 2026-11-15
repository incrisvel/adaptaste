from recommendation.relevance_scorer import RRF_K

MECHANISM_WEIGHTS = {
    "description_bert": 1.0,
    "description_tfidf": 1.0,
    "review_bert": 1.0,
    "review_tfidf": 1.0,
}

RELEVANCE_WEIGHT = 0.80
ANALYSIS_WEIGHTS = {"dietary": 0.07, "aspect": 0.07, "quality": 0.06}


def fuse(
    results: dict[str, dict],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """RRF sobre os rankings de estabelecimentos, normalizado por cobertura de dados.

    O denominador inclui todo mecanismo para o qual o estabelecimento tem dado, mesmo
    quando ele não aparece no ranking: ter avaliações e nenhuma relevante é evidência
    contrária, diferente de não ter descrição cadastrada.
    """
    weights = weights or MECHANISM_WEIGHTS
    places: dict[str, dict] = {}

    def entry_for(place_id: str) -> dict:
        return places.setdefault(
            place_id,
            {"place_id": place_id, "score": 0.0, "coverage": 0.0, "mechanisms": {}},
        )

    for mechanism, result in results.items():
        if result.get("skipped"):
            continue

        weight = weights[mechanism]

        for place_id in result["place_ids_available"]:
            entry_for(place_id)["coverage"] += weight

        for place in result["places"]:
            entry = entry_for(place["place_id"])
            entry["score"] += weight / (k + place["position"])
            entry["mechanisms"][mechanism] = place

    ranked = []

    for place in places.values():
        if not place["mechanisms"]:
            # tinha dado, mas nenhum documento relevante: não é candidato a recomendação
            continue

        place["score"] = place["score"] / place["coverage"] if place["coverage"] else 0.0
        ranked.append(place)

    ranked.sort(key=lambda place: place["score"], reverse=True)

    for position, place in enumerate(ranked, start=1):
        place["position"] = position

    return ranked


def rerank(
    candidates: list[dict],
    analyses: dict[str, dict],
    names: dict[str, str],
    relevance_weight: float = RELEVANCE_WEIGHT,
) -> list[dict]:
    """Combina a relevância normalizada com as análises consolidadas por estabelecimento.

    Componente sem registro sai do numerador e do denominador: sem nenhuma análise o
    resultado é a relevância pura.
    """
    weights = {"relevance": relevance_weight, **ANALYSIS_WEIGHTS}
    top = max((candidate["score"] for candidate in candidates), default=0.0) or 1.0
    ranked = []

    for candidate in candidates:
        scores = analyses.get(candidate["place_id"]) or {}
        components = {"relevance": candidate["score"] / top}
        components.update({name: scores.get(name) for name in ANALYSIS_WEIGHTS})

        available = [name for name in weights if components[name] is not None]
        total = sum(weights[name] for name in available)
        score = sum(weights[name] * components[name] for name in available) / total

        ranked.append({
            **candidate,
            "name": names.get(candidate["place_id"]),
            "score": score,
            "components": components,
        })

    ranked.sort(key=lambda candidate: candidate["score"], reverse=True)

    for position, candidate in enumerate(ranked, start=1):
        candidate["position"] = position

    return ranked
