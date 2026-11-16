from math import asin, cos, radians, sin, sqrt

from recommendation.relevance_scorer import RRF_K

MECHANISM_WEIGHTS = {
    "description_bert": 1.0,
    "description_tfidf": 1.0,
    "review_bert": 1.0,
    "review_tfidf": 1.0,
}

RELEVANCE_WEIGHT = 0.80
ANALYSIS_WEIGHTS = {"dietary": 0.07, "aspect": 0.07, "quality": 0.06}
PRIORITY_CRITERIA = (
    "dietary_suitability",
    "distance",
    "rating",
    "meal_time",
)


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


def _coordinates(value: dict | None) -> tuple[float, float] | None:
    if not value:
        return None

    try:
        return float(value["latitude"]), float(value["longitude"])
    except (KeyError, TypeError, ValueError):
        return None


def _distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    latitude_1, longitude_1 = map(radians, first)
    latitude_2, longitude_2 = map(radians, second)
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1
    value = (
        sin(delta_latitude / 2) ** 2
        + cos(latitude_1) * cos(latitude_2) * sin(delta_longitude / 2) ** 2
    )
    return 6371 * 2 * asin(sqrt(value))


def prioritize(
    candidates: list[dict],
    priorities: dict[str, int],
    metadata: dict[str, dict],
    analyses: dict[str, dict],
    location: dict | None,
) -> list[dict]:
    """Combina relevância com as prioridades do usuário, sem critérios dominantes absolutos."""
    user_location = _coordinates(location)
    distances = {}

    if user_location:
        for candidate in candidates:
            place = metadata.get(candidate["place_id"], {})
            coordinates = _coordinates(
                place.get("coordinates") or place.get("location")
            )
            if coordinates:
                distances[candidate["place_id"]] = _distance_km(
                    user_location, coordinates
                )

    max_distance = max(distances.values(), default=0.0)
    ranked = []

    for candidate in candidates:
        place_id = candidate["place_id"]
        place = metadata.get(place_id, {})
        scores = analyses.get(place_id) or {}
        criteria = {
            "relevance": candidate["score"],
            "dietary_suitability": scores.get("dietary"),
            "distance": (
                1 - distances[place_id] / max_distance
                if place_id in distances and max_distance
                else 1.0
                if place_id in distances
                else None
            ),
            "rating": (
                min(max(float(place["rating"]) / 5, 0.0), 1.0)
                if place.get("rating") is not None
                else None
            ),
            "meal_time": place.get("meal_time"),
        }
        weights = {"relevance": 1, **priorities}
        available = [
            criterion
            for criterion, value in criteria.items()
            if value is not None and weights.get(criterion, 0) > 0
        ]
        total_weight = sum(weights[criterion] for criterion in available)
        score = (
            sum(weights[criterion] * criteria[criterion] for criterion in available)
            / total_weight
            if total_weight
            else 0.0
        )
        ranked.append({**candidate, "score": score, "criteria": criteria})

    ranked.sort(key=lambda candidate: candidate["score"], reverse=True)
    for position, candidate in enumerate(ranked, start=1):
        candidate["position"] = position

    return ranked
