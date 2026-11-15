from collections import Counter
from functools import cache

from nlp.artifacts import load_review_index
from nlp.corpus import place_analyses, place_names
from recommendation import candidate_retriever
from recommendation.ranking_service import fuse, rerank
from recommendation.relevance_scorer import RRF_K, aggregate


@cache
def _reviews_indexed() -> Counter:
    """Volume de avaliações por estabelecimento — diagnóstico de viés, não sinal de ranking."""
    return Counter(entry["place_id"] for entry in load_review_index())


async def recommend(query: str, k: int = 10, rrf_k: int = RRF_K) -> list[dict]:
    """Estabelecimentos para a consulta, com as evidências de cada mecanismo."""
    encoded = candidate_retriever.encode_query(query)
    results = {
        mechanism: {
            **result,
            "places": aggregate(result["documents"], rrf_k),
        }
        for mechanism, result in candidate_retriever.retrieve_all(encoded).items()
    }

    candidates = fuse(results, rrf_k)
    place_ids = [candidate["place_id"] for candidate in candidates]

    names = await place_names(place_ids)
    analyses = await place_analyses(place_ids)

    ranked = rerank(candidates, analyses, names)
    reviews_indexed = _reviews_indexed()

    for candidate in ranked:
        candidate["reviews_indexed"] = reviews_indexed[candidate["place_id"]]

    return ranked[:k]
