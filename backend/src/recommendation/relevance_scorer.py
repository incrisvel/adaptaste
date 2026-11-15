from collections import defaultdict

from recommendation.candidate_retriever import MAX_DOCUMENTS_PER_PLACE

RRF_K = 60


def aggregate(documents: list[dict], k: int = RRF_K) -> list[dict]:
    """Documentos de um mecanismo → ranking de estabelecimentos.

    A contribuição de cada estabelecimento satura em `MAX_DOCUMENTS_PER_PLACE`
    documentos: quantidade de avaliações deixa de ser vantagem depois do terceiro.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)

    for document in documents:
        grouped[document["place_id"]].append(document)

    places = []

    for place_id, place_documents in grouped.items():
        best = sorted(
            place_documents, key=lambda document: document["position"]
        )[:MAX_DOCUMENTS_PER_PLACE]
        places.append({
            "place_id": place_id,
            "score": sum(1 / (k + document["position"]) for document in best),
            "documents": best,
        })

    places.sort(key=lambda place: place["score"], reverse=True)

    for position, place in enumerate(places, start=1):
        place["position"] = position

    return places
