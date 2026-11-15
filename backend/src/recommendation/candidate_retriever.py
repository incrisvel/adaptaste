from collections import Counter

import numpy as np

from nlp import artifacts
from nlp.preprocessing import preprocess
from nlp.representations import bert_embeddings
from recommendation.strategies import lexical_search, semantic_search

MAX_DOCUMENTS_PER_PLACE = 3

MECHANISMS = (
    "description_bert",
    "description_tfidf",
    "review_bert",
    "review_tfidf",
)


def encode_query(query: str) -> dict:
    """Uma codificação por consulta; os dois mecanismos de cada tipo reaproveitam."""
    if not query.strip():
        raise ValueError("consulta vazia")

    return {
        "raw": query,
        "lemmatized": preprocess(query)["lemmatized"],
        "bert": bert_embeddings([query])[0],
    }


def documents(
    order: np.ndarray,
    scores: np.ndarray,
    index: list[dict],
    limit: int = MAX_DOCUMENTS_PER_PLACE,
) -> list[dict]:
    """Documentos do ranking, no máximo `limit` por estabelecimento.

    `position` é a posição no ranking completo do mecanismo, não a posição na lista
    devolvida: é ela que mede o quanto o documento é relevante.
    """
    total_places = len({entry["place_id"] for entry in index})
    per_place: Counter = Counter()
    selected = []

    for position, (row, score) in enumerate(zip(order, scores), start=1):
        entry = index[row]
        place_id = entry["place_id"]

        if per_place[place_id] >= limit:
            continue

        per_place[place_id] += 1
        selected.append({
            "doc_id": entry.get("id", place_id),
            "place_id": place_id,
            "position": position,
            "score": float(score),
        })

        if len(selected) == total_places * limit:
            break

    return selected


def check_alignment(label: str, index: list[dict], rows: int) -> None:
    if len(index) != rows:
        raise ValueError(
            f"artefatos de {label} desalinhados: índice={len(index)}, matriz={rows}"
        )


def retrieve(mechanism: str, encoded: dict) -> dict:
    """Um mecanismo: documentos, estabelecimentos com dado e motivo de eventual skip."""
    if mechanism == "description_bert":
        index = artifacts.load_place_index()
        embeddings = artifacts.load_description_embeddings()
        check_alignment("descrições", index, embeddings.shape[0])
        order, scores = semantic_search.ranking(encoded, embeddings)
    elif mechanism == "description_tfidf":
        index = artifacts.load_place_index()
        matrix = artifacts.load_description_matrix()
        check_alignment("descrições", index, matrix.shape[0])
        order, scores = lexical_search.ranking(
            encoded, artifacts.load_description_vectorizer(), matrix
        )
    elif mechanism == "review_bert":
        index = artifacts.load_review_index()
        embeddings = artifacts.load_bert_embeddings()
        check_alignment("reviews", index, embeddings.shape[0])
        order, scores = semantic_search.ranking(encoded, embeddings)
    elif mechanism == "review_tfidf":
        index = artifacts.load_review_index()
        matrix = artifacts.load_tfidf_matrix()
        check_alignment("reviews", index, matrix.shape[0])
        order, scores = lexical_search.ranking(
            encoded, artifacts.load_tfidf_vectorizer(), matrix
        )
    else:
        raise ValueError(f"mecanismo desconhecido: {mechanism}")

    available = {entry["place_id"] for entry in index}

    if mechanism.endswith("_tfidf") and not encoded["lemmatized"]:
        # sem termos não há sinal lexical: o mecanismo sai da fusão em vez de empatar
        return {
            "mechanism": mechanism,
            "documents": [],
            "place_ids_available": available,
            "skipped": "consulta sem termos após normalização",
        }

    return {
        "mechanism": mechanism,
        "documents": documents(order, scores, index),
        "place_ids_available": available,
        "skipped": None,
    }


def retrieve_all(encoded: dict) -> dict[str, dict]:
    return {mechanism: retrieve(mechanism, encoded) for mechanism in MECHANISMS}
