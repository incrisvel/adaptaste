import argparse
import asyncio
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.artifacts import artifacts_dir
from nlp.corpus import description_text, fetch_places
from nlp.representations import bert_embeddings, build_tfidf


def index_for(places: list[dict]) -> list[dict]:
    return [{"place_id": place["id"], "name": place.get("name")} for place in places]


def build(
    places: list[dict],
) -> tuple[TfidfVectorizer, sparse.csr_matrix, np.ndarray]:
    missing = sum(
        1 for place in places if place.get("description_lemmatized") is None
    )

    if missing:
        raise ValueError(
            f"{missing} lugares sem description_lemmatized — rode `make preprocess` antes"
        )

    vectorizer, matrix = build_tfidf(
        [place["description_lemmatized"].split() for place in places]
    )
    embeddings = bert_embeddings([description_text(place) for place in places])

    return vectorizer, matrix, embeddings


def save(
    places: list[dict],
    vectorizer: TfidfVectorizer,
    matrix: sparse.csr_matrix,
    embeddings: np.ndarray,
    directory: Path | None = None,
) -> Path:
    directory = directory or artifacts_dir()
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, directory / "description_vectorizer.joblib")
    sparse.save_npz(directory / "description_tfidf.npz", matrix)
    np.save(directory / "description_embeddings.npy", embeddings)
    (directory / "place_index.json").write_text(
        json.dumps(index_for(places), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera índices de descrição dos estabelecimentos."
    )
    parser.parse_args()

    places = asyncio.run(fetch_places())
    vectorizer, matrix, embeddings = build(places)
    directory = save(places, vectorizer, matrix, embeddings)

    print(
        f"descrições: {matrix.shape[0]} lugares x {matrix.shape[1]} termos, "
        f"{embeddings.shape[1]} dims -> {directory}"
    )


if __name__ == "__main__":
    main()
