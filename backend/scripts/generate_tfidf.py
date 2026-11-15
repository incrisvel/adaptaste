import argparse
import asyncio
from pathlib import Path

import joblib
from scipy import sparse

from nlp.artifacts import artifacts_dir
from nlp.corpus import fetch_reviews
from nlp.representations import build_tfidf


def build(texts: list[str]):
    return build_tfidf(texts)


def save(vectorizer, matrix, directory: Path | None = None) -> Path:
    directory = directory or artifacts_dir()
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, directory / "tfidf_vectorizer.joblib")
    sparse.save_npz(directory / "tfidf_matrix.npz", matrix)

    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera vetorizador e matriz TF-IDF.")
    parser.add_argument("--limit", type=int, default=None, help="usa as N primeiras reviews")
    args = parser.parse_args()

    reviews = asyncio.run(fetch_reviews(args.limit))
    vectorizer, matrix = build([review["text_raw"] for review in reviews])
    directory = save(vectorizer, matrix)

    print(f"TF-IDF: {matrix.shape[0]} reviews x {matrix.shape[1]} termos -> {directory}")


if __name__ == "__main__":
    main()
