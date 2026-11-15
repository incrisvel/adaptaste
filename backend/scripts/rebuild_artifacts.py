import argparse
import asyncio
import json

from generate_embeddings import build as build_embeddings
from generate_embeddings import save as save_embeddings
from generate_tfidf import build as build_tfidf
from generate_tfidf import save as save_tfidf

from nlp.artifacts import artifacts_dir
from nlp.corpus import fetch_reviews


def write_index(reviews: list[dict]) -> int:
    directory = artifacts_dir()
    directory.mkdir(parents=True, exist_ok=True)

    index = [
        {"id": review["id"], "place_id": review["place_id"]}
        for review in reviews
    ]
    (directory / "review_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera todos os artefatos de busca.")
    parser.add_argument("--limit", type=int, default=None, help="usa as N primeiras reviews")
    args = parser.parse_args()

    reviews = asyncio.run(fetch_reviews(args.limit))
    texts = [review["text_raw"] for review in reviews]

    vectorizer, matrix = build_tfidf(texts)
    save_tfidf(vectorizer, matrix)

    embeddings = build_embeddings(texts)
    save_embeddings(embeddings)

    rows = write_index(reviews)

    if not rows == matrix.shape[0] == embeddings.shape[0]:
        raise ValueError(
            "artefatos desalinhados: "
            f"índice={rows}, tfidf={matrix.shape[0]}, bert={embeddings.shape[0]}"
        )

    print(f"artefatos em {artifacts_dir()}: {rows} reviews")


if __name__ == "__main__":
    main()
