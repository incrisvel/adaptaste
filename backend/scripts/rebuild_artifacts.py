import argparse
import asyncio
import json

from generate_embeddings import build as build_embeddings
from generate_embeddings import save as save_embeddings
from generate_places_artifacts import build as build_places
from generate_places_artifacts import index_for
from generate_places_artifacts import save as save_places
from generate_tfidf import build as build_tfidf
from generate_tfidf import save as save_tfidf

from nlp.artifacts import artifacts_dir
from nlp.corpus import fetch_places, fetch_reviews


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


async def fetch_all(limit: int | None) -> tuple[list[dict], list[dict]]:
    # asyncio.run não pode ser chamado duas vezes: o cliente Mongo fica preso ao loop
    return await fetch_reviews(limit), await fetch_places()


def check(label: str, rows: int, *shapes: int) -> None:
    if not all(rows == shape for shape in shapes):
        counts = ", ".join(str(shape) for shape in (rows, *shapes))
        raise ValueError(f"artefatos de {label} desalinhados: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera todos os artefatos de busca.")
    parser.add_argument("--limit", type=int, default=None, help="usa as N primeiras reviews")
    args = parser.parse_args()

    reviews, places = asyncio.run(fetch_all(args.limit))

    vectorizer, matrix = build_tfidf(reviews)
    save_tfidf(vectorizer, matrix)

    embeddings = build_embeddings(reviews)
    save_embeddings(embeddings)

    rows = write_index(reviews)
    check("reviews", rows, matrix.shape[0], embeddings.shape[0])

    place_vectorizer, place_matrix, place_embeddings = build_places(places)
    save_places(places, place_vectorizer, place_matrix, place_embeddings)
    check("descrições", len(index_for(places)), place_matrix.shape[0], place_embeddings.shape[0])

    print(
        f"artefatos em {artifacts_dir()}: {rows} reviews, "
        f"{len(places)} descrições"
    )


if __name__ == "__main__":
    main()
