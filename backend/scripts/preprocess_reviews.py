import argparse
import asyncio

from core.db.mongo import db
from nlp.corpus import DESCRIPTION_FIELDS, description_text
from nlp.preprocessing import preprocess


def build_fields(text: str) -> dict[str, str]:
    result = preprocess(text)

    return {
        "text_nostopwords": " ".join(result["no_stopwords"]),
        "text_lemmatized": " ".join(result["lemmatized"]),
    }


async def run(limit: int | None = None, redo: bool = False) -> int:
    query = {"text_raw": {"$ne": None}}

    if not redo:
        query["text_lemmatized"] = None

    cursor = db.reviews.find(query, {"text_raw": 1}).sort("_id", 1)

    if limit is not None:
        cursor = cursor.limit(limit)

    updated = 0

    async for document in cursor:
        await db.reviews.update_one(
            {"_id": document["_id"]},
            {"$set": build_fields(document["text_raw"])},
        )
        updated += 1

        if updated % 500 == 0:
            print(f"{updated} reviews preprocessadas", flush=True)

    return updated


def places_query(redo: bool) -> dict:
    query = {"$or": [{field: {"$ne": None}} for field in DESCRIPTION_FIELDS]}

    if not redo:
        query["description_lemmatized"] = None

    return query


async def run_places(limit: int | None = None, redo: bool = False) -> int:
    """Mesma cadeia das reviews, sobre a descrição consolidada de cada lugar."""
    projection = {**{field: 1 for field in DESCRIPTION_FIELDS}, "name": 1}
    cursor = db.places.find(places_query(redo), projection).sort("_id", 1)

    if limit is not None:
        cursor = cursor.limit(limit)

    updated = 0

    async for place in cursor:
        await db.places.update_one(
            {"_id": place["_id"]},
            {"$set": {
                "description_lemmatized": " ".join(
                    preprocess(description_text(place))["lemmatized"]
                ),
            }},
        )
        updated += 1

    return updated


async def report(limit: int | None = None, redo: bool = False) -> None:
    reviews = await run(limit, redo)
    places = await run_places(limit, redo)
    pending_reviews = await db.reviews.count_documents(
        {"text_raw": {"$ne": None}, "text_lemmatized": None}
    )
    pending_places = await db.places.count_documents(places_query(redo=False))

    print(
        f"reviews: {reviews} | lugares: {places} | "
        f"pendentes: {pending_reviews} reviews, {pending_places} lugares"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grava text_nostopwords/text_lemmatized (reviews) e description_lemmatized (places)."
    )
    parser.add_argument("--limit", type=int, default=None, help="processa só os N primeiros")
    parser.add_argument("--all", action="store_true", help="refaz mesmo os já processados")
    args = parser.parse_args()

    asyncio.run(report(args.limit, args.all))


if __name__ == "__main__":
    main()
