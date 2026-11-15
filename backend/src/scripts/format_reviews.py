import asyncio

from core.db.mongo import db

SOURCE = db.google_reviews
TARGET = db.reviews

PROJECTION = {
    "_id": 0,
    "review_id": 1,
    "author": 1,
    "review_date": 1,
    "review_text": 1,
    "rating": 1,
    "place_id": 1,
}


def to_review(doc: dict) -> dict:
    # review_text vem como {idioma: texto}; o texto original é sempre o único valor.
    return {
        "_id": doc["review_id"],
        "id": doc["review_id"],
        "author": doc["author"],
        "date": doc["review_date"],
        "text_raw": next(iter(doc.get("review_text", {}).values()), None),
        "text_nostopwords": None,
        "text_lemmatized": None,
        "rating": doc["rating"],
        "place_id": doc["place_id"],
    }


async def main() -> None:
    written = 0
    async for doc in SOURCE.find({}, PROJECTION):
        await TARGET.replace_one({"_id": doc["review_id"]}, to_review(doc), upsert=True)
        written += 1
        if written % 5000 == 0:
            print(f"{written} reviews migradas", flush=True)

    print(
        f"origem: {await SOURCE.count_documents({})} | "
        f"destino: {await TARGET.count_documents({})} | "
        f"lidas nesta execucao: {written}"
    )


if __name__ == "__main__":
    asyncio.run(main())
