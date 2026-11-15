from core.db.mongo import db

PROJECTION = {
    "_id": 0,
    "review_id": 1,
    "place_id": 1,
    "author": 1,
    "rating": 1,
    "review_date": 1,
    "review_text": 1,
}


async def list_reviews(limit: int, place_id: str | None = None) -> list[dict]:
    query = {"place_id": place_id} if place_id else {}
    return await db.google_reviews.find(query, PROJECTION).limit(limit).to_list(None)
