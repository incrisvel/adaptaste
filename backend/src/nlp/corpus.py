from typing import Any

from core.db.mongo import db

PROJECTION = {"_id": 0, "id": 1, "place_id": 1, "text_raw": 1}


async def fetch_reviews(limit: int | None = None) -> list[dict[str, Any]]:
    """Reviews com texto, na ordem estável de _id — base da linha i dos artefatos."""
    cursor = db.reviews.find({"text_raw": {"$ne": None}}, PROJECTION).sort("_id", 1)

    if limit is not None:
        cursor = cursor.limit(limit)

    return await cursor.to_list(None)
