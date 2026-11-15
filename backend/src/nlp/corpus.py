from typing import Any

from core.db.mongo import db

PROJECTION = {
    "_id": 0,
    "id": 1,
    "place_id": 1,
    "text_raw": 1,
    "text_nostopwords": 1,
    "text_lemmatized": 1,
}

DESCRIPTION_FIELDS = (
    "google_maps_description",
    "tripadvisor_description",
    "gemini_summary",
)

PLACE_PROJECTION = {
    "_id": 0,
    "id": 1,
    "name": 1,
    "description_lemmatized": 1,
    **{field: 1 for field in DESCRIPTION_FIELDS},
}


async def fetch_reviews(limit: int | None = None) -> list[dict[str, Any]]:
    """Reviews com texto, na ordem estável de _id — base da linha i dos artefatos."""
    cursor = db.reviews.find({"text_raw": {"$ne": None}}, PROJECTION).sort("_id", 1)

    if limit is not None:
        cursor = cursor.limit(limit)

    return await cursor.to_list(None)


def description_text(place: dict[str, Any]) -> str:
    """Une as fontes de descrição que existirem, sempre na mesma ordem."""
    return "\n".join(
        place[field].strip()
        for field in DESCRIPTION_FIELDS
        if (place.get(field) or "").strip()
    )


async def fetch_places() -> list[dict[str, Any]]:
    """Lugares com descrição, na ordem de _id — base da linha i dos artefatos de descrição."""
    query = {"$or": [{field: {"$ne": None}} for field in DESCRIPTION_FIELDS]}
    cursor = db.places.find(query, PLACE_PROJECTION).sort("_id", 1)
    places = await cursor.to_list(None)

    return [place for place in places if description_text(place)]


async def place_names(place_ids: list[str]) -> dict[str, str]:
    if not place_ids:
        return {}

    cursor = db.places.find(
        {"id": {"$in": place_ids}},
        {"_id": 0, "id": 1, "name": 1},
    )
    return {place["id"]: place.get("name") for place in await cursor.to_list(None)}


async def place_analyses(place_ids: list[str]) -> dict[str, dict]:
    """Scores consolidados por estabelecimento — o que o ranking consome."""
    if not place_ids:
        return {}

    cursor = db.place_analyses.find(
        {"place_id": {"$in": place_ids}},
        {"_id": 0, "place_id": 1, "scores": 1},
    )
    return {
        document["place_id"]: document.get("scores") or {}
        for document in await cursor.to_list(None)
    }
