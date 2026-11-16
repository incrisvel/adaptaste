from fastapi import APIRouter, Query

from application import repository
from application.dto import RecommendationRequest
from recommendation import engine

router = APIRouter()


@router.get("/reviews")
async def list_reviews(limit: int = Query(5, ge=1, le=50), place_id: str | None = None):
    return await repository.list_reviews(limit, place_id)


@router.get("/recommendations")
async def list_recommendations(
    q: str,
    k: int = Query(10, ge=1, le=50),
    dietary_suitability: int = Query(1, ge=1, le=5),
    distance: int = Query(1, ge=1, le=5),
    rating: int = Query(1, ge=1, le=5),
    meal_time: int = Query(1, ge=1, le=5),
    latitude: float | None = None,
    longitude: float | None = None,
):
    location = (
        {"latitude": latitude, "longitude": longitude}
        if latitude is not None and longitude is not None
        else None
    )
    return await engine.recommend(
        q,
        k,
        priorities={
            "dietary_suitability": dietary_suitability,
            "distance": distance,
            "rating": rating,
            "meal_time": meal_time,
        },
        location=location,
    )


@router.post("/recommendations")
async def create_recommendations(
    request: RecommendationRequest,
    k: int = Query(10, ge=1, le=50),
):
    query = " ".join(
        part
        for part in (
            request.profile.eating_summary,
            request.search.preferences_text,
            " ".join(request.search.meal_categories),
        )
        if part
    )
    return await engine.recommend(
        query,
        k,
        priorities=request.priorities.model_dump(),
        location=request.location.model_dump() if request.location else None,
    )
