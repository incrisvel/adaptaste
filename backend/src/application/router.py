from fastapi import APIRouter, Query

from application import repository
from recommendation import engine

router = APIRouter()


@router.get("/reviews")
async def list_reviews(limit: int = Query(5, ge=1, le=50), place_id: str | None = None):
    return await repository.list_reviews(limit, place_id)


@router.get("/recommendations")
async def list_recommendations(q: str, k: int = Query(10, ge=1, le=50)):
    return await engine.recommend(q, k)
