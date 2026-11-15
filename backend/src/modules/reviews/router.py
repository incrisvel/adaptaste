from fastapi import APIRouter, Query

from modules.reviews import repository

router = APIRouter()


@router.get("/reviews")
async def list_reviews(limit: int = Query(5, ge=1, le=50), place_id: str | None = None):
    return await repository.list_reviews(limit, place_id)
