from pydantic import BaseModel, Field

from domain.models.coordinates import Coordinates


class EatingProfile(BaseModel):
    eating_conditions: list[str] = []
    eating_summary: str | None = None


class PlaceSearch(BaseModel):
    preferences_text: str = ""
    meal_categories: list[str] = []


class RecommendationPriorities(BaseModel):
    distance: int = Field(1, ge=1, le=5)
    dietary_suitability: int = Field(1, ge=1, le=5)
    rating: int = Field(1, ge=1, le=5)
    meal_time: int = Field(1, ge=1, le=5)


class RecommendationRequest(BaseModel):
    profile: EatingProfile
    search: PlaceSearch
    priorities: RecommendationPriorities
    location: Coordinates | None = None
