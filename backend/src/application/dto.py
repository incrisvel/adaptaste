from pydantic import BaseModel

from domain.models.coordinates import Coordinates


class EatingProfile(BaseModel):
    eating_conditions: list[str] = []
    eating_summary: str | None = None


class PlaceSearch(BaseModel):
    preferences_text: str = ""
    meal_categories: list[str] = []


class RecommendationPriorities(BaseModel):
    distance: int = 1
    dietary_suitability: int = 1
    rating: int = 1
    meal_time: int = 1


class RecommendationRequest(BaseModel):
    profile: EatingProfile
    search: PlaceSearch
    priorities: RecommendationPriorities
    location: Coordinates | None = None
