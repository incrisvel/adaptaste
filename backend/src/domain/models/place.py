from pydantic import BaseModel

from domain.models.coordinates import GeoPoint


class Place(BaseModel):
    id: str
    name: str
    rating: float
    address: str
    google_maps_url: str
    google_maps_description: str | None = None
    gemini_summary: str | None = None
    tripadvisor_description: str | None = None
    image_url: str
    coordinates: GeoPoint
    opening_hours: dict[str, list[str]]
    meal_categories: list[str]
