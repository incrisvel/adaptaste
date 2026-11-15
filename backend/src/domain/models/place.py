from pydantic import BaseModel

from domain.models.coordinates import Coordinates


class Place(BaseModel):
    id: str
    name: str
    rating: str
    address: str
    coordinates: Coordinates
    image_url: str
    google_maps_url: str
    gemini_sumary: str
    tripadvisor_description: str
