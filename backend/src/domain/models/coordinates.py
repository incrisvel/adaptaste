from typing import Literal

from pydantic import BaseModel


class Coordinates(BaseModel):
    latitude: str
    longitude: str


class GeoPoint(BaseModel):
    type: Literal["Point"] = "Point"
    coordinates: tuple[float, float]  # (latitude, longitude), como gravado no banco
