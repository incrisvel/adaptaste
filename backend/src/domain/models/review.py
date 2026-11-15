from datetime import datetime

from pydantic import BaseModel


class Review(BaseModel):
    id: str
    author: str
    date: datetime
    rating: int
    place_id: str
    text_raw: str | None = None
    text_nostopwords: str | None = None
    text_lemmatized: str | None = None
