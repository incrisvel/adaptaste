from pydantic import BaseModel


class EatingCondition(BaseModel):
    id: str
    name: str
    related_terms: list[str]
