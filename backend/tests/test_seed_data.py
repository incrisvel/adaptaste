import json
from pathlib import Path


def test_eating_conditions_seed_has_condicoes_basicas():
    path = Path(__file__).parents[1] / "data/seeds/eating_conditions.json"
    conditions = json.loads(path.read_text(encoding="utf-8"))
    ids = {condition["id"] for condition in conditions}

    assert {"diabetes", "celiac", "vegan", "vegetarian", "lactose_intolerance"} <= ids
    assert all(condition["related_terms"] for condition in conditions)
