"""Compara mecanismos, agregações e fusões contra um conjunto pequeno de consultas rotuladas.

Não escolhe configuração: imprime a tabela e o diagnóstico de viés para decisão humana.
"""

import argparse
import asyncio
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from core.db.mongo import db
from nlp.artifacts import load_review_index
from recommendation import candidate_retriever
from recommendation.ranking_service import fuse
from recommendation.relevance_scorer import RRF_K, aggregate

QUERIES = [
    (
        "comida vegana",
        ["Vegecetera", "Restaurante Vegetariano/Vegano - Bodhilyn", "Âme Gastronomia Funcional"],
    ),
    (
        "opções sem glúten",
        ["E pronto! - Sem Glúten e Proteico", "Leves Protein Coffee - Cafeteria Saudável", "Âme Gastronomia Funcional"],
    ),
    (
        "saladas frescas e saudáveis",
        ["Green Life - Real Food", "Elô Gastronomia - Restaurante", "Restaurante Per Tutti"],
    ),
    (
        "café da manhã com brunch",
        ["Quiero Café", "Salvador Cozinha | Do Brunch ao Happy Hour"],
    ),
    (
        "buffet self-service com variedade",
        ["Elô Gastronomia - Restaurante", "BERLIM RESTAURANTE - Velha", "Saint Peter Restaurante"],
    ),
    (
        "marmitas congeladas com comida caseira",
        ["300 Natural Food - Marmitas Congeladas", "Marmita Blumenau"],
    ),
    (
        "massas e menu executivo",
        ["Mailde Bistrô", "Restaurante Per Tutti"],
    ),
    (
        "pescados e frutos do mar",
        ["Sabor Imperial", "Saint Peter Restaurante"],
    ),
]

AGGREGATIONS = ("best", "top3", "mean3")


def establishment_ranking(documents: list[dict], k: int, aggregation: str) -> list[dict]:
    """Ranking de estabelecimentos com uma das três agregações avaliadas."""
    if aggregation == "top3":
        return aggregate(documents, k)

    positions: dict[str, list[int]] = defaultdict(list)
    for document in documents:
        positions[document["place_id"]].append(document["position"])

    places = []

    for place_id, found in positions.items():
        best = sorted(found)[:3]
        if aggregation == "best":
            score = 1 / (k + best[0])
        elif aggregation == "mean3":
            score = sum(1 / (k + position) for position in best) / len(best)
        else:
            raise ValueError(f"agregação desconhecida: {aggregation}")

        places.append({"place_id": place_id, "score": score, "documents": []})

    places.sort(key=lambda place: place["score"], reverse=True)

    for position, place in enumerate(places, start=1):
        place["position"] = position

    return places


def fused_ranking(results: dict[str, dict], k: int, aggregation: str) -> list[str]:
    with_places = {
        mechanism: {
            **result,
            "places": establishment_ranking(result["documents"], k, aggregation),
        }
        for mechanism, result in results.items()
    }

    return [place["place_id"] for place in fuse(with_places, k)]


def borda_ranking(results: dict[str, dict], k: int, aggregation: str = "top3") -> list[str]:
    """Linha de base: média do score de Borda normalizado, só nos mecanismos com dado."""
    totals: dict[str, float] = defaultdict(float)
    coverage: dict[str, float] = defaultdict(float)

    for result in results.values():
        if result["skipped"]:
            continue

        places = establishment_ranking(result["documents"], k, aggregation)
        total = len(places)

        for place in places:
            totals[place["place_id"]] += 1 - (place["position"] - 1) / total

        for place_id in result["place_ids_available"]:
            coverage[place_id] += 1

    return sorted(totals, key=lambda place_id: totals[place_id] / coverage[place_id], reverse=True)


def metrics(ranked: list[str], expected: list[str]) -> dict[str, float]:
    ranks = [ranked.index(place_id) + 1 for place_id in expected if place_id in ranked]
    best = min(ranks) if ranks else None

    return {
        "hit@1": float(best == 1),
        "hit@3": float(best is not None and best <= 3),
        "hit@5": float(best is not None and best <= 5),
        "rr": 1 / best if best else 0.0,
    }


def average(per_query: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in per_query]))
        for name in per_query[0]
    }


def format_row(label: str, row: dict[str, float]) -> str:
    return (
        f"{label:<28}"
        f"{row['hit@1']:>8.2f}{row['hit@3']:>8.2f}{row['hit@5']:>8.2f}{row['rr']:>8.3f}"
    )


def bias_report(top_lists: list[list[str]], reviews: Counter) -> None:
    positions: dict[str, list[int]] = defaultdict(list)
    for top in top_lists:
        for position, place_id in enumerate(top, start=1):
            positions[place_id].append(position)

    ids = list(positions)
    rho = spearmanr(
        [reviews[place_id] for place_id in ids],
        [float(np.mean(positions[place_id])) for place_id in ids],
    ).statistic
    biggest = {place_id for place_id, _ in reviews.most_common(3)}
    shares = []

    for top in top_lists:
        total = sum(reviews[place_id] for place_id in top)
        shares.append(
            sum(reviews[place_id] for place_id in top if place_id in biggest) / total
            if total
            else 0.0
        )

    print(f"\nSpearman(avaliações, posição média no top-10): {rho:.3f}")
    print(
        "fração das avaliações do top-10 vinda dos 3 maiores lugares: "
        f"{np.mean(shares):.2f}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia a recuperação híbrida.")
    parser.add_argument("--k", type=int, default=10, help="tamanho da lista avaliada")
    args = parser.parse_args()

    cursor = db.places.find({}, {"_id": 0, "id": 1, "name": 1})
    ids_by_name: dict[str, list[str]] = defaultdict(list)
    for place in await cursor.to_list(None):
        if place.get("name"):
            ids_by_name[place["name"]].append(place["id"])

    reviews = Counter(entry["place_id"] for entry in load_review_index())
    dataset = []

    for query, names in QUERIES:
        expected = [place_id for name in names for place_id in ids_by_name[name]]
        encoded = candidate_retriever.encode_query(query)
        results = candidate_retriever.retrieve_all(encoded)
        dataset.append((query, expected, results))

    configs: dict[str, list[dict[str, float]]] = defaultdict(list)

    for _, expected, results in dataset:
        for mechanism in candidate_retriever.MECHANISMS:
            ranked = [
                place["place_id"]
                for place in establishment_ranking(results[mechanism]["documents"], RRF_K, "top3")
            ]
            configs[mechanism].append(metrics(ranked, expected))

        for k in (10, RRF_K):
            for aggregation in AGGREGATIONS:
                configs[f"rrf k={k} {aggregation}"].append(
                    metrics(fused_ranking(results, k, aggregation), expected)
                )

        configs["borda"].append(metrics(borda_ranking(results, RRF_K), expected))

    print(f"Hit@1   Hit@3   Hit@5   MRR   (topo {args.k}, {len(dataset)} consultas)\n")
    for label, rows in configs.items():
        print(format_row(label, average(rows)))

    print("\nposição do primeiro lugar esperado, por consulta — rrf k=60 top3")
    chosen = "rrf k=60 top3"
    top_lists: list[list[str]] = []

    for (query, expected, results), _ in zip(dataset, configs[chosen]):
        ranked = fused_ranking(results, RRF_K, "top3")
        ranks = [ranked.index(place_id) + 1 for place_id in expected if place_id in ranked]
        print(f"  {query:<40} {min(ranks) if ranks else 'fora':>5}")
        top_lists.append(ranked[:10])

    bias_report(top_lists, reviews)


if __name__ == "__main__":
    asyncio.run(main())
