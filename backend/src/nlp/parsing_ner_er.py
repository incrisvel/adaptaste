from collections import Counter
from collections.abc import Callable, Sequence

from nlp import EXTREME_RATINGS
from nlp.preprocessing import get_nlp

SUBJECT_DEPS = {"nsubj", "nsubj:pass", "csubj"}
OBJECT_DEPS = {"obj", "dobj", "iobj", "obl"}
OPINION_DEPS = {"amod", "acomp", "oprd"}


def dependencies(text: str) -> list[dict]:
    return [
        {
            "token": token.text,
            "lemma": token.lemma_,
            "grammar_class": token.pos_,
            "dependency": token.dep_,
            "governance_token": token.head.text,
        }
        for token in get_nlp()(text)
    ]


def entities(doc) -> list[dict]:
    return [
        {
            "entity": entity.text,
            "category": entity.label_,
            "start": entity.start_char,
            "end": entity.end_char,
        }
        for entity in doc.ents
    ]


def corpus_entities(
    reviews: Sequence[dict],
    batch_size: int = 32,
) -> list[dict]:
    """Entidades de cada review; `reviews` = [{"id", "place_id", "text"}]."""
    texts = [review.get("text") or "" for review in reviews]
    records = []

    for review_index, (review, doc) in enumerate(
        zip(reviews, get_nlp().pipe(texts, batch_size=batch_size))
    ):
        for entity in entities(doc):
            records.append({
                **entity,
                "review_index": review_index,
                "review_id": review.get("id"),
                "place_id": review.get("place_id"),
                "review_text": review.get("text"),
            })

    return records


def extract_svo(doc) -> list[dict]:
    """Triplas sujeito-verbo-objeto explícitas na árvore de dependências."""
    relations = []

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subjects = [child for child in token.children if child.dep_ in SUBJECT_DEPS]
        objects = [child for child in token.children if child.dep_ in OBJECT_DEPS]

        for subject in subjects:
            for obj in objects:
                relations.append({
                    "subject": subject.text,
                    "verb": token.lemma_,
                    "object": obj.text,
                    "sentence": token.sent.text,
                })

    return relations


def aspect_opinions(doc) -> list[dict]:
    """Pares aspecto-opinião: "comida deliciosa" e "a comida estava deliciosa"."""
    records = []

    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue

        opinions = {
            child
            for child in token.children
            if child.pos_ == "ADJ" and child.dep_ in OPINION_DEPS
        }

        # Com verbo de ligação o adjetivo fica na cabeça do sujeito, não no substantivo.
        if token.dep_ in {"nsubj", "nsubj:pass"}:
            head = token.head
            opinions.update(
                sibling for sibling in head.children if sibling.pos_ == "ADJ"
            )
            if head.pos_ == "ADJ":
                opinions.add(head)

        for opinion in opinions:
            records.append({
                "aspect": token.text,
                "opinion": opinion.text,
                "aspect_lemma": token.lemma_,
                "opinion_lemma": opinion.lemma_,
                "sentence": token.sent.text,
            })

    return records


def polarity_lexicon(
    aspects: Sequence[dict],
    ratings: Sequence[float],
    train_review: Callable[[int], bool] = lambda index: index % 2 == 0,
) -> dict[str, float]:
    """Taxa de opiniões positivas por lemma, derivada do rating (não é léxico de domínio)."""
    counts: dict[str, Counter] = {}

    for record in aspects:
        review_index = record["review_index"]

        if not train_review(review_index):
            continue

        rating = ratings[review_index]

        if rating not in EXTREME_RATINGS:
            continue

        counts.setdefault(record["opinion_lemma"], Counter())[rating >= 4] += 1

    return {
        lemma: counter[True] / sum(counter.values())
        for lemma, counter in counts.items()
    }


def aspect_polarity(
    aspects: Sequence[dict],
    lexicon: dict[str, float],
    min_opinions: int = 5,
) -> list[dict]:
    """Polaridade média de cada aspecto, do mais negativo ao mais positivo."""
    grouped: dict[str, list[float]] = {}

    for record in aspects:
        if record["opinion_lemma"] in lexicon:
            grouped.setdefault(record["aspect_lemma"], []).append(
                lexicon[record["opinion_lemma"]]
            )

    return sorted(
        (
            {
                "aspect": aspect,
                "opinions": len(rates),
                "polarity": sum(rates) / len(rates),
            }
            for aspect, rates in grouped.items()
            if len(rates) >= min_opinions
        ),
        key=lambda row: row["polarity"],
    )
