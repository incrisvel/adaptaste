from collections.abc import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp import EXTREME_RATINGS
from nlp.preprocessing import preprocess

SPARSE_REPRESENTATIONS = ("BoW", "TF-IDF")
NGRAM_RANGES = ((1, 1), (1, 2))
DENSE_REPRESENTATIONS = ("BERT",)
EmbeddingFn = Callable[[Sequence[str]], np.ndarray]


def labelled(texts: Sequence[str], ratings: Sequence[int]) -> tuple[list[str], list[int]]:
    """Notas 1-2 -> negativa (0), 4-5 -> positiva (1). Nota 3 fica de fora."""
    pairs = [
        (text, int(rating >= 4))
        for text, rating in zip(texts, ratings)
        if rating in EXTREME_RATINGS
    ]
    return [text for text, _ in pairs], [label for _, label in pairs]


def prepare_documents(texts: Sequence[str]) -> list[str]:
    """Texto lematizado e juntado — entrada dos vetorizadores esparsos."""
    return [" ".join(preprocess(text)["lemmatized"]) for text in texts]


def build_vectorizer(representation: str, ngram_range: tuple[int, int]):
    common = dict(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        ngram_range=ngram_range,
        lowercase=False,
        min_df=1,
    )

    if representation == "BoW":
        return CountVectorizer(**common)

    if representation == "TF-IDF":
        return TfidfVectorizer(sublinear_tf=True, **common)

    raise ValueError(f"representação esparsa desconhecida: {representation}")


def build_model(classifier: str, class_weight: str | None):
    if classifier == "logistic":
        return LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=42,
        )

    if classifier == "svm":
        return LinearSVC(class_weight=class_weight, random_state=42)

    raise ValueError(f"classificador desconhecido: {classifier}")


def experiments() -> list[dict]:
    """Grid representação × n-grama × classificador × class_weight."""
    return [
        {
            "representation": representation,
            "ngram_range": ngram_range,
            "classifier": classifier,
            "class_weight": class_weight,
        }
        for representation in (*SPARSE_REPRESENTATIONS, *DENSE_REPRESENTATIONS)
        for ngram_range in (
            NGRAM_RANGES if representation in SPARSE_REPRESENTATIONS else (None,)
        )
        for classifier in ("logistic", "svm")
        for class_weight in (None, "balanced")
    ]


def experiment_name(experiment: dict) -> str:
    return " | ".join(
        str(experiment[key])
        for key in ("representation", "ngram_range", "classifier", "class_weight")
    )


def _metrics(y_true, predictions) -> dict:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(y_true, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
        "f1_negativa": f1_score(y_true, predictions, labels=[0], average="macro", zero_division=0),
        "f1_positiva": f1_score(y_true, predictions, labels=[1], average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def run_experiments(
    texts: Sequence[str],
    ratings: Sequence[int],
    embeddings: EmbeddingFn,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict]:
    """Treina e avalia todas as configurações no mesmo split estratificado.

    `embeddings` gera a matriz densa usada na representação BERT.
    """
    from sklearn.model_selection import train_test_split

    texts, labels = labelled(texts, ratings)
    documents = prepare_documents(texts)

    train_index, test_index = train_test_split(
        range(len(texts)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    y_train = [labels[row] for row in train_index]
    y_test = [labels[row] for row in test_index]
    x_train_text = [documents[row] for row in train_index]
    x_test_text = [documents[row] for row in test_index]
    embedding_matrix = embeddings(texts)

    results = []

    for experiment in experiments():
        representation = experiment["representation"]
        name = experiment_name(experiment)

        if representation in DENSE_REPRESENTATIONS:
            model = build_model(experiment["classifier"], experiment["class_weight"])
            model.fit(embedding_matrix[train_index], y_train)
            predictions = model.predict(embedding_matrix[test_index])
        else:
            model = Pipeline([
                ("vectorizer", build_vectorizer(representation, experiment["ngram_range"])),
                ("model", build_model(experiment["classifier"], experiment["class_weight"])),
            ])
            model.fit(x_train_text, y_train)
            predictions = model.predict(x_test_text)

        results.append({
            "experiment": name,
            **experiment,
            "class_weight": str(experiment["class_weight"]),
            "ngram_range": str(experiment["ngram_range"]),
            **_metrics(y_test, predictions),
            "model": model,
        })

    return results


def cross_validate(
    texts: Sequence[str],
    ratings: Sequence[int],
    experiment: dict,
    embeddings: EmbeddingFn,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """F1 macro em folds estratificados — um split só é frágil com poucas negativas."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    texts, labels = labelled(texts, ratings)
    representation = experiment["representation"]
    model = build_model(experiment["classifier"], experiment["class_weight"])

    if representation in DENSE_REPRESENTATIONS:
        x_data = embeddings(texts)
        estimator = model
    else:
        x_data = prepare_documents(texts)
        estimator = Pipeline([
            ("vectorizer", build_vectorizer(representation, experiment["ngram_range"])),
            ("model", model),
        ])

    scores = cross_val_score(
        estimator,
        x_data,
        np.asarray(labels),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring="f1_macro",
    )

    return {
        "experiment": experiment_name(experiment),
        "f1_macro_medio": float(scores.mean()),
        "desvio_padrao": float(scores.std()),
        "folds": [float(score) for score in scores],
    }
