import json
from functools import cache
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from core.settings import settings


def artifacts_dir() -> Path:
    return Path(settings.artifacts_dir)


# shortcut: cache vive no processo, regerar artefatos exige reiniciar a API.
@cache
def load_bert_config() -> dict:
    path = artifacts_dir() / "bert_model_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def load_review_index() -> list[dict]:
    path = artifacts_dir() / "review_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def load_tfidf_vectorizer() -> TfidfVectorizer:
    return joblib.load(artifacts_dir() / "tfidf_vectorizer.joblib")


@cache
def load_tfidf_matrix() -> sparse.csr_matrix:
    return sparse.load_npz(artifacts_dir() / "tfidf_matrix.npz")


@cache
def load_bert_embeddings() -> np.ndarray:
    return np.load(artifacts_dir() / "bert_embeddings.npy")


@cache
def load_place_index() -> list[dict]:
    path = artifacts_dir() / "place_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def load_description_vectorizer() -> TfidfVectorizer:
    return joblib.load(artifacts_dir() / "description_vectorizer.joblib")


@cache
def load_description_matrix() -> sparse.csr_matrix:
    return sparse.load_npz(artifacts_dir() / "description_tfidf.npz")


@cache
def load_description_embeddings() -> np.ndarray:
    return np.load(artifacts_dir() / "description_embeddings.npy")

