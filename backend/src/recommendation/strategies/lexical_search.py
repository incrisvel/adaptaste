import numpy as np

from nlp.representations import cosine_ranking, transform_tfidf


def ranking(encoded: dict, vectorizer, matrix) -> tuple[np.ndarray, np.ndarray]:
    """Linhas da matriz TF-IDF em ordem de similaridade com a consulta."""
    vector = transform_tfidf(vectorizer, [encoded["lemmatized"]])

    return cosine_ranking(matrix, vector)
