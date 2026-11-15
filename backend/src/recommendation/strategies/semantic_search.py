import numpy as np

from nlp.representations import cosine_ranking


def ranking(encoded: dict, embeddings) -> tuple[np.ndarray, np.ndarray]:
    """Linhas da matriz de embeddings em ordem de similaridade com a consulta."""
    vector = encoded["bert"].reshape(1, -1)

    return cosine_ranking(embeddings, vector)
