from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nlp import artifacts
from nlp.preprocessing import preprocess
from nlp.representations import bert_embeddings, transform_tfidf

Representation = Literal["bert", "tfidf"]


def search(
    query: str,
    k: int = 5,
    representation: Representation = "bert",
) -> list[dict]:
    """k reviews mais similares à consulta, com id e place_id do índice."""
    index = artifacts.load_review_index()

    if representation == "tfidf":
        # mesma cadeia que gerou o text_lemmatized do corpus
        documents = [preprocess(query)["lemmatized"]]
        matrix = artifacts.load_tfidf_matrix()
        vector = transform_tfidf(artifacts.load_tfidf_vectorizer(), documents)
        similarities = cosine_similarity(vector, matrix)[0]
    else:
        embeddings = artifacts.load_bert_embeddings()
        vector = bert_embeddings([query])[0].reshape(1, -1)
        similarities = cosine_similarity(vector, embeddings)[0]

    if len(index) != similarities.shape[0]:
        raise ValueError(
            f"artefatos desalinhados: índice={len(index)}, "
            f"matriz={similarities.shape[0]}"
        )

    order = np.argsort(similarities)[::-1][:k]

    return [
        {
            "position": position,
            **index[row],
            "similarity": float(similarities[row]),
        }
        for position, row in enumerate(order, start=1)
    ]
