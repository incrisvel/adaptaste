import numpy as np
from scipy import sparse

from nlp.representations import build_tfidf, cosine_ranking
from recommendation.strategies import lexical_search, semantic_search


def test_cosine_ranking_ordena_por_similaridade_decrescente():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    order, scores = cosine_ranking(matrix, np.array([[1.0, 0.0]]))

    assert order.tolist() == [0, 2, 1]
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)


def test_lexical_ranking_encontra_o_documento_com_os_tokens_da_consulta():
    vectorizer, matrix = build_tfidf([["comida", "boa"], ["atendimento", "ruim"]])

    order, scores = lexical_search.ranking(
        {"lemmatized": ["comida"]}, vectorizer, matrix
    )

    assert order[0] == 0
    assert scores[0] > scores[1]


def test_semantic_ranking_encontra_o_vetor_mais_proximo():
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0]])

    order, _ = semantic_search.ranking({"bert": np.array([1.0, 0.0])}, embeddings)

    assert order.tolist() == [1, 0]


def test_lexical_ranking_funciona_com_matriz_esparsa():
    vectorizer, matrix = build_tfidf([["comida", "boa"], ["atendimento", "ruim"]])

    assert sparse.issparse(matrix)
    order, _ = lexical_search.ranking({"lemmatized": ["ruim"]}, vectorizer, matrix)
    assert order[0] == 1
