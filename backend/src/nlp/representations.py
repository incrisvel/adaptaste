from collections.abc import Iterable, Sequence
from functools import cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
BERT_MAX_LENGTH = 128


def cosine_ranking(matrix, vector) -> tuple[np.ndarray, np.ndarray]:
    """Linhas da matriz em ordem de similaridade decrescente, com as similaridades."""
    similarities = cosine_similarity(vector, matrix)[0]
    order = np.argsort(-similarities)

    return order, similarities[order]


def identity_tokens(tokens: Iterable[str]) -> list[str]:
    # analyzer precisa ser função de módulo: joblib não serializa lambda
    return list(tokens)


def build_tfidf(
    documents: Iterable[Sequence[str]],
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """`documents` já vem tokenizado (text_lemmatized.split(), por exemplo)."""
    vectorizer = TfidfVectorizer(analyzer=identity_tokens)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def transform_tfidf(
    vectorizer: TfidfVectorizer,
    documents: Iterable[Sequence[str]],
) -> sparse.csr_matrix:
    return vectorizer.transform(documents)


@cache
def get_bert(model_name: str = BERT_MODEL):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device


def mean_pooling(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()
    total = (last_hidden_state * mask).sum(dim=1)

    return total / torch.clamp(mask.sum(dim=1), min=1e-9)


def bert_embeddings(
    texts: Iterable[str],
    batch_size: int = 8,
    model_name: str = BERT_MODEL,
    max_length: int = BERT_MAX_LENGTH,
) -> np.ndarray:
    """Um vetor por texto: BERTimbau + mean pooling."""
    import torch

    tokenizer, model, device = get_bert(model_name)
    texts = list(texts)
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        embeddings.append(
            mean_pooling(
                output.last_hidden_state,
                inputs["attention_mask"],
            ).cpu().numpy()
        )

    if not embeddings:
        raise ValueError("nenhum texto para gerar embeddings")

    return np.vstack(embeddings)
