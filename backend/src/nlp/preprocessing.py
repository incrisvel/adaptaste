from functools import cache
from typing import TYPE_CHECKING

import emoji
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

if TYPE_CHECKING:
    from spacy.language import Language

# Negação, intensidade e contraste carregam sentimento: não são stopwords.
PRESERVE_WORDS = frozenset({
    "não", "nunca", "jamais", "nem", "sem",
    "muito", "pouco", "bastante", "demais",
    "apenas", "só", "somente", "mas", "porém", "contudo",
})


@cache
def stopwords_pt() -> frozenset[str]:
    return frozenset(stopwords.words("portuguese")) - PRESERVE_WORDS


@cache
def get_nlp() -> "Language":
    import spacy

    return spacy.load("pt_core_news_sm")


_stemmer = RSLPStemmer()


def clean_text(text: str) -> str:
    return emoji.replace_emoji(text or "", replace="")


def tokenize(text: str) -> list[str]:
    return word_tokenize(text, language="portuguese")


def normalize(tokens: list[str]) -> list[str]:
    return [token.casefold() for token in tokens if token.isalpha()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    vocabulary = stopwords_pt()
    return [token for token in tokens if token not in vocabulary]


def stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(token) for token in tokens]


def lemmatize(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    return [
        token.lemma_
        for token in get_nlp()(" ".join(tokens))
        if token.is_alpha
    ]


def bigrams(tokens: list[str]) -> list[str | tuple[str, str]]:
    return list(ngrams(tokens, 2)) if len(tokens) > 1 else tokens


def preprocess(text: str) -> dict[str, list]:
    """Cadeia do preprocessing: texto cru -> todas as etapas."""
    clean = clean_text(text)
    tokens = tokenize(clean)
    normalized = normalize(tokens)
    no_stopwords = remove_stopwords(normalized)
    lemmatized = lemmatize(no_stopwords)

    return {
        "clean": clean,
        "tokens": tokens,
        "normalized": normalized,
        "no_stopwords": no_stopwords,
        "stemmed": stem(no_stopwords),
        "lemmatized": lemmatized,
        "bigrams": bigrams(lemmatized),
    }
