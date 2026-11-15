import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from nlp.artifacts import artifacts_dir
from nlp.corpus import fetch_reviews
from nlp.representations import BERT_MAX_LENGTH, BERT_MODEL, bert_embeddings


def build(
    texts: list[str],
    batch_size: int = 8,
    model_name: str = BERT_MODEL,
) -> np.ndarray:
    return bert_embeddings(texts, batch_size=batch_size, model_name=model_name)


def save(
    embeddings: np.ndarray,
    directory: Path | None = None,
    model_name: str = BERT_MODEL,
    max_length: int = BERT_MAX_LENGTH,
) -> Path:
    directory = directory or artifacts_dir()
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "bert_embeddings.npy", embeddings)

    config = {
        "model": model_name,
        "max_length": max_length,
        "dimension": int(embeddings.shape[1]),
        "documents": int(embeddings.shape[0]),
        "dtype": str(embeddings.dtype),
    }
    (directory / "bert_model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera os embeddings do BERTimbau.")
    parser.add_argument("--limit", type=int, default=None, help="usa as N primeiras reviews")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    reviews = asyncio.run(fetch_reviews(args.limit))
    embeddings = build(
        [review["text_raw"] for review in reviews],
        batch_size=args.batch_size,
    )
    directory = save(embeddings)

    print(f"BERT: {embeddings.shape[0]} reviews x {embeddings.shape[1]} dims -> {directory}")


if __name__ == "__main__":
    main()
