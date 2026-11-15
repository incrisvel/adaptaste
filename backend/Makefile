.PHONY: run format-db artifacts tfidf embeddings nlp-setup

run:
	uv run uvicorn main:app --app-dir src --reload

format-db:
	PYTHONPATH=src uv run python scripts/format_reviews.py

artifacts:
	PYTHONPATH=src uv run python scripts/rebuild_artifacts.py

tfidf:
	PYTHONPATH=src uv run python scripts/generate_tfidf.py

embeddings:
	PYTHONPATH=src uv run python scripts/generate_embeddings.py

nlp-setup:
	uv run python -m nltk.downloader punkt punkt_tab stopwords rslp
