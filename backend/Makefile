.PHONY: run format-db preprocess artifacts places tfidf embeddings evaluate nlp-setup

run:
	uv run uvicorn main:app --app-dir src --reload

format-db:
	PYTHONPATH=src uv run python scripts/format_reviews.py

preprocess:
	PYTHONPATH=src uv run python scripts/preprocess_reviews.py

artifacts:
	PYTHONPATH=src uv run python scripts/rebuild_artifacts.py

places:
	PYTHONPATH=src uv run python scripts/generate_places_artifacts.py

tfidf:
	PYTHONPATH=src uv run python scripts/generate_tfidf.py

embeddings:
	PYTHONPATH=src uv run python scripts/generate_embeddings.py

evaluate:
	PYTHONPATH=src uv run python scripts/evaluate_recommendation.py

nlp-setup:
	uv run python -m nltk.downloader punkt punkt_tab stopwords rslp
