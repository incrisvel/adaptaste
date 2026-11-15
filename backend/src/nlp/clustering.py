import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

MAX_CLUSTERS = 20


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def similarity(matrix) -> np.ndarray:
    return cosine_similarity(matrix)


def cluster_sweep(
    matrix,
    max_k: int = MAX_CLUSTERS,
    random_state: int = 42,
) -> list[dict]:
    """Inércia e silhouette de cosseno para k = 2..max_k."""
    n_samples = matrix.shape[0]
    dense = _dense(matrix)
    results = []

    for k in range(2, min(max_k, n_samples)):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels = model.fit_predict(matrix)
        n_labels = len(np.unique(labels))

        if 1 < n_labels < n_samples:
            silhouette = silhouette_score(
                dense,
                labels,
                metric="cosine",
                sample_size=min(2000, n_samples),
                random_state=random_state,
            )
        else:
            silhouette = float("nan")

        results.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_cosine": float(silhouette),
            "cluster_sizes": np.bincount(labels).tolist(),
        })

    return results


def fit_kmeans(matrix, k: int, random_state: int = 42) -> KMeans:
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    model.fit(matrix)
    return model


def representative_terms(model: KMeans, terms, n_terms: int = 10) -> list[dict]:
    """Termos de maior peso em cada centróide — só para matrizes de BoW/TF-IDF."""
    centroids = model.cluster_centers_
    results = []

    for cluster_id in range(model.n_clusters):
        indices = np.argsort(centroids[cluster_id])[::-1][:n_terms]
        results.append({
            "cluster": cluster_id,
            "representative_tokens": ", ".join(str(terms[index]) for index in indices),
        })

    return results


def centroid_distances(model: KMeans) -> np.ndarray:
    return 1 - cosine_similarity(model.cluster_centers_)


def cluster_cohesion(similarity_matrix: np.ndarray, labels) -> list[dict]:
    """Similaridade média entre pares do mesmo cluster."""
    labels = np.asarray(labels)
    results = []

    for cluster in sorted(np.unique(labels).tolist()):
        indices = np.where(labels == cluster)[0]
        submatrix = similarity_matrix[np.ix_(indices, indices)]
        values = submatrix[np.triu_indices_from(submatrix, k=1)]

        results.append({
            "cluster": int(cluster),
            "n_documents": int(len(indices)),
            "mean_cohesion": float(values.mean()) if len(values) else 1.0,
        })

    return results


def pca_coordinates(matrix, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    return PCA(n_components=n_components, random_state=random_state).fit_transform(_dense(matrix))


def closest_examples(
    matrix,
    model: KMeans,
    texts,
    n_reviews: int = 10,
) -> dict[int, list[dict]]:
    """Reviews mais próximas do centróide de cada cluster, sem textos repetidos."""
    similarities = cosine_similarity(matrix, model.cluster_centers_)
    results = {}

    for cluster in range(model.n_clusters):
        indices = np.where(model.labels_ == cluster)[0]
        ranked = indices[np.argsort(similarities[indices, cluster])[::-1]]
        selected = []
        seen = set()

        for row in ranked:
            text = str(texts[row]).strip().lower()

            if text in seen:
                continue

            seen.add(text)
            selected.append({
                "index": int(row),
                "similarity_to_centroid": float(similarities[row, cluster]),
                "distance_to_centroid": float(1 - similarities[row, cluster]),
            })

            if len(selected) >= n_reviews:
                break

        results[cluster] = selected

    return results
