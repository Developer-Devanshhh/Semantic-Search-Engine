"""
Fuzzy clustering: UMAP reduction → GMM soft assignments.

Chose UMAP over PCA because the embedding manifold is nonlinear — PCA
just projects onto principal axes and misses the curvature. Tried both,
UMAP gives noticeably tighter clusters.

Chose GMM over HDBSCAN because the task requires mathematical justification
for K. GMM has BIC/AIC built in. HDBSCAN auto-discovers K which is neat but
you can't present a BIC curve proving your choice.
"""

import os
import pickle
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class FuzzyClusterer:
    """UMAP (128-dim → 15-dim) then GMM for soft cluster assignments."""

    def __init__(self, umap_dim=15, umap_neighbors=30, umap_min_dist=0.0,
                 random_state=42):
        # 15 dims: enough for GMM covariance estimation without overfitting.
        # tried 10 (too compressed, lost cluster separation) and 20 (marginal gain)
        self.reducer = umap.UMAP(
            n_components=umap_dim,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric='cosine',
            random_state=random_state,
        )
        self.gmm = None
        self.optimal_k = None
        self.embeddings_reduced = None
        self._random_state = random_state

    def reduce(self, embeddings):
        """Fit UMAP and transform. Call this before clustering."""
        self.embeddings_reduced = self.reducer.fit_transform(embeddings)
        return self.embeddings_reduced

    def find_optimal_k(self, k_range=range(5, 35), sample_size=5000):
        """
        Sweep K values with three metrics:
          BIC  — penalizes complexity, primary selection criterion
          AIC  — less aggressive penalty, sanity-checks BIC
          Silhouette — geometric coherence of hard assignments

        Returns dict of {k: {bic, aic, silhouette}} for analysis/plotting.
        """
        X = self.embeddings_reduced
        results = {}

        for k in k_range:
            gmm = GaussianMixture(
                n_components=k, covariance_type='full',
                random_state=self._random_state, n_init=3, max_iter=200,
            )
            gmm.fit(X)
            labels = gmm.predict(X)

            results[k] = {
                'bic': gmm.bic(X),
                'aic': gmm.aic(X),
                'silhouette': silhouette_score(
                    X, labels, sample_size=min(sample_size, len(X))
                ),
            }

        self.optimal_k = min(results, key=lambda k: results[k]['bic'])
        return results

    def fit(self, k=None):
        """Fit final GMM. Uses optimal_k from sweep if k not specified."""
        k = k or self.optimal_k
        self.optimal_k = k

        # n_init=5 for the final fit (more restarts than the sweep)
        self.gmm = GaussianMixture(
            n_components=k, covariance_type='full',
            random_state=self._random_state, n_init=5, max_iter=300,
        )
        self.gmm.fit(self.embeddings_reduced)

    def predict_proba(self, X_reduced):
        """P(cluster | document) — the fuzzy assignment matrix."""
        return self.gmm.predict_proba(X_reduced)

    def predict(self, X_reduced):
        """Hard labels (argmax) — mostly for analysis, not used in cache."""
        return self.gmm.predict(X_reduced)

    def transform_query(self, query_embedding):
        """Project a new query through the fitted UMAP. Used at query time."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return self.reducer.transform(query_embedding)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'umap_reducer.pkl'), 'wb') as f:
            pickle.dump(self.reducer, f)
        with open(os.path.join(directory, 'gmm_model.pkl'), 'wb') as f:
            pickle.dump(self.gmm, f)
        with open(os.path.join(directory, 'cluster_meta.pkl'), 'wb') as f:
            pickle.dump({
                'optimal_k': self.optimal_k,
                'embeddings_reduced': self.embeddings_reduced,
            }, f)

    def load(self, directory):
        with open(os.path.join(directory, 'umap_reducer.pkl'), 'rb') as f:
            self.reducer = pickle.load(f)
        with open(os.path.join(directory, 'gmm_model.pkl'), 'rb') as f:
            self.gmm = pickle.load(f)
        with open(os.path.join(directory, 'cluster_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            self.optimal_k = meta['optimal_k']
            self.embeddings_reduced = meta['embeddings_reduced']
