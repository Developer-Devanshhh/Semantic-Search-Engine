"""
Step 2 — UMAP reduction + GMM clustering + analysis.

Run after step 1:  python -m scripts.02_cluster

Outputs:
  artifacts/umap_reducer.pkl, gmm_model.pkl, cluster_meta.pkl
  artifacts/k_selection.pkl   — BIC/AIC/Silhouette for all K
  artifacts/memberships.npy   — (N, K) soft assignment matrix
"""

import os
import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from src.clustering import FuzzyClusterer

ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


def main():
    print("[1/5] Loading data...")
    embeddings = np.load(os.path.join(ARTIFACTS, 'embeddings.npy'))
    with open(os.path.join(ARTIFACTS, 'documents.pkl'), 'rb') as f:
        documents = pickle.load(f)
    texts = [d['text'] for d in documents]
    categories = [d['category'] for d in documents]
    print(f"  {len(documents)} docs, {embeddings.shape[1]}-dim")

    print("[2/5] UMAP reduction...")
    clusterer = FuzzyClusterer()
    X_reduced = clusterer.reduce(embeddings)
    print(f"  Reduced: {X_reduced.shape}")

    print("[3/5] K selection sweep (5..34)...")
    k_results = clusterer.find_optimal_k(k_range=range(5, 35))
    k = clusterer.optimal_k

    # show top candidates
    top3 = sorted(k_results.items(), key=lambda x: x[1]['bic'])[:3]
    for ki, s in top3:
        print(f"    K={ki:2d}  BIC={s['bic']:.0f}  "
              f"Sil={s['silhouette']:.3f}")
    print(f"  → Selected K={k}")

    print(f"[4/5] Fitting GMM (K={k})...")
    clusterer.fit()
    memberships = clusterer.predict_proba(X_reduced)
    labels = clusterer.predict(X_reduced)

    # --- analysis ---
    print(f"[5/5] Cluster report\n")

    # TF-IDF keywords per cluster
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())

    for c in range(k):
        mask = labels == c
        size = mask.sum()

        # top keywords by cluster-average TF-IDF
        avg_tfidf = tfidf_matrix[mask].mean(axis=0).A1
        top_words = vocab[avg_tfidf.argsort()[-6:][::-1]].tolist()

        # dominant ground-truth categories
        cats = Counter(cat for cat, m in zip(categories, mask) if m).most_common(3)
        cat_str = ', '.join(f"{c_}({n})" for c_, n in cats)

        print(f"  Cluster {c:2d} ({size:4d}): [{', '.join(top_words)}]  ← {cat_str}")

    # boundary docs — where GMM is uncertain (max prob < 0.4)
    max_probs = memberships.max(axis=1)
    n_boundary = (max_probs < 0.4).sum()
    print(f"\n  Boundary docs (max P < 0.4): {n_boundary} "
          f"({100 * n_boundary / len(documents):.1f}%)")

    boundary_ids = np.where(max_probs < 0.4)[0][:3]
    for idx in boundary_ids:
        top2 = memberships[idx].argsort()[-2:][::-1]
        snippet = documents[idx]['text'][:80].replace('\n', ' ')
        print(f"    [{idx}] cluster {top2[0]}({memberships[idx][top2[0]]:.2f}) / "
              f"{top2[1]}({memberships[idx][top2[1]]:.2f}): \"{snippet}...\"")

    # save
    clusterer.save(ARTIFACTS)
    np.save(os.path.join(ARTIFACTS, 'memberships.npy'), memberships)
    with open(os.path.join(ARTIFACTS, 'k_selection.pkl'), 'wb') as f:
        pickle.dump(k_results, f)

    print(f"\nDone — clustering artifacts saved.")


if __name__ == '__main__':
    main()
