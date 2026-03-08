"""
Step 1 — clean corpus, generate embeddings, build FAISS index.

Run once:  python -m scripts.01_preprocess

Takes ~10-15 min (mostly embedding generation). Outputs:
  artifacts/documents.pkl   — cleaned docs with metadata
  artifacts/embeddings.npy  — (N, 128) float32
  artifacts/faiss.index     — IVF-PQ index
"""

import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

from src.cleaner import clean_document, is_valid
from src.embedder import Embedder
from src.vector_store import VectorStore

ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    # sklearn handles the first layer of cleaning (headers/footers/quotes)
    print("[1/4] Loading 20 Newsgroups...")
    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=False,
    )
    print(f"  Raw: {len(dataset.data)} documents")

    # our regex pipeline handles the rest
    print("[2/4] Deep cleaning...")
    documents = []
    for i, raw_text in enumerate(tqdm(dataset.data, desc="  Cleaning")):
        text = clean_document(raw_text)
        if is_valid(text):
            documents.append({
                'text': text,
                'original_idx': i,
                'label': int(dataset.target[i]),
                'category': dataset.target_names[dataset.target[i]],
            })

    dropped = len(dataset.data) - len(documents)
    print(f"  Kept {len(documents)} docs (dropped {dropped} noisy/empty)")

    texts = [d['text'] for d in documents]

    # 128-dim MRL-truncated embeddings
    print("[3/4] Embedding (128-dim MRL)...")
    embedder = Embedder()
    embeddings = embedder.encode(texts, batch_size=256)
    print(f"  Shape: {embeddings.shape}")

    print("[4/4] Building FAISS index...")
    store = VectorStore(dim=embeddings.shape[1])
    store.build(embeddings)

    # save artifacts
    with open(os.path.join(ARTIFACTS, 'documents.pkl'), 'wb') as f:
        pickle.dump(documents, f)
    np.save(os.path.join(ARTIFACTS, 'embeddings.npy'), embeddings)
    store.save(os.path.join(ARTIFACTS, 'faiss.index'))

    print(f"\nDone — artifacts in {os.path.abspath(ARTIFACTS)}/")


if __name__ == '__main__':
    main()
