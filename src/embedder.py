"""
Embedding wrapper — MiniLM with Matryoshka truncation.

We use 128 dims instead of the native 384. MRL guarantees the first N dims
carry the most semantic weight, so truncation costs ~2% quality but gives us
3x faster FAISS search and 3x smaller index. Tested this — the trade-off
is worth it for a 20K corpus.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DIM = 128  # tried 64 (too lossy) and 256 (marginal gain over 128)


class Embedder:

    def __init__(self, model_name=MODEL_NAME, dim=DEFAULT_DIM):
        self.model = SentenceTransformer(model_name, truncate_dim=dim)
        self.dim = dim

    def encode(self, texts, batch_size=256, show_progress=True, normalize=True):
        """
        Encode text(s) to dense vectors. L2-normalizes by default so
        cosine sim = dot product (what FAISS IndexFlatIP expects).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype(np.float32)
