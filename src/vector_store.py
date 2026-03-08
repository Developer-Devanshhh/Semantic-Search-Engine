# vector_store.py — FAISS IVF-PQ index wrapper
#
# IVF-PQ over flat search because:
#   - IVF partitions the space into Voronoi cells → searches fewer vectors
#   - PQ compresses each 128-dim vector into compact codes → smaller memory
#   - For 20K docs this is arguably overkill, but it demonstrates the right
#     approach for production scale. A flat index would work too, just slower.

import os
import faiss
import numpy as np


class VectorStore:

    def __init__(self, dim=128, nlist=32, m=16, nbits=8, nprobe=8):
        """
        nlist=32  — Voronoi cells. sqrt(20K)≈141 but 32 is plenty here.
        m=16      — sub-quantizers. 128/16 = 8 dims each.
        nprobe=8  — search 25% of cells. More = better recall, slower.
        """
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.index = None

    def build(self, embeddings):
        """Train IVF-PQ on the corpus and add all vectors."""
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFPQ(
            quantizer, self.dim, self.nlist, self.m, self.nbits
        )
        self.index.nprobe = self.nprobe
        self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query_vec, top_k=5):
        """Returns (scores, indices) — scores are inner product (higher=better)."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        return distances[0], indices[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
        self.index.nprobe = self.nprobe
