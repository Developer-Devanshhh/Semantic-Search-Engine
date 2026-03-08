# cache.py — cluster-routed semantic cache (no Redis, no Memcached)
#
# The idea: instead of comparing a query against every cached entry (O(N)),
# use the GMM cluster assignments to route it to the right bucket first.
# With K clusters, lookup becomes O(N/K). For K=15, that's 15x fewer
# comparisons as the cache grows.
#
# Two other things we do differently from a basic dict cache:
#   1. Adaptive thresholds per cluster (dense clusters need stricter matching)
#   2. Frequency-weighted eviction (not just LRU)

import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    result: list
    cluster_id: int
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1


class SemanticCache:

    def __init__(self, n_clusters, base_threshold=0.75, max_entries=500):
        self.buckets = defaultdict(list)  # cluster_id → [CacheEntry, ...]
        self.n_clusters = n_clusters
        self.base_threshold = base_threshold
        self.max_entries = max_entries
        self.thresholds = {}  # per-cluster adaptive τ

        self.hit_count = 0
        self.miss_count = 0

    def compute_adaptive_thresholds(self, embeddings, memberships,
                                    membership_cutoff=0.3):
        """
        Compute per-cluster thresholds based on how tight each cluster is.

        Dense clusters (high avg intra-similarity) get a strict τ — otherwise
        we'd return false cache hits between subtly different queries.
        Sparse clusters get a looser τ because in a spread-out space,
        even moderate similarity means a genuine match.
        """
        for c in range(self.n_clusters):
            mask = memberships[:, c] > membership_cutoff
            cluster_vecs = embeddings[mask]

            if len(cluster_vecs) < 2:
                self.thresholds[c] = self.base_threshold
                continue

            # subsample for speed — full pairwise on 1000+ vectors is slow
            if len(cluster_vecs) > 500:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(cluster_vecs), 500, replace=False)
                cluster_vecs = cluster_vecs[idx]

            sims = cosine_similarity(cluster_vecs)
            upper = sims[np.triu_indices_from(sims, k=1)]
            mean_sim = float(np.mean(upper))

            # simple linear scaling around the base threshold
            tau = self.base_threshold + 0.12 * (mean_sim - 0.5)
            self.thresholds[c] = float(np.clip(tau, 0.60, 0.92))

    def lookup(self, query_embedding, cluster_probs):
        """
        Route query to top-2 cluster buckets and find the best match.
        Returns (entry, similarity) on hit, (None, best_sim) on miss.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # only search the 2 most probable clusters
        top_clusters = np.argsort(cluster_probs)[-2:][::-1]

        best_sim = -1.0
        best_entry = None

        for c in top_clusters:
            for entry in self.buckets[int(c)]:
                sim = float(cosine_similarity(
                    query_embedding, entry.query_embedding.reshape(1, -1)
                )[0, 0])
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        # check against the cluster-specific threshold
        if best_entry and best_sim >= self.thresholds.get(
            best_entry.cluster_id, self.base_threshold
        ):
            self.hit_count += 1
            best_entry.access_count += 1
            best_entry.last_accessed = time.time()
            return best_entry, best_sim

        self.miss_count += 1
        return None, best_sim

    def store(self, query_text, query_embedding, result, cluster_probs):
        """Add a new entry to the dominant cluster's bucket."""
        cluster_id = int(np.argmax(cluster_probs))
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding.flatten(),
            result=result,
            cluster_id=cluster_id,
        )
        self.buckets[cluster_id].append(entry)
        self._maybe_evict()

    def _maybe_evict(self):
        """
        Frequency-weighted eviction.
        Score = (1 / access_count) * age. High score = rarely used + old → evict.
        This beats pure LRU because popular queries stay cached even if not
        accessed in the last few seconds.
        """
        total = sum(len(b) for b in self.buckets.values())
        if total <= self.max_entries:
            return

        now = time.time()
        scored = []
        for c, entries in self.buckets.items():
            for i, entry in enumerate(entries):
                age = now - entry.last_accessed
                score = (1.0 / max(entry.access_count, 1)) * age
                scored.append((score, c, i))

        scored.sort(reverse=True)
        n_remove = total - self.max_entries

        # collect removals per bucket, remove from end to keep indices stable
        removals = defaultdict(list)
        for _, c, i in scored[:n_remove]:
            removals[c].append(i)

        for c, idxs in removals.items():
            for i in sorted(idxs, reverse=True):
                if i < len(self.buckets[c]):
                    self.buckets[c].pop(i)

    def flush(self):
        """Clear everything. Returns how many entries were dropped."""
        total = sum(len(b) for b in self.buckets.values())
        self.buckets.clear()
        self.hit_count = 0
        self.miss_count = 0
        return total

    @property
    def total_entries(self):
        return sum(len(b) for b in self.buckets.values())

    @property
    def hit_rate(self):
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def stats(self):
        return {
            'total_entries': self.total_entries,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': round(self.hit_rate, 4),
        }
