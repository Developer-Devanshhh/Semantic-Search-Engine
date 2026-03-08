"""
FastAPI service — loads everything into memory at startup and serves queries.

The important concurrency detail: SentenceTransformer.encode() is a blocking
PyTorch call. If we run it directly in an async route, the event loop freezes
and concurrent requests queue up. asyncio.to_thread() offloads it to a worker
thread so the API stays responsive under load.
"""

import os
import asyncio
import pickle
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.clustering import FuzzyClusterer
from src.cache import SemanticCache
from src.models import QueryRequest, QueryResponse, CacheStatsResponse, CacheFlushResponse

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models + data once at startup. No per-request loading."""

    app.state.embedder = Embedder()

    with open(os.path.join(ARTIFACTS_DIR, 'documents.pkl'), 'rb') as f:
        app.state.documents = pickle.load(f)

    app.state.embeddings = np.load(os.path.join(ARTIFACTS_DIR, 'embeddings.npy'))

    app.state.vector_store = VectorStore()
    app.state.vector_store.load(os.path.join(ARTIFACTS_DIR, 'faiss.index'))

    app.state.clusterer = FuzzyClusterer()
    app.state.clusterer.load(ARTIFACTS_DIR)

    memberships = np.load(os.path.join(ARTIFACTS_DIR, 'memberships.npy'))

    # build cache with per-cluster adaptive thresholds
    app.state.cache = SemanticCache(
        n_clusters=app.state.clusterer.optimal_k,
        base_threshold=0.75,
        max_entries=500,
    )
    app.state.cache.compute_adaptive_thresholds(
        app.state.embeddings, memberships
    )

    n_docs = len(app.state.documents)
    k = app.state.clusterer.optimal_k
    print(f"Ready: {n_docs} docs, K={k} clusters, FAISS loaded")

    yield


app = FastAPI(
    title="Semantic Search Engine",
    description="20 Newsgroups semantic search with fuzzy clustering and cluster-routed caching",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def search_query(request: QueryRequest):
    embedder = app.state.embedder
    clusterer = app.state.clusterer
    cache = app.state.cache
    store = app.state.vector_store
    docs = app.state.documents

    # offload to thread — keeps the event loop free for concurrent requests
    query_vec = await asyncio.to_thread(
        embedder.encode, request.query, 1, False
    )

    # cluster assignment for routing
    query_reduced = clusterer.transform_query(query_vec)
    probs = clusterer.predict_proba(query_reduced)[0]
    dominant = int(np.argmax(probs))

    # cache check (O(N/K) not O(N))
    cached, sim = cache.lookup(query_vec.flatten(), probs)

    if cached is not None:
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=cached.query_text,
            similarity_score=round(float(sim), 4),
            result=cached.result,
            dominant_cluster=dominant,
        )

    # miss — hit FAISS
    distances, indices = store.search(query_vec, top_k=5)

    results = []
    for dist, idx in zip(distances, indices):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        doc = docs[int(idx)]
        results.append({
            'text': doc['text'][:500],
            'category': doc['category'],
            'score': round(float(dist), 4),
        })

    cache.store(request.query, query_vec, results, probs)

    return QueryResponse(
        query=request.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=results,
        dominant_cluster=dominant,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    return CacheStatsResponse(**app.state.cache.stats())


@app.delete("/cache", response_model=CacheFlushResponse)
async def flush_cache():
    cleared = app.state.cache.flush()
    return CacheFlushResponse(
        message="Cache flushed successfully",
        entries_cleared=cleared,
    )
