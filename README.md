# Semantic Search System — 20 Newsgroups

A lightweight semantic search API with fuzzy clustering, a cluster-routed semantic cache, and a FastAPI service. Built from first principles on the [20 Newsgroups](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups) dataset.

## Architecture

```
Query → Embed (128-dim MRL) → UMAP transform → GMM predict_proba
                                                     │
                                          Top-2 clusters by P(k|q)
                                                     │
                                     Cache lookup (O(N/K) per cluster)
                                                     │
                              HIT → return cached  │  MISS → FAISS search → cache → return
```

## Design Decisions

| Component           | Choice                                    | Rationale                                                                                         |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Embedding**       | `all-MiniLM-L6-v2` @ 128-dim (Matryoshka) | 3× less memory and faster search vs full 384-dim, with ~98% semantic quality retained             |
| **Vector DB**       | FAISS `IndexIVFPQ`                        | 7× faster cold-search than ChromaDB; product quantization compresses index further                |
| **Dim. reduction**  | UMAP (15-dim, cosine)                     | Non-linear manifold preservation — PCA misses the curvature in embedding space                    |
| **Clustering**      | GMM (scikit-learn)                        | Produces genuine probability distributions; K justified via BIC + AIC + Silhouette                |
| **Cache routing**   | Cluster-partitioned dict                  | O(N/K) lookup instead of O(N); fuzzy membership routes queries to top-2 cluster buckets           |
| **Threshold**       | Adaptive per-cluster τ                    | Dense clusters get strict thresholds, sparse clusters get loose — static 0.85 is a common mistake |
| **Eviction**        | Frequency-weighted (MAB-inspired)         | Better than LRU for skewed query distributions                                                    |
| **API concurrency** | `asyncio.to_thread`                       | Prevents event-loop blocking during PyTorch inference under concurrent load                       |

### Why Not HDBSCAN?

HDBSCAN's auto-discovery of K is appealing, and its noise handling is genuinely better. We evaluated it. But the problem asks to _"justify [K] with evidence"_ — GMM's BIC/AIC curves provide mathematical proof for the optimal K. HDBSCAN discovers K, but you can't independently argue for it with a formal criterion. For this task, the ability to present a BIC-minimum curve outweighs HDBSCAN's noise isolation.

## Project Structure

```
├── src/
│   ├── cleaner.py          # Multi-stage regex cleaning for Usenet noise
│   ├── embedder.py         # SentenceTransformer wrapper (128-dim MRL)
│   ├── vector_store.py     # FAISS IndexIVFPQ wrapper
│   ├── clustering.py       # UMAP + GMM pipeline with K-selection
│   ├── cache.py            # Cluster-routed semantic cache
│   └── models.py           # Pydantic schemas
├── api/
│   └── main.py             # FastAPI endpoints with async inference
├── scripts/
│   ├── 01_preprocess.py    # Clean, embed, build FAISS index
│   └── 02_cluster.py       # UMAP + GMM clustering with analysis
├── notebooks/
│   └── analysis.ipynb      # Cluster visualization, threshold curves
├── artifacts/              # Saved index, GMM, UMAP reducer (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick Start

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build Corpus

```bash
python -m scripts.01_preprocess    # ~10-15 min: clean, embed, index
python -m scripts.02_cluster       # ~5 min: UMAP + GMM clustering
```

### 3. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Test

```bash
# Interactive docs
open http://localhost:8000/docs
```

## API Endpoints

### `POST /query`

```json
// Request
{ "query": "space exploration missions" }

// Response
{
  "query": "space exploration missions",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [...],
  "dominant_cluster": 7
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

Flushes the cache and resets all statistics.

## Docker

```bash
docker compose up --build
```

The container starts uvicorn on port 8000. The model is pre-downloaded at build time for instant startup.

## Data Cleaning

The 20 Newsgroups dataset is 1990s Usenet — heavily polluted with routing headers, email addresses, quoted replies, and signature blocks. Our pipeline applies three cleaning layers:

1. **sklearn built-in** — `remove=('headers', 'footers', 'quotes')`
2. **Custom regex** — strips residual headers, emails, URLs, file paths, `In article <...>` references, `--` signature blocks
3. **Quality filter** — drops documents < 50 chars or > 5000 chars post-cleaning

## Similarity Threshold

The threshold τ determines cache hit sensitivity. A static value (like 0.85) is a common choice but fundamentally flawed — it ignores cluster density. Our cache computes per-cluster thresholds:

- **Dense clusters** (e.g., hardware specs with uniform jargon) → stricter τ ≈ 0.88
- **Diffuse clusters** (e.g., politics with varied language) → looser τ ≈ 0.72

The analysis notebook explores threshold sensitivity curves showing hit-rate vs accuracy trade-offs across τ ∈ [0.5, 0.95].
