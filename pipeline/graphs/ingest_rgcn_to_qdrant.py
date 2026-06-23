"""
DSA Engine -- RGCN Step 4: Ingest into Qdrant
=============================================
Writes the learned embeddings into Qdrant -- the primary, DB-fed output.

    problems_rgcn  : 128-d pure RGCN embedding   (graph-structural similarity)
    problems_full  : (problem_feature + RGCN) fused embedding

By default this reads the ARTIFACTS (graph.pt + rgcn_problem_embeddings.npy) so
no parquet is required: ids and payload come from the graph's problem_meta, the
RGCN vector from the npy, and `full` = L2-norm(concat(problem_feature, rgcn)).
Use --from-parquet to ingest from the parquet artifact instead.

Run:
    python rgcn/ingest_rgcn_to_qdrant.py
    python rgcn/ingest_rgcn_to_qdrant.py --no-full
    python rgcn/ingest_rgcn_to_qdrant.py --from-parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import config as C

EMBED_KEYS = {"question_embedding", "solution_embedding", "rgcn_embedding",
              "question_solution_embedding", "full_embedding"}


# ---------------------------------------------------------------------------
# Core upsert
# ---------------------------------------------------------------------------

def ingest_embeddings(client, collection, ids, vectors, payloads, batch_size=128):
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, OptimizersConfigDiff,
    )
    if len(vectors) == 0:
        print(f"[X] nothing to ingest into '{collection}'")
        return
    dim = len(vectors[0])
    print(f"\n[->] {collection}: dim={dim}  points={len(vectors)}")

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        print(f"[OK] created '{collection}'")
    else:
        print(f"[->] '{collection}' exists -- upserting")

    points = [PointStruct(id=i, vector=list(map(float, vectors[i])),
                          payload=payloads[i]) for i in range(len(vectors))]
    t0 = time.time()
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[start:start + batch_size])
        pct = min(100, int(100 * (start + batch_size) / max(len(points), 1)))
        print(f"\r  {pct}%", end="", flush=True)
    client.update_collection(
        collection_name=collection,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000))
    cnt = client.count(collection_name=collection).count
    print(f"\n[OK] '{collection}' now has {cnt} points  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Build arrays from artifacts (DB-fed default)
# ---------------------------------------------------------------------------

def _clean_payload(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if k in EMBED_KEYS or v is None:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, float) and np.isnan(v):
            continue
        else:
            out[k] = v
    return out


def from_artifacts():
    import torch
    if not C.GRAPH_PATH.exists() or not C.EMB_NPY.exists():
        raise FileNotFoundError(
            f"Need {C.GRAPH_PATH} and {C.EMB_NPY} -- run build_graph.py + train_rgcn.py.")
    graph = torch.load(C.GRAPH_PATH, weights_only=False)
    rgcn = np.load(C.EMB_NPY).astype(np.float32)
    ids = graph["problem_ids"]
    meta = graph.get("problem_meta", [{} for _ in ids])
    feats = graph["problem_features"].numpy().astype(np.float32)

    full = []
    for i in range(len(ids)):
        v = np.concatenate([feats[i], rgcn[i]]).astype(np.float32)
        n = np.linalg.norm(v)
        full.append(v / (n or 1.0))

    payloads = []
    for i, pid in enumerate(ids):
        pl = _clean_payload(meta[i]); pl.setdefault("problem_id", pid)
        payloads.append(pl)
    return ids, [r for r in rgcn], full, payloads


def from_parquet(path: Path):
    import pandas as pd
    df = pd.read_parquet(path)
    ids, rgcn, full, payloads = [], [], [], []
    for _, row in df.iterrows():
        r = row.get("rgcn_embedding")
        if r is None:
            continue
        ids.append(str(row.get("problem_id", "")))
        rgcn.append(np.asarray(r, dtype=np.float32))
        f = row.get("full_embedding")
        full.append(np.asarray(f, dtype=np.float32) if f is not None else None)
        payloads.append(_clean_payload(
            {k: row[k] for k in df.columns if k not in EMBED_KEYS}))
    return ids, rgcn, full, payloads


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default=C.QDRANT_URL)
    ap.add_argument("--rgcn-collection", default=C.QDRANT_COLLECTION_RGCN)
    ap.add_argument("--full-collection", default=C.QDRANT_COLLECTION_FULL)
    ap.add_argument("--from-parquet", action="store_true",
                    help="ingest from the parquet artifact instead of graph+npy")
    ap.add_argument("--no-rgcn", action="store_true")
    ap.add_argument("--no-full", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 62)
    print("  RGCN STEP 4 -- QDRANT INGEST")
    print("=" * 62)

    if args.from_parquet:
        ids, rgcn, full, payloads = from_parquet(C.OUTPUT_PARQUET)
        print(f"[->] {len(ids)} rows from {C.OUTPUT_PARQUET.name}")
    else:
        ids, rgcn, full, payloads = from_artifacts()
        print(f"[->] {len(ids)} problems from artifacts (graph.pt + npy)")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.qdrant_url, api_key=C.QDRANT_API_KEY, timeout=30)
        client.get_collections()
    except Exception as e:
        print(f"[X] cannot reach Qdrant at {args.qdrant_url}: "
              f"{e.__class__.__name__}: {str(e)[:120]}")
        print("    start it:  docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    if not args.no_rgcn:
        ingest_embeddings(client, args.rgcn_collection, ids, rgcn, payloads)
    if not args.no_full:
        valid = [(i, v) for i, v in enumerate(full) if v is not None]
        if valid:
            idx = [i for i, _ in valid]
            ingest_embeddings(client, args.full_collection,
                              [ids[i] for i in idx], [full[i] for i in idx],
                              [payloads[i] for i in idx])

    print("\n" + "=" * 62)
    print("  QDRANT INGEST COMPLETE")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
