"""
run_experiment.py
=================

Run #GNN ablation on a dataset in the official drhash-cn format.

This script:
  1. Reads `features.txt` and `<name>_<ratio>.mat` from the dataset folder
     (so it uses the SAME train/test split as the official C++ binary).
  2. Runs my Python reimplementation of the original algorithm, plus
     three modifications:
         A. Jumping-Knowledge concatenation
         B. Self-influence parameter alpha
         C. Random neighbor cap
     plus an "all three" combination.
  3. Computes AUC the SAME way as the official MATLAB script
     `lp_evaluation.m`:
         sample 10,000 random non-edges (pairs with network[i,j] = 0)
         sample 10,000 test edges from testGraph (with replacement)
         AUC = (#test_sims > non_sims  +  0.5 * #ties) / 10,000
     This matches the official protocol so the comparison is fair.
  4. Optionally evaluates an embedding file produced by the official
     C++ binary (`--official-embedding path`) using the SAME protocol.
  5. Writes a CSV with one row per config.

Usage:
    # Run my Python code (5 configs) on a subset
    python run_experiment.py --data data/twitter500 --ratio 0.8 \\
        --T 3 --K 200 --seed 42 --out results.csv

    # ALSO score the official binary's embedding on the same subset
    # (after running ./hashgnn ... -embedding official_emb.txt)
    python run_experiment.py --data data/twitter500 --ratio 0.8 \\
        --T 3 --K 200 --seed 42 --out results.csv \\
        --official-embedding official_emb.txt

    # Run a single config instead of the full ablation
    python run_experiment.py --data data/twitter500 --ratio 0.8 \\
        --no-ablation --jk concat --alpha 0.7 --neigh-cap 30
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import scipy.sparse as sp

import data_io as io
from hashgnn import hash_gnn


# ---------------------------------------------------------------------------
# AUC protocol matching lp_evaluation.m
# ---------------------------------------------------------------------------

def hamming_sim_pairs(H: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Hamming similarity (fraction of matching positions) for an array
    of (u, v) pairs. H has shape (n, D); pairs has shape (m, 2)."""
    return np.mean(H[pairs[:, 0]] == H[pairs[:, 1]], axis=1)


def sample_non_edges(network: sp.spmatrix, n_samples: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Sample n_samples random pairs (i, j) where network[i, j] == 0.
    Matches the loop in lp_evaluation.m (uniform over all pairs)."""
    n = network.shape[0]
    net_csr = network.tocsr()
    out = np.empty((n_samples, 2), dtype=np.int64)
    filled = 0
    # Vectorized rejection sampling for speed
    while filled < n_samples:
        need = n_samples - filled
        # Oversample to reduce loop iterations
        cand = rng.integers(0, n, size=(need * 2, 2))
        # Drop self-loops
        cand = cand[cand[:, 0] != cand[:, 1]]
        # Drop pairs that are edges
        # net_csr[i, j] returns a 1x1 matrix; do it in bulk via fancy indexing
        rows = cand[:, 0]; cols = cand[:, 1]
        vals = np.asarray(net_csr[rows, cols]).ravel()
        cand = cand[vals == 0]
        if len(cand) == 0:
            continue
        take = min(len(cand), need)
        out[filled:filled + take] = cand[:take]
        filled += take
    return out


def sample_test_edges(test_graph: sp.spmatrix, n_samples: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Sample n_samples edges from testGraph (i.e. positions where it's 1),
    with replacement. Matches lp_evaluation.m's `randi` over `testedEdges`.
    Only takes the upper-triangular part (each undirected edge once)."""
    coo = sp.triu(test_graph, k=1).tocoo()
    if coo.nnz == 0:
        raise ValueError("testGraph has no upper-triangular edges to sample.")
    idx = rng.integers(0, coo.nnz, size=n_samples)
    return np.column_stack([coo.row[idx], coo.col[idx]])


def auc_paired(test_sims: np.ndarray, non_sims: np.ndarray) -> float:
    """AUC formula from lp_evaluation.m:
        AUC = (#( test > non ) + 0.5 * #( test == non )) / N
    where the two sample arrays are paired (same length).
    """
    assert len(test_sims) == len(non_sims)
    great = float(np.sum(test_sims > non_sims))
    equal = float(np.sum(test_sims == non_sims))
    return (great + 0.5 * equal) / len(test_sims)


def evaluate_embedding(H: np.ndarray, network: sp.spmatrix,
                       test_graph: sp.spmatrix, *, n_pairs: int = 10000,
                       seed: int = 0) -> float:
    """Compute AUC using the official protocol."""
    rng = np.random.default_rng(seed)
    non_pairs = sample_non_edges(network, n_pairs, rng)
    test_pairs = sample_test_edges(test_graph, n_pairs, rng)
    non_sims = hamming_sim_pairs(H, non_pairs)
    test_sims = hamming_sim_pairs(H, test_pairs)
    return auc_paired(test_sims, non_sims)


# ---------------------------------------------------------------------------
# Read official-binary embedding output
# ---------------------------------------------------------------------------

def read_embedding_file(path: str) -> np.ndarray:
    """The official C++ binary writes one line per node with K
    space-separated unsigned integers."""
    return np.loadtxt(path, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(name, adj_train, features, num_attrs, T, K, seed,
            network, test_graph, eval_seed, **kw):
    t0 = time.perf_counter()
    H = hash_gnn(adj_train, features, T=T, K=K, num_attrs=num_attrs,
                 seed=seed, **kw)
    t_embed = time.perf_counter() - t0
    score = evaluate_embedding(H, network, test_graph, seed=eval_seed)
    return {
        "config": name,
        "AUC": round(score, 4),
        "embed_time_s": round(t_embed, 3),
        "embed_dim": H.shape[1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to dataset folder (e.g. data/twitter500)")
    p.add_argument("--ratio", type=float, default=0.8,
                   help="Training ratio (must match an existing split file)")
    p.add_argument("--T", type=int, default=3)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=0,
                   help="Seed for the AUC sampling (held fixed across configs "
                        "for a fair comparison)")
    p.add_argument("--n-eval-pairs", type=int, default=10000)
    p.add_argument("--jk", choices=["last", "concat"], default="last")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--neigh-cap", type=int, default=None)
    p.add_argument("--no-ablation", action="store_true",
                   help="Run only the single config given by the flags above")
    p.add_argument("--official-embedding", type=str, default=None,
                   help="If set, also compute AUC for an embedding file "
                        "produced by the official C++ binary, on the same "
                        "split (for direct comparison)")
    p.add_argument("--out", type=str, default="results.csv")
    args = p.parse_args()

    # ---- load dataset ----
    print(f"Loading {args.data} at ratio {args.ratio}...")
    features, adj_train, network, train_graph, test_graph, name = \
        io.load_dataset(args.data, args.ratio)
    n = len(features)
    num_attrs = max((max(f) for f in features if f), default=0) + 1
    train_edges = train_graph.nnz // 2
    test_edges = test_graph.nnz // 2
    avg_deg = sum(len(a) for a in adj_train) / max(n, 1)
    print(f"  dataset name: {name}")
    print(f"  nodes: {n}, attributes ≤ {num_attrs}")
    print(f"  train edges: {train_edges}, test edges: {test_edges}")
    print(f"  avg train degree: {avg_deg:.2f}")

    # ---- pick configs ----
    if args.no_ablation:
        configs = [(
            f"jk={args.jk}|alpha={args.alpha}|cap={args.neigh_cap}",
            dict(jk=args.jk, alpha=args.alpha, neigh_cap=args.neigh_cap),
        )]
    else:
        configs = [
            ("+alpha=0.7",     dict(jk="last",   alpha=0.7,  neigh_cap=None)),
            ("original",       dict(jk="last",   alpha=None, neigh_cap=None)),
            ("+JK concat",     dict(jk="concat", alpha=None, neigh_cap=None)),
            ("+neigh-cap=30",  dict(jk="last",   alpha=None, neigh_cap=30)),
            ("all three",      dict(jk="concat", alpha=0.7,  neigh_cap=30)),
        ]

    # ---- run my Python configs ----
    print("\nRunning Python experiments...")
    rows = []
    for cfg_name, kw in configs:
        r = run_one(cfg_name, adj_train, features, num_attrs,
                    T=args.T, K=args.K, seed=args.seed,
                    network=network, test_graph=test_graph,
                    eval_seed=args.eval_seed, **kw)
        rows.append(r)
        print(f"  {r['config']:<18s}  AUC={r['AUC']:.4f}  "
              f"embed={r['embed_time_s']:.2f}s  dim={r['embed_dim']}")

    # ---- (optional) score the official binary's embedding ----
    if args.official_embedding:
        print(f"\nScoring official embedding: {args.official_embedding}")
        H_off = read_embedding_file(args.official_embedding)
        if H_off.shape[0] != n:
            print(f"  WARNING: embedding has {H_off.shape[0]} rows, "
                  f"dataset has {n} nodes")
        score = evaluate_embedding(H_off, network, test_graph,
                                   n_pairs=args.n_eval_pairs,
                                   seed=args.eval_seed)
        rows.insert(0, {  # put it at the top of the table for visibility
            "config": "official C++",
            "AUC": round(score, 4),
            "embed_time_s": float("nan"),
            "embed_dim": H_off.shape[1],
        })
        print(f"  official C++       AUC={score:.4f}  dim={H_off.shape[1]}")

    # ---- summary ----
    print("\n" + "=" * 60)
    print(f"{'config':<20s}  {'AUC':>6s}  {'embed(s)':>10s}  {'dim':>5s}")
    print("=" * 60)
    for r in rows:
        et = "  -" if (isinstance(r["embed_time_s"], float)
                        and np.isnan(r["embed_time_s"])) \
            else f"{r['embed_time_s']:>10.3f}"
        print(f"{r['config']:<20s}  {r['AUC']:>6.4f}  "
              f"{et}  {r['embed_dim']:>5d}")
    print("=" * 60)

    # ---- write csv ----
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
