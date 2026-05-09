"""
hashgnn.py
==========

Reference implementation of #GNN ("HashGNN") from:
    Wei Wu, Bin Li, Chuan Luo, Wolfgang Nejdl.
    "Hashing-Accelerated Graph Neural Networks for Link Prediction." WWW 2021.
    arXiv:2105.14280

Plus three optional modifications (each toggleable independently):
    A. Jumping-Knowledge concatenation across iterations  (`jk='concat'`)
    B. Explicit self-influence parameter alpha            (`alpha` in (0, 1))
    C. Random neighborhood subsampling                    (`neigh_cap=int`)

With all three modifications disabled (the defaults), this reproduces
Algorithm 1 of the paper exactly.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _next_prime(n: int) -> int:
    """Smallest prime >= n. Used as the modulus c in the universal hash family
    pi(i) = (a*i + b) mod c, where the paper requires c >= |U| (§3.3)."""
    while not _is_prime(n):
        n += 1
    return n


def hash_gnn(
    adj: Sequence[Sequence[int]],
    features: Sequence[Sequence[int]],
    T: int,
    K: int,
    num_attrs: int,
    *,
    seed: int = 42,
    jk: str = "last",
    alpha: Optional[float] = None,
    neigh_cap: Optional[int] = None,
) -> np.ndarray:
    """
    Run #GNN (Algorithm 1) and return node embeddings.

    Parameters
    ----------
    adj : list of lists of ints
        Adjacency list. adj[v] = neighbors of node v in the training graph.
    features : list of lists of ints
        features[v] = active attribute IDs for node v.
    T : int
        Number of message-passing iterations.
    K : int
        Embedding dimension.
    num_attrs : int
        |A|, the size of the attribute set.
    seed : int
        Random seed.

    jk : "last" or "concat"
        "last" = paper default; "concat" = concatenate iterations 1..T.
    alpha : float in (0,1) or None
        If set, with prob alpha use argmin over self only, else neighbors only.
    neigh_cap : int or None
        If set, cap each neighborhood at this size by random sampling.

    Returns
    -------
    H : np.ndarray, (n_nodes, K) if jk='last' else (n_nodes, T*K)
    """
    if jk not in ("last", "concat"):
        raise ValueError(f"jk must be 'last' or 'concat', got {jk!r}")
    use_alpha = alpha is not None and 0.0 <= alpha < 1.0

    rng = np.random.default_rng(seed)
    n = len(adj)
    c = _next_prime(2 * num_attrs + 100)

    x: List[np.ndarray] = [np.asarray(f, dtype=np.int64) for f in features]
    H_iters: List[np.ndarray] = []

    for t in range(T):
        a1 = rng.integers(1, c, size=K); b1 = rng.integers(0, c, size=K)
        a2 = rng.integers(1, c, size=K); b2 = rng.integers(0, c, size=K)
        a3 = rng.integers(1, c, size=K); b3 = rng.integers(0, c, size=K)

        gate_self = (rng.random(K) < alpha) if use_alpha else None
        new_x: List[np.ndarray] = [np.zeros(K, dtype=np.int64) for _ in range(n)]

        for k in range(K):
            # Phase 1: per-node diffused message
            diffused = np.zeros(n, dtype=np.int64)
            valid = np.ones(n, dtype=bool)
            for v in range(n):
                xv = x[v]
                if xv.size == 0:
                    valid[v] = False
                    continue
                hv = (a3[k] * xv + b3[k]) % c
                diffused[v] = xv[int(np.argmin(hv))]

            # Phase 2: combine self + neighbor messages
            for v in range(n):
                neighbors = adj[v]
                if neigh_cap is not None and len(neighbors) > neigh_cap:
                    neighbors = rng.choice(np.asarray(neighbors),
                                            size=neigh_cap, replace=False)

                xv = x[v]
                if xv.size > 0:
                    h_self = (a1[k] * xv + b1[k]) % c
                    e_self = xv
                else:
                    h_self = np.empty(0, dtype=np.int64)
                    e_self = np.empty(0, dtype=np.int64)

                neigh_msgs = [diffused[u] for u in neighbors if valid[u]]
                if neigh_msgs:
                    e_neigh = np.asarray(neigh_msgs, dtype=np.int64)
                    h_neigh = (a2[k] * e_neigh + b2[k]) % c
                else:
                    e_neigh = np.empty(0, dtype=np.int64)
                    h_neigh = np.empty(0, dtype=np.int64)

                if use_alpha:
                    if gate_self[k] and e_self.size > 0:
                        new_x[v][k] = int(e_self[int(np.argmin(h_self))])
                        continue
                    if (not gate_self[k]) and e_neigh.size > 0:
                        new_x[v][k] = int(e_neigh[int(np.argmin(h_neigh))])
                        continue

                if h_self.size + h_neigh.size > 0:
                    all_h = np.concatenate([h_self, h_neigh])
                    all_e = np.concatenate([e_self, e_neigh])
                    new_x[v][k] = int(all_e[int(np.argmin(all_h))])
                else:
                    new_x[v][k] = 0

        x = new_x
        if jk == "concat":
            H_iters.append(np.stack(x))

    if jk == "concat":
        return np.concatenate(H_iters, axis=1)
    return np.stack(x)
