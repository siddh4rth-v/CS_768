"""
data_io.py
==========

Read and write datasets in the format used by the official drhash-cn/
graph-hashing repo. Each dataset lives in a directory `data/<name>/`
containing:

    features.txt                   — line per node, space-separated attr IDs
    <name>.adjlist.<ratio>          — line per node, neighbors in training graph
                                      ratio in {0.5, 0.6, 0.7, 0.8, 0.9}
    <name>.mat                      — MATLAB file with `network`, `attributes`
    <name>_<ratio>.mat              — MATLAB file with `trainGraph`, `testGraph`,
                                      `network`, `attributes`

Notes on the format:

* Node IDs are implicit: 0-indexed by line number in features.txt and the
  adjlist files.
* features.txt may have empty lines (nodes with no attributes).
* The `network` adjacency in .mat files is symmetric (undirected); the
  diagonal is set to 0 by lp_evaluation.m before AUC computation.
* The C++ binary uses sentinel `MAX_FEATURE_NUM = 10000000` for "no
  features" — we don't write that explicitly; an empty list is fine.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# features.txt
# ---------------------------------------------------------------------------

def read_features_txt(path: str) -> List[List[int]]:
    """Each line is space-separated attribute IDs. Empty line = no attrs."""
    features: List[List[int]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                features.append([])
            else:
                features.append([int(x) for x in line.split()])
    return features


def write_features_txt(path: str, features: List[List[int]]) -> None:
    with open(path, "w") as f:
        for fs in features:
            if fs:
                f.write(" ".join(str(int(x)) for x in fs))
            f.write("\n")


# ---------------------------------------------------------------------------
# <name>.adjlist.<ratio>
# ---------------------------------------------------------------------------

def read_adjlist(path: str, n_nodes: Optional[int] = None
                 ) -> List[List[int]]:
    """Each line is space-separated neighbor IDs of the row's node."""
    adj: List[List[int]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                adj.append([])
            else:
                adj.append([int(x) for x in line.split()])
    # Some adjlist files have fewer lines than n_nodes (trailing empties);
    # pad if a target node count was given.
    if n_nodes is not None and len(adj) < n_nodes:
        adj.extend([] for _ in range(n_nodes - len(adj)))
    return adj


def write_adjlist(path: str, adj: List[List[int]]) -> None:
    with open(path, "w") as f:
        for nbrs in adj:
            if nbrs:
                f.write(" ".join(str(int(x)) for x in nbrs))
            f.write("\n")


def adjlist_from_sparse(M: sp.spmatrix) -> List[List[int]]:
    """Convert a symmetric sparse adjacency matrix to an adjacency list."""
    Mc = M.tocsr()
    n = Mc.shape[0]
    adj = []
    for v in range(n):
        nbrs = Mc.indices[Mc.indptr[v]:Mc.indptr[v + 1]].tolist()
        adj.append(nbrs)
    return adj


# ---------------------------------------------------------------------------
# .mat files
# ---------------------------------------------------------------------------

def read_full_mat(path: str) -> Dict[str, sp.spmatrix]:
    """Read <name>.mat. Returns dict with 'network' and 'attributes' as
    sparse CSR matrices (and any other arrays present)."""
    raw = sio.loadmat(path)
    out: Dict[str, sp.spmatrix] = {}
    for key in ("network", "attributes", "labels"):
        if key in raw:
            arr = raw[key]
            out[key] = arr.tocsr() if sp.issparse(arr) else sp.csr_matrix(arr)
    return out


def read_split_mat(path: str) -> Dict[str, sp.spmatrix]:
    """Read <name>_<ratio>.mat with trainGraph and testGraph."""
    raw = sio.loadmat(path)
    out: Dict[str, sp.spmatrix] = {}
    for key in ("network", "attributes", "labels", "trainGraph", "testGraph"):
        if key in raw:
            arr = raw[key]
            out[key] = arr.tocsr() if sp.issparse(arr) else sp.csr_matrix(arr)
    return out


def write_mat(path: str, **arrays) -> None:
    """Save the given sparse arrays into a MATLAB v5 .mat file."""
    sio.savemat(path, {k: v for k, v in arrays.items()}, do_compression=True)


# ---------------------------------------------------------------------------
# Convenience: load a full dataset directory
# ---------------------------------------------------------------------------

def find_dataset_name(data_dir: str) -> str:
    """Infer the dataset name from the folder. The convention is that
    files are named `<name>.adjlist.<ratio>` and `<name>.mat`. We look at
    the first .mat file we find."""
    for fn in os.listdir(data_dir):
        if fn.endswith(".mat") and "_" not in fn:
            return fn[:-4]  # strip .mat
    # Fall back to folder basename
    return os.path.basename(os.path.normpath(data_dir))


def load_dataset(data_dir: str, ratio: float
                 ) -> Tuple[List[List[int]], List[List[int]],
                            sp.spmatrix, sp.spmatrix, sp.spmatrix, str]:
    """Load everything needed to run an experiment at the given ratio.

    Returns
    -------
    features : list of lists of ints
        features[v] = active attribute IDs for node v
    adj_train : list of lists of ints
        Adjacency list of the training graph at the given ratio
        (read from <name>.adjlist.<ratio>)
    network : scipy sparse (n, n)
        Full undirected adjacency matrix
    train_graph : scipy sparse (n, n)
        Train-edges adjacency matrix
    test_graph : scipy sparse (n, n)
        Test-edges adjacency matrix
    name : str
        Inferred dataset name
    """
    name = find_dataset_name(data_dir)
    ratio_str = str(ratio)

    features = read_features_txt(os.path.join(data_dir, "features.txt"))
    n_nodes = len(features)

    adj_train = read_adjlist(
        os.path.join(data_dir, f"{name}.adjlist.{ratio_str}"), n_nodes)

    split_path = os.path.join(data_dir, f"{name}_{ratio_str}.mat")
    full_path = os.path.join(data_dir, f"{name}.mat")
    if os.path.exists(split_path):
        m = read_split_mat(split_path)
        network = m["network"]
        train_graph = m["trainGraph"]
        test_graph = m["testGraph"]
    else:
        # Older format: only <name>.mat exists. We'd have to derive a
        # split, but the official data always ships with the split files.
        m = read_full_mat(full_path)
        network = m["network"]
        # Fallback: build trainGraph from the adjlist we just read
        rows, cols = [], []
        for u, nbrs in enumerate(adj_train):
            for v in nbrs:
                rows.append(u); cols.append(v)
        train_graph = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_nodes, n_nodes))
        test_graph = (network - train_graph).maximum(0)

    return features, adj_train, network, train_graph, test_graph, name
