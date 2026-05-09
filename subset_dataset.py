"""
subset_dataset.py
=================

Take the first N nodes of a dataset and write a complete subset in the
official drhash-cn/graph-hashing data format. After running this, the
subset is a *drop-in* replacement for any of the original datasets:

  * Both the official C++ binary and my Python code can read it.
  * The official MATLAB evaluation scripts can read it.
  * The train/test split is preserved (induced from the original split).

Usage:
    python subset_dataset.py \\
        --in graph-hashing/hash-gnn/data/twitter \\
        --out graph-hashing/hash-gnn/data/twitter500 \\
        --n 500

After this you'll have:
    twitter500/
        features.txt
        twitter500.adjlist.0.5
        twitter500.adjlist.0.6
        twitter500.adjlist.0.7
        twitter500.adjlist.0.8
        twitter500.adjlist.0.9
        twitter500.mat
        twitter500_0.5.mat   ... 0.9.mat

The output dataset name (`twitter500` here) comes from the --out folder name.

We do NOT renumber attribute IDs — the attribute vocabulary is unchanged
(keeps the data values identical to the source). We just keep fewer rows.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp

import data_io as io


RATIOS = ["0.5", "0.6", "0.7", "0.8", "0.9"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True,
                   help="Source dataset directory, e.g. data/twitter")
    p.add_argument("--out", dest="out_dir", required=True,
                   help="Destination directory; basename becomes the new "
                        "dataset name, e.g. data/twitter500")
    p.add_argument("--n", type=int, required=True,
                   help="Keep the first N nodes (0..N-1)")
    p.add_argument("--ratios", nargs="+", default=RATIOS,
                   help=f"Training ratios to subset; default {RATIOS}")
    args = p.parse_args()

    src = args.in_dir
    dst = args.out_dir
    new_name = os.path.basename(os.path.normpath(dst))
    os.makedirs(dst, exist_ok=True)

    src_name = io.find_dataset_name(src)
    print(f"Source dataset: {src_name}  →  Output name: {new_name}")

    # ---- features.txt: take first N lines, attributes unchanged ----
    features = io.read_features_txt(os.path.join(src, "features.txt"))
    n_full = len(features)
    n_keep = min(args.n, n_full)
    print(f"Nodes: {n_full} → {n_keep}")
    sub_features = features[:n_keep]
    io.write_features_txt(os.path.join(dst, "features.txt"), sub_features)

    # ---- <name>.mat: induced subgraph on first N nodes ----
    full = io.read_full_mat(os.path.join(src, f"{src_name}.mat"))
    network = full["network"][:n_keep, :n_keep]
    attributes = full["attributes"][:n_keep, :]
    out_full = {"network": network, "attributes": attributes}
    if "labels" in full:
        out_full["labels"] = full["labels"][:n_keep, :]
    io.write_mat(os.path.join(dst, f"{new_name}.mat"), **out_full)
    print(f"  network: {network.shape[0]} x {network.shape[1]}, "
          f"nnz={int(network.nnz)}")

    # ---- per-ratio split files ----
    for r in args.ratios:
        # Try the existing split mat
        split_src = os.path.join(src, f"{src_name}_{r}.mat")
        adjlist_src = os.path.join(src, f"{src_name}.adjlist.{r}")
        if not os.path.exists(split_src):
            print(f"  (skip ratio {r}: no split file found)")
            continue

        m = io.read_split_mat(split_src)
        train_graph = m["trainGraph"][:n_keep, :n_keep]
        test_graph = m["testGraph"][:n_keep, :n_keep]
        # Keep network/attributes consistent with the .mat schema
        out = {
            "network": network,
            "attributes": attributes,
            "trainGraph": train_graph,
            "testGraph": test_graph,
        }
        if "labels" in m:
            out["labels"] = m["labels"][:n_keep, :]

        io.write_mat(os.path.join(dst, f"{new_name}_{r}.mat"), **out)

        # Adjlist file: write from the trainGraph submatrix
        sub_adj = io.adjlist_from_sparse(train_graph)
        io.write_adjlist(
            os.path.join(dst, f"{new_name}.adjlist.{r}"), sub_adj)

        print(f"  ratio {r}: train_edges={int(train_graph.nnz)//2}, "
              f"test_edges={int(test_graph.nnz)//2}")

    print(f"\nDone. Subset written to: {dst}")
    print(f"To run my Python code on it:")
    print(f"   python run_experiment.py --data {dst} --ratio 0.8")
    print(f"To run the official C++ code on it (after compiling):")
    print(f"   ./hashgnn -network {dst}/{new_name}.adjlist.0.8 \\")
    print(f"       -feature {dst}/features.txt -hashdim 200 -iteration 3 \\")
    print(f"       -embedding emb.txt -time t.txt")


if __name__ == "__main__":
    main()
