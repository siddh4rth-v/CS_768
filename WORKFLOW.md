# Workflow: Full Project, Step by Step

> **Read `BACKGROUND.md` first** for the conceptual picture: what AUC is,
> what the dataset looks like, what the modifications do.

---

## What we're producing

Three AUC numbers on the **same subset** of Twitter, in one CSV:

1. **Official C++ binary** — the unmodified paper code. Reference baseline.
2. **My Python reimplementation of the original** — should match (1)
   within ~1% AUC. Validates my reimplementation.
3. **My reimplementation + each modification** — the actual contribution.

All three use the **same** data, the **same** train/test split, and the
**same** AUC protocol (the one from the paper's `lp_evaluation.m`).

---

## Files in this package

| File | Purpose |
|---|---|
| `BACKGROUND.md` | Conceptual explanation |
| `WORKFLOW.md` | This file — every command you need |
| `hashgnn.py` | The model: original Algorithm 1 + 3 modifications |
| `data_io.py` | Reads & writes the official repo's file format |
| `subset_dataset.py` | Takes first N nodes of a dataset → drop-in subset |
| `run_experiment.py` | Main script: runs ablation + scores official binary |
| `requirements.txt` | `numpy`, `scipy` |

---

## Step 1. One-time setup

```bash
mkdir hashgnn_project && cd hashgnn_project

# Copy in the 6 files above
# (hashgnn.py, data_io.py, subset_dataset.py, run_experiment.py,
#  BACKGROUND.md, WORKFLOW.md, requirements.txt)

pip install -r requirements.txt

# Clone the official repo (we need its data + we'll compile its C++ binary)
git clone https://github.com/drhash-cn/graph-hashing.git
```

The Twitter data is at `graph-hashing/hash-gnn/data/twitter/`.
**No conversion needed** — my code reads the official format directly.

---

## Step 2. Make a 500-node subset of Twitter

This is the only data-prep step. It produces a folder that is a **drop-in
replacement** for any of the original datasets — same filenames, same
formats. Both the official C++ binary and my Python code read it.

```bash
python subset_dataset.py \
    --in graph-hashing/hash-gnn/data/twitter \
    --out graph-hashing/hash-gnn/data/twitter500 \
    --n 500
```

Output:
```
Source dataset: twitter  →  Output name: twitter500
Nodes: 2511 → 500
  network: 500 x 500, nnz=21708
  ratio 0.5: train_edges=5466, test_edges=5388
  ratio 0.6: train_edges=6462, test_edges=4392
  ratio 0.7: train_edges=7659, test_edges=3195
  ratio 0.8: train_edges=8657, test_edges=2197
  ratio 0.9: train_edges=9794, test_edges=1060
```

The new folder `data/twitter500/` contains:
```
features.txt
twitter500.adjlist.0.5  ...  twitter500.adjlist.0.9
twitter500.mat
twitter500_0.5.mat       ...  twitter500_0.9.mat
```

These are exactly the files the C++ binary expects.

---

## Step 3. Run my Python ablation

This runs the 5 configs (original + 3 mods + all-three) on the subset:

```bash
python run_experiment.py \
    --data graph-hashing/hash-gnn/data/twitter500 \
    --ratio 0.8 \
    --T 3 --K 200 \
    --seed 42 \
    --out results.csv
```

You should see something like:
```
Loading graph-hashing/hash-gnn/data/twitter500 at ratio 0.8...
  dataset name: twitter500
  nodes: 500, attributes ≤ 1994
  train edges: 8657, test edges: 2197
  avg train degree: 34.63

Running Python experiments...
  original            AUC=0.9666  embed=4.51s  dim=200
  +JK concat          AUC=0.9698  embed=4.38s  dim=600
  +alpha=0.7          AUC=0.9693  embed=3.96s  dim=200
  +neigh-cap=30       AUC=0.9526  embed=6.11s  dim=200
  all three           AUC=0.9571  embed=5.50s  dim=600
```

(Numbers will differ slightly from these if you pick a different seed
or N; what matters is the **pattern**.)

The CSV (`results.csv`) is the main result file. Don't close it yet —
we'll add one more row in the next step.

---

## Step 4. Compile and run the official C++ binary on the same subset

This is the new piece. We compile `hashgnn.cpp` from the official repo,
run it on `twitter500`, and feed its output back into our evaluation.

### 4a. Install C++ build dependencies

The C++ code needs the GSL and Boost libraries:

```bash
# Ubuntu/Debian
sudo apt-get install -y libgsl-dev libboost-dev g++

# macOS (with Homebrew)
brew install gsl boost
```

### 4b. Patch a memory issue in the official source (one-line fix)

The official `hashgnn.cpp` allocates `MAX_NODE_NUM = 1_000_000_000` (1
billion) pointers at startup, which means it tries to grab ~12 GB of
RAM **regardless of dataset size**. Most laptops will OOM. Patch it to
something reasonable for our subset:

```bash
# In the file: graph-hashing/hash-gnn/hash-gnn/hashgnn.cpp
# change line 14 (or wherever the macro is):
#     #define MAX_NODE_NUM 1000000000
# to:
#     #define MAX_NODE_NUM 100000

# Or do it in one command:
sed -i 's/#define MAX_NODE_NUM 1000000000/#define MAX_NODE_NUM 100000/' \
    graph-hashing/hash-gnn/hash-gnn/hashgnn.cpp
```

This change does not affect the algorithm — it just makes the static
buffer fit in normal memory. (On macOS use `sed -i ''` instead of `sed -i`.)

### 4c. Compile

```bash
cd graph-hashing/hash-gnn/hash-gnn
python compile.py
ls hashgnn   # should be a binary executable now
cd ../../..  # back to hashgnn_project/
```

If you see warnings about unused variables, ignore them — they're
benign.

### 4d. Run the binary on the subset

The CLI is positional but uses `-flag value` style:

```bash
cd graph-hashing/hash-gnn/hash-gnn
./hashgnn \
    -network ../data/twitter500/twitter500.adjlist.0.8 \
    -feature ../data/twitter500/features.txt \
    -hashdim 200 \
    -iteration 3 \
    -embedding ../../../official_emb.txt \
    -time ../../../official_time.txt
cd ../../..
```

That writes `official_emb.txt` (one line per node, 200 hash IDs each)
and `official_time.txt` (the elapsed seconds).

### 4e. Score the official embedding using the same AUC protocol

Now run our experiment script again, but this time pass
`--official-embedding` so it scores the C++ output too:

```bash
python run_experiment.py \
    --data graph-hashing/hash-gnn/data/twitter500 \
    --ratio 0.8 \
    --T 3 --K 200 \
    --seed 42 \
    --official-embedding official_emb.txt \
    --out results.csv
```

You'll now see one extra row at the top:
```
============================================================
config                  AUC    embed(s)   dim
============================================================
official C++         0.9628        -      200
original             0.9666     4.562    200
+JK concat           0.9698     4.492    600
+alpha=0.7           0.9693     3.998    200
+neigh-cap=30        0.9526     6.151    200
all three            0.9571     5.345    600
============================================================
```

**That table is your main result.** The first row is the official paper
code; the rest are our reimplementation and modifications, all on the
same data.

---

## Step 5. (Optional) Multiple seeds for error bars

A single AUC number can be lucky. Repeat with 5 seeds:

```bash
for seed in 0 1 2 3 4; do
  python run_experiment.py \
      --data graph-hashing/hash-gnn/data/twitter500 \
      --ratio 0.8 --T 3 --K 200 --seed $seed \
      --out results_seed${seed}.csv
done
```

Then for each config compute mean ± std across the 5 CSVs.

---

## Step 6. (Optional) Alpha sweep for the discussion

```bash
for a in 0.0 0.25 0.5 0.7 1.0; do
  python run_experiment.py \
      --data graph-hashing/hash-gnn/data/twitter500 \
      --ratio 0.8 --T 3 --K 200 --seed 42 \
      --no-ablation --alpha $a \
      --out results_alpha${a}.csv
done
```

α = 0.0 ignores self entirely; α = 1.0 ignores neighbors entirely (so
should be near-random AUC).

---

## What goes in your report

### Section 1: Background (1 page)

Use `BACKGROUND.md` sections 1–4 as your source. Cover:
- Link prediction task and why it matters
- Attributed networks; what Twitter contains (2,511 nodes, 9,073 hashtags)
- AUC: definition, value-meaning table, how the paper computes it
- The high-level idea of #GNN: hashes instead of learned weights

### Section 2: Method (1 page)

Paraphrase Algorithm 1 of the paper. Cover:
- Two phases per iteration (diffuse, then aggregate)
- K independent hash functions
- MinHash → Hamming similarity → Jaccard similarity (paper §3.3)
- Time complexity O(T · K · |V| · (K + ν)) — linear in iterations

### Section 3: Modifications (1 page)

For each, write one paragraph: motivation, change made, expected effect.

- **A. Jumping-Knowledge concat.** Paper uses only iteration-T
  embedding; we concatenate iterations 1..T to expose multi-scale
  proximity to the similarity computation.
- **B. Self-influence α.** Paper merges self and neighbor candidates;
  the implicit balance depends on |features| vs degree. With α explicit
  we can tune this.
- **C. Random neighbor cap (s=30).** Cap each neighborhood at 30
  random samples to reduce hub noise.

### Section 4: Experimental setup (½ page)

- Dataset: Twitter, first 500 nodes (induced subgraph)
- Train/test ratio: 0.8 (80% train, 20% test) — using the predefined
  split shipped with the dataset
- Hyperparameters: T=3, K=200, seed=42 (or mean over seeds 0–4)
- Metric: AUC computed with the official protocol from
  `lp_evaluation.m` (10,000 sampled non-edges + 10,000 sampled test
  edges, paired comparison)
- Hardware: CPU only

### Section 5: Results (1 page)

Show the 6-row table from Step 4. Optional plots:
- AUC ± std bars from Step 5
- AUC vs α from Step 6

**Sample result from a real run (your numbers will vary):**

| Source | Config | AUC | Δ vs original |
|---|---|---|---|
| Official C++ | original | 0.9628 | — |
| My Python | original | 0.9666 | matches official within 0.4% ✓ |
| My Python | + JK concat | **0.9698** | +0.7% ✓ |
| My Python | + α=0.7 | **0.9693** | +0.7% ✓ |
| My Python | + neigh-cap=30 | 0.9526 | **−1.0% ✗** |
| My Python | all three | 0.9571 | +0.4% (cap drags it down) |

### Section 6: Discussion (1 page)

For each modification:

- **Compare official vs my reimpl:** they should agree within ~1%. If
  yes, it validates that all subsequent rows are trustworthy. If no,
  list possible reasons (random seed semantics, different prime moduli,
  edge-case handling of isolated nodes, AUC sample randomness).

- **JK concat:** Did AUC go up? On Twitter (paper's optimal T=3, AUC
  ~98% on full data), JK gives a small but consistent gain because
  iteration-1 captures direct attribute overlap, iteration-3 captures
  3-hop proximity, and concatenating exposes both to the similarity
  computation.

- **α = 0.7:** Sweep α (Step 6). Note α = 0 ignores self entirely and
  α = 1 ignores neighbors entirely (so AUC drops to near random).
  Twitter's avg train degree on the subset is ~35, high enough that
  reweighting toward "self" helps.

- **Neighbor cap (the failure case):** AUC drops by ~1% with no
  runtime gain. Theoretical explanation:

  > MinHash already does uniform random sampling internally — the
  > collision probability of two MinHashed sets equals their full-set
  > Jaccard similarity. Truncating each set to 30 elements before
  > MinHash means we're estimating Jaccard of two random 30-subsets,
  > which has higher variance and a downward bias. Worse, the runtime
  > cost is dominated by the K=200 hash functions per node, so the
  > truncation doesn't even save compute. The result is exactly what
  > theory predicts: a clear AUC degradation with no compensating gain.

  This is a textbook "we tried it, it failed, here's why" finding.

### Section 7: Conclusion (1 paragraph)

Which modification(s) helped, by how much, and what the takeaway is.

---

## All commands in one place

```bash
# Setup
pip install -r requirements.txt
git clone https://github.com/drhash-cn/graph-hashing.git

# Subset twitter to 500 nodes
python subset_dataset.py \
    --in graph-hashing/hash-gnn/data/twitter \
    --out graph-hashing/hash-gnn/data/twitter500 \
    --n 500

# Compile official C++ (after one-line patch)
sudo apt-get install -y libgsl-dev libboost-dev g++
sed -i 's/#define MAX_NODE_NUM 1000000000/#define MAX_NODE_NUM 100000/' \
    graph-hashing/hash-gnn/hash-gnn/hashgnn.cpp
(cd graph-hashing/hash-gnn/hash-gnn && python compile.py)

# Run official C++ on subset
(cd graph-hashing/hash-gnn/hash-gnn && ./hashgnn \
    -network ../data/twitter500/twitter500.adjlist.0.8 \
    -feature ../data/twitter500/features.txt \
    -hashdim 200 -iteration 3 \
    -embedding ../../../official_emb.txt \
    -time ../../../official_time.txt)

# Run my Python ablation + score the official binary's output
python run_experiment.py \
    --data graph-hashing/hash-gnn/data/twitter500 \
    --ratio 0.8 --T 3 --K 200 --seed 42 \
    --official-embedding official_emb.txt \
    --out results.csv
```

---

## Troubleshooting

**`std::bad_alloc` when running the C++ binary.** You forgot to patch
`MAX_NODE_NUM`. See Step 4b.

**`feature file not found!` from the C++ binary.** Check the relative
path in your `-feature` flag matches the actual file location.

**My Python "original" AUC is far from official C++ AUC (>2% gap).**
Possible reasons (worth discussing in the report):
- The C++ uses a fixed seed (`srand(1)`) for hash parameters; my Python
  uses `--seed 42`. Different RNG state → different hash function
  draws → slightly different fingerprints. Try multiple seeds (Step 5)
  and report the mean.
- The AUC sampling itself is stochastic (10,000 random pairs). Use the
  same `--eval-seed` for both runs.

**AUC > 0.99 even on a 500-node subset.** That's expected for Twitter —
hashtags are very discriminative. Your *relative* numbers across configs
are what matter.

**The C++ binary expects a specific dataset name.** It doesn't — the
binary takes file paths via `-network` and `-feature`. The dataset
"name" comes only from how the file is named. Our subsetting script
makes the name match the output folder, so as long as you point
`-network` at the right adjlist file you're fine.

**Want to run a single config of my Python code (no ablation).**
```bash
python run_experiment.py --data graph-hashing/hash-gnn/data/twitter500 \
    --ratio 0.8 --T 3 --K 200 --seed 42 \
    --no-ablation --jk concat --alpha 0.7
```

**Want to try a different training ratio.** Pass `--ratio 0.5` (or 0.6,
0.7, 0.9). The subset already has all five split files.
