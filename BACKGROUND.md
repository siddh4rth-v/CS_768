# Background: What This Project Is About

Read this first if you want the conceptual picture before touching code.

---

## 1. The task: link prediction on attributed networks

A **network** (or **graph**) is a set of **nodes** connected by **edges**.
For example, on Twitter, each user is a node and a "follows" relationship
is an edge between two nodes.

An **attributed** network means each node also has a **feature vector** —
extra information besides who it's connected to. On Twitter, a user's
feature vector is the set of hashtags they've posted, encoded as a 0/1
vector over a 9,073-word hashtag vocabulary.

**Link prediction** asks: given the network as it is today, can we
predict which pairs of nodes are *most likely* to form an edge that
isn't there yet? Equivalently, given that we hide some edges and pretend
they don't exist, can we still rank them as more likely than random
non-edges? Applications:

- Friend recommendation on social networks
- Product recommendation in e-commerce
- Predicting protein-protein interactions in biology
- Knowledge graph completion ("Paris — capital of — ?")

---

## 2. How #GNN solves it (intuition)

A standard **Graph Neural Network (GNN)** assigns each node a learned
embedding vector (e.g. 200 floats). Embeddings start random; the network
trains them so that connected nodes end up close in vector space, and
unconnected ones end up far apart. Training takes time and a GPU.

The paper's idea — **#GNN** — is: **skip the training entirely**.
Instead of *learning* a mapping from a node's neighborhood to its
embedding, use a **random hash function** as the mapping. If you pick
the hash family carefully (specifically: MinHash), then nodes with
similar neighborhoods will probabilistically end up with similar hash
codes — exactly what we want.

The mechanism, simplified:

1. Each node starts with a "representation" = its set of active
   attribute IDs (e.g. the hashtags it uses).
2. **Iteration `t`:** For each node `v`, look at `v` and all its
   neighbors. Hash all their representations. Take the element with the
   smallest hash value. That's `v`'s new representation at iteration
   `t`. Do this **K times** with K independent random hash functions to
   get a length-K vector.
3. Repeat for `T` iterations. After iteration `t`, each node's
   representation reflects its `t`-hop neighborhood.

The output: each node has a length-K vector of integer "hash codes."

To predict if there's a link between node `u` and node `v`: compute the
**Hamming similarity** of their vectors — the fraction of positions
where the two vectors agree. Higher = more likely to be linked.

**Why this works:** for two nodes with similar neighborhoods, the
MinHash collision probability theoretically equals their Jaccard
similarity (paper §3.3). So Hamming similarity in the hash space
≈ Jaccard similarity in the neighborhood space. No learning required.

---

## 3. What is AUC?

AUC stands for **Area Under the (ROC) Curve**. It's the standard metric
for ranking-based tasks like link prediction.

**The setup.** We have:
- A list of **positive test edges**: real edges we hid from the model.
- A list of **negative test edges**: random pairs that aren't edges.

The model assigns a similarity score to every pair. Ideally the
positive pairs all score higher than the negatives.

**AUC's interpretation.** AUC is the probability that a randomly chosen
positive pair scores higher than a randomly chosen negative pair.

| AUC value | What it means |
|---|---|
| 1.00 | Perfect: every positive ranks above every negative |
| 0.97 | Excellent (about what #GNN achieves on Twitter) |
| 0.90 | Very good |
| 0.80 | Good |
| 0.70 | OK |
| 0.50 | Random — model has no skill |

**The official protocol** (from `lp_evaluation.m` in the repo) computes
AUC by sampling: draw 10,000 random non-edges and 10,000 random test
edges, count how often a test edge scores higher, and use the formula
`AUC = (#test_higher + 0.5 × #ties) / 10,000`. Our `run_experiment.py`
uses the **same** protocol so the comparison is fair.

---

## 4. The dataset

The paper uses 5 attributed networks. We'll use **one** of them.

| Name | Nodes | Edges | Attributes |
|---|---|---|---|
| **twitter** | 2,511 | 37,154 | 9,073 |
| facebook | 4,039 | 88,234 | 1,403 |
| blog | 5,196 | 171,743 | 8,189 |
| flickr | 7,564 | 239,365 | 12,047 |
| googleplus | 7,856 | 321,268 | 2,024 |

**We use Twitter** — smallest, fastest, paper reports highest AUC on it,
so the original-vs-modified contrast is clear.

### How the dataset is stored on disk (official format)

The official repo's `data/twitter/` folder contains:

| File | Contents |
|---|---|
| `features.txt` | 2,511 lines (one per node). Each line is space-separated **attribute IDs** for that node. |
| `twitter.adjlist.0.5` | 2,511 lines. Each line is the neighbors of that node in the **training graph** when 50% of edges are kept for training. |
| `twitter.adjlist.0.8` | Same but at 80% training ratio. (Also `0.6`, `0.7`, `0.9`.) |
| `twitter.mat` | MATLAB v5 file with the **full** network adjacency matrix (`network`), the attribute matrix (`attributes`), and node labels. |
| `twitter_0.8.mat` | MATLAB v5 file with `trainGraph` (80% edges) and `testGraph` (20% edges) for ratio 0.8. (Also one per other ratio.) |

The train/test split is **predefined** — you don't generate your own.

---

## 5. Why we subset the data

The paper's full datasets work fine on a server. On a CPU laptop, taking
the **first 500 nodes** of Twitter (instead of all 2,511) makes each
experiment finish in seconds instead of minutes.

We **don't change the data values** — we just keep the first N node IDs
and the edges that happen to be among those N nodes. Each kept node
still has its original feature vector. The attribute vocabulary is the
same. The trainGraph/testGraph split is preserved as the induced
subgraph of the original split.

The trade-off: AUC numbers on the subset won't match the paper's
numbers on the full data exactly. **That's fine** — what matters is the
*relative* comparison between the original algorithm and our
modifications, all run on the same subset.

---

## 6. The three modifications we test

(Full details in `WORKFLOW.md`. Here's the one-line version of each.)

| # | Name | Hypothesis |
|---|---|---|
| A | **Jumping-Knowledge concat** | Concatenating embeddings from all iterations gives the similarity access to short-range AND long-range proximity. Should help. |
| B | **Self-influence parameter α** | Making the self-vs-neighbor balance an explicit knob lets us tune for high-degree graphs. Likely helps. |
| C | **Random neighbor cap** | Capping each neighborhood at 30 random samples should reduce hub noise. **Likely fails** because MinHash already does uniform sampling — we're just biasing the Jaccard estimator. |

The expected outcome — A and B improve AUC, C makes it worse — is
exactly the structure your project asks for: things that work, AND a
modification that doesn't, with a theoretical explanation of why.

---

## 7. The three-way comparison in your report

For your project to be solid, you'll have **three** AUC numbers to
compare on the same subset:

1. **Official C++ binary** (the unmodified paper code, compiled from
   `hashgnn.cpp`) → "ground truth" baseline.
2. **My Python reimplementation of the original** → should match (1)
   within ~1% AUC. Confirms my code faithfully implements the paper.
3. **My reimplementation + each modification** → the actual experiment.

If (1) and (2) agree, then (3) vs (2) is a fair comparison and your
conclusions about each modification are valid.

The next file (`WORKFLOW.md`) tells you exactly how to compute all
three.
