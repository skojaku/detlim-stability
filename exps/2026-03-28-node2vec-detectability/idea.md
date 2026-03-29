# Ideas: Why does node2vec detect communities near the detectability limit?

**One-sentence goal**: Determine whether the detectability power of node2vec near the SBM detectability limit arises from (a) lack of orthogonality constraints, (b) the log-nonlinearity in its objective vs linear factorization of R, or (c) implicit SGD regularization (batch size effects), by comparing NMI of spectral clustering, BP, unconstrained matrix factorizations, and linearized node2vec variants on a sparse 2-community SBM.

---

## Background

node2vec achieves NMI on par with Belief Propagation near the SBM detectability limit, while spectral clustering (normalized Laplacian) fails. This is paradoxical because node2vec is theoretically equivalent to spectral clustering of the normalized Laplacian in the limit of infinite training and infinite dimensions.

**Key observation**: node2vec learns many near-parallel dimensions in 64-d space. Effective dimensionality is small. Each dimension encodes community + noise, but cosine similarity across near-parallel dimensions amplifies community signal while noise cancels.

**Open question**: What causes node2vec to learn near-parallel (not orthogonal) dimensions?

---

## Evaluation framing

**IMPORTANT**: Methods are expected to fail above the detectability limit (mu > mu*=0.553). Failure above mu* is not a bug — it is the theoretical prediction. The interesting comparison is how well methods detect communities **below** mu*, especially near mu* (mu ≈ 0.4–0.52). Node2vec supposedly keeps detecting near-limit while spectral fails.

## Hypotheses

- [✗] **(H2) Lack of orthogonality constraint**: RULED OUT. Unconstrained MSE factorization of M=(I-L_norm) does not outperform spectral. Free factorization of log(R) (free_logR) is *worse* than SVD of log(R). Removing orthogonality alone does not help.
- [~] **(H3) Log nonlinearity**: PARTIALLY CONFIRMED. log matters: svd_logR NMI=0.248 vs svd_R NMI=0.001 at mu=0.3. But single-hop log(R) is not what node2vec factorizes — node2vec uses the multi-hop NetMF matrix. The correct test is: SVD of NetMF matrix vs free factorization of NetMF matrix (iter-003 next).
- [ ] **(H1) Implicit SGD regularization**: OPEN. Node2vec (N=500) achieves NMI=0.635/0.326/0.104 at mu=0.3/0.4/0.5 vs free_logR=0.006/0.003/0.001. Large gap suggests SGD matters. But N=500 vs N=2000 confounds the comparison.
- [~] **(H4) Objective matrix matters**: CONFIRMED for log. But the right comparison is NetMF multi-hop matrix vs single-hop R, not just log vs linear. Node2VecMatrixFactorization (embcom) uses the NetMF matrix (window=10 hops), which is much better than single-hop R.

---

## Approaches to Try

**[DONE] Priority 1 — Test H2: unconstrained MSE factorization of (I - L_norm)**
- Result: No improvement over spectral. H2 ruled out for this matrix. (iter-001)

**[DONE] Priority 2+3 — Test H3: log vs linear × orthogonal vs free on R**
- Result: Log helps (svd_logR NMI=0.248 vs svd_R≈0 at mu=0.3). Free hurts (free_logR NMI=0.006). Single-hop R is wrong comparison for node2vec — use NetMF matrix. (iter-002)

**Priority 3 (revised) — Fair same-N comparison at N=2000** ← NEXT
- All methods at N=2000:
  - BP, spectral, n2vec_mf (embcom), node2vec (embcom), svd_logR, free_logR
- Focus mu: [0.3, 0.4, 0.45, 0.5, 0.52] (all below mu*=0.553; ~5 values saves ~30% vs 7-value sweep)
- 10 samples per mu
- node2vec runtime: ~5s × 50 runs ≈ 250s — acceptable
- **Key question**: Does node2vec (random walk SGNS) outperform n2vec_mf (SVD of same NetMF matrix) at matched N=2000? If yes → SGD/stochastic effect (H1).
- Also test: free factorization of the NetMF matrix (same target, no orthogonality) vs SVD

**Priority 4 — Test H1: batch size effect**
- Repeat best unconstrained factorization of NetMF matrix with: full-batch Adam, mini-batch (256, 1024)
- Does NMI improve with mini-batch? If yes, supports H1

**Priority 5 — Visualize embedding geometry**
- Compute pairwise cosine similarity of dimension vectors (columns of U) for each method
- Histogram of off-diagonal cosines — are they near-parallel (|cos| ≈ 1)?
- Effective dimensionality via participation ratio: PR = (sum_i lambda_i)^2 / sum_i(lambda_i^2) on Gram matrix U^T U
- Compare PR across methods: node2vec should have much lower PR than spectral

---

## Data / Setup

- **Network**: SBM, 2 communities, N=2000 (1000 per community)
- **Parameters**: cave=5.0 (average degree); mu swept: [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
  - Detectability limit at mu* = 1 - 1/sqrt(cave) ≈ 0.553
- **Samples**: 30 independent realizations per mu value
- **Embedding dim**: 64 (match node2vec default)
- **Baselines**:
  - Spectral: `embcom.LaplacianEigenMap().fit(A).transform(dim=64)` → k-means
  - Belief Propagation: `libs/BeliefPropagation/belief_propagation.detect(A, q=2)`
  - node2vec: `embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10)` → k-means
- **Metric**: NMI (normalized_mutual_info_score from sklearn)
- **Classification**: k-means (k=2) for embedding methods; centroid cosine similarity as secondary

---

## SBM Generation

**CORRECTED parameterization** (original code was wrong — gave mu*≈0.276, not 0.553):

```python
import igraph as ig
import scipy.sparse as sp
import numpy as np

N, n = 2000, 1000
cave = 5.0
mu = 0.5

# Correct: c_out = mu*cave, c_in = 2*cave - c_out
# This gives mu* = 1 - 1/sqrt(cave) = 0.553 for cave=5
c_out = mu * cave
c_in = 2 * cave - c_out
p_in = c_in / N
p_out = c_out / N

g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n], directed=False)
A = g.get_adjacency_sparse()
A = sp.csr_matrix(A, dtype=float)
membership = np.array([0]*n + [1]*n)
```

## R_ij matrix (symmetric PMI)

```python
d = np.array(A.sum(axis=1)).flatten()
m = d.sum() / 2
# R_ij = (2m) * A_ij / (d_i * d_j)  [symmetric; factorized form of SGNS PMI]
D_inv = sp.diags(1.0 / d)
R = (2 * m) * D_inv @ A @ D_inv
```

## Normalized adjacency

```python
# I - L_norm where L_norm = I - D^{-1/2} A D^{-1/2}
# So I - L_norm = D^{-1/2} A D^{-1/2}  [normalized adjacency]
D_inv_sqrt = sp.diags(1.0 / np.sqrt(d))
M_norm_adj = D_inv_sqrt @ A @ D_inv_sqrt  # = I - L_norm
```

---

## Existing Code to Reuse

- SBM generation: `exps/2026-03-13-where-wrong/workflow/scripts/generate_network.py`
- BP: `libs/BeliefPropagation/belief_propagation.py` — `detect(A, q=2)`
- Eigenvector computation: `scipy.sparse.linalg.eigsh(M, k=64, which="LA")`
- NMI: `sklearn.metrics.normalized_mutual_info_score`
- node2vec: `pecanpy` — see pecanpy docs for API

---

## References / Prior Work

- Levy & Goldberg (2014): word2vec implicitly factorizes PMI matrix
- Qiu et al. (2018): NetMF — node2vec factorizes closed-form random walk matrix
- Decelle et al. (2011): detectability limit for SBM
- Krzakala et al. (2013): spectral methods and the detectability limit (Bethe Hessian)
- Prior exps: `exps/2026-02-27-equivalence`, `exps/2026-03-13-where-wrong`, `exps/2026-03-20-eigvec-explore`

---

## Constraints

- Use: numpy, scipy, scikit-learn, igraph, matplotlib/seaborn
- **node2vec implementation**: `embcom` library at `libs/embcom_repo/libs/embcom/` (already installed as editable dep)
  - `import embcom` — available classes: `Node2Vec`, `LinearizedNode2Vec`, `Node2VecMatrixFactorization`, `LaplacianEigenMap`, `AdjacencySpectralEmbedding`
  - All follow: `model.fit(A); emb = model.transform(dim=64)`  where A is scipy sparse CSR
- NO graph-tool, NO pecanpy
- BP via `libs/BeliefPropagation/belief_propagation.detect(A, q=2)`
- Run scripts with: `~/.local/bin/uv run python script.py`
- Runtime: gradient descent factorizations should converge in <5 min per network
- For initial tests, use 10 samples per mu, then expand to 30 once methods work

## embcom API Quick Reference

```python
import embcom
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

# node2vec (random walk + word2vec)
model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10, p=1.0, q=1.0)
model.fit(A)  # A: scipy sparse CSR
emb = model.transform(dim=64)  # shape (N, 64)

# Linearized node2vec — factorizes R_ij (no log), no orthogonality constraint
model = embcom.LinearizedNode2Vec(window_length=10)
model.fit(A)
emb = model.transform(dim=64)

# Matrix factorization of log(R_ij) — closest analytic approximation to node2vec
model = embcom.Node2VecMatrixFactorization(window_length=10, num_blocks=500)
model.fit(A)
emb = model.transform(dim=64)

# Spectral (normalized Laplacian eigenvectors — with orthogonality)
model = embcom.LaplacianEigenMap()
model.fit(A)
emb = model.transform(dim=64)

# Classify after embedding
labels = KMeans(n_clusters=2, n_init=10).fit_predict(emb)
nmi = normalized_mutual_info_score(true_labels, labels)
```
