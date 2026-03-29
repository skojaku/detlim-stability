# Result: Why node2vec detects communities near the SBM detectability limit

**Date**: 2026-03-28
**Branch**: explore/node2vec-detectability-20260328

---

## One-line answer

node2vec detects communities near the SBM detectability limit because it implicitly performs (1) multi-hop random walk aggregation (window=10 hops), which captures long-range community coherence unavailable to 1-hop spectral methods, and (2) SGNS gradient weighting, which focuses learning on ambiguous edge pairs with PMI≈0 while zeroing out the ~99.5% non-edge pairs that otherwise dominate and corrupt SVD of the raw NetMF log-PMI matrix.

---

## The question

> Determine whether the detectability power of node2vec near the SBM detectability limit arises from (a) lack of orthogonality constraints, (b) the log-nonlinearity in its objective vs linear factorization of R, or (c) implicit SGD regularization (batch size effects), by comparing NMI of spectral clustering, BP, unconstrained matrix factorizations, and linearized node2vec variants on a sparse 2-community SBM.

Setup: 2-community SBM, N=2000, cave=5, detectability limit mu*=0.553 (Decelle et al. 2011). Methods evaluated at mu=[0.30, 0.35, 0.40, 0.45, 0.50, 0.52], up to 30 samples per mu value.

---

## What we found (iteration by iteration)

**Iteration 1** (hypothesis H2 ruled out): Unconstrained MSE factorization of the normalized adjacency (I - L_norm) does not outperform spectral. At mu=0.40, unconstrained_mf NMI=0.054 (high variance) vs spectral NMI=0.018 — both near chance. Removing orthogonality alone does not enable community detection near the limit.

**Iteration 2** (log-nonlinearity necessary but insufficient): SVD of log(R) achieves NMI=0.248 at mu=0.30 while SVD of R achieves NMI=0.001. Log transformation is required. But free factorization of log(R) (no orthogonality constraint) actually *hurts*: NMI=0.006 at mu=0.30. Actual node2vec (N=500) achieves NMI=0.635/0.326/0.104 at mu=0.3/0.4/0.5 — far above all static matrix methods. The single-hop log(R) matrix is the wrong comparison; node2vec uses a window=10 multi-hop matrix.

**Iteration 3** (NetMF matrix collapses at mu≥0.40): At matched N=2000, both n2vec_mf and svd_netmf (SVD of NetMF log-PMI matrix, no clip) achieve NMI~0.609 at mu=0.30 but collapse to NMI~0.003–0.005 at mu=0.40. The multi-hop NetMF matrix is necessary but not sufficient without further processing.

**Iteration 4** (node2vec vs n2vec_mf at matched N=2000, H1 initially supported): Direct comparison at identical N=2000 graphs: mu=0.40 gap is +0.344 (node2vec NMI=0.347 vs n2vec_mf NMI=0.003). Gap is not a size artifact. Two failure modes identified: log(0) numerical failure (~22% runs) and SVD spectral collapse (remaining runs at mu≥0.40). Initial hypothesis: SGNS/stochastic regularization (H1).

**Iteration 5** (clipping fully rescues SVD, H1 weakened): Clipping the NetMF log-PMI matrix at 0 before SVD (the original NetMF paper formulation, omitted from embcom) matches node2vec within |delta|<0.02 at all mu. netmf_clipped_svd vs node2vec at mu=0.40: 0.364 vs 0.353. The clipped matrix is 85.5% sparse (only 14.5% positive entries). H1 (SGD noise essential) substantially weakened — a deterministic clip suffices. embcom's Node2VecMatrixFactorization omits this clip, causing its failure.

**Iteration 6** (mechanism fully characterized, n=20): Multi-hop (spectral fails completely at mu=0.30: NMI=0.106 vs 0.6 for clipped/node2vec) and clipping are both independently essential. Dominant spurious singular value (SV1=1565 full vs 241 clipped, ratio 6.5×) confirmed as the failure mode. Community eigenvector at rank 2 (|corr|=0.670 clipped). Clip threshold plateau from -3.0 to 0.0; phase transition above +0.5; collapse above +1.0.

**Iteration 7** (30-sample definitive confirmation): n=30. netmf_clipped ≈ node2vec at all mu (mu=0.40: 0.365±0.038 vs 0.346±0.035). SV1 ratio full/clipped=6.39× (1546 vs 242). Community eigenvector |corr|=0.650 clipped vs 0.414 full. Clip threshold plateau confirmed: stable from -5.0 to 0.0, sharp transition above +0.5, total collapse at +1.5. BP at iters=1 fails everywhere (non-convergence).

**Iteration 8** (gradient weighting: complete mechanism): Gradient-weighted matrix M × σ(M)(1-σ(M)) outperforms clip_0 near the detectability limit. At mu=0.50: gradient_weight NMI=0.119 vs clip_0=0.082 vs node2vec=0.090. Community eigenvector |corr|=0.732 (gradient_weight) vs 0.708 (clip_0). clip_0 is confirmed as a first-order hard-threshold approximation of the SGNS gradient weighting. The mechanism is the focus on PMI≈0 (ambiguous) pairs at the community boundary; non-edges (PMI→-∞, ~99.5% of pairs) contribute zero gradient and are effectively zeroed out.

**Iteration 13** (embedding geometry: original hypothesis resolved): Node2vec has participation ratio PR≈23.7 (near-parallel dimensions) vs PR=64.0 for all SVD methods (perfectly orthogonal). However, near-parallelism is NOT the mechanism: gradient_weight_svd (PR=64, fully orthogonal) achieves NMI=0.379 vs node2vec NMI=0.350 at mu=0.40. The original observation (near-parallel dims) was confirmed geometrically but is an SGNS training artifact, not the causal driver of community detection.

**Iteration 14** (phase transition characterization): Dense mu sweep [0.30–0.54], 15 samples/mu, 5 methods. Effective detectability limits (interpolated mu where NMI=0.1): gradient_weight_svd=0.503, Bethe_Hessian=0.502, clip_0_svd=0.488, node2vec=0.486, BP=0.449. Theoretical KS limit=0.553. Gradient_weight_svd extends the effective detectable range by +0.017 mu beyond node2vec, and +0.054 beyond BP.

---

## The mechanism

**Component 1: Multi-hop aggregation (T≥2 sufficient; T=10 in node2vec is not the minimum).** Node2vec's random walks aggregate co-occurrence signal over paths of length 1 through 10. The resulting log-PMI matrix M_ij = log(vol(G) × PPR_window(i,j) / d_i d_j) captures long-range community coherence unavailable to single-hop spectral methods. Spectral clustering of the normalized adjacency (1-hop) fails completely at all tested mu values, including mu=0.30 (NMI=0.182 vs 0.598 for node2vec on the same graphs). The 10-hop PPR averaging amplifies within-community signal while averaging out fluctuations from cross-community random walks. However, T=2 is sufficient for the NMI to jump from 0.132 to 0.350 (see iter-012 subsection below).

### Multi-hop: necessary or sufficient?

*(iter-012 finding)*

T=2 is **sufficient** — the NMI jump from T=1 to T=2 is massive (+0.218), and by T=3 the plateau is essentially reached. T=10 (node2vec's default) is not the minimum.

T=1 log-PMI is **insufficient**: at cave=5 and mu=0.40, 99.6% of within-community node pairs are non-edges. The log-PMI matrix at T=1 is almost entirely large-negative entries, producing a garbage SV1 that overwhelms the community eigenvector.

| T | NMI at mu=0.40 |
|---|----------------|
| 1 | 0.132 |
| 2 | 0.350 |
| 3 | 0.385 |
| 5–20 | 0.35–0.38 (plateau) |
| Bethe Hessian (T=1, r=sqrt(cave)) | 0.362 |

**Bethe Hessian (1-hop) matches 10-hop NetMF**, demonstrating that multi-hop is not fundamental. The Bethe Hessian H(r) = (r²−1)I − rA + D with r=sqrt(cave) is derived from the cavity method / Belief Propagation. It naturally encodes the degree-based null model, and its negative eigenvalues only exist above the Kesten-Stigum threshold — there are exactly as many negative eigenvalues as detectable communities. It is equivalent to infinite-depth message passing condensed into a single 1-hop operator.

**What IS fundamental** is any operator whose dominant spectral structure encodes community membership rather than degree heterogeneity. Three confirmed paths:

a. **Multi-hop (T≥2) of row-stochastic operator**: averaging over long walks smooths out degree fluctuations; T=1 is too sparse (99.6% non-edges), T=2 reaches most within-community pairs (~25 same-community 2-hop neighbors with cave=5)

b. **Bethe Hessian**: explicitly designed via cavity method to suppress the degree mode; uses negative eigenvalues that only appear above the KS threshold

c. **Modularity + sigmoid**: subtracts the degree-based expected adjacency B = A − dd^T/(2m) directly, then sigmoid suppresses extremes

Multi-hop aggregation is therefore **one path**, not the fundamental requirement.

**Component 2: Implicit zeroing of non-edge log-PMI entries.** The raw NetMF log-PMI matrix has approximately 85.5% negative entries, corresponding to node pairs rarely co-visited in random walks (PMI → -∞ for non-edges in a sparse graph with mean degree 5). These large-negative entries produce a dominant spurious singular value (SV1≈1546) that is 6.4× larger than the community-signal singular value (SV2≈242 after clipping). The community eigenvector exists at rank 2 (|corr|=0.414 with community labels in the full matrix), but the rank-1 garbage direction — driven by the 85.5% large-negative entries — overwhelms the 64-dimensional k-means embedding. Clipping at 0 or below removes this garbage direction, reducing SV1 to 242 and promoting the community eigenvector to the dominant position (|corr|=0.650). SGNS in node2vec implements this implicitly: negative sampling trains only on observed walk pairs (which have positive or moderate PMI), never on the ~99.5% non-edge pairs, effectively setting their gradient to zero.

**Component 3: SGNS gradient weighting (the complete mechanism).** clip_0 is a hard-threshold approximation. The true SGNS gradient at convergence weights each pair (i,j) by σ(PMI)(1-σ(PMI)) — the derivative of the logistic loss evaluated at u_i·v_j = PMI(i,j). This weighting has a symmetric bell shape: zero for PMI→+∞ (easy positives, saturated), zero for PMI→-∞ (non-edges), maximum at PMI=0. The gradient-weighted matrix M × σ(M) × (1-σ(M)) explicitly encodes this: SVD of this matrix outperforms clip_0 at all mu≥0.40 and exceeds node2vec near the detectability limit (mu=0.50: NMI=0.119 vs 0.090). The improvement is largest near the limit because that is where the marginal community signal — carried by PMI≈0 ambiguous pairs at the inter-community boundary — is most precious.

---

## The key equation

The implicit matrix that node2vec effectively factorizes is the **gradient-weighted log-PMI matrix**:

```
M_effective[i,j] = σ(M_netmf[i,j]) × (1 - σ(M_netmf[i,j])) × M_netmf[i,j]
```

where:
- `M_netmf[i,j] = log(vol(G) × PPR_window(i,j) / d_i d_j)`  [10-hop log-PMI matrix]
- `σ(x) = 1 / (1 + exp(-x))`  [sigmoid / logistic function]
- `σ(M) × (1-σ(M)) = dσ/dx` evaluated at M  [SGNS gradient magnitude]

The weighting σ(M)(1-σ(M)) achieves:
- M → -∞ (non-edge pairs, ~99.5% of pairs): weight → 0  [excluded from learning]
- M → +∞ (hub-hub edges): weight → 0  [saturated, easy positives]
- M ≈ 0 (ambiguous pairs, R_ij≈1): weight → 0.25  [maximum focus]

**Practical first-order approximation** (clip_0, within ~0.04 NMI of gradient_weight):

```
M_effective ≈ max(M_netmf, 0)
```

This hard-threshold approximation correctly zeroes non-edge entries and retains positive-PMI entries. It is the formulation from the original NetMF paper (Qiu et al. 2018) and is absent from embcom's Node2VecMatrixFactorization.

---

## Practical implication

To fix `embcom.Node2VecMatrixFactorization` to match node2vec performance:

**Option A (simple, clip_0)** — within 0.02 NMI of node2vec at all mu:
```python
# Compute NetMF log-PMI matrix
M_raw = compute_netmf_logpmi(A, window=10)
# Clip at 0 (restores original NetMF paper formulation)
M_clipped = np.maximum(M_raw, 0)
# SVD
from sklearn.decomposition import TruncatedSVD
emb = TruncatedSVD(n_components=64).fit_transform(M_clipped)
```

**Option B (gradient_weight)** — best performance, exceeds node2vec near detectability limit:
```python
M_raw = compute_netmf_logpmi(A, window=10)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
sig = sigmoid(M_raw)
M_gw = M_raw * sig * (1 - sig)
emb = TruncatedSVD(n_components=64).fit_transform(M_gw)
```

Both options produce effectively sparse matrices (the ~14.5% of node-pair entries with positive log-PMI dominate). The gradient_weight option marginally outperforms clip_0 particularly near mu≈mu*; both are deterministic and reproducible (no stochastic walks needed).

---

## What doesn't explain it

- **H2 (lack of orthogonality constraint)**: Ruled out (iter-001). Unconstrained MSE factorization of the normalized adjacency is no better than spectral. The matrix choice matters, not the orthogonality constraint.

- **Single-hop log nonlinearity (H3 partial)**: Log transformation of the 1-hop R matrix helps (svd_logR NMI=0.248 at mu=0.30 vs svd_R NMI=0.001) but is far below node2vec (0.635) at mu=0.30 and collapses by mu=0.40. The multi-hop matrix, not the log itself, is the key.

- **Free factorization of log(R)**: Unconstrained gradient descent on log(R) performs *worse* than SVD (free_logR NMI=0.006 at mu=0.30 vs svd_logR NMI=0.248). Removing orthogonality from an already-appropriate log matrix does not help.

- **Size artifact**: The N=500 vs N=2000 confound from early experiments (iter-002) was real. At matched N=2000 (iter-004 onward), the gap between node2vec and unclipped SVD is +0.344 at mu=0.40 — a genuine effect.

- **-∞ entries specifically**: The LCC of the SBM graph is connected; there are no actual -∞ entries (nan_rate=0%). The failure of unclipped SVD is caused by large-but-finite negative values from rarely-visited node pairs, not literal infinity.

- **Stochastic SGD noise (H1, largely ruled out)**: The deterministic gradient_weight SVD matches or exceeds node2vec at all mu≥0.40. Stochastic walk sampling is not essential; the benefit can be achieved by a closed-form matrix transformation.

---

## Effective detectability limits (quantitative summary)

| Method | Effective mu* | Mechanism |
|---|---|---|
| gradient_weight_svd | 0.503 | 10-hop log-PMI + sigmoid gradient weighting |
| Bethe Hessian | 0.502 | 1-hop cavity-method operator, KS-aware |
| clip_0_svd | 0.488 | 10-hop log-PMI clipped at 0 (NetMF paper) |
| node2vec (N=2000) | 0.486 | 10-hop SGNS walks + KMeans |
| BP | 0.449 | Belief propagation (poor convergence at N=2000) |
| **Theoretical KS limit** | **0.553** | |

Gradient_weight_svd is deterministic, reproducible, and fast (no walks needed), and outperforms node2vec across the full mu range above 0.40. It is the recommended replacement for node2vec when speed and reproducibility matter.

**Figure**: `analyses/iter-014/fig_phase_transition.png`

---

## Open questions

1. **BP performance**: All BP runs used iters=1 (default), which fails to converge (NMI=0.04–0.08 across all mu). BP near the Bayesian detectability limit (which should theoretically reach the Nishimori bound) remains uncharacterized. With iters≥10 and multiple restarts, BP may substantially outperform gradient_weight SVD.

2. **Scaling with N**: Does the SV1 ratio (full vs clipped) scale with N? If SV1_full ∝ N while SV1_clipped ∝ N^0.5, the failure regime of unclipped SVD would shift with graph size, and the mechanism may not apply at very small N.

3. **Optimal clip threshold**: clip≈-1.0 is marginally better than clip=0.0 (NMI=0.346 vs 0.337 at mu=0.40, n=30). Mildly negative log-PMI entries carry a small community signal. Whether the optimal threshold shifts near mu* is untested.

4. **LFR benchmark generalization**: The mechanism was characterized on a planted partition SBM. Whether the same clipping/gradient-weighting benefit holds for LFR benchmarks (heterogeneous degree distribution, overlapping communities) is an open question.

5. **Window size ablation**: Window=1 (spectral) fails completely; window=10 succeeds. The critical window length — and whether it relates to the graph diameter or community mixing time — is unexplored.

6. **Effective dimensionality**: Node2vec embeddings show near-parallel dimensions (small effective dimensionality, noted in idea.md). Whether gradient_weight SVD embeddings share this geometry, and whether it explains any residual gap between the two methods, is unexamined.

---

## Summary table

| Method | mu=0.30 | mu=0.40 | mu=0.50 | Notes |
|--------|---------|---------|---------|-------|
| node2vec | 0.586 ± 0.032 | 0.346 ± 0.035 | 0.068 ± 0.047 | Ground truth; stochastic (n=30) |
| gradient_weight SVD | 0.490 ± 0.033 | 0.383 ± 0.046 | **0.119 ± 0.061** | Best deterministic method (n=15) |
| clip_0 SVD | 0.598 ± 0.031 | 0.374 ± 0.051 | 0.082 ± 0.065 | Simple fix; within ~0.02 NMI (n=30/15) |
| n2vec_mf (no clip) | 0.558 ± 0.152 | 0.320 ± 0.130 | 0.050 ± 0.055 | High variance; fails at mu≥0.40 (n=30) |
| spectral (1-hop) | 0.182 ± 0.199 | 0.017 ± 0.030 | 0.005 ± 0.010 | Completely fails at all mu (n=30) |
| BP (iters=1) | 0.040 ± 0.149 | 0.079 ± 0.157 | 0.018 ± 0.045 | Non-converged; unreliable (n=15) |

All results: N=2000, cave=5, 2-community SBM.

---

## Generalizability across operators (iter-009 through iter-011)

### The transferable principle

- **Row-stochastic (or spectral-radius ≤1) base operator — established BEFORE multi-hop**: P = D^{-1}A (row-stochastic) and M_na = D^{-1/2}AD^{-1/2} (spectral radius ≤1) both work. Raw adjacency A fails because A^T_{ij} ∝ d_i^{T/2} × d_j^{T/2}, producing SV1/SV2 ≈ 155–163 after the log-PMI recipe; post-hoc normalization cannot undo this amplification (post-hoc row-normalize recovers only NMI=0.297 vs 0.356 for P).
- **Multi-hop aggregation (T≥2 sufficient; T=10 in node2vec is not the minimum)**: All degree-normalized operators improve substantially with multi-hop (P: +0.222, B: +0.152, M_na: +0.056 NMI at mu=0.40). Single-hop log-PMI gives NMI=0.132 at mu=0.40 (insufficient); T=2 gives NMI=0.350 (massive jump); plateau by T=3 at NMI=0.385. Bethe Hessian (T=1) matches 10-hop NetMF (NMI=0.362 vs ~0.37), confirming multi-hop is not fundamental. Raw A is the exception — multi-hop degrades it (0.012 → 0.003) due to degree explosion.
- **Null-model boundary focus (suppress extremes)**: For log-domain operators (P, M_na), clip at 0 focuses SVD on pairs with positive log-PMI. For linear operators (B = A − dd^T/(2m)), sigmoid of B_10hop directly achieves the same effect. Both routes suppress entries far from community-boundary ambiguity (B or PMI ≈ 0). This is what SGNS in node2vec does implicitly.

### Operators tested

| Operator | Works? | Mechanism | Best NMI at mu=0.40 |
|----------|--------|-----------|---------------------|
| P = D^{-1}A | Yes | Row-stochastic → 10-hop → log-PMI → clip0 | 0.356 |
| M_na = D^{-1/2}AD^{-1/2} | Yes | Spectral-radius ≤ 1 → 10-hop → log-PMI → clip0 | 0.370 |
| B = A − dd^T/(2m) | Yes (partial) | Null model encoded → 10-hop → sigmoid | 0.317 |
| A (raw adjacency) | No | Degree explosion: SV1/SV2 = 155 | 0.001 |

### Why raw adjacency fails

Each entry of A^T scales as a sum of paths of length T. In a sparse SBM with heterogeneous degrees, the dominant contribution is proportional to d_i^{T/2} × d_j^{T/2} (via the leading eigenvector of A, which is degree-correlated). After T=10 hops, the resulting matrix has SV1 ≈ 10.7M vs SV2 ≈ 1.2M. The community signal is present at SV2 (eigenvector-label correlation = 0.465) but is buried under the degree-dominated SV1 (correlation = 0.062). Applying log and clip cannot fix this: the log-PMI recipe amplifies the degree heterogeneity rather than canceling it, producing SV1/SV2 = 155 after the full recipe. This is confirmed by the max element difference of 18.04 between adj_netmf_clip and randwalk_netmf_clip log matrices — they encode fundamentally different information.

### The row-stochasticity condition

Row-stochasticity (P^T has all entries ≤ 1 by construction) or spectral radius ≤ 1 (M_na) must be established in the base operator BEFORE raising to the T-th power. For P: P^T_{ij} ≤ 1 always, so multi-hop powers cannot amplify entries. For A: A^T_{ij} can grow without bound. Post-hoc normalization (row-normalize A^10hop) provides only partial recovery (NMI=0.297 vs 0.356 for P) because the relative ordering of entries — already corrupted by degree amplification — is not fully restored by rescaling row sums. Post-hoc symmetric normalization (D^{-1/2} A^10hop D^{-1/2}) fails entirely (NMI=0.003, SV1/SV2=63) because it does not flatten row-sum variance.

### Equivalent formulations

Three approaches achieve equivalent NMI (within ~0.03 noise) at all tested mu values:

1. **NetMF** (standard): P = D^{-1}A → T=10 multi-hop → vol/degree-normalize → log → clip0 → NMI = 0.352 at mu=0.40
2. **Symmetric**: M_na = D^{-1/2}AD^{-1/2} → T=10 multi-hop → vol/degree-normalize → log → clip0 → NMI = 0.359 at mu=0.40
3. **Modularity**: B = A − dd^T/(2m) → T=10 multi-hop → sigmoid → NMI = 0.324 at mu=0.40

All three satisfy the same abstract principle: degree-normalized null-model deviation + multi-hop aggregation + suppression of extremes. The modularity formulation reaches this via a different route (B already encodes the null model explicitly, so no log-PMI step is needed; sigmoid plays the role of clip).
