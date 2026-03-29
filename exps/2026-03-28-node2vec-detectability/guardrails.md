# Guardrails: {topic}

Approaches that have been tried and failed. Analyst subagents must not repeat these.

---

## Failed Approaches

(appended by interpreter subagent when an approach fails badly or produces no signal)

<!-- Format:
### YYYY-MM-DD
**Approach**: what was tried
**Why it failed**: what went wrong
**Lesson**: what to do instead
-->

### 2026-03-28 — Framing
**Approach**: Treating methods failing at mu > mu*=0.553 as a bug or problem.
**Why it failed**: Methods are *theoretically expected* to fail above the detectability limit. This is not a bug.
**Lesson**: Only evaluate and compare methods for mu < mu*=0.553. The interesting question is how well methods detect communities below the limit, especially near mu* (mu ≈ 0.4–0.52).

### 2026-03-28 — Wrong SBM parameterization
**Approach**: Using `p_total = cave*2/N; p_out = mu*p_total; p_in = p_total - p_out` from the original idea.md code.
**Why it failed**: This gives a detectability limit of mu*≈0.276, not 0.553. The formula mu* = 1-1/sqrt(cave) only holds for the corrected parameterization.
**Lesson**: Always use: `c_out = mu*cave; c_in = 2*cave - c_out; p_in = c_in/N; p_out = c_out/N`.

### 2026-03-28 — embcom.LaplacianEigenMap bug
**Approach**: Using `embcom.LaplacianEigenMap().fit(A).transform(dim=64)`.
**Why it failed**: The embcom implementation applies an extra D^{-1/2} transform that corrupts the embedding, giving NMI≈0 even in easy regimes.
**Lesson**: Use direct `scipy.sparse.linalg.eigsh(M, k=64, which='LM')` on M=D^{-1/2}AD^{-1/2}, then skip the trivial (all-ones) eigenvector (largest eigenvalue). This gives correct spectral embeddings.

### 2026-03-28 — node2vec full 7-mu sweep at N=2000
**Approach**: Running walk-based node2vec across all 7 mu values × 10 samples at N=2000.
**Why it failed**: ~5s per graph × 70 runs ≈ 350s. Unnecessary to sweep above mu*=0.553 since all methods fail there.
**Lesson**: Limit node2vec sweeps to mu < mu* (5 values × 10 samples = ~250s, acceptable). N=2000 is required for all methods — do not reduce N.

### 2026-03-28
**Approach**: Linear factorization of the co-occurrence matrix R = (2m) D^{-1} A D^{-1}, either via SVD (svd_R) or unconstrained MSE (free_R).
**Why it failed**: Both methods produce NMI~0.001 (near chance) at every mu value including mu=0.3, well below any useful detectability signal. The linear target R contains community structure, but neither SVD nor gradient-based free factorization can extract it — likely because R is dominated by degree noise in the sparse regime.
**Lesson**: Do not test any variant of linear factorization of R (adjacency-based co-occurrence without log). Only log-transformed targets (log(R) or log(PPR matrix)) yield non-trivial community embeddings.

### 2026-03-28 — free_netmf via torch
**Approach**: Running free_netmf (unconstrained Adam factorization of the NetMF matrix) using torch/PyTorch in the uv environment.
**Why it failed**: torch is not installed in /workspace/.venv. ModuleNotFoundError on every attempt.
**Lesson**: Do not attempt torch-based methods without first confirming installation via `uv add torch`. Alternatively, implement free factorization using scipy.optimize.minimize or numpy gradient descent — no torch dependency required.

### 2026-03-28 — n2vec_mf (SVD of raw NetMF) at mu >= 0.40
**Approach**: Using embcom.Node2VecMatrixFactorization (SVD of the NetMF log-PPR matrix) as a substitute for node2vec at mu >= 0.40.
**Why it failed**: Two failure modes: (1) ~22% of runs produce -inf in the NetMF matrix due to log(0) on zero co-occurrence pairs, causing SVD to output NaN/inf-contaminated embeddings; (2) even numerically valid runs collapse to near-zero NMI at mu >= 0.40 — the community signal in the NetMF matrix is too weak for SVD to extract in the N=2000 sparse regime.
**Lesson**: n2vec_mf is not a reliable substitute for node2vec near the detectability limit. If testing matrix factorization of the NetMF target, always clip M_netmf at 0 (replace log(0) with 0, as in the original NetMF paper) before SVD to eliminate numerical failures. Even then, expect SVD to underperform node2vec at mu >= 0.40.

### 2026-03-28 — n2vec_mf_full (embcom's Node2VecMatrixFactorization without clipping)
**Approach**: Using embcom's Node2VecMatrixFactorization directly (SVD of the full unclipped NetMF log-PPR matrix) for any experiment near the detectability limit.
**Why it failed**: Even when nan_rate=0% (LCC is connected, no -inf entries), the full unclipped matrix contains large negative log values for rarely co-visited node pairs. These large-magnitude negative entries dominate SVD's Frobenius objective and prevent recovery of community eigenvectors. NMI collapses to ~0.001 at mu>=0.40 despite the community signal being present in the matrix (clipped version recovers it fully). embcom's implementation is missing the max(M,0) clip from the original NetMF paper (Qiu et al. 2018).
**Lesson**: Do not use n2vec_mf_full / embcom's Node2VecMatrixFactorization for experiments near the detectability limit. Use netmf_clipped_svd instead: compute the NetMF raw matrix, apply max(log(M/k), 0) elementwise, convert to sparse, then run SVD. This matches node2vec NMI within ~0.016 at all tested mu values.

### 2026-03-28 — Over-clipping (clip threshold above 0)
**Approach**: Clipping the NetMF log matrix at a threshold above 0 (e.g., clip=+1.0) to reduce matrix size or noise.
**Why it failed**: Confirmed by iter-006 (n=10 at mu=0.40): clip=+1.0 gives NMI=0.233 with high variance (std=0.113) vs clip=0.0 giving NMI=0.374. Positive log-PMI entries encode community co-occurrence enrichment — they are the community signal, not noise. Clipping them away discards the information that enables community detection.
**Lesson**: Never clip the NetMF log matrix above 0. The optimal threshold is in the range [−1, 0]. Clip at 0 is the natural default (original NetMF paper); clip at −1 is marginally better (NMI=0.389 vs 0.374 at mu=0.40). Do not test thresholds above 0.

### 2026-03-28 — Raw adjacency A with any subsequent log-PMI recipe
**Approach**: Applying the NetMF-style recipe (multi-hop → degree-normalize → log → clip) directly to raw adjacency A (adj_netmf_clip), or combining any recipe with A as the base operator.
**Why it failed**: A^10hop entries scale as d_i^5 × d_j^5 — degree amplification explodes during multi-hop. The resulting log matrix has SV1/SV2 = 155–163 and community signal is buried at rank 2 under a dominant degree-driven SV1. NMI = 0.001–0.002 at all mu values tested (iter-010 confirmed with n=10). The max element difference between adj_netmf_clip and randwalk_netmf_clip is 18.04 — these are not similar matrices despite the identical recipe.
**Lesson**: Never use raw A as a base operator for any log-PMI recipe. Only row-stochastic (P = D^{-1}A) or spectral-radius-≤1 (M_na = D^{-1/2}AD^{-1/2}) operators produce working results. Even gradient-weight and sigmoid on A^10hop fail completely (NMI < 0.001).

### 2026-03-28 — Post-hoc symmetric normalization of A^10hop
**Approach**: Computing A^10hop first (without pre-normalization), then applying symmetric degree normalization (D^{-1/2} A^10hop D^{-1/2}) before log+clip, hoping to recover the same result as M_na → 10hop → log+clip.
**Why it failed**: Post-hoc symmetric normalization still fails (NMI=0.003, SV1/SV2=63 at mu=0.40, iter-011 n=10). The degree amplification during multi-hop corrupts the relative entry ordering in a way that symmetric rescaling cannot undo. Specifically, D^{-1/2} A^10hop D^{-1/2} ≠ (D^{-1/2} A D^{-1/2})^10 = M_na^10, so the resulting matrix is structurally different from the pre-normalized version.
**Lesson**: Degree normalization must happen BEFORE raising to the T-th power. Post-hoc normalization is not equivalent and does not recover community structure.

### 2026-03-28 — Post-hoc row normalization of A^10hop (partial recovery only)
**Approach**: Computing A^10hop first, then row-normalizing (divide each row by its sum) before log+clip, as an approximation of P^10hop = (D^{-1}A)^10.
**Why it failed (partially)**: Row-normalizing A^10hop gives NMI=0.297 (SV1/SV2=2.93 at mu=0.40, iter-011), compared to P_mh_log_clip NMI=0.356. This is substantially below the pre-normalized version and well below NetMF. Row-normalizing post-hoc does not produce the same matrix as P^T because P^T_{ij} accumulates probability mass differently than row_normalize(A^T)_{ij} — the multi-hop averaging mixes paths of different lengths and the two operations do not commute.
**Lesson**: Post-hoc row normalization is a suboptimal substitute that wastes ~0.06 NMI at mu=0.40. If row-stochasticity is needed, compute P = D^{-1}A first, then raise to the T-th power. Only use post-hoc row normalization if the pre-normalization is computationally infeasible.

### 2026-03-28 — T=1 log-PMI (1-hop NetMF): insufficient near detectability limit
**Approach**: Using log-PMI NetMF with T=1 (single-hop, no multi-hop aggregation) as a baseline or surrogate for 10-hop NetMF.
**Why it failed**: At mu=0.40 on N=2000 SBM with cave=5, T=1 log-PMI achieves NMI=0.132 — far below T=2 (NMI=0.350) or T=10 (NMI≈0.37). The failure is structural: 99.6% of within-community node pairs are non-edges. The T=1 matrix is nearly all large-negative log-PMI, producing a garbage SV1 that overwhelms the community eigenvector. There is simply not enough positive co-occurrence signal at 1 hop in a sparse graph. (iter-012)
**Lesson**: Always use T≥2 for log-PMI NetMF near the detectability limit. T=2 is sufficient for near-plateau performance; T=3 is the practical minimum for robustness. Do not treat T=1 log-PMI as a proxy for "just a normalization issue" — the sparsity at 1 hop is the fundamental problem.

### 2026-03-28 — Bethe Hessian with wrong r parameter
**Approach**: Running Bethe Hessian H(r) = (r²−1)I − rA + D with an incorrectly chosen r (e.g., r=1, r=cave, or r chosen by optimization on wrong quantity).
**Why it failed**: The Bethe Hessian's negative eigenvalues (which encode community structure) only exist above the KS threshold AND only for r in the correct range around r=sqrt(cave) = sqrt(mean_degree). With r=1, the matrix reduces to D − A (standard Laplacian), which does not have the right null-model suppression. With r=cave, entries are over-weighted. The number of negative eigenvalues equals the number of detectable communities only at r = sqrt(cave) (equivalently, sqrt(mean_degree) — these are numerically identical for the SBM since cave is the actual mean degree). (iter-012)
**Lesson**: Always set r = sqrt(cave) = sqrt(mean_degree) for Bethe Hessian. Cluster using the eigenvectors corresponding to negative eigenvalues only (not the smallest magnitude eigenvalues of the full spectrum). Bethe Hessian is a 1-hop operator that matches 10-hop NetMF when r is set correctly — it is not necessary to tune r further.
