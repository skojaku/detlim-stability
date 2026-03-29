status: DONE

# Progress: node2vec-detectability

**Branch**: explore/node2vec-detectability-20260328
**Started**: 2026-03-28

---

## Goal

> Determine whether the detectability power of node2vec near the SBM detectability limit arises from (a) lack of orthogonality constraints, (b) the log-nonlinearity in its objective vs linear factorization of R, or (c) implicit SGD regularization (batch size effects), by comparing NMI of spectral clustering, BP, unconstrained matrix factorizations, and linearized node2vec variants on a sparse 2-community SBM.

---

## Latest Finding

> Iteration 13: Near-parallelism of embedding dimensions (PR≈24 vs PR=64) is a geometric signature of SGNS optimization but is NOT what gives node2vec its detectability advantage. Gradient_weight_svd (NMI=0.379) and clip_0_svd (NMI=0.370) both match or exceed node2vec (NMI=0.350) at mu=0.40 while maintaining perfectly orthogonal embeddings (PR=64). The detectability advantage comes from the log-nonlinearity + implicit clipping of negative log-PMI entries — recoverable deterministically via SVD — not from lack of orthogonality constraints or SGD stochasticity. Core research question is answered; suggested next step is phase-transition characterization (NMI vs mu sweep for top methods).

---

## Guardrails

- **embcom.Node2VecMatrixFactorization (no clip)**: always fails at mu>=0.40 on N=2000 SBM with cave=5. The raw log NetMF matrix has 85.5% negative entries producing a dominant spurious singular value (SV1~1546 vs ~242 clipped). Use clip_0 SVD instead: `SVD(max(M_netmf, 0))`. (Confirmed iters 004–007)
- **Belief Propagation with iters=1**: always fails — outputs near-trivial result. NMI=0.04–0.08 (std=0.15–0.18) is non-convergence. Use iters≥10 with multiple restarts. (Confirmed iter-007)

---

## Iterations

(appended by interpreter subagent after each iteration)

### Iteration 1 — 2026-03-28
**Tried**: Unconstrained MSE factorization of normalized adjacency (I - L_norm), compared against spectral and n2vec_mf, sweeping mu from 0.3 to 0.7 on a 2-community SBM (N=2000, c_avg=5, detectability limit mu=0.553).
**Found**: All methods fail near-identically at mu≥0.5 (NMI < 0.003); unconstrained_mf is marginally better at mu=0.4 (NMI=0.054 vs spectral 0.018) but none approach the detectability limit, ruling out lack of orthogonality constraints (hypothesis a) as the key driver.
**Next**: Compare unconstrained MSE factorization of the true node2vec implicit log-PPR matrix vs the normalized adjacency to test whether log-nonlinearity (hypothesis b) confers detectability near mu=0.553.

### Iteration 2 — 2026-03-28
**Tried**: Compared svd_R, free_R, svd_logR, free_logR, spectral, BP, and actual node2vec (N=500) across mu=0.3–0.7 on a 2-community SBM (N=2000 for matrix methods, c_avg=5, detectability limit mu*=0.553).
**Found**: Log-nonlinearity is necessary (svd_logR NMI=0.248 vs svd_R NMI=0.001 at mu=0.3), but free factorization of log(R) *hurts* vs SVD (free_logR NMI=0.006 at mu=0.3); node2vec achieves NMI=0.635/0.326/0.104 at mu=0.3/0.4/0.5, far above all matrix methods including free_logR, with the gap likely attributable to SGD/stochastic regularization (hypothesis c) — though N=500 vs N=2000 confounds the comparison.
**Next**: Fair comparison at N=2000 — run BP, spectral, n2vec_mf, node2vec, svd_logR, free_logR all at N=2000 for mu in [0.3, 0.4, 0.45, 0.5, 0.52] (all below mu*). Key question: does node2vec outperform n2vec_mf (SVD of same NetMF matrix) at matched N=2000? If yes, SGD/stochastic noise (H1) is the driver.

### Iteration 3 — 2026-03-28
**Tried**: Fair N=2000 comparison: BP, spectral, n2vec_mf, svd_netmf across mu=[0.30, 0.35, 0.40, 0.45, 0.50, 0.52]. free_netmf and node2vec not run (torch unavailable; node2vec was deferred to iter-004 to avoid partial comparison).
**Found**: n2vec_mf and svd_netmf both achieve NMI~0.609 at mu=0.30, collapse sharply by mu=0.40 (NMI~0.003–0.005). BP achieves NMI=0.487/0.438/0.304/0.222 at mu=0.30/0.35/0.40/0.45 but fails at mu=0.50 (NMI~0.0004). Spectral fails across all mu (NMI~0.003–0.093). Confirms the N=2000 confound from iter-002 was real: all static-matrix methods fail near mu*=0.553 even at correct N.
**Next**: Run node2vec at N=2000 on the same graphs to test H1 (SGD/stochastic regularization) directly.

### Iteration 4 — 2026-03-28
**Tried**: Direct node2vec (walk+SGNS) vs n2vec_mf (SVD NetMF) at matched N=2000, identical graphs, mu=[0.30, 0.35, 0.40, 0.45, 0.50, 0.52], 10 seeds each. Spectral included as baseline.
**Found**: At mu=0.30 both methods comparable (node2vec 0.585, n2vec_mf 0.601). At mu=0.40 node2vec NMI=0.347 vs n2vec_mf NMI=0.003 (gap +0.344). At mu=0.45, 0.202 vs 0.002. H1 (SGNS/stochastic regularization) strongly supported — gap is not a size artifact. n2vec_mf has two failure modes: ~22% runs fail numerically (log(0) -> -inf in NetMF matrix) and remaining runs near mu>=0.40 produce near-zero NMI. Spectral also fails. free_netmf could not run (torch unavailable).
**Next**: Test clipped NetMF SVD (clip M_netmf at 0 before SVD) at mu=[0.35, 0.40, 0.45] to separate numerical corruption from insufficient spectral signal.

### Iteration 5 — 2026-03-28
**Tried**: Clipped NetMF SVD (max(log(M), 0) before SVD, original NetMF paper formulation) vs n2vec_mf_full (embcom, no clip) vs node2vec at mu=[0.35, 0.40, 0.45, 0.50], N=2000, 10 samples. nan_rate=0% confirmed (no -inf entries — LCC is connected).
**Found**: Clipping at 0 FULLY rescues SVD. netmf_clipped_svd matches node2vec within ~0.016 NMI at all mu values (non-systematic gap). n2vec_mf_full collapses to NMI~0.001 at mu>=0.40. The clipped matrix is ~85.5% sparse (only ~14.5% of node-pair entries positive). The failure of n2vec_mf_full is caused by large negative log values in the full matrix dominating SVD's Frobenius objective — not by -inf/-nan entries. SGNS implicitly clips by only training on observed walk pairs. embcom's Node2VecMatrixFactorization is missing the max(M,0) clip from the original paper. H1 (SGD noise essential) is weakened — a simple deterministic clip suffices.
**Next**: Confirm at 30 samples and characterize mechanism: (1) NMI vs clip threshold (0, -0.5, -1, -2) to show 0 is optimal; (2) show clipped matrix preserves community eigenvectors while full matrix has them dominated by noise; (3) compare effective rank of clipped vs full matrix.

### Iteration 6 — 2026-03-28
**Tried**: Three sub-analyses with n=20 samples: (A) Full confirmation of netmf_clipped_svd vs node2vec vs n2vec_mf_full vs spectral across mu=[0.30, 0.35, 0.40, 0.45, 0.50, 0.52]; (B) Clip threshold sensitivity at mu=0.40 (thresholds: +1, 0, -1, -2, -3, None); (C) Eigenvalue structure — top-5 singular values and eigenvector-label correlations for full vs clipped matrices at mu=0.40 (n=5).
**Found**: (A) CONFIRMED with n=20: netmf_clipped_svd ≈ node2vec at all mu; n2vec_mf_full is high-variance (std=0.184 at mu=0.30) vs clipped (std=0.028); spectral fails completely even at mu=0.30 (NMI=0.106 vs 0.6 for others), proving multi-hop aggregation is independently essential. (B) Clip threshold: -1.0 is optimal (NMI=0.389), 0.0 near-optimal (0.374), plateau below -1.0; clipping above 0 is harmful (0.233 at +1.0) — positive PMI entries encode community structure. (C) Eigenvalue mechanism: full matrix dominant singular value is 1565 vs 241 for clipped (6.5× ratio). Community eigenvector is rank-2 in BOTH matrices (|corr|=0.670 full, 0.653 clipped). The rank-1 garbage direction in the full matrix (|corr|=0.033 with labels) destabilizes the 64-d k-means embedding. Complete mechanism: multi-hop aggregation + negative log-PMI clipping together reproduce the detectability of node2vec.
**Next**: Final confirmation run with 30 samples + definitive mechanism figure: (a) NMI vs mu for node2vec/clipped/BP/spectral, (b) eigenvalue spectra full vs clipped at mu=0.40, (c) NMI vs clip threshold. This is the final result.

### Iteration 7 — 2026-03-28
**Tried**: 30-sample definitive confirmation. Five methods: node2vec, netmf_clipped (clip=0), netmf_full (no clip), spectral (1-hop), BP (iters=1, n=15). Full mu sweep [0.30, 0.35, 0.40, 0.45, 0.50, 0.52]. Plus mechanism analysis: top-20 singular values for full vs clipped matrix, top-5 eigenvector-label correlations, NMI vs clip threshold (10 thresholds from -5.0 to +2.0) at mu=0.40.
**Found**: CONFIRMED with n=30: netmf_clipped ≈ node2vec at all mu (mu=0.40: clipped=0.365±0.038, node2vec=0.346±0.035). netmf_full collapses at mu≥0.40 (NMI=0.320 with high variance std=0.130). Spectral fails at ALL mu including mu=0.30 (NMI=0.182 vs 0.598 for clipped). BP with iters=1 fails completely (NMI=0.04–0.08, std=0.15–0.18 — non-convergence). Mechanism quantified: SV1 ratio full/clipped = 6.39× (1546 vs 242). Community signal at rank 2: |corr|=0.650 clipped vs 0.414 full. Clip threshold plateau from -5.0 to 0.0 (NMI≈0.335–0.346), sharp transition above +0.5 (NMI=0.307), collapse at +1.5 (NMI=0.007). BP at iters=1 added to guardrails as always-failing.
**Next**: Test sigmoid-transformed matrix variants to determine if gradient_weight = PMI × σ(PMI)(1-σ(PMI)) outperforms clip_0 and better captures the SGNS gradient weighting mechanism.

### Iteration 8 — 2026-03-28
**Tried**: Five SVD-based matrix variants vs node2vec: clip_0, sigmoid_M, sigmoid_centered, gradient_weight [M × σ(M)(1-σ(M))], sigmoid_minus_half_clipped. N=2000, mu=[0.35, 0.40, 0.45, 0.50], 15 samples each. Also measured SV spectra and eigenvector-label correlations for each variant.
**Found**: All sigmoid variants outperform clip_0 at mu≥0.40. gradient_weight is strongest near detectability limit: mu=0.50 NMI=0.119 vs clip_0=0.082 vs node2vec=0.090; mu=0.45 NMI=0.240 matching node2vec=0.239. gradient_weight achieves highest community eigenvector correlation (0.732 vs 0.708 for clip_0). SV1 for gradient_weight=263 (close to clip_0=243, far from full_M=1535). SGNS gradient weighting confirmed as the mechanism: σ(u·v)(1-σ(u·v)) weights down easy positives (PMI→+∞) and non-edges (PMI→-∞), focusing gradient on ambiguous pairs (PMI≈0) at the community boundary. clip_0 is a hard-threshold first-order approximation of this soft weighting. COMPLETE ANSWER REACHED.
**Next**: Write result.md.

### Iteration 9 — 2026-03-28
**Tried**: Generalizability of multi-hop and sigmoid/clip principles across four operators (raw A, normalized adjacency M_na, modularity B, random walk P) on N=2000 SBM, mu=[0.30, 0.40, 0.45, 0.50], 10 samples each. Tested 1-hop vs 10-hop, plus clip0/sigmoid/gradient-weight transforms for each operator.
**Found**: Multi-hop helps all degree-normalized operators (P: +0.222, B: +0.152, M_na: +0.056 at mu=0.40) but HURTS raw A (0.012 → 0.003). Clip-at-0 and sigmoid are no-ops for A, M_na, P after multi-hop (fraction_positive=1.0 for all three; no negatives to clip). Sigmoid helps only modularity B (NMI +0.139, reaching 0.337). Gradient-weight collapses on B (NMI≈0.001) because B≈0 at non-edges, not because they are community-boundary-ambiguous.
**Next**: Apply the full NetMF log-PMI recipe (multi-hop → degree-normalize → log → clip) to each operator to test whether the recipe is operator-invariant.

### Iteration 10 — 2026-03-28
**Tried**: Unified log-PMI recipe (T=10 multi-hop → vol/degree normalize → log → clip0) applied to each operator: A (adj_netmf_clip), M_na (norm_adj_netmf_clip), B (mod_netmf_clip), P (randwalk_netmf_clip = standard NetMF). Plus modularity variants (raw, clip0, sigmoid, gradweight). N=2000, mu=[0.30, 0.40, 0.45, 0.50], 10 samples.
**Found**: norm_adj_netmf_clip exactly matches NetMF (NMI=0.341 vs 0.341 at mu=0.40). adj_netmf_clip catastrophically fails (NMI=0.002, SV1/SV2=163). mod_netmf_clip also fails (SV1/SV2=366). Best modularity approach: mod_10hop_sigmoid at NMI=0.317. The max element difference between adj and randwalk log-matrices is 18.04, confirming they encode fundamentally different information even with the same recipe.
**Next**: Ablate degree normalization: test post-hoc row-normalize and symmetric-normalize of A^10hop to determine if normalization timing (before vs after multi-hop) is the critical factor.

### Iteration 11 — 2026-03-28
**Tried**: Degree normalization ablation at mu=0.40 (10 samples): P_mh_log_clip, Msym_mh_log_clip, A_mh_log_clip, A_degnorm_post (symmetric normalize A^10hop), A_row_post (row-normalize A^10hop), Dsym_post. Full SV analysis of A^10hop and P^10hop, plus full mu sweep (20 samples) for best 4 methods.
**Found**: Row-normalizing A^10hop then log+clip gives NMI=0.297 (partial recovery, SV1/SV2=2.93). Post-hoc symmetric normalize of A^10hop still fails (NMI=0.003, SV1/SV2=63). A^10hop raw: SV1=10.7M, SV2=1.2M; community signal at SV2 (corr=0.465) is buried under degree-dominated SV1 (corr=0.062). P^10hop: SV1=1.085, SV2=0.445; clean. Row-stochasticity must precede multi-hop — it cannot be recovered post-hoc. Three equivalent formulations (NetMF NMI=0.352, norm_adj NMI=0.359, mod_sigmoid NMI=0.324) all reach within 0.03 NMI of each other at mu=0.40.
**Next**: Characterize the SV1/SV2 threshold as an empirical diagnostic criterion for whether a given matrix representation will support community detection.

### Iteration 12 — 2026-03-28
**Tried**: NMI vs hop count T sweep (T=1,2,3,5,10,20) for log-PMI NetMF at mu=0.40; Bethe Hessian with r=sqrt(cave) as a 1-hop baseline; comparison of T=1 log-PMI failure vs T=2 recovery.
**Found**: T=1 log-PMI (NMI=0.132) fails because 99.6% of within-community pairs are non-edges — signal too sparse at 1 hop. T=2 recovers massively (NMI=0.350): 2-hop paths reach ~25 same-community neighbors in a graph with cave=5. Plateau by T=3 (NMI=0.385); T=5,10,20 stable at 0.35–0.38. Bethe Hessian (1-hop, r=sqrt(cave)) achieves NMI=0.362 — statistically equivalent to T=10. Multi-hop is NOT fundamental: Bethe Hessian achieves the same result using only direct edges, via the cavity method derivation that naturally encodes the degree null model and uses negative eigenvalues that only appear above the KS threshold (equivalent to infinite-depth message passing).
**Next**: Refined principle confirmed — the fundamental requirement is any operator whose dominant spectral structure encodes community membership rather than degree heterogeneity. Three paths: (a) multi-hop T≥2 of row-stochastic operator, (b) Bethe Hessian, (c) modularity + sigmoid.

### Iteration 13 — 2026-03-28
**Tried**: Embedding geometry analysis — participation ratio (PR = (Σλ)²/Σλ² of the Gram matrix) and off-diagonal cosine similarity across four methods (node2vec, gradient_weight_svd, clip_0_svd, spectral) at mu=0.30 and mu=0.40, N=2000 SBM, dim=64, 10 samples each. Tested whether near-parallel embedding dimensions are required for or predictive of superior community detection.
**Found**: Node2vec has dramatically lower PR than all SVD/spectral methods: PR≈23.7 (mu=0.40) vs PR≈64.0 for gradient_weight_svd, clip_0_svd, and spectral (perfectly orthogonal). Node2vec dimensions are near-parallel (mean off-diagonal |cos|=0.122), while SVD methods are effectively orthogonal (|cos|≈0.0001). The node2vec Gram eigenvalue spectrum is dominated by a single large eigenvalue (~11.1 at mu=0.40), suggesting most of the 64 embedding dimensions collapse onto the same community direction. Critically, near-parallelism is NOT required for strong community detection: gradient_weight_svd (NMI=0.379) and clip_0_svd (NMI=0.370) match or exceed node2vec (NMI=0.350) at mu=0.40 while maintaining PR=64 (perfect orthogonality). Near-parallelism is a property of the SGNS optimization geometry (asymmetric context/target matrices, stochastic updates), not a prerequisite for community detection. The hypothesis from idea.md — that orthogonality constraints limit spectral methods — is CONFIRMED for the comparison with standard spectral, but REFUTED as the key factor distinguishing node2vec from gradient_weight_svd: both overcome orthogonality, but only via different geometric routes.
**Next**: The core research question is now fully answered. The detectability advantage of node2vec arises primarily from (b) log-nonlinearity in the SGNS objective combined with implicit negative-entry suppression (equivalent to clipping), not from (a) lack of orthogonality constraints (which can be matched by gradient_weight_svd) and not from (c) SGD batch-size regularization per se (the gradient weighting function σ(u·v)(1−σ(u·v)) is the key quantity, recoverable deterministically). The remaining open question with highest scientific value is the phase transition characterization: sweep mu densely (e.g., 0.30 to 0.55 in steps of 0.02) for the top four methods (gradient_weight_svd, clip_0_svd, Bethe Hessian, node2vec) to determine the exact mu at which each crosses NMI=0.1, quantifying the "effective detectability extension" per method. This would yield the clearest summary figure for the final result.

### Iteration 14 — 2026-03-28
**Tried**: Dense mu sweep [0.30–0.54] with 15 samples/mu for 5 methods: gradient_weight_svd, Bethe Hessian, clip_0_svd, node2vec, BP. N=2000, cave=5.
**Found**:
- Effective detectability limits (mu where NMI crosses 0.1): gradient_weight=0.503, bethe_hessian=0.502, clip_0=0.488, node2vec=0.486, BP=0.449
- gradient_weight_svd and Bethe Hessian are tied as best methods, ~0.05 below theoretical mu*=0.553
- node2vec and clip_0 are nearly identical (0.486 vs 0.488) — clip_0 fully captures node2vec's mechanism
- BP performs worst (mu*=0.449) due to high variance / poor convergence at N=2000
- At mu=0.50: gradient_weight NMI=0.106 vs node2vec NMI=0.065 (+63% improvement)

**Next**: DONE — core question fully answered. Recommend writing final result.md.
