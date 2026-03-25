# Where is Community Information Encoded in Eigenvectors near the Detectability Limit?

## Question

Near the detectability limit (mu~0.5, cave=5, N=2000), community structure is hard to detect.
The second eigenvector (v2) of the adjacency matrix carries the main community signal, but
previous analysis suggests that eigenvectors v3, v4, ... also contain community structure information.
We want to:

1. Understand exactly where and how community information is encoded in adjacency eigenvectors.
2. Develop an algorithm using eigenvectors that matches or exceeds belief propagation (NMI ~0.112).

## Setup

- N=2000 nodes (2 communities of 1000 each)
- cave=5.0, mu=0.5 (detectability limit ≈ 0.553 for cave=5)
- 30 independent SBM samples
- 10 eigenvectors computed per sample (adjacency)
- Baselines: sign(v2) NMI=0.047, K-means NMI=0.027, BP NMI=0.112

## Log

### [2026-03-25] Data generation
Generated 30 SBM samples with mu=0.5, cave=5, N=2000. Computed top-10 eigenvectors.
Reference: sign(v2) NMI=0.047±0.035, K-means(5 eigvecs) NMI=0.027±0.023, BP NMI=0.112±0.062.

### [2026-03-25] Round 1: Localization and structure of higher eigenvectors

**Scripts**: 01_localization_community.py, 02_node_degree_structure.py, 03_sign_combination.py

**Localization (IPR)**:
- v1: IPR=0.00135 (0.9x bulk) — delocalized, encodes degree
- v2: IPR=0.01859 (12.4x bulk) — most localized
- v3-v10: IPR=0.005-0.010 (3.5-7x bulk) — moderately localized

**Community content of large-magnitude nodes**:
- v2 top-5% nodes: NMI=0.29 (but std=0.21, very noisy!)
- v3-v10 top-5% nodes: NMI=0.04-0.12 (declining)
- Cross-sample Jaccard of top-5% nodes: ≈ chance (0.026) — large-magnitude nodes are DIFFERENT across samples, no structural consistency

**Degree structure of large-magnitude nodes**:
- Top-10% nodes by |v_k| have degree ~1.5x average for ALL eigvecs k=2..8 (consistent)
- Community bias of top-10% nodes: ~0.06-0.08 (near zero — equally split across communities)
- The localization in v3-v10 is DEGREE-DRIVEN, not community-driven

**Sign combination results**:
- Combining v3-v10 with v2 HURTS performance (NMI drops from 0.047 to 0.016)
- Label propagation from v2 seeds: max NMI=0.044 at 50% seeds
- Majority vote: NMI=0.013-0.016

**Conclusion Round 1**: The "community information" in v3-v10 is an artifact of degree localization.
High-degree nodes (hubs) dominate the large-magnitude entries of all eigvecs, and hubs are
roughly equally split between communities. Naive combination of higher eigvecs kills performance.

### [2026-03-25] Round 2: Bethe Hessian and spectral variants

**Scripts**: 04_bethe_hessian.py, 05_score_regularized.py, 06_iterative_refinement.py

**Bethe Hessian H(r) = (r²-1)I - rA + D** at r=sqrt(cave)=2.236:
- sign of 2nd-smallest eigvec: NMI=0.0949 (best r=2.236, much better than v2!)
- BH + K-means (3 eigvecs): NMI=0.1011 (nearly matches BP=0.112)

**Other variants (none beat BP)**:
- SCORE (v2/v1 ratio): NMI=0.047 — no improvement
- Regularized adjacency A + tau/N * 11^T: NMI≈0.047 (flat for all tau)
- Normalized adjacency D^{-1/2}AD^{-1/2}: NMI=0.010 — worse
- Centered adjacency A - cave/N * 11^T: NMI=0.022
- Label propagation (personalized PageRank from v2): NMI=0.048
- Deflated power iteration on A: NMI=0.047 (identical, as expected)

### [2026-03-25] Round 3: Relationship between BH and adjacency eigenvectors

**Scripts**: 07_bh_adjacency_decomposition.py, 08_degree_community_mixing.py, 09_eigvec_algorithm.py

**BH eigvec decomposition in adjacency basis**:
The BH community eigvec u_BH projected onto adjacency eigvecs:
- v2 (community): |c_2|² = 0.317 (31.7%)
- v3: |c_3|² = 0.135 (13.5%)
- v4: |c_4|² = 0.075 (7.5%)
- v5: |c_5|² = 0.059 (5.9%)
- Top-20 adj eigvecs explain 80.7% of u_BH power
- ~19% lives in the spectral bulk (below top-20)

**D acts nearly diagonally on eigvecs**: The degree-correction term D@v_k - d_avg*v_k
projects almost entirely back onto v_k itself (>97% self-projection, <3% cross-mixing).
D does NOT mix different adjacency eigenvectors — it scales each independently.

**Why BH community signal is NOT just v2**:
- u_BH ≈ 0.52*v2 (adj) + 0.50*v1 (adj) + smaller contributions from v3-v20
- Only 26.7% of u_BH's power comes from v2
- 25.4% comes from v1 (degree eigvec)
- Residual after removing v2: f = u_BH - c2*v2 has NMI=0.071 — REAL community signal!
- This residual is dominated by v1 (adjacency degree vector)

**Degree contamination of v2**:
- Community explains only 6.7% of v2 variance (R²=0.064)
- Degree explains 0% of v2 variance independently (R²=0.000)
- Degree normalization (v2/sqrt(degree)) doesn't improve NMI
- BUT: high-boundary-fraction nodes are misclassified more (r=-0.170)

**Best algorithm found**: sign(u2_BH / u1_BH) = NMI=0.1164 > BP=0.1118 ✓
- Equivalently: sign(u2_BH / v1_adj) gives the same result
- v1_adj is a good proxy for u1_BH (BH degree eigvec)

## Gotchas

- Higher adjacency eigvecs (v3-v10): their large-magnitude nodes are degree-driven hubs,
  not community-driven. Combining them with v2 destroys performance.
- The Bethe Hessian "second eigvec" is the one near zero (second-smallest), NOT most-negative.
  The most-negative is a bulk mode from the (r²-1) shift.
- v2 alone: only 6.7% of its energy is actual community signal. The rest is noise.
- D@v_k is nearly diagonal in eigvec space — D doesn't mix eigvecs significantly.

## Learnings

1. **Localization in v3-v10 is degree-driven**: The large-magnitude entries concentrate on
   high-degree nodes (hubs), not on community-specific nodes. This explains why combining
   higher eigvecs hurts: you're adding degree noise, not community signal.

2. **BH extracts community signal through a specific reweighting**: H(r)=D-rA+(r²-1)I.
   The D term shifts each node's eigenvalue contribution by its degree, effectively
   "separating" the community signal (which sits in a specific combination of v1, v2, and
   higher eigvecs) from the bulk noise.

3. **Community information spreads across many adjacency eigvecs**: The BH community eigvec
   draws ~32% from v2, ~25% from v1, and ~43% from higher eigvecs + bulk. The adjacency
   eigenvectors individually are poor community representations because the community direction
   does not align well with any single eigvec.

4. **The correct combination**: The Bethe Hessian finds the optimal linear combination of
   all adjacency eigvecs that corresponds to the "community direction," by using the degree
   matrix D to define a generalized eigenvalue problem.

## Conclusions

**Where is community information in adjacency eigenvectors?**
- Primary carrier: v2 (second eigvec), but only 6.7% of its energy is community signal
- The "true" community direction is a specific linear combination spanning v1, v2, v3...v20+
- Higher eigvecs (v3-v10) contain community information BUT also degree localization that
  cancels the community signal when combined naively

**Why does Bethe Hessian work?**
- H(r) = (r²-1)I - rA + D defines a generalized eigenproblem that finds the community direction
- The D term corrects for degree heterogeneity, transforming from the adjacency eigenvector basis
  into a basis where the community direction is more cleanly isolated
- The BH community eigvec is approximately: u_BH ≈ 0.52*v2 + 0.50*v1 + mix of higher eigvecs
- Crucially, the v1 component (degree) plays a constructive role: sign(u_BH / v1) beats BP

**Best algorithm from eigenvectors**: sign(u_BH / u1_BH) where:
- u_BH = 2nd-smallest eigvec of H(sqrt(cave))
- u1_BH = smallest eigvec of H(sqrt(cave)) ≈ v1_adj (degree eigvec of adjacency)
- NMI = 0.1164 > BP = 0.1118 ✓
