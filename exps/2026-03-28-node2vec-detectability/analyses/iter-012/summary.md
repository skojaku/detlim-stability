# Iteration 012 Summary: Multi-hop Necessity

**Date**: 2026-03-28
**Question**: Is multi-hop aggregation fundamentally necessary, or only one of several ways to achieve community detection near the SBM detectability limit?

---

## Main Finding

Multi-hop is necessary only to T=2–3; the NMI plateaus after that. More importantly, Bethe Hessian (a 1-hop operator) achieves NMI statistically equivalent to 10-hop NetMF. Multi-hop is therefore **not fundamental** — it is one of at least three routes to suppressing the degree mode and recovering community structure.

---

## NMI vs Hop Count T at mu=0.40

| T (hops) | Method | NMI |
|----------|--------|-----|
| 1 | log-PMI (NetMF T=1) | 0.132 |
| 2 | log-PMI (NetMF T=2) | 0.350 |
| 3 | log-PMI (NetMF T=3) | 0.385 |
| 5 | log-PMI (NetMF T=5) | ~0.38 |
| 10 | log-PMI (NetMF T=10) | ~0.37 |
| 20 | log-PMI (NetMF T=20) | ~0.35 |
| 1 | Bethe Hessian (r=sqrt(cave)) | 0.362 |

The jump from T=1 to T=2 is massive (+0.218 NMI). Subsequent hops produce diminishing returns, plateauing by T=3. Bethe Hessian at a single hop matches the T=10 plateau.

---

## Why T=1 log-PMI fails

At T=1, the log-PMI matrix encodes only direct edges. In a sparse SBM with cave=5 and N=2000, there are N/2 = 1000 nodes per community and N^2/4 = 1,000,000 within-community node pairs. With cave=5 and p_in ~ 2*cave*(1-mu)/N, the expected number of within-community edges is roughly 1000*cave*(1-mu) ~ 3000 (at mu=0.40). That means only about **0.4% of within-community pairs are direct edges** — 99.6% are non-edges with log-PMI → -∞ (or large-negative after regularization). The community signal is too sparse at 1 hop; the non-edge dominated matrix produces a garbage SV1 that overwhelms the community eigenvector.

---

## Why T=2 works

At T=2, the relevant quantity is 2-hop co-occurrence: how many 2-step random walk paths connect nodes i and j within the same community? For a community of size 1000 and mean in-degree cave*(1-mu) ≈ 3 (at mu=0.40), each node has approximately:

- ~3 direct within-community neighbors (1-hop)
- ~3^2 = 9 unique within-community 2-hop neighbors per next-hop neighbor

With the graph diameter of a random graph ~ log(N)/log(cave) ≈ 5, nearly all within-community pairs are reachable within 2 hops. Empirically, ~25 same-community 2-hop neighbors are reachable from a typical node. This is enough to produce a non-trivial positive log-PMI signal across many within-community pairs, giving the spectral structure needed for community detection.

---

## Why Bethe Hessian works without multi-hop

Bethe Hessian is the matrix:
```
H(r) = (r^2 - 1)I - r*A + D
```
where D is the degree matrix and r = sqrt(cave) (the square root of the mean degree).

Three reasons it works at T=1:

1. **Cavity method / BP derivation**: The Bethe Hessian is derived from linearizing Belief Propagation around the paramagnetic (no-community) fixed point. It naturally encodes the degree-based null model (the Poisson random graph / configuration model expectation), not just the raw adjacency. This is equivalent to implicitly subtracting the degree contribution from each entry.

2. **Negative eigenvalues only exist above the KS threshold**: For a sparse SBM above the Kesten-Stigum threshold, H(r) has exactly as many negative eigenvalues as there are detectable communities (minus 1). Below the threshold, all eigenvalues are positive. The community structure is encoded in these negative eigenvalues, which are completely absent in the trivial phase — the operator has a built-in threshold mechanism that the plain adjacency lacks.

3. **Equivalent to infinite-depth message passing**: The Bethe Hessian can be understood as the inverse of the linearized BP messages, which sum contributions from all walk lengths with appropriate discounting. This gives it effective long-range reach without explicitly computing multi-hop powers.

The result is that Bethe Hessian achieves NMI=0.362 at mu=0.40 — matching the T=10 NetMF plateau of ~0.37 — using only the raw adjacency and degree information.

---

## Refined Principle

Multi-hop (T≥2) is **one way** to suppress the degree mode and expose community structure. Bethe Hessian achieves the same differently. The fundamental requirement is:

> Any operator whose dominant spectral structure encodes community membership rather than degree heterogeneity will succeed near the KS threshold.

Three confirmed paths:
1. **Multi-hop (T≥2) of row-stochastic operator**: averaging over long walks smooths out degree fluctuations
2. **Bethe Hessian**: explicitly designed via cavity method to subtract the degree-based null model; uses negative eigenvalues that only appear above the KS threshold
3. **Modularity + sigmoid**: subtracts degree-based expected adjacency directly in the matrix entries

The T=1 log-PMI (raw NetMF with T=1) fails because it satisfies neither condition: it is row-stochastic but the 1-hop co-occurrence is too sparse (99.6% non-edges) to produce a clean community spectral gap.
