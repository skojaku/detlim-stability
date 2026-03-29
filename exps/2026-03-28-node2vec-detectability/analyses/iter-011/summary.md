# Summary: Iteration 11

## What Was Found

Post-hoc normalization of A^10hop partially recovers community detection but cannot fully substitute for pre-normalization. Row-normalizing A^10hop then applying log+clip yields NMI=0.297 (SV1/SV2=2.93), a partial recovery from 0.001. Post-hoc symmetric normalization (D^{-1/2} A^10hop D^{-1/2}) still fails (NMI=0.003, SV1/SV2=63). The direct SV analysis explains why: A^10hop has SV1=10.7M vs SV2=1.2M (ratio 8.8), and community signal IS present — at SV2 (eigvec-label correlation=0.465) and SV5 (corr=0.001..0.576) — but is buried under the degree-dominated SV1 (corr=0.062). In contrast, P^10hop has SV1=1.085 (≈1, trivial stationary distribution), SV2=0.445 — clean separation, with community signal at SV5 (corr=0.576). The conclusion is unambiguous: row-stochasticity (or spectral radius ≤1) must be established BEFORE multi-hop; post-hoc normalization cannot undo degree amplification after A^T accumulates d_i^{T/2} × d_j^{T/2} contributions.

## What This Means for the Goal

The three necessary ingredients for detecting communities near the SBM detectability limit are now fully characterized and tested across all major graph operators: (1) row-stochastic/spectral-radius-≤1 base operator, (2) multi-hop aggregation T≈10, (3) suppression of extremes (clip or sigmoid). These together form a transferable principle. Three equivalent formulations (NetMF/P, symmetric/M_na, modularity/B-sigmoid) achieve NMI within 0.03 of each other at all tested mu values.

## Recommended Next Task

Characterize the SV1/SV2 threshold empirically: at what ratio does community detection consistently fail? Sweep SV1/SV2 from ~1 to ~200 across the ablation variants and fit a threshold. This would give a diagnostic criterion for whether a matrix representation will support community detection before running k-means.
