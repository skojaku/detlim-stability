# Summary: Iteration 9

## What Was Found

Multi-hop aggregation (T=10) helps all degree-normalized operators: random walk P improved by +0.222 NMI, modularity B by +0.152, and normalized adjacency M_na by +0.056 at mu=0.40. Raw adjacency A is the exception — its NMI *degrades* with multi-hop (0.012 → 0.003) because A^10 entries scale as d_i^5 × d_j^5, causing a runaway degree-dominated SV1/SV2 ratio of 7.2. Clip-at-0 and sigmoid transformations are no-ops for P, M_na, and A after multi-hop, because their 10-hop matrices are entirely non-negative (fraction_positive=1.0); only modularity B (fraction_positive=0.35) benefits from sigmoid (NMI +0.139, reaching 0.337 at mu=0.40 vs NetMF reference 0.369). The gradient-weight transform completely fails on modularity (NMI≈0.001) because B≈0 at non-edges, making σ(B)(1-σ(B)) ≈ 0.25 for pairs that are not community-boundary-ambiguous.

## What This Means for the Goal

This establishes that degree-normalized operators are a precondition for multi-hop to work, not just a performance tweak. The log-clip mechanism from NetMF is not directly portable to other operators without accounting for their sign structure and whether clip-at-0 is a no-op.

## Recommended Next Task

Apply the full NetMF recipe (multi-hop → degree-normalize → log → clip) to each operator (A, M_na, B, P) to test whether the recipe is operator-invariant or only works for P. Compare NMI and SV1/SV2 ratios across all four operators with the identical pipeline.
