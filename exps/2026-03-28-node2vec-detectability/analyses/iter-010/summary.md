# Summary: Iteration 10

## What Was Found

Applying the unified NetMF recipe (multi-hop → degree-normalize → log → clip) to each operator reveals that the recipe is NOT operator-invariant. norm_adj_netmf_clip (M_na → 10-hop → log → clip) achieves NMI=0.341 at mu=0.40, exactly matching standard NetMF (P → same recipe). But adj_netmf_clip (A → same recipe) achieves NMI=0.002 with SV1/SV2=163, a catastrophic failure. mod_netmf_clip (B → offset-then-log) also fails with SV1/SV2=366. The best modularity approach remains mod_10hop_sigmoid at NMI=0.317 (mu=0.40). The critical diagnostic: adj_netmf_clip vs randwalk_netmf_clip have a max element difference of 18.04 in the resulting log matrices, confirming they encode completely different quantities even with the same recipe applied.

## What This Means for the Goal

The recipe failure on raw A narrows the essential condition: it is not simply "apply log-PMI to multi-hop matrix" — the base operator must be degree-normalized (row-stochastic or spectral-radius ≤ 1) before multi-hop. This rules out a simple recipe-transfer explanation and points to a structural property of the operator.

## Recommended Next Task

Ablate degree normalization directly: test (1) A^10hop → log → clip [fails], (2) A^10hop → post-hoc symmetric normalize → log → clip, (3) A^10hop → post-hoc row-normalize → log → clip. If post-hoc row-normalization rescues community detection, it confirms that row-stochasticity is the key property rather than when normalization happens.
