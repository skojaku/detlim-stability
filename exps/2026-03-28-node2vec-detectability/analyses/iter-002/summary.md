# Summary: Iteration 2

## What Was Found

This iteration compared six methods on a 2-community SBM (N=2000 for matrix methods, N=500 for node2vec, c_avg=5, detectability limit mu*=0.553). Below the detectability limit, the log-nonlinearity of the objective is the key differentiator: svd_logR (SVD of log(R), orthogonal) achieves NMI=0.248 at mu=0.3 and NMI=0.023 at mu=0.4, while svd_R (SVD of R, orthogonal) and free_R (unconstrained MSE of R) both flatline at NMI~0.001 across all mu values. This confirms that the log transformation — not freedom from orthogonality — is what matters. Surprisingly, free_logR (unconstrained MSE of log(R)) performs *worse* than svd_logR, with NMI=0.006 at mu=0.3 and 0.003 at mu=0.4, showing that removing orthogonality constraints from the log objective actually hurts. Actual node2vec (N=500) achieves NMI=0.635/0.326/0.104 at mu=0.3/0.4/0.5 — dramatically better than all matrix factorization methods including free_logR, which is the closest analytical proxy to its SGNS objective.

## What This Means for the Goal

Log-nonlinearity (hypothesis b) is confirmed as necessary but not sufficient: svd_logR shows log helps, but the gap between node2vec and free_logR (the best log-nonlinear free factorization) is enormous at mu=0.5 (NMI=0.104 vs 0.001). This points strongly to SGD/stochastic regularization (hypothesis c) as the additional source of node2vec's power near the detectability limit — the full SGNS training process with random walks provides something the batch matrix factorization cannot. The note about N=500 vs N=2000 means some of node2vec's advantage could be a size effect; this must be controlled.

## Recommended Next Task

Run node2vec (N=500) alongside svd_logR and free_logR at the *same* graph size (N=500) for mu in {0.3, 0.4, 0.5, 0.52} with n_samples=10 each, to determine whether node2vec's advantage over free_logR at mu=0.5 survives a fair size-controlled comparison. If the gap persists at N=500, it implicates SGD/stochastic noise as the driver; if it closes, the iter-002 advantage was a size artifact.
