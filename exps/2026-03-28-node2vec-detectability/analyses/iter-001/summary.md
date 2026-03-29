# Summary: Iteration 1

## What Was Found

Three methods were compared on a 2-community SBM with N=2000, c_avg=5, detectability limit at mu=0.553: spectral (top eigenvectors of normalized adjacency), n2vec_mf (embcom's Node2Vec matrix factorization), and unconstrained_mf (MSE factorization of the same normalized adjacency matrix with no orthogonality constraint, trained with Adam for 500 steps). All three methods collapsed to near-chance NMI (< 0.002) at mu=0.5 and above. At mu=0.3, n2vec_mf performed best (NMI=0.597 ± 0.032), followed by unconstrained_mf (NMI=0.293 ± 0.239, high variance), and spectral worst (NMI=0.197 ± 0.182). At mu=0.4, all methods effectively failed: spectral NMI=0.018, n2vec_mf NMI=0.001, unconstrained_mf NMI=0.054 (large variance). The actual node2vec walk-based method was too slow to run (estimated 355 s for full sweep). Notably, the task spec's SBM parameterization had a bug giving mu*~0.276 instead of the correct mu*~0.553; the experiment used the correct Decelle limit.

## What This Means for the Goal

Hypothesis H2 — that removing orthogonality constraints from matrix factorization of the normalized adjacency is sufficient to push detectability toward the theoretical limit — is not supported: the unconstrained MSE factorizer fails just as badly as spectral at mu≥0.5, ruling out (a) as the sole driver. The log-nonlinearity (b) and implicit SGD regularization (c) remain untested because actual node2vec was too slow to run and no linearized variant was included.

## Recommended Next Task

Run a controlled comparison between (i) unconstrained MSE factorization of the node2vec PPR matrix M_n2v = log(vol(G) * PPR_window / d_i d_j) - log(k) (the true node2vec implicit matrix) vs (ii) unconstrained MSE factorization of the normalized adjacency I-L_norm used in iter-001, across mu in [0.3, 0.7], to isolate whether the log-nonlinearity of the node2vec target matrix (hypothesis b) confers detectability near the limit. Use the same Adam optimizer, 500 steps, lr=0.01, dim=64 setup as iter-001. If the log-PPR factorization outperforms the linear factorization near mu=0.553, hypothesis (b) is confirmed.
