# SGD Eigenvector Solver — Robustness Near Detection Limit

## Question

Can stochastic gradient descent yield a more robust embedding than direct eigendecomposition in sparse SBMs near the detection limit?

**Hypothesis**: SGD's implicit regularization (mini-batch noise, early stopping) filters bulk-eigenvalue contamination that drowns community signal in exact spectral methods. A linear autoencoder (or symmetric matrix factorization via SGD) should learn a low-rank embedding that is better conditioned near the KS threshold.

## Theory

A **linear autoencoder** minimizing MSE on the adjacency matrix is equivalent to PCA — at convergence the embedding spans the top-k eigenvectors (Baldi & Hornik 1989). The key differences from direct eigendecomposition:

- **Oja's rule**: online stochastic PCA on rows of A; updates one node at a time.
- **SGD symmetric factorization**: minimize `||A - ZZ^T||_F^2` with negative sampling; the negative samples act as implicit Tikhonov regularization, penalizing large off-diagonal scores.
- **Degree-normalized variant**: factorize `D^{-1/2} A D^{-1/2}` which suppresses high-degree hub contamination (analogous to regularized Laplacian).

## Setup

- Two-community SBM: N=2000, cave=5.0, mu=0.5 (below mu_c ≈ 0.553)
- 30 independent samples
- Methods compared: spectral sign, spectral K-means, Oja's rule, SGD(A), SGD(normed), BP

## Log

### 2026-03-24 — Initial implementation

Implemented three SGD variants in pure NumPy (no PyTorch available):
- `ojas_rule()`: Oja's rule with Gram-Schmidt deflation, 20 epochs, lr=0.005
- `sgd_factorization(A)`: mini-batch SGD on `||A - ZZ^T||^2`, neg_ratio=2, 80 epochs, lr=0.005
- `sgd_factorization(A_norm)`: same on degree-normalized adjacency

Script: `draft-scripts/01_sgd_autoencoder.py`

## Gotchas

- `belief_propagation` module must be imported via sys.path from `libs/BeliefPropagation/`
- igraph SBM API (v1.0.0): `ig.Graph.SBM(pref_matrix, block_sizes)` with row-probability matrix

## Learnings

(to be updated after first run)

## Conclusions

(to be updated after first run)
