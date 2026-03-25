# SGD Eigenvector Solver — Robustness Near Detection Limit

## Question

Can stochastic gradient descent yield a more robust embedding than direct eigendecomposition in sparse SBMs near the detection limit?

**Hypothesis**: SGD's implicit regularization (mini-batch noise, early stopping) filters bulk-eigenvalue contamination that drowns community signal in exact spectral methods.

## Theory

A **linear autoencoder** minimizing MSE on the adjacency matrix is equivalent to PCA — at convergence the embedding spans the top-k eigenvectors (Baldi & Hornik 1989). The key differences from direct eigendecomposition:

- **Oja's rule**: online stochastic PCA on rows of A; updates one node at a time.
- **SGD symmetric factorization**: minimize `||A - ZZ^T||_F^2` with negative sampling.
- **DeepWalk**: random walk + skip-gram SGD; converges to normalized Laplacian eigenvectors (Qiu et al. 2019).
- **Bethe-Hessian**: H(r) = (r²-1)I - rA + D; all noise eigenvalues positive, community signal in negative eigenvalues; provably reaches KS threshold (Saade et al. 2014).

## Setup

- Two-community SBM: N=2000, cave=5.0, mu=0.5 (below mu_c ≈ 0.553)
- 20 independent samples per mu value; mu swept 0→0.7

## Log

### 2026-03-24 — Script 01: Basic SGD methods

Implemented three SGD variants in pure NumPy:
- `ojas_rule()`: Oja's rule with Gram-Schmidt deflation, 20 epochs
- `sgd_factorization(A)`: mini-batch SGD on `||A - ZZ^T||^2`, 80 epochs
- `sgd_factorization(A_norm)`: same on degree-normalized adjacency

**Results at mu=0.5 (30 samples):**

| Method | NMI |
|---|---|
| BP | 0.116 ± 0.063 |
| Spectral (sign v₂) | 0.042 ± 0.037 |
| Oja's rule | 0.002 ± 0.004 |
| SGD factorize A | 0.000 |
| SGD factorize D⁻½AD⁻½ | 0.000 |

**Conclusion**: Hypothesis wrong — naive SGD is far worse than direct eigendecomposition.

### 2026-03-24 — Script 02: Diagnosis

- **Symmetric factorization collapses**: Z initialized near zero; gradient ∝ Z → stuck at init forever. Loss frozen at ~20 (= n_edges × 1²) all epochs.
- **Oja's rule too slow**: NMI 0.001→0.008 over 200 epochs. Near threshold, spectral gap is tiny → convergence rate ∝ gap² per step.
- **Rayleigh quotient SGD bug**: divided by n_edges twice → effective lr ≈ 2×10⁻⁶; QR reset dominated initialization each step.

### 2026-03-24 — Script 03: Mini-batch power iteration sweep

Swept mu 0→0.7 for mini-batch power iteration on A at 4 batch fractions.

Key findings:
- **Full batch beats eigsh at low mu** (strong communities, disconnected blocks): eigsh returns arbitrary rotation of degenerate eigenspace; power iteration's random init naturally picks community-aligned directions.
- **More noise = always worse near threshold**: convergence slower, not better.
- **Crossover at mu ≈ 0.3**: full-batch power iteration worse than Lanczos (eigsh) for mu > 0.3 because Lanczos builds Krylov subspace and handles small spectral gaps efficiently.

### 2026-03-24 — Script 04: Bethe-Hessian MBPI

BH spectral (eigsh with `which='SA'`): **NMI ≈ 0.10–0.19** at mu=0.5, close to BP.

BH power iteration (on -H(r)) fails because:
- Eigenvalues of -H(r): community modes at +2.94, +0.025; but bulk modes at ~−23 (large magnitude)
- Power iteration converges to LARGEST MAGNITUDE, which is the bulk (−23), not community (+2.94)
- Shifted BH power iteration: works but spectral gap ratio = 0.991 after shift → needs ~1000s epochs

**Root cause**: Community eigenvectors are not dominant in ANY simple power iteration on H(r) or -H(r). Lanczos succeeds by building the full Krylov subspace without being dominated by large-magnitude eigenvalues.

### 2026-03-24 — Script 06: Final comparison across mu values

**Results (N=2000, cave=5, 20 samples):**

| mu | BP | BH spectral | NL eigvecs | DeepWalk | Spectral (A) |
|---|---|---|---|---|---|
| 0.00 | 0.827 | **0.968** | 0.261 | — | 0.344 |
| 0.10 | 0.909 | 0.887 | 0.896 | — | 0.769 |
| 0.20 | 0.799 | 0.745 | 0.760 | — | 0.663 |
| 0.30 | 0.633 | 0.564 | 0.577 | — | 0.484 |
| 0.40 | 0.408 | 0.349 | 0.352 | — | 0.183 |
| 0.45 | 0.261 | 0.239 | 0.227 | — | 0.083 |
| 0.50 | 0.118 | **0.108** | 0.077 | 0.052 | 0.028 |
| 0.55 | 0.024 | 0.014 | 0.023 | — | 0.012 |
| 0.60 | 0.008 | 0.005 | 0.007 | — | 0.005 |

## Gotchas

- `belief_propagation` module: import via `sys.path.insert(0, 'libs/BeliefPropagation/')`
- igraph SBM API (v1.0.0): `ig.Graph.SBM(pref_matrix, block_sizes)` with row-probability matrix
- **Double add.at bug**: symmetric A (stored with both i→j and j→i) + `np.add.at(AZ, ei, Z[ej])` already computes A@Z correctly. Adding the symmetric line `np.add.at(AZ, ej, Z[ei])` doubles it to 2*(A@Z).
- `matplotlib.use("TkAgg")` hangs in background tasks without display — use `"Agg"` for scripts run non-interactively.

## Learnings

1. **The operator matters more than the optimizer.** Bethe-Hessian + Lanczos (eigsh SA) achieves NMI ≈ 0.91×BP at mu=0.5. Raw adjacency + Lanczos achieves only 0.24×BP.

2. **SGD does not help near the KS threshold.** The community eigenvectors near threshold are not dominant in any simple power-iteration sense — they are buried in the bulk by magnitude. Neither noise nor implicit regularization from SGD can recover them.

3. **DeepWalk ≈ normalized Laplacian eigvecs at convergence.** DeepWalk SGD (n_walks=20, 5 epochs, mu=0.5) achieved NMI=0.052±0.041 — high variance, between raw spectral (0.028) and NL eigvecs (0.077). With more training it should approach the NL limit.

4. **Lanczos vs power iteration**: Lanczos builds the full Krylov subspace and can extract eigenvectors with tiny spectral gaps. Power iteration needs gap⁻² steps. Near mu_c, gap → 0 → exponentially slow convergence for power iteration.

5. **Degenerate eigenspace at mu=0**: BH spectral (0.968) far outperforms raw spectral (0.344). With two disconnected blocks, A's community eigenvectors are degenerate — eigsh returns arbitrary rotation. BH breaks this degeneracy.

## Conclusions

The hypothesis is **FALSE**: SGD does not provide robustness over direct eigendecomposition near the detection limit.

**Hierarchy near the KS threshold (mu=0.5, cave=5):**
```
BP           0.118   ← oracle (message passing on generative model)
BH spectral  0.108   ← 91% of BP; Bethe-Hessian + Lanczos
NL eigvecs   0.077   ← normalized Laplacian; DeepWalk's theoretical limit
DeepWalk SGD 0.052   ← random walk SGD; approaches NL limit with more training
Spectral (A) 0.028   ← raw adjacency eigvecs; worst
```

**Practical recommendation**: Use the Bethe-Hessian spectral method (`eigsh(H(r), which='SA')` where r=√(mean_degree)) for community detection near the threshold. It closes ~91% of the gap to BP with no model fitting.

**Why SGD can't match Lanczos here**: The community eigenvalue of -H(r) is λ₂≈+2.94, while bulk eigenvalues range to −23. After shifting to make community modes dominant, the remaining spectral gap is only λ₂−λ₃≈0.23, giving a power iteration convergence ratio of ≈0.991 — requiring thousands of epochs. Lanczos bypasses this by constructing the Krylov subspace without being misled by eigenvalue magnitude ordering.
