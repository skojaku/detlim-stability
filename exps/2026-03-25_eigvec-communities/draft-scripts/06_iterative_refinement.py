"""
Iterative Refinement and Graph-Diffusion Methods for Community Detection
N=2000, 2 communities (0-999 = comm 0, 1000-1999 = comm 1)
cave=5.0, mu=0.5, 30 samples

Baselines: sign(v2) NMI~0.047, K-means NMI~0.027, BP NMI~0.112
"""

import numpy as np
import scipy.sparse as sp
import os
from math import log

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000

# True community labels: 0-999 = comm 0, 1000-1999 = comm 1
true_labels = np.array([0] * 1000 + [1] * 1000)


def load_net(sample_idx):
    path = os.path.join(DATA_DIR, f"net_sample_{sample_idx:03d}.npz")
    return sp.load_npz(path)


def load_eigvecs(sample_idx):
    path = os.path.join(DATA_DIR, f"eigvecs_sample_{sample_idx:03d}.npz")
    d = np.load(path)
    return d["vals"], d["V"]  # (10,), (2000, 10)


def compute_nmi(pred_labels, true_lab):
    """NMI between predicted and true labels (both {0,1})."""
    pred_binary = (np.array(pred_labels) >= 0).astype(int)
    true_binary = np.array(true_lab)

    n = len(pred_binary)
    contingency = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            contingency[i, j] = np.sum((pred_binary == i) & (true_binary == j))

    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    total = contingency.sum()

    if total == 0:
        return 0.0

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if contingency[i, j] > 0 and row_sums[i] > 0 and col_sums[j] > 0:
                mi += (contingency[i, j] / total) * log(
                    (contingency[i, j] * total) / (row_sums[i] * col_sums[j])
                )

    def entropy(probs):
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * log(p)
        return h

    h_pred = entropy(row_sums / total)
    h_true = entropy(col_sums / total)
    denom = (h_pred + h_true) / 2.0
    if denom == 0:
        return 0.0
    return mi / denom


# ============================================================
# 1. Soft Label Propagation from Eigvec Seeds
# ============================================================
print("=" * 65)
print("1. SOFT LABEL PROPAGATION (Personalized PageRank-like)")
print("   f_new = alpha * A @ f + (1-alpha) * f_init,  then normalize")
print("   Classify by sign(f_converged), 20 iterations")
print("=" * 65)

alphas = [0.5, 0.8, 0.9]
nmi_lp = {a: [] for a in alphas}

for s in range(N_SAMPLES):
    A = load_net(s).astype(float)
    vals, V = load_eigvecs(s)
    v2 = V[:, 1]  # second eigenvector (index 1, largest is index 0)

    # Row-normalize A for propagation (use degree normalization)
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = sp.diags(deg_inv)
    A_norm = D_inv @ A  # row-stochastic

    f_init = v2.copy()
    norm = np.linalg.norm(f_init)
    if norm > 0:
        f_init = f_init / norm

    for alpha in alphas:
        f = f_init.copy()
        for _ in range(20):
            f_new = alpha * (A_norm @ f) + (1 - alpha) * f_init
            norm = np.linalg.norm(f_new)
            if norm > 0:
                f_new = f_new / norm
            f = f_new
        nmi = compute_nmi(f, true_labels)
        nmi_lp[alpha].append(nmi)

print(f"\n{'alpha':>8}  {'mean NMI':>10}  {'std NMI':>10}")
print("-" * 35)
for alpha in alphas:
    arr = np.array(nmi_lp[alpha])
    print(f"{alpha:>8.1f}  {arr.mean():>10.4f}  {arr.std():>10.4f}")

print()


# ============================================================
# 2. Power Iteration on Modified Adjacency (Deflated)
# ============================================================
print("=" * 65)
print("2. POWER ITERATION WITH DEFLATION OF LEADING EIGENVALUE")
print("   x = A @ x - lambda1 * dot(v1, x) * v1,  x /= ||x||")
print("   Starting from x_0 = v2 (second eigvec)")
print("=" * 65)

nmi_power = {it: [] for it in range(1, 21)}

for s in range(N_SAMPLES):
    A = load_net(s).astype(float)
    vals, V = load_eigvecs(s)
    v1 = V[:, 0]  # leading eigvec
    v2 = V[:, 1]  # second eigvec
    lambda1 = vals[0]

    # Normalize v1
    v1 = v1 / np.linalg.norm(v1)

    x = v2.copy()
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm

    for it in range(1, 21):
        x_new = A @ x - lambda1 * np.dot(v1, x) * v1
        norm = np.linalg.norm(x_new)
        if norm > 0:
            x_new = x_new / norm
        x = x_new
        nmi = compute_nmi(x, true_labels)
        nmi_power[it].append(nmi)

print(f"\n{'iter':>6}  {'mean NMI':>10}  {'std NMI':>10}")
print("-" * 32)
for it in range(1, 21):
    arr = np.array(nmi_power[it])
    print(f"{it:>6}  {arr.mean():>10.4f}  {arr.std():>10.4f}")

print()


# ============================================================
# 3. Spectral Clustering with Warm-Started Lanczos on A^2
# ============================================================
print("=" * 65)
print("3. WARM-STARTED LANCZOS ON A^2 (v2 as starting vector)")
print("   Apply eigsh to A^2 with v0=v2; classify by sign of result")
print("=" * 65)

from scipy.sparse.linalg import eigsh, LinearOperator

nmi_lanczos = []

for s in range(N_SAMPLES):
    A = load_net(s).astype(float)
    vals, V = load_eigvecs(s)
    v2 = V[:, 1]

    # Build A^2 as a LinearOperator to avoid materializing dense matrix
    n = A.shape[0]

    def matvec_A2(x, _A=A):
        return _A @ (_A @ x)

    A2_op = LinearOperator((n, n), matvec=matvec_A2, dtype=float)

    v0 = v2.copy()
    norm = np.linalg.norm(v0)
    if norm > 0:
        v0 = v0 / norm

    try:
        # Request 2 eigenvalues: leading (lambda1^2) and second (lambda2^2)
        evals, evecs = eigsh(A2_op, k=2, v0=v0, which="LM", tol=1e-6, maxiter=500)
        # Sort by descending eigenvalue
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        refined_v = evecs[:, 1]  # second eigenvector of A^2
        nmi = compute_nmi(refined_v, true_labels)
    except Exception as e:
        nmi = float("nan")
    nmi_lanczos.append(nmi)

arr = np.array(nmi_lanczos)
valid = arr[~np.isnan(arr)]
print(f"\n  mean NMI = {valid.mean():.4f}  std = {valid.std():.4f}  (n={len(valid)}/{N_SAMPLES})")
print()


# ============================================================
# 4. Walk-Based Community Detection: A^t @ v2
# ============================================================
print("=" * 65)
print("4. WALK-BASED AMPLIFICATION: A^t @ v2, classify by sign")
print("   t = 2, 3, 4")
print("=" * 65)

t_values = [2, 3, 4]
nmi_walk = {t: [] for t in t_values}

for s in range(N_SAMPLES):
    A = load_net(s).astype(float)
    vals, V = load_eigvecs(s)
    v2 = V[:, 1]

    x = v2.copy()
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm

    for step in range(1, max(t_values) + 1):
        x = A @ x
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        if step in t_values:
            nmi = compute_nmi(x, true_labels)
            nmi_walk[step].append(nmi)

print(f"\n{'t':>4}  {'mean NMI':>10}  {'std NMI':>10}")
print("-" * 30)
for t in t_values:
    arr = np.array(nmi_walk[t])
    print(f"{t:>4}  {arr.mean():>10.4f}  {arr.std():>10.4f}")

print()

# ============================================================
# Summary
# ============================================================
print("=" * 65)
print("SUMMARY  (baselines: sign(v2)~0.047, K-means~0.027, BP~0.112)")
print("=" * 65)
print(f"\n{'Method':<45}  {'mean NMI':>10}")
print("-" * 60)
for alpha in alphas:
    arr = np.array(nmi_lp[alpha])
    print(f"  Label propagation (alpha={alpha:.1f})               {arr.mean():>10.4f}")
best_power_iter = max(nmi_power, key=lambda it: np.mean(nmi_power[it]))
arr_best = np.array(nmi_power[best_power_iter])
print(f"  Deflated power iteration (best iter={best_power_iter})       {arr_best.mean():>10.4f}")
arr_lanczos = np.array(nmi_lanczos)
valid_lanczos = arr_lanczos[~np.isnan(arr_lanczos)]
print(f"  Warm Lanczos on A^2                          {valid_lanczos.mean():>10.4f}")
for t in t_values:
    arr = np.array(nmi_walk[t])
    print(f"  Walk amplification (t={t})                     {arr.mean():>10.4f}")
print()
