"""
02_node_degree_structure.py

Investigates what makes a node have large magnitude in higher eigenvectors.
N=2000, communities: nodes 0-999 = community 0, nodes 1000-1999 = community 1
Eigenvectors: columns of V sorted by decreasing eigenvalue (k=1 is leading)
We analyze k=2..8 (0-indexed columns 1..7)
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from itertools import combinations

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000
# Community assignment: 0-999 = community 0, 1000-1999 = community 1
COMMUNITY = np.array([0] * 1000 + [1] * 1000)

# Eigenvector indices to analyze (0-indexed column in V)
# k=2..8 means columns 1..7
EIG_COLS = list(range(1, 8))   # columns 1,2,3,4,5,6,7  -> k=2..8
EIG_LABELS = [f"k={c+1}" for c in EIG_COLS]  # k=2..8

TOP_FRAC = 0.10  # top 10%
TOP_N = int(N_NODES * TOP_FRAC)  # 200 nodes


# ============================================================
# 1. Degree vs eigenvector magnitude (Spearman correlation)
# ============================================================
print("=" * 60)
print("1. Degree vs |v_i,k| Spearman correlation")
print("=" * 60)

spearman_per_sample = {c: [] for c in EIG_COLS}

for s in range(N_SAMPLES):
    A = sp.load_npz(f"{DATA_DIR}/net_sample_{s:03d}.npz")
    degree = np.array(A.sum(axis=1)).flatten()

    e = np.load(f"{DATA_DIR}/eigvecs_sample_{s:03d}.npz")
    V = e['V']  # shape (2000, 10)

    for c in EIG_COLS:
        mag = np.abs(V[:, c])
        r, _ = spearmanr(degree, mag)
        spearman_per_sample[c].append(r)

print(f"\n{'Eigvec':>8}  {'Mean Spearman r':>16}  {'Std':>8}")
print("-" * 38)
for c in EIG_COLS:
    rs = np.array(spearman_per_sample[c])
    print(f"  k={c+1:>2}    {rs.mean():>+16.4f}  {rs.std():>8.4f}")


# ============================================================
# 2. High-magnitude node community bias
# ============================================================
print()
print("=" * 60)
print("2. Community bias of top-10% magnitude nodes")
print("   (bias = |fraction_community0 - 0.5|, range 0..0.5)")
print("=" * 60)

community_bias = {c: [] for c in EIG_COLS}

for s in range(N_SAMPLES):
    e = np.load(f"{DATA_DIR}/eigvecs_sample_{s:03d}.npz")
    V = e['V']

    for c in EIG_COLS:
        mag = np.abs(V[:, c])
        top_idx = np.argsort(mag)[-TOP_N:]
        frac_c0 = np.mean(COMMUNITY[top_idx] == 0)
        bias = abs(frac_c0 - 0.5)
        community_bias[c].append(bias)

print(f"\n{'Eigvec':>8}  {'Mean bias':>12}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print("-" * 55)
for c in EIG_COLS:
    b = np.array(community_bias[c])
    print(f"  k={c+1:>2}    {b.mean():>12.4f}  {b.std():>8.4f}  {b.min():>8.4f}  {b.max():>8.4f}")


# ============================================================
# 3. Jaccard similarity of top-10% nodes across eigenvectors
#    (only k=2..6, i.e., columns 1..5)
# ============================================================
print()
print("=" * 60)
print("3. Jaccard similarity of top-10% nodes across eigenvectors")
print("   (k=2..6, columns 1..5)")
print("=" * 60)

JAC_COLS = list(range(1, 6))  # columns 1..5 -> k=2..6
JAC_LABELS = [f"k={c+1}" for c in JAC_COLS]
n_jac = len(JAC_COLS)
jaccard_matrices = []

for s in range(N_SAMPLES):
    e = np.load(f"{DATA_DIR}/eigvecs_sample_{s:03d}.npz")
    V = e['V']

    # Compute top-10% sets for each eigvec
    top_sets = {}
    for c in JAC_COLS:
        mag = np.abs(V[:, c])
        top_idx = set(np.argsort(mag)[-TOP_N:])
        top_sets[c] = top_idx

    # Jaccard matrix
    J = np.zeros((n_jac, n_jac))
    for i, ci in enumerate(JAC_COLS):
        for j, cj in enumerate(JAC_COLS):
            if i == j:
                J[i, j] = 1.0
            else:
                inter = len(top_sets[ci] & top_sets[cj])
                union = len(top_sets[ci] | top_sets[cj])
                J[i, j] = inter / union if union > 0 else 0.0
    jaccard_matrices.append(J)

mean_J = np.mean(jaccard_matrices, axis=0)

print(f"\nMean Jaccard matrix (rows/cols = {JAC_LABELS}):")
header = f"{'':>6}" + "".join(f"{lb:>8}" for lb in JAC_LABELS)
print(header)
print("-" * (6 + 8 * n_jac))
for i, lb in enumerate(JAC_LABELS):
    row = f"{lb:>6}" + "".join(f"{mean_J[i, j]:>8.4f}" for j in range(n_jac))
    print(row)

# Also print off-diagonal summary
off_diag = mean_J[np.tril_indices(n_jac, k=-1)]
print(f"\nOff-diagonal Jaccard (mean ± std): {off_diag.mean():.4f} ± {off_diag.std():.4f}")
print(f"Range: {off_diag.min():.4f} .. {off_diag.max():.4f}")


# ============================================================
# 4. Degree of large-magnitude nodes vs all nodes
# ============================================================
print()
print("=" * 60)
print("4. Mean degree: top-10% magnitude nodes vs all nodes")
print("   (eigenvectors k=2, 3, 5, 8)")
print("=" * 60)

CHECK_COLS = [1, 2, 4, 7]  # 0-indexed -> k=2, 3, 5, 8
CHECK_LABELS = [f"k={c+1}" for c in CHECK_COLS]

top_degree_stats = {c: [] for c in CHECK_COLS}
all_degree_stats = []

for s in range(N_SAMPLES):
    A = sp.load_npz(f"{DATA_DIR}/net_sample_{s:03d}.npz")
    degree = np.array(A.sum(axis=1)).flatten()
    all_degree_stats.append(degree.mean())

    e = np.load(f"{DATA_DIR}/eigvecs_sample_{s:03d}.npz")
    V = e['V']

    for c in CHECK_COLS:
        mag = np.abs(V[:, c])
        top_idx = np.argsort(mag)[-TOP_N:]
        top_degree_stats[c].append(degree[top_idx].mean())

all_deg_mean = np.mean(all_degree_stats)
all_deg_std = np.std(all_degree_stats)

print(f"\n{'Eigvec':>8}  {'Mean deg (top10%)':>20}  {'Std':>8}  {'vs all nodes':>14}")
print(f"{'All nodes':>8}  {all_deg_mean:>20.3f}  {all_deg_std:>8.3f}  {'(reference)':>14}")
print("-" * 60)
for c, lb in zip(CHECK_COLS, CHECK_LABELS):
    d = np.array(top_degree_stats[c])
    ratio = d.mean() / all_deg_mean
    print(f"  {lb:>6}  {d.mean():>20.3f}  {d.std():>8.3f}  {'ratio':>6} {ratio:>6.3f}x")

print()
print("Done.")
