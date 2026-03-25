"""
Localization and Community Correlation Analysis for Eigenvectors
N=2000, 2 communities (0-999 = comm 0, 1000-1999 = comm 1)
cave=5.0, mu=0.5, 30 samples
"""

import numpy as np
from itertools import combinations
import os

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000
N_EIGVECS = 10
THRESHOLDS = [5, 10, 20]  # percent

# True community labels: 0-999 = comm 0, 1000-1999 = comm 1
true_labels = np.array([0] * 1000 + [1] * 1000)


def load_eigvecs(sample_idx):
    path = os.path.join(DATA_DIR, f"eigvecs_sample_{sample_idx:03d}.npz")
    d = np.load(path)
    return d["vals"], d["V"]  # (10,), (2000, 10)


def compute_ipr(v):
    """IPR = sum(v_i^4). v should be normalized."""
    return np.sum(v**4)


def compute_purity(node_indices, true_labels):
    """Fraction of nodes belonging to majority community."""
    labels = true_labels[node_indices]
    counts = np.bincount(labels, minlength=2)
    return counts.max() / len(node_indices)


def compute_nmi(pred_labels, true_labels_subset):
    """
    NMI between predicted labels (signs) and true community labels
    for a subset of nodes.
    Uses manual computation to avoid sklearn dependency.
    """
    from math import log

    n = len(pred_labels)
    if n == 0:
        return 0.0

    # Map pred_labels to {0,1} (they are +1/-1 signs)
    pred_binary = (np.array(pred_labels) > 0).astype(int)
    true_binary = np.array(true_labels_subset)

    # Contingency table
    classes_pred = [0, 1]
    classes_true = [0, 1]
    contingency = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            contingency[i, j] = np.sum((pred_binary == i) & (true_binary == j))

    # Marginals
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    total = contingency.sum()

    if total == 0:
        return 0.0

    # Mutual information
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if contingency[i, j] > 0 and row_sums[i] > 0 and col_sums[j] > 0:
                mi += (contingency[i, j] / total) * log(
                    (contingency[i, j] * total) / (row_sums[i] * col_sums[j])
                )

    # Entropies
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


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ============================================================
# Load all eigenvectors
# ============================================================
all_V = []  # list of (2000, 10) arrays
all_vals = []

print("Loading eigenvectors from 30 samples...")
for s in range(N_SAMPLES):
    vals, V = load_eigvecs(s)
    all_V.append(V)
    all_vals.append(vals)

print(f"Loaded {N_SAMPLES} samples. V shape: {all_V[0].shape}")
print()

# ============================================================
# 1. IPR Analysis
# ============================================================
print("=" * 60)
print("1. INVERSE PARTICIPATION RATIO (IPR)")
print("   Bulk random: IPR ~ 3/N = {:.5f}".format(3 / N_NODES))
print("=" * 60)

ipr_all = np.zeros((N_SAMPLES, N_EIGVECS))
for s in range(N_SAMPLES):
    for k in range(N_EIGVECS):
        v = all_V[s][:, k]
        # Ensure normalized
        v = v / np.linalg.norm(v)
        ipr_all[s, k] = compute_ipr(v)

ipr_mean = ipr_all.mean(axis=0)
ipr_std = ipr_all.std(axis=0)

print(f"{'k':>4} {'mean IPR':>12} {'std IPR':>12} {'ratio to bulk':>15}")
print("-" * 45)
bulk_ipr = 3.0 / N_NODES
for k in range(N_EIGVECS):
    ratio = ipr_mean[k] / bulk_ipr
    print(f"{k+1:>4} {ipr_mean[k]:>12.6f} {ipr_std[k]:>12.6f} {ratio:>15.2f}x")

print()

# ============================================================
# 2. Community Purity of Large-Magnitude Nodes
# ============================================================
print("=" * 60)
print("2. COMMUNITY PURITY OF LARGE-MAGNITUDE NODES (k=2..10)")
print("=" * 60)

purity_all = {}  # (k, X) -> list of purities across samples
for k in range(1, N_EIGVECS):  # k=1 is index 1, eigvec 2
    for X in THRESHOLDS:
        purity_all[(k, X)] = []

for s in range(N_SAMPLES):
    V = all_V[s]
    for k in range(1, N_EIGVECS):  # eigvec index 1..9 = k=2..10
        v = V[:, k]
        abs_v = np.abs(v)
        for X in THRESHOLDS:
            n_top = max(1, int(N_NODES * X / 100))
            top_indices = np.argsort(abs_v)[-n_top:]
            purity = compute_purity(top_indices, true_labels)
            purity_all[(k, X)].append(purity)

print(f"\n{'k':>4} {'X%':>5}", end="")
for X in THRESHOLDS:
    print(f"  purity@{X:2d}%", end="")
print()
print("-" * (4 + 5 + len(THRESHOLDS) * 13))

for k in range(1, N_EIGVECS):
    eigvec_num = k + 1
    print(f"{eigvec_num:>4}", end="")
    print(f"{'':>5}", end="")
    for X in THRESHOLDS:
        purities = purity_all[(k, X)]
        mean_p = np.mean(purities)
        print(f"  {mean_p:>8.4f}   ", end="")
    print()

# Also print with std
print()
print("Purity mean ± std:")
print(f"{'k':>4}", end="")
for X in THRESHOLDS:
    print(f"  {'@'+str(X)+'%':>14}", end="")
print()
for k in range(1, N_EIGVECS):
    eigvec_num = k + 1
    print(f"{eigvec_num:>4}", end="")
    for X in THRESHOLDS:
        purities = purity_all[(k, X)]
        mean_p = np.mean(purities)
        std_p = np.std(purities)
        print(f"  {mean_p:.3f}±{std_p:.3f}  ", end="")
    print()

print()

# ============================================================
# 3. NMI of Large-Magnitude Subsets
# ============================================================
print("=" * 60)
print("3. NMI (sign(v_i) vs true label) FOR TOP-X% NODES (k=2..10)")
print("=" * 60)

nmi_all = {}  # (k, X) -> list of NMIs
for k in range(1, N_EIGVECS):
    for X in THRESHOLDS:
        nmi_all[(k, X)] = []

for s in range(N_SAMPLES):
    V = all_V[s]
    for k in range(1, N_EIGVECS):
        v = V[:, k]
        abs_v = np.abs(v)
        for X in THRESHOLDS:
            n_top = max(1, int(N_NODES * X / 100))
            top_indices = np.argsort(abs_v)[-n_top:]
            pred_labels = np.sign(v[top_indices])
            true_subset = true_labels[top_indices]
            nmi = compute_nmi(pred_labels, true_subset)
            nmi_all[(k, X)].append(nmi)

print(f"\n{'k':>4}", end="")
for X in THRESHOLDS:
    print(f"  {'NMI@'+str(X)+'%':>12}", end="")
print()
print("-" * (4 + len(THRESHOLDS) * 14))

for k in range(1, N_EIGVECS):
    eigvec_num = k + 1
    print(f"{eigvec_num:>4}", end="")
    for X in THRESHOLDS:
        nmis = nmi_all[(k, X)]
        mean_nmi = np.mean(nmis)
        std_nmi = np.std(nmis)
        print(f"  {mean_nmi:.3f}±{std_nmi:.3f} ", end="")
    print()

# Best NMI per threshold
print()
print("Best NMI per threshold (eigvec with highest mean NMI):")
for X in THRESHOLDS:
    best_k = max(range(1, N_EIGVECS), key=lambda k: np.mean(nmi_all[(k, X)]))
    best_nmi = np.mean(nmi_all[(best_k, X)])
    print(f"  @{X:2d}%: eigvec {best_k+1} -> NMI = {best_nmi:.4f}")

print()

# ============================================================
# 4. Cross-Sample Consistency (Jaccard of Top-5% Node Sets)
# ============================================================
print("=" * 60)
print("4. CROSS-SAMPLE JACCARD SIMILARITY OF TOP-5% NODES (k=2..5)")
print("=" * 60)

X_jaccard = 5
n_top_j = max(1, int(N_NODES * X_jaccard / 100))

# Expected chance Jaccard: if two random sets of size m out of N nodes
# E[Jaccard] = m / (2N - m) for random sets
chance_jaccard = n_top_j / (2 * N_NODES - n_top_j)
print(f"\nTop-{X_jaccard}% = {n_top_j} nodes out of {N_NODES}")
print(f"Expected chance Jaccard ~ {chance_jaccard:.4f}")
print()

print(f"{'k':>4} {'mean Jaccard':>14} {'std Jaccard':>12} {'ratio to chance':>16}")
print("-" * 50)

for k in range(1, 5):  # eigvecs 2-5 = indices 1-4
    eigvec_num = k + 1
    # Collect top-X% node sets for each sample
    top_sets = []
    for s in range(N_SAMPLES):
        v = all_V[s][:, k]
        abs_v = np.abs(v)
        top_indices = frozenset(np.argsort(abs_v)[-n_top_j:])
        top_sets.append(top_indices)

    # Compute pairwise Jaccard for all pairs
    jaccards = []
    for i, j in combinations(range(N_SAMPLES), 2):
        jaccards.append(jaccard(top_sets[i], top_sets[j]))

    mean_j = np.mean(jaccards)
    std_j = np.std(jaccards)
    ratio = mean_j / chance_jaccard if chance_jaccard > 0 else float("inf")
    print(f"{eigvec_num:>4} {mean_j:>14.4f} {std_j:>12.4f} {ratio:>16.2f}x")

print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nBulk IPR reference: {bulk_ipr:.5f} (3/N for N={N_NODES})")
print(f"\nEigvec 1 (leading) IPR: {ipr_mean[0]:.6f} ({ipr_mean[0]/bulk_ipr:.1f}x bulk)")
print(f"Eigvec 2 IPR:           {ipr_mean[1]:.6f} ({ipr_mean[1]/bulk_ipr:.1f}x bulk)")
print(f"Eigvec 3 IPR:           {ipr_mean[2]:.6f} ({ipr_mean[2]/bulk_ipr:.1f}x bulk)")
print(f"Eigvec 10 IPR:          {ipr_mean[9]:.6f} ({ipr_mean[9]/bulk_ipr:.1f}x bulk)")

print(f"\nHighest NMI (top-5%): eigvec {max(range(1,N_EIGVECS), key=lambda k: np.mean(nmi_all[(k,5)]))+1}")
print(f"Highest NMI (top-10%): eigvec {max(range(1,N_EIGVECS), key=lambda k: np.mean(nmi_all[(k,10)]))+1}")
print(f"Highest NMI (top-20%): eigvec {max(range(1,N_EIGVECS), key=lambda k: np.mean(nmi_all[(k,20)]))+1}")

print("\nDone.")
