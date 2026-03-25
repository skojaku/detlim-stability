"""
03_sign_combination.py

Sign pattern analysis and simple combination algorithms for community detection
near the detectability limit.

Setup: N=2000 nodes, 2 communities (nodes 0-999 = comm 0, 1000-1999 = comm 1)
       cave=5, mu=0.5, 30 samples, 10 eigenvectors per sample
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score
import os

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N = 2000
N_COMM = 1000  # nodes per community
N_SAMPLES = 30
K_EIGVECS = 10  # columns 0..9, sorted by decreasing eigenvalue

# True community membership
membership = np.array([0] * N_COMM + [1] * N_COMM)

# Baselines from CSV
import csv
baselines = {}
with open(os.path.join(DATA_DIR, "baselines.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        baselines[int(row["sample"])] = {
            "nmi_sign_v2": float(row["nmi_sign_v2"]),
            "nmi_kmeans":  float(row["nmi_kmeans"]),
            "nmi_bp":      float(row["nmi_bp"]),
        }

bp_mean  = np.mean([baselines[s]["nmi_bp"]      for s in range(N_SAMPLES)])
sv2_mean = np.mean([baselines[s]["nmi_sign_v2"] for s in range(N_SAMPLES)])
km_mean  = np.mean([baselines[s]["nmi_kmeans"]  for s in range(N_SAMPLES)])

print("=" * 65)
print("BASELINES (mean NMI over 30 samples)")
print(f"  sign(v2):          {sv2_mean:.4f}")
print(f"  K-means:           {km_mean:.4f}")
print(f"  Belief propagation:{bp_mean:.4f}  <-- TARGET")
print("=" * 65)


# ------------------------------------------------------------------
# Helper: align sign of a vector so that mean(v[0:1000]) > mean(v[1000:2000])
# i.e., positive side corresponds to community 0.
# ------------------------------------------------------------------
def align_to_community0(v):
    if np.mean(v[:N_COMM]) < np.mean(v[N_COMM:]):
        return -v
    return v


# ------------------------------------------------------------------
# Helper: align sign of v_k so that Pearson corr with reference is positive.
# ------------------------------------------------------------------
def align_to_reference(v, ref):
    if np.dot(v, ref) < 0:
        return -v
    return v


# ------------------------------------------------------------------
# Load all samples
# ------------------------------------------------------------------
all_vals = []   # shape (N_SAMPLES, K_EIGVECS)
all_V    = []   # shape (N_SAMPLES, N, K_EIGVECS)

for s in range(N_SAMPLES):
    d = np.load(os.path.join(DATA_DIR, f"eigvecs_sample_{s:03d}.npz"))
    all_vals.append(d["vals"])
    all_V.append(d["V"])

all_vals = np.array(all_vals)  # (30, 10)
all_V    = np.array(all_V)     # (30, 2000, 10)


# ==================================================================
# PART 1: Per-eigenvector NMI (full set, k=1..10, i.e. columns 0..9)
# ==================================================================
print("\n" + "=" * 65)
print("PART 1: Per-eigenvector NMI (aligned to community 0)")
print("=" * 65)

nmi_per_eigvec = np.zeros((N_SAMPLES, K_EIGVECS))
for s in range(N_SAMPLES):
    for k in range(K_EIGVECS):
        v = align_to_community0(all_V[s, :, k])
        pred = (v >= 0).astype(int)
        nmi_per_eigvec[s, k] = normalized_mutual_info_score(membership, pred)

mean_nmi = nmi_per_eigvec.mean(axis=0)
std_nmi  = nmi_per_eigvec.std(axis=0)

print(f"{'Eigvec':>8}  {'Mean NMI':>10}  {'Std':>8}  {'> BP?':>6}")
print("-" * 42)
for k in range(K_EIGVECS):
    flag = "YES" if mean_nmi[k] > bp_mean else ""
    print(f"  v{k+1:02d}     {mean_nmi[k]:10.4f}  {std_nmi[k]:8.4f}  {flag:>6}")


# ==================================================================
# PART 2: Weighted sign combination  (k=2..10, columns 1..9)
# ==================================================================
print("\n" + "=" * 65)
print("PART 2: Weighted sign combination (k=2..10)")
print("=" * 65)

nmi_eigenval_weighted = np.zeros(N_SAMPLES)
nmi_uniform_weighted  = np.zeros(N_SAMPLES)

for s in range(N_SAMPLES):
    V   = all_V[s]     # (2000, 10)
    lam = all_vals[s]  # (10,)

    # Reference for sign alignment: v2 (column 1) aligned to community 0
    v2_ref = align_to_community0(V[:, 1].copy())

    score_eig = np.zeros(N)
    score_uni = np.zeros(N)

    for k in range(1, K_EIGVECS):  # columns 1..9 => eigvecs v2..v10
        vk = align_to_reference(V[:, k].copy(), v2_ref)
        w_eig = lam[k]
        w_uni = 1.0
        contrib = np.sign(vk) * np.abs(vk)   # sign * |v_{i,k}|
        score_eig += w_eig * contrib
        score_uni += w_uni * contrib

    pred_eig = (score_eig >= 0).astype(int)
    pred_uni = (score_uni >= 0).astype(int)

    nmi_eigenval_weighted[s] = normalized_mutual_info_score(membership, pred_eig)
    nmi_uniform_weighted[s]  = normalized_mutual_info_score(membership, pred_uni)

print(f"  Eigenvalue-weighted:  mean NMI = {nmi_eigenval_weighted.mean():.4f} ± {nmi_eigenval_weighted.std():.4f}  "
      f"{'> BP!' if nmi_eigenval_weighted.mean() > bp_mean else ''}")
print(f"  Uniform-weighted:     mean NMI = {nmi_uniform_weighted.mean():.4f} ± {nmi_uniform_weighted.std():.4f}  "
      f"{'> BP!' if nmi_uniform_weighted.mean() > bp_mean else ''}")


# ==================================================================
# PART 3: Large-magnitude nodes + neighbor propagation
# ==================================================================
print("\n" + "=" * 65)
print("PART 3: Large-|v2| seed nodes + majority-neighbor propagation")
print("=" * 65)

P_VALUES = [5, 10, 15, 20, 30, 50]
MAX_ITER = 50

nmi_propagation = {p: np.zeros(N_SAMPLES) for p in P_VALUES}

for s in range(N_SAMPLES):
    # Load adjacency matrix (sparse)
    A = sp.load_npz(os.path.join(DATA_DIR, f"net_sample_{s:03d}.npz"))
    A_bool = A.astype(bool)  # treat as unweighted

    v2 = align_to_community0(all_V[s, :, 1].copy())
    magnitude = np.abs(v2)

    for P in P_VALUES:
        n_seed = max(1, int(N * P / 100))
        seed_idx = np.argsort(magnitude)[-n_seed:]  # top P% by |v2|

        labels = np.full(N, -1, dtype=int)
        labels[seed_idx] = (v2[seed_idx] >= 0).astype(int)

        # Propagation
        unseen = np.where(labels == -1)[0]
        for _ in range(MAX_ITER):
            if len(unseen) == 0:
                break
            new_labels = labels.copy()
            changed = False
            for i in unseen:
                # neighbors of node i
                row = A_bool.getrow(i)
                nbrs = row.indices
                if len(nbrs) == 0:
                    continue
                nbr_labels = labels[nbrs]
                known = nbr_labels[nbr_labels >= 0]
                if len(known) == 0:
                    continue
                # majority vote among known neighbors
                n0 = np.sum(known == 0)
                n1 = np.sum(known == 1)
                if n0 > n1:
                    new_labels[i] = 0
                elif n1 > n0:
                    new_labels[i] = 1
                else:
                    new_labels[i] = int(np.random.rand() > 0.5)
                changed = True
            labels = new_labels
            unseen = np.where(labels == -1)[0]
            if not changed:
                break

        # Any still unlabeled -> assign randomly
        if np.any(labels == -1):
            labels[labels == -1] = (np.random.rand(np.sum(labels == -1)) > 0.5).astype(int)

        nmi_propagation[P][s] = normalized_mutual_info_score(membership, labels)

print(f"{'P (%)':>6}  {'Mean NMI':>10}  {'Std':>8}  {'> BP?':>6}")
print("-" * 38)
for P in P_VALUES:
    m = nmi_propagation[P].mean()
    sd = nmi_propagation[P].std()
    flag = "YES" if m > bp_mean else ""
    print(f"  {P:4d}%  {m:10.4f}  {sd:8.4f}  {flag:>6}")


# ==================================================================
# PART 4: Majority vote across eigenvectors (k=2..10)
# ==================================================================
print("\n" + "=" * 65)
print("PART 4: Majority vote across eigenvectors (k=2..10)")
print("=" * 65)

nmi_majority_vote      = np.zeros(N_SAMPLES)
nmi_weighted_majority  = np.zeros(N_SAMPLES)

for s in range(N_SAMPLES):
    V = all_V[s]  # (2000, 10)

    v2_ref = align_to_community0(V[:, 1].copy())

    # Collect sign votes for each node across k=2..10 (columns 1..9)
    votes    = np.zeros(N)        # unweighted count
    w_votes  = np.zeros(N)        # weighted by |v_{i,k}|

    for k in range(1, K_EIGVECS):
        vk = align_to_reference(V[:, k].copy(), v2_ref)
        sign_k = np.sign(vk)      # +1 or -1 per node

        votes   += sign_k
        w_votes += sign_k * np.abs(vk)

    # Positive score => community 0, negative => community 1
    pred_mv = (votes   >= 0).astype(int)
    pred_wv = (w_votes >= 0).astype(int)

    nmi_majority_vote[s]     = normalized_mutual_info_score(membership, pred_mv)
    nmi_weighted_majority[s] = normalized_mutual_info_score(membership, pred_wv)

print(f"  Unweighted majority vote:  mean NMI = {nmi_majority_vote.mean():.4f} ± {nmi_majority_vote.std():.4f}  "
      f"{'> BP!' if nmi_majority_vote.mean() > bp_mean else ''}")
print(f"  |v_ik|-weighted vote:      mean NMI = {nmi_weighted_majority.mean():.4f} ± {nmi_weighted_majority.std():.4f}  "
      f"{'> BP!' if nmi_weighted_majority.mean() > bp_mean else ''}")


# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "=" * 65)
print("SUMMARY: Mean NMI across 30 samples")
print("=" * 65)
print(f"  Baseline sign(v2):              {sv2_mean:.4f}")
print(f"  Baseline K-means:               {km_mean:.4f}")
print(f"  Baseline BP (target):           {bp_mean:.4f}")
print()
print(f"  Best single eigvec (v{np.argmax(mean_nmi)+1}):        {np.max(mean_nmi):.4f}")
print(f"  Eigenvalue-weighted combo:      {nmi_eigenval_weighted.mean():.4f}")
print(f"  Uniform-weighted combo:         {nmi_uniform_weighted.mean():.4f}")
best_P = max(P_VALUES, key=lambda p: nmi_propagation[p].mean())
print(f"  Propagation best (P={best_P}%):    {nmi_propagation[best_P].mean():.4f}")
print(f"  Unweighted majority vote:       {nmi_majority_vote.mean():.4f}")
print(f"  |v_ik|-weighted majority vote:  {nmi_weighted_majority.mean():.4f}")
print()
methods_above_bp = []
if nmi_eigenval_weighted.mean() > bp_mean:
    methods_above_bp.append("Eigenvalue-weighted combo")
if nmi_uniform_weighted.mean() > bp_mean:
    methods_above_bp.append("Uniform-weighted combo")
for P in P_VALUES:
    if nmi_propagation[P].mean() > bp_mean:
        methods_above_bp.append(f"Propagation P={P}%")
if nmi_majority_vote.mean() > bp_mean:
    methods_above_bp.append("Unweighted majority vote")
if nmi_weighted_majority.mean() > bp_mean:
    methods_above_bp.append("|v_ik|-weighted majority vote")
if np.max(mean_nmi) > bp_mean:
    best_k = np.argmax(mean_nmi) + 1
    methods_above_bp.append(f"Single eigvec v{best_k}")

if methods_above_bp:
    print("  Methods EXCEEDING BP:")
    for m in methods_above_bp:
        print(f"    - {m}")
else:
    print(f"  No method exceeds BP NMI = {bp_mean:.4f}")
print("=" * 65)
