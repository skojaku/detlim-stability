"""
Bethe Hessian Spectral Clustering
N=2000, 2 communities (0-999 = comm 0, 1000-1999 = comm 1)
cave=5.0, mu=0.5, 30 samples

Baseline NMI values from data/baselines.csv:
  sign(v2):  ~0.047
  K-means:   ~0.026
  BP:        ~0.112
"""

import numpy as np
import os
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000
CAVE = 5.0
R_OPT = np.sqrt(CAVE)  # ~2.236

# True community labels
true_labels = np.array([0] * 1000 + [1] * 1000)


def load_network(sample_idx):
    path = os.path.join(DATA_DIR, f"net_sample_{sample_idx:03d}.npz")
    return sp.load_npz(path)


def build_bethe_hessian(A, r):
    """H(r) = (r^2 - 1)*I - r*A + D"""
    N = A.shape[0]
    D_diag = np.array(A.sum(axis=1)).flatten()
    H = (r**2 - 1) * sp.eye(N) - r * A + sp.diags(D_diag)
    return H


def classify_sign(v):
    """Classify nodes by sign of eigenvector."""
    return (v >= 0).astype(int)


def compute_nmi(pred, true):
    return nmi_score(true, pred, average_method="arithmetic")


# ============================================================
# Load baselines
# ============================================================
import csv

baseline_sign_v2 = []
baseline_bp = []
with open(os.path.join(DATA_DIR, "baselines.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        baseline_sign_v2.append(float(row["nmi_sign_v2"]))
        baseline_bp.append(float(row["nmi_bp"]))

print("=" * 65)
print("BASELINES (from baselines.csv)")
print("=" * 65)
print(f"  sign(v2)  mean NMI = {np.mean(baseline_sign_v2):.4f}  (std {np.std(baseline_sign_v2):.4f})")
print(f"  BP        mean NMI = {np.mean(baseline_bp):.4f}  (std {np.std(baseline_bp):.4f})")
print()

# ============================================================
# 1. Bethe Hessian at r = sqrt(cave)
# ============================================================
print("=" * 65)
print(f"1. BETHE HESSIAN at r = sqrt(cave) = {R_OPT:.4f}")
print("   Using sign of eigvec for most negative eigenvalue")
print("=" * 65)

nmi_bh_sign = []

for s in range(N_SAMPLES):
    A = load_network(s)
    H = build_bethe_hessian(A, R_OPT)
    # k=2 to get the smallest two eigenvalues (most negative)
    vals, vecs = eigsh(H, k=2, which="SA")
    # Sort by ascending eigenvalue
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    # Community vector: eigvec with most negative eigenvalue
    v_comm = vecs[:, 0]
    pred = classify_sign(v_comm)
    nmi = compute_nmi(pred, true_labels)
    nmi_bh_sign.append(nmi)
    if s < 5 or s == N_SAMPLES - 1:
        print(f"  sample {s:02d}: eigenvalues = {vals[0]:.4f}, {vals[1]:.4f}  |  NMI = {nmi:.4f}")

print(f"\n  Mean NMI (BH sign, r=sqrt(cave)) = {np.mean(nmi_bh_sign):.4f}  "
      f"(std {np.std(nmi_bh_sign):.4f})")
print()

# ============================================================
# 2. Sweep over r values
# ============================================================
print("=" * 65)
print("2. SWEEP OVER r VALUES")
print("   Using sign of second-smallest eigvec of H(r)")
print("=" * 65)

r_values = [1.0, 1.5, 2.0, 2.236, 2.5, 3.0]
r_nmi_results = {}

print(f"\n{'r':>8}  {'mean NMI':>10}  {'std NMI':>9}  {'vs BP':>8}")
print("-" * 45)

for r in r_values:
    nmis = []
    for s in range(N_SAMPLES):
        A = load_network(s)
        H = build_bethe_hessian(A, r)
        vals, vecs = eigsh(H, k=2, which="SA")
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        # Use second-smallest eigvec (index 1) which changes sign across communities
        # If only 1 negative eigenvalue exists, use the most negative one (index 0)
        n_negative = np.sum(vals < 0)
        if n_negative >= 2:
            v_comm = vecs[:, 1]
        else:
            v_comm = vecs[:, 0]
        pred = classify_sign(v_comm)
        nmi = compute_nmi(pred, true_labels)
        nmis.append(nmi)
    r_nmi_results[r] = nmis
    mean_nmi = np.mean(nmis)
    std_nmi = np.std(nmis)
    bp_diff = mean_nmi - np.mean(baseline_bp)
    marker = " <-- optimal r" if abs(r - R_OPT) < 0.01 else ""
    print(f"{r:>8.3f}  {mean_nmi:>10.4f}  {std_nmi:>9.4f}  {bp_diff:>+8.4f}{marker}")

print()

# Also try: for each r, use eigvec with MOST negative eigenvalue (index 0)
print("  [Cross-check: using most-negative eigvec (index 0) for each r]")
print(f"  {'r':>8}  {'mean NMI':>10}")
print("  " + "-" * 22)
for r in r_values:
    nmis_v0 = []
    for s in range(N_SAMPLES):
        A = load_network(s)
        H = build_bethe_hessian(A, r)
        vals, vecs = eigsh(H, k=2, which="SA")
        order = np.argsort(vals)
        vecs = vecs[:, order]
        v_comm = vecs[:, 0]
        pred = classify_sign(v_comm)
        nmi = compute_nmi(pred, true_labels)
        nmis_v0.append(nmi)
    print(f"  {r:>8.3f}  {np.mean(nmis_v0):>10.4f}")

print()

# ============================================================
# 3. Bethe Hessian + K-means (top-3 smallest eigvecs)
# ============================================================
print("=" * 65)
print(f"3. BETHE HESSIAN + K-MEANS (r = {R_OPT:.4f}, k=3 eigvecs)")
print("=" * 65)

nmi_bh_kmeans = []

for s in range(N_SAMPLES):
    A = load_network(s)
    H = build_bethe_hessian(A, R_OPT)
    vals, vecs = eigsh(H, k=3, which="SA")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    # Use top-3 smallest eigenvectors as features
    features = vecs  # shape (N, 3)
    km = KMeans(n_clusters=2, n_init=10, random_state=s)
    pred = km.fit_predict(features)
    nmi = compute_nmi(pred, true_labels)
    nmi_bh_kmeans.append(nmi)
    if s < 5 or s == N_SAMPLES - 1:
        n_neg = np.sum(vals < 0)
        print(f"  sample {s:02d}: eigenvalues = {vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}  "
              f"(neg={n_neg})  |  NMI = {nmi:.4f}")

print(f"\n  Mean NMI (BH + KMeans, r=sqrt(cave)) = {np.mean(nmi_bh_kmeans):.4f}  "
      f"(std {np.std(nmi_bh_kmeans):.4f})")
print()

# ============================================================
# 4. Comparison Summary
# ============================================================
print("=" * 65)
print("4. COMPARISON SUMMARY")
print("=" * 65)

best_r = max(r_values, key=lambda r: np.mean(r_nmi_results[r]))
best_r_nmi = np.mean(r_nmi_results[best_r])

print()
print(f"  {'Method':<35} {'Mean NMI':>10}  {'Std NMI':>9}")
print("  " + "-" * 58)
print(f"  {'Baseline: sign(v2) [adjacency]':<35} {np.mean(baseline_sign_v2):>10.4f}  {np.std(baseline_sign_v2):>9.4f}")
print(f"  {'Baseline: Belief Propagation':<35} {np.mean(baseline_bp):>10.4f}  {np.std(baseline_bp):>9.4f}")
bh_sign_label = f"BH sign (r=sqrt(cave)={R_OPT:.3f})"
bh_km_label = "BH + KMeans (r=sqrt(cave), 3 vecs)"
bh_best_label = f"BH sign (best r={best_r:.3f})"
print(f"  {bh_sign_label:<35} {np.mean(nmi_bh_sign):>10.4f}  {np.std(nmi_bh_sign):>9.4f}")
print(f"  {bh_km_label:<35} {np.mean(nmi_bh_kmeans):>10.4f}  {np.std(nmi_bh_kmeans):>9.4f}")
print(f"  {bh_best_label:<35} {best_r_nmi:>10.4f}  {np.std(r_nmi_results[best_r]):>9.4f}")

print()
print("  r-sweep results:")
for r in r_values:
    mean_nmi = np.mean(r_nmi_results[r])
    marker = " *" if r == best_r else ""
    print(f"    r={r:.3f}  mean NMI = {mean_nmi:.4f}{marker}")

print()
print("  Note: BP NMI target ~ 0.112 (from baselines.csv)")
print()
print("Done.")
