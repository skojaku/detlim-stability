"""
05_score_regularized.py

SCORE (Spectral Clustering On Ratios-of-Eigenvectors) and
Regularized Spectral Methods for community detection.

Setup: N=2000 nodes, 2 communities (0-999 = comm 0, 1000-1999 = comm 1)
       cave=5.0, mu=0.5, 30 samples
Baselines: sign(v2) NMI~0.047, K-means NMI~0.027, BP NMI~0.112 (target)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import os
import csv

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N = 2000
N_COMM = 1000
N_SAMPLES = 30
CAVE = 5.0

# True community membership
membership = np.array([0] * N_COMM + [1] * N_COMM)

# ------------------------------------------------------------------
# Load baselines
# ------------------------------------------------------------------
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
print(f"  sign(v2):           {sv2_mean:.4f}")
print(f"  K-means:            {km_mean:.4f}")
print(f"  Belief propagation: {bp_mean:.4f}  <-- TARGET")
print("=" * 65)

# ------------------------------------------------------------------
# Load precomputed eigenvectors
# ------------------------------------------------------------------
all_vals = []
all_V    = []

print("\nLoading precomputed eigenvectors...")
for s in range(N_SAMPLES):
    d = np.load(os.path.join(DATA_DIR, f"eigvecs_sample_{s:03d}.npz"))
    all_vals.append(d["vals"])   # (10,) sorted by decreasing eigenvalue
    all_V.append(d["V"])         # (2000, 10)

all_vals = np.array(all_vals)    # (30, 10)
all_V    = np.array(all_V)       # (30, 2000, 10)
print(f"Loaded {N_SAMPLES} samples. V shape: {all_V[0].shape}")


# ------------------------------------------------------------------
# Helper: align sign so community 0 gets mean positive value
# ------------------------------------------------------------------
def align_to_community0(v):
    if np.mean(v[:N_COMM]) < np.mean(v[N_COMM:]):
        return -v
    return v


# ==================================================================
# PART 1: SCORE method (Jin 2015)
#   r_i = v2[i] / v1[i]   (ratio of 2nd to 1st eigenvector)
#   classify by sign(r_i)
# ==================================================================
print("\n" + "=" * 65)
print("PART 1: SCORE (Ratios-of-Eigenvectors, Jin 2015)")
print("  r_i = v2[i] / v1[i];  classify by sign(r_i)")
print("=" * 65)

nmi_score_sign   = np.zeros(N_SAMPLES)
nmi_score_kmeans = np.zeros(N_SAMPLES)

for s in range(N_SAMPLES):
    v1 = all_V[s, :, 0]   # leading eigenvector (community signal)
    v2 = all_V[s, :, 1]   # second eigenvector

    # Avoid division by near-zero: use a small epsilon guard
    eps = 1e-12
    ratio = v2 / (v1 + np.sign(v1 + eps) * eps)

    # Sign classification — align so community 0 gets positive ratio
    pred_sign = (ratio >= 0).astype(int)
    if normalized_mutual_info_score(membership, pred_sign) < \
       normalized_mutual_info_score(membership, 1 - pred_sign):
        pred_sign = 1 - pred_sign
    nmi_score_sign[s] = normalized_mutual_info_score(membership, pred_sign)

    # K-means on ratios
    ratio_col = ratio.reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=s)
    pred_km = km.fit_predict(ratio_col)
    # align
    if normalized_mutual_info_score(membership, pred_km) < \
       normalized_mutual_info_score(membership, 1 - pred_km):
        pred_km = 1 - pred_km
    nmi_score_kmeans[s] = normalized_mutual_info_score(membership, pred_km)

print(f"  SCORE sign(r):  mean NMI = {nmi_score_sign.mean():.4f} ± {nmi_score_sign.std():.4f}"
      f"  {'> BP!' if nmi_score_sign.mean() > bp_mean else ''}")
print(f"  SCORE K-means:  mean NMI = {nmi_score_kmeans.mean():.4f} ± {nmi_score_kmeans.std():.4f}"
      f"  {'> BP!' if nmi_score_kmeans.mean() > bp_mean else ''}")


# ==================================================================
# PART 2: Regularized adjacency  A_tau = A + (tau/N) * 11^T
#   Applied as LinearOperator to avoid N^2 memory.
#   For each tau: top-2 eigvecs, classify by sign of 2nd.
# ==================================================================
print("\n" + "=" * 65)
print("PART 2: Regularized adjacency  A_tau = A + (tau/N)*11^T")
print("  Classify by sign of 2nd eigenvector of A_tau")
print("=" * 65)

TAU_VALUES = [0.0, 0.1, 0.5, 1.0, CAVE, 10.0, 50.0]

nmi_reg = {tau: np.zeros(N_SAMPLES) for tau in TAU_VALUES}

print(f"\nProcessing {N_SAMPLES} samples x {len(TAU_VALUES)} tau values...")
for s in range(N_SAMPLES):
    A = sp.load_npz(os.path.join(DATA_DIR, f"net_sample_{s:03d}.npz")).astype(float)
    A = A.tocsr()

    for tau in TAU_VALUES:
        if tau == 0.0:
            # Use precomputed eigenvectors for tau=0 (A itself)
            v2 = all_V[s, :, 1]
        else:
            # LinearOperator: x -> A x + (tau/N) * sum(x) * 1
            shift = tau / N

            def _matvec(x, _A=A, _shift=shift):
                return _A @ x + _shift * np.sum(x) * np.ones(N)

            A_op = LinearOperator((N, N), matvec=_matvec, dtype=float)
            vals_reg, vecs_reg = eigsh(A_op, k=2, which='LA', tol=1e-6, maxiter=3000)
            # eigsh returns in ascending order; last column is largest eigenvector
            idx_sorted = np.argsort(vals_reg)[::-1]   # descending
            v2 = vecs_reg[:, idx_sorted[1]]            # 2nd largest

        pred = (align_to_community0(v2) >= 0).astype(int)
        nmi_reg[tau][s] = normalized_mutual_info_score(membership, pred)

print(f"\n{'tau':>8}  {'Mean NMI':>10}  {'Std':>8}  {'> BP?':>6}")
print("-" * 40)
for tau in TAU_VALUES:
    m  = nmi_reg[tau].mean()
    sd = nmi_reg[tau].std()
    label = f"{tau:.1f}" if tau != CAVE else f"{tau:.1f} (=cave)"
    flag  = "YES" if m > bp_mean else ""
    print(f"  {label:>12}  {m:10.4f}  {sd:8.4f}  {flag:>6}")


# ==================================================================
# PART 3: Row-normalized adjacency  D^{-1/2} A D^{-1/2}
#   Top-2 eigvecs, classify by sign of 2nd.
# ==================================================================
print("\n" + "=" * 65)
print("PART 3: Symmetric normalized adjacency  D^{-1/2} A D^{-1/2}")
print("  Classify by sign of 2nd eigenvector")
print("=" * 65)

nmi_norm = np.zeros(N_SAMPLES)

for s in range(N_SAMPLES):
    A = sp.load_npz(os.path.join(DATA_DIR, f"net_sample_{s:03d}.npz")).astype(float)
    A = A.tocsr()

    # Degree vector; handle isolated nodes
    deg = np.array(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    # D^{-1/2} A D^{-1/2} as LinearOperator
    def _matvec_norm(x, _A=A, _d=deg_inv_sqrt):
        return _d * (_A @ (_d * x))

    A_norm_op = LinearOperator((N, N), matvec=_matvec_norm, dtype=float)
    vals_n, vecs_n = eigsh(A_norm_op, k=2, which='LA', tol=1e-6, maxiter=3000)

    idx_sorted = np.argsort(vals_n)[::-1]
    v2 = vecs_n[:, idx_sorted[1]]

    pred = (align_to_community0(v2) >= 0).astype(int)
    nmi_norm[s] = normalized_mutual_info_score(membership, pred)

print(f"\n  D^{{-1/2}} A D^{{-1/2}}:  mean NMI = {nmi_norm.mean():.4f} ± {nmi_norm.std():.4f}"
      f"  {'> BP!' if nmi_norm.mean() > bp_mean else ''}")


# ==================================================================
# PART 4: Centered adjacency  A_shift = A - (cave/N) * 11^T
#   Shifts bulk eigenvalues down; use sign of 2nd eigvec.
# ==================================================================
print("\n" + "=" * 65)
print("PART 4: Centered adjacency  A_shift = A - (cave/N)*11^T")
print("  (mean-field bulk shift)")
print("  Classify by sign of 2nd eigenvector")
print("=" * 65)

nmi_shift = np.zeros(N_SAMPLES)
shift_val = CAVE / N   # subtract this from every entry

for s in range(N_SAMPLES):
    A = sp.load_npz(os.path.join(DATA_DIR, f"net_sample_{s:03d}.npz")).astype(float)
    A = A.tocsr()

    def _matvec_shift(x, _A=A, _s=shift_val):
        return _A @ x - _s * np.sum(x) * np.ones(N)

    A_shift_op = LinearOperator((N, N), matvec=_matvec_shift, dtype=float)
    vals_sh, vecs_sh = eigsh(A_shift_op, k=2, which='LA', tol=1e-6, maxiter=3000)

    idx_sorted = np.argsort(vals_sh)[::-1]
    v2 = vecs_sh[:, idx_sorted[1]]

    pred = (align_to_community0(v2) >= 0).astype(int)
    nmi_shift[s] = normalized_mutual_info_score(membership, pred)

print(f"\n  A_shift (cave/N shift):  mean NMI = {nmi_shift.mean():.4f} ± {nmi_shift.std():.4f}"
      f"  {'> BP!' if nmi_shift.mean() > bp_mean else ''}")


# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "=" * 65)
print("SUMMARY: Mean NMI across 30 samples  (BP target = {:.4f})".format(bp_mean))
print("=" * 65)
print(f"  Baseline sign(v2):                 {sv2_mean:.4f}")
print(f"  Baseline K-means (on v2):          {km_mean:.4f}")
print(f"  Baseline BP (target):              {bp_mean:.4f}")
print()
print(f"  SCORE sign(r=v2/v1):               {nmi_score_sign.mean():.4f}  {'> BP!' if nmi_score_sign.mean() > bp_mean else ''}")
print(f"  SCORE K-means on r:                {nmi_score_kmeans.mean():.4f}  {'> BP!' if nmi_score_kmeans.mean() > bp_mean else ''}")
print()
print("  Regularized adjacency (sign of 2nd eigvec):")
for tau in TAU_VALUES:
    label = f"tau={tau:.1f}" + (" (cave)" if tau == CAVE else "")
    m = nmi_reg[tau].mean()
    print(f"    {label:<20} {m:.4f}  {'> BP!' if m > bp_mean else ''}")
print()
print(f"  Normalized adj D^{{-1/2}}AD^{{-1/2}}:   {nmi_norm.mean():.4f}  {'> BP!' if nmi_norm.mean() > bp_mean else ''}")
print(f"  Centered adj (cave/N shift):       {nmi_shift.mean():.4f}  {'> BP!' if nmi_shift.mean() > bp_mean else ''}")
print()

# Collect all methods that beat BP
methods_above_bp = []
if nmi_score_sign.mean()   > bp_mean: methods_above_bp.append(f"SCORE sign(r)            NMI={nmi_score_sign.mean():.4f}")
if nmi_score_kmeans.mean() > bp_mean: methods_above_bp.append(f"SCORE K-means on r       NMI={nmi_score_kmeans.mean():.4f}")
for tau in TAU_VALUES:
    if nmi_reg[tau].mean()  > bp_mean:
        methods_above_bp.append(f"Reg adj tau={tau:<5.1f}         NMI={nmi_reg[tau].mean():.4f}")
if nmi_norm.mean()          > bp_mean: methods_above_bp.append(f"Norm adj D^-1/2 A D^-1/2 NMI={nmi_norm.mean():.4f}")
if nmi_shift.mean()         > bp_mean: methods_above_bp.append(f"Centered adj (cave/N)    NMI={nmi_shift.mean():.4f}")

if methods_above_bp:
    print("  Methods EXCEEDING BP target:")
    for m in methods_above_bp:
        print(f"    - {m}")
else:
    print(f"  No method exceeds BP NMI = {bp_mean:.4f}")

print("=" * 65)
print("Done.")
