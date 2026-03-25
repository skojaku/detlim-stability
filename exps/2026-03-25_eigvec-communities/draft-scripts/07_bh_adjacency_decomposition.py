"""
07_bh_adjacency_decomposition.py

Relationship Between Bethe Hessian and Adjacency Eigenvectors.
N=2000, 2 communities (0-999=comm0, 1000-1999=comm1), cave=5.0, mu=0.5, 30 samples.

Goal: Understand what the BH community eigenvector IS in terms of adjacency eigenvectors.
"""

import numpy as np
import os
import scipy.sparse as sp
from scipy.sparse import eye, diags
from scipy.sparse.linalg import eigsh
from sklearn.metrics import normalized_mutual_info_score as nmi_score

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000
CAVE = 5.0
R = np.sqrt(CAVE)  # ~2.236
K_ADJ = 20         # number of adjacency eigvecs to use
K_BH = 4           # number of BH eigvecs to compute

# True community labels
true_labels = np.array([0] * 1000 + [1] * 1000)


def load_network(sample_idx):
    path = os.path.join(DATA_DIR, f"net_sample_{sample_idx:03d}.npz")
    return sp.load_npz(path)


def nmi(pred, true):
    return nmi_score(true, pred, average_method="arithmetic")


def classify_sign(v):
    return (v >= 0).astype(int)


# ============================================================
# Accumulators across samples
# ============================================================
# 1. Projection coefficients: |c_k|^2 for k=1..K_ADJ
overlap_sq = np.zeros(K_ADJ)          # mean |c_k|^2 over samples

# 2. Residual norms
residual_norms = []

# 3. Degree-correction decomposition for v_2..v_10
K_DC = 10  # analyze degree correction for first 10 adj eigvecs
dc_overlap_sq = np.zeros((K_DC, K_ADJ))  # dc_overlap_sq[j, k] = |<D@v_j - d_avg*v_j, v_k>|^2

# 4. Coefficient on v_2
coeff_v2 = []          # c_2 = dot(u_BH, v_2)
coeff_other_sum = []   # sum_{k != 2} |c_k|^2

# 5. NMI comparisons
nmi_bh_list = []
nmi_v2_list = []
nmi_recon_list = []

print("=" * 70)
print(f"BH-ADJACENCY DECOMPOSITION  (r = sqrt({CAVE}) = {R:.4f})")
print(f"N={N_NODES}, K_adj={K_ADJ}, 30 samples")
print("=" * 70)
print()

for s in range(N_SAMPLES):
    A = load_network(s)
    N = A.shape[0]

    # --------------------------------------------------------
    # Compute BH second-smallest eigvec
    # --------------------------------------------------------
    D_diag = np.array(A.sum(axis=1)).flatten()
    H = (R**2 - 1) * eye(N) - R * A + diags(D_diag)
    bh_vals, bh_vecs = eigsh(H, k=K_BH, which='SA')
    idx = np.argsort(bh_vals)
    bh_vals = bh_vals[idx]
    bh_vecs = bh_vecs[:, idx]
    u_BH = bh_vecs[:, 1]  # second-smallest = community eigenvector

    # --------------------------------------------------------
    # Compute top-K_ADJ adjacency eigvecs (by largest eigenvalue)
    # --------------------------------------------------------
    adj_vals, adj_vecs = eigsh(A, k=K_ADJ, which='LA')
    # Sort by DECREASING eigenvalue: v_1 = largest, v_2 = 2nd largest, ...
    idx_adj = np.argsort(adj_vals)[::-1]
    adj_vals = adj_vals[idx_adj]
    adj_vecs = adj_vecs[:, idx_adj]   # shape (N, K_ADJ)

    # --------------------------------------------------------
    # SECTION 1: Project u_BH onto adjacency eigvecs
    # --------------------------------------------------------
    c = adj_vecs.T @ u_BH   # shape (K_ADJ,), c[k] = dot(u_BH, v_{k+1})
    overlap_sq += c**2

    # --------------------------------------------------------
    # SECTION 2: Residual analysis
    # --------------------------------------------------------
    u_BH_recon = adj_vecs @ c   # reconstructed from top-K_ADJ adj eigvecs
    res_norm = np.linalg.norm(u_BH - u_BH_recon) / np.linalg.norm(u_BH)
    residual_norms.append(res_norm)

    # --------------------------------------------------------
    # SECTION 3: Degree correction D@v_k - d_avg*v_k
    # --------------------------------------------------------
    d_avg = D_diag.mean()
    D_mat = diags(D_diag)
    for j in range(min(K_DC, K_ADJ)):
        v_j = adj_vecs[:, j]
        Dv_j = D_mat @ v_j                    # D @ v_j
        dc_j = Dv_j - d_avg * v_j             # degree-correction residual
        # Project dc_j onto all K_ADJ adjacency eigvecs
        dc_coeffs = adj_vecs.T @ dc_j          # shape (K_ADJ,)
        dc_overlap_sq[j] += dc_coeffs**2

    # --------------------------------------------------------
    # SECTION 4: Coefficient on v_2 (index 1 in 0-based)
    # --------------------------------------------------------
    coeff_v2.append(c[1])                          # c[1] = projection on v_2
    other_sum = np.sum(c**2) - c[1]**2
    coeff_other_sum.append(other_sum)

    # --------------------------------------------------------
    # SECTION 5: NMI comparisons
    # --------------------------------------------------------
    # NMI of sign(u_BH)
    nmi_bh_list.append(nmi(classify_sign(u_BH), true_labels))
    # NMI of sign(v_2_adj)
    v2_adj = adj_vecs[:, 1]
    nmi_v2_list.append(nmi(classify_sign(v2_adj), true_labels))
    # NMI of sign(reconstructed u_BH from top-20 adj eigvecs)
    nmi_recon_list.append(nmi(classify_sign(u_BH_recon), true_labels))

    if s % 10 == 0:
        print(f"  sample {s:02d} done  |  res_norm={res_norm:.4f}  |  "
              f"NMI_BH={nmi_bh_list[-1]:.4f}  NMI_v2={nmi_v2_list[-1]:.4f}  NMI_recon={nmi_recon_list[-1]:.4f}")

# Normalize by number of samples
overlap_sq_mean = overlap_sq / N_SAMPLES
dc_overlap_sq_mean = dc_overlap_sq / N_SAMPLES

print()
print("=" * 70)
print("SECTION 1: Projection of u_BH onto adjacency eigvecs (mean |c_k|^2)")
print("=" * 70)
print(f"  {'k':>4}  {'mean |c_k|^2':>14}  {'cumulative':>12}  {'bar'}")
print("  " + "-" * 60)
cumul = 0.0
for k in range(K_ADJ):
    cumul += overlap_sq_mean[k]
    bar_len = int(overlap_sq_mean[k] * 400)
    bar = "#" * min(bar_len, 50)
    print(f"  {k+1:>4}  {overlap_sq_mean[k]:>14.6f}  {cumul:>12.6f}  {bar}")

total_explained = overlap_sq_mean.sum()
print(f"\n  Total |c_k|^2 (all 20 eigvecs) = {total_explained:.6f}  "
      f"(sum should be <= 1 since u_BH is unit-norm)")
print()

print("=" * 70)
print("SECTION 2: Residual analysis")
print("=" * 70)
print(f"  Residual norm ||u_BH - u_BH_recon|| / ||u_BH||")
print(f"  (fraction of u_BH NOT captured by top-{K_ADJ} adjacency eigvecs)")
print(f"\n  Mean residual norm = {np.mean(residual_norms):.6f}")
print(f"  Std  residual norm = {np.std(residual_norms):.6f}")
print(f"  Min  residual norm = {np.min(residual_norms):.6f}")
print(f"  Max  residual norm = {np.max(residual_norms):.6f}")
frac_in_bulk = np.mean(np.array(residual_norms)**2)
frac_in_top20 = 1.0 - frac_in_bulk
print(f"\n  Fraction of ||u_BH||^2 in top-{K_ADJ} adj eigvecs = {frac_in_top20:.4f}")
print(f"  Fraction in spectral bulk (not top-{K_ADJ})        = {frac_in_bulk:.4f}")
print()

print("=" * 70)
print("SECTION 3: Degree-correction D@v_k - d_avg*v_k  decomposed in adj eigvecs")
print("  (shows which eigvecs are 'mixed in' by the D term)")
print("=" * 70)
for j in range(K_DC):
    top_idx = np.argsort(dc_overlap_sq_mean[j])[::-1][:5]
    top_vals = dc_overlap_sq_mean[j][top_idx]
    total_dc = dc_overlap_sq_mean[j].sum()
    print(f"\n  D@v_{j+1} - d_avg*v_{j+1}  (total |c|^2 in top-{K_ADJ} eigvecs = {total_dc:.4f}):")
    for rank, (ki, val) in enumerate(zip(top_idx, top_vals)):
        print(f"    top-{rank+1}: v_{ki+1:>2d}  |c|^2 = {val:.6f}")

print()
print("=" * 70)
print("SECTION 4: Is u_BH mostly v_2 with small corrections?")
print("=" * 70)
coeff_v2_arr = np.array(coeff_v2)
coeff_other_arr = np.array(coeff_other_sum)
print(f"\n  Coefficient on v_2 (adjacency 2nd eigvec):")
print(f"    mean |c_2|^2 = {np.mean(coeff_v2_arr**2):.6f}  (mean |c_2| = {np.mean(np.abs(coeff_v2_arr)):.4f})")
print(f"  Sum of |c_k|^2 for k != 2 (other adj eigvecs):")
print(f"    mean = {np.mean(coeff_other_arr):.6f}")
frac_v2 = np.mean(coeff_v2_arr**2) / overlap_sq_mean.sum()
print(f"\n  Fraction of top-{K_ADJ} overlap coming from v_2 alone:")
print(f"    |c_2|^2 / sum_k |c_k|^2 = {frac_v2:.4f}")
print()
if frac_v2 > 0.7:
    print("  INTERPRETATION: u_BH is PREDOMINANTLY v_2 (with small corrections).")
elif frac_v2 > 0.3:
    print("  INTERPRETATION: u_BH has SIGNIFICANT overlap with v_2, but other eigvecs matter too.")
else:
    print("  INTERPRETATION: u_BH is a SUBSTANTIALLY DIFFERENT direction from v_2.")

print()
print("=" * 70)
print("SECTION 5: NMI comparison")
print("=" * 70)
print(f"\n  {'Method':<50}  {'Mean NMI':>10}  {'Std NMI':>9}")
print("  " + "-" * 74)
print(f"  {'sign(u_BH)  [Bethe Hessian 2nd eigvec]':<50}  {np.mean(nmi_bh_list):>10.4f}  {np.std(nmi_bh_list):>9.4f}")
print(f"  {'sign(v_2_adj)  [adjacency 2nd eigvec]':<50}  {np.mean(nmi_v2_list):>10.4f}  {np.std(nmi_v2_list):>9.4f}")
print(f"  {'sign(u_BH_recon)  [recon from top-20 adj eigvecs]':<50}  {np.mean(nmi_recon_list):>10.4f}  {np.std(nmi_recon_list):>9.4f}")
print()
print("  Reference baselines:")
print("    sign(v2) NMI ~ 0.047")
print("    BH+Kmeans    ~ 0.101")
print("    BP           ~ 0.112  (target)")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  1. BH 2nd eigvec (u_BH) projected onto adjacency top-{K_ADJ} eigvecs:")
top3_k = np.argsort(overlap_sq_mean)[::-1][:3]
for rank, k in enumerate(top3_k):
    print(f"       rank {rank+1}: adj eigvec v_{k+1}  mean |c_{k+1}|^2 = {overlap_sq_mean[k]:.4f}")
print(f"       Total power in top-{K_ADJ} adj eigvecs: {overlap_sq_mean.sum():.4f}")
print()
print(f"  2. Fraction of u_BH in spectral bulk (NOT top-{K_ADJ}): "
      f"{np.mean(np.array(residual_norms)**2):.4f}")
print()
print(f"  3. Degree-correction mixes v_2 mainly into:")
top3_dc = np.argsort(dc_overlap_sq_mean[1])[::-1][:3]  # for j=1 (v_2)
for rank, k in enumerate(top3_dc):
    print(f"       v_{k+1}  |c|^2 = {dc_overlap_sq_mean[1][k]:.4f}")
print()
print(f"  4. u_BH = {np.mean(np.abs(coeff_v2_arr)):.4f} * v_2 + (other contributions)")
print(f"     Fraction from v_2 alone: {frac_v2:.4f}")
print()
print(f"  5. NMI  u_BH={np.mean(nmi_bh_list):.4f}  "
      f"v_2={np.mean(nmi_v2_list):.4f}  "
      f"recon={np.mean(nmi_recon_list):.4f}  "
      f"(BP target~0.112)")
print()
print("Done.")
