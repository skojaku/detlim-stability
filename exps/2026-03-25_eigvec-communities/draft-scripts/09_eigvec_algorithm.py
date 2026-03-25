"""
09_eigvec_algorithm.py

Algorithm to Extract Community Information from Adjacency Eigenvectors.

Setup: N=2000 nodes, 2 communities (0-999 = comm 0, 1000-1999 = comm 1)
       cave=5.0, mu=0.5, 30 samples

Baselines: sign(v2) NMI~0.047, BH+Kmeans NMI~0.101, BP NMI~0.112 (target)

Key insight: The leading eigenvector v1 of the adjacency matrix encodes degree
structure. The community signal in v2 is partially masked by degree variation.
We isolate the community signal by:
  Approach 1: v2/v1 correction (SCORE-like applied to BH)
  Approach 2: Degree-reweighted adjacency eigenvectors
  Approach 3: BH eigvec decomposition in adjacency eigvec basis +
              degree mixing matrix analysis
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.cluster import KMeans
import os
import csv

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000
N_COMM = 1000
CAVE = 5.0
R_OPT = np.sqrt(CAVE)   # ~2.2361

true_labels = np.array([0] * N_COMM + [1] * N_COMM)


# ------------------------------------------------------------------ helpers --

def load_network(s):
    return sp.load_npz(os.path.join(DATA_DIR, f"net_sample_{s:03d}.npz")).astype(float)


def load_eigvecs(s):
    d = np.load(os.path.join(DATA_DIR, f"eigvecs_sample_{s:03d}.npz"))
    return d["vals"], d["V"]   # (10,), (2000, 10)


def build_bethe_hessian(A, r):
    """H(r) = (r^2-1)*I - r*A + D"""
    n = A.shape[0]
    d = np.array(A.sum(axis=1)).ravel()
    return (r**2 - 1) * sp.eye(n) - r * A + sp.diags(d)


def best_nmi(v, true_lab=true_labels):
    """Sign-ambiguity-aware NMI."""
    p = (v >= 0).astype(int)
    n1 = nmi_score(true_lab, p,     average_method="arithmetic")
    n2 = nmi_score(true_lab, 1 - p, average_method="arithmetic")
    return max(n1, n2)


# ---------------------------------------------------------------- baselines --

baseline_sign_v2 = []
baseline_bh_km   = []
baseline_bp      = []

with open(os.path.join(DATA_DIR, "baselines.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        baseline_sign_v2.append(float(row["nmi_sign_v2"]))
        baseline_bp.append(float(row["nmi_bp"]))
        # bh+kmeans column may not exist; fall back gracefully
        if "nmi_bh_kmeans" in row:
            baseline_bh_km.append(float(row["nmi_bh_kmeans"]))

bp_mean   = np.mean(baseline_bp)
sv2_mean  = np.mean(baseline_sign_v2)
bhkm_mean = np.mean(baseline_bh_km) if baseline_bh_km else None

print("=" * 70)
print("BASELINES (mean NMI over 30 samples)")
print("=" * 70)
print(f"  sign(v2)            : {sv2_mean:.4f}")
if bhkm_mean is not None:
    print(f"  BH + Kmeans         : {bhkm_mean:.4f}")
else:
    print("  BH + Kmeans         : ~0.101  (from task description)")
print(f"  Belief Propagation  : {bp_mean:.4f}  <-- TARGET")
print()


# =============================================================================
# STEP 0: Verify BH second eigvec performance (sanity check)
# =============================================================================
print("=" * 70)
print("STEP 0: BH SECOND EIGVEC SIGN (sanity check)")
print("=" * 70)

nmi_bh_sign2 = []
for s in range(N_SAMPLES):
    A  = load_network(s)
    H  = build_bethe_hessian(A, R_OPT)
    vals_h, vecs_h = eigsh(H, k=3, which="SA")
    order = np.argsort(vals_h)
    vals_h = vals_h[order]; vecs_h = vecs_h[:, order]
    # second-smallest eigvec (community vector)
    n_neg = np.sum(vals_h < 0)
    v_comm = vecs_h[:, 1] if n_neg >= 2 else vecs_h[:, 0]
    nmi_bh_sign2.append(best_nmi(v_comm))

print(f"  BH sign(u2)  mean NMI = {np.mean(nmi_bh_sign2):.4f}  std={np.std(nmi_bh_sign2):.4f}")
print()


# =============================================================================
# APPROACH 1: v2/v1 and u_BH2/u_BH1 ratios  (SCORE-like)
# =============================================================================
print("=" * 70)
print("APPROACH 1: SCORE-LIKE RATIO CORRECTION")
print("  (a) Adjacency:   r_adj = v2_adj / v1_adj")
print("  (b) BH:          r_bh  = u2_bh  / u1_bh")
print("  (c) Cross:       r_cross = u2_bh / v1_adj  (BH comm vec / adj degree vec)")
print("=" * 70)

nmi_ratio_adj   = []
nmi_ratio_bh    = []
nmi_ratio_cross = []
nmi_ratio_km_adj = []
nmi_ratio_km_bh  = []

for s in range(N_SAMPLES):
    A = load_network(s)
    _, V = load_eigvecs(s)
    v1_adj = V[:, 0]   # leading adj eigvec (degree-like)
    v2_adj = V[:, 1]   # second adj eigvec (community-like but masked)

    H = build_bethe_hessian(A, R_OPT)
    vals_h, vecs_h = eigsh(H, k=3, which="SA")
    order = np.argsort(vals_h)
    vecs_h = vecs_h[:, order]
    u1_bh = vecs_h[:, 0]   # most-negative BH eigvec (degree-like)
    u2_bh = vecs_h[:, 1]   # second-smallest BH eigvec (community)

    eps = 1e-12

    # (a) adj ratio
    r_adj = v2_adj / (np.abs(v1_adj) + eps)
    nmi_ratio_adj.append(best_nmi(r_adj))
    km_adj = KMeans(n_clusters=2, n_init=10, random_state=s).fit_predict(r_adj.reshape(-1,1))
    nmi_ratio_km_adj.append(nmi_score(true_labels, km_adj, average_method="arithmetic"))

    # (b) BH ratio
    r_bh = u2_bh / (np.abs(u1_bh) + eps)
    nmi_ratio_bh.append(best_nmi(r_bh))
    km_bh = KMeans(n_clusters=2, n_init=10, random_state=s).fit_predict(r_bh.reshape(-1,1))
    nmi_ratio_km_bh.append(nmi_score(true_labels, km_bh, average_method="arithmetic"))

    # (c) cross ratio: BH community vec / adj degree vec
    r_cross = u2_bh / (np.abs(v1_adj) + eps)
    nmi_ratio_cross.append(best_nmi(r_cross))

print(f"  Adj ratio   sign(v2/v1)    mean NMI = {np.mean(nmi_ratio_adj):.4f}  std={np.std(nmi_ratio_adj):.4f}")
print(f"  Adj ratio   K-means        mean NMI = {np.mean(nmi_ratio_km_adj):.4f}  std={np.std(nmi_ratio_km_adj):.4f}")
print(f"  BH  ratio   sign(u2/u1)    mean NMI = {np.mean(nmi_ratio_bh):.4f}  std={np.std(nmi_ratio_bh):.4f}")
print(f"  BH  ratio   K-means        mean NMI = {np.mean(nmi_ratio_km_bh):.4f}  std={np.std(nmi_ratio_km_bh):.4f}")
print(f"  Cross ratio sign(u2_bh/v1) mean NMI = {np.mean(nmi_ratio_cross):.4f}  std={np.std(nmi_ratio_cross):.4f}")
print()


# =============================================================================
# APPROACH 2: Degree-reweighted adjacency eigenvectors
#
# From BH equation: A @ u_BH = (1/r) * [D - (mu_BH - r^2+1)*I] @ u_BH
# => u_BH is a generalized eigvec:  A @ u = lam_eff * D_eff @ u
#
# Practical: compute D^{-1} A u2_BH and compare to u2_BH itself.
# Also try: D^{-1/2} A D^{-1/2} eigenvectors (symmetric normalised).
# =============================================================================
print("=" * 70)
print("APPROACH 2: DEGREE-REWEIGHTED ADJACENCY EIGENVECTORS")
print("  (a) D^{-1} A u2_BH  (one step of degree-normalised power iter from BH seed)")
print("  (b) D^{-1/2} A D^{-1/2}  2nd eigvec")
print("  (c) Generalised: (A - lam_eff * D) u = 0, vary lam_eff in {0..2/cave}")
print("=" * 70)

nmi_reweight_step  = []
nmi_dnorm          = []
nmi_gen_best       = []
lam_eff_values     = np.linspace(0.0, 0.6, 13)   # 1/sqrt(cave) ~ 0.447
nmi_gen_grid       = np.zeros((N_SAMPLES, len(lam_eff_values)))

for s in range(N_SAMPLES):
    A = load_network(s)
    _, V = load_eigvecs(s)

    deg = np.array(A.sum(axis=1)).ravel()
    d_inv      = np.where(deg > 0, 1.0 / deg, 0.0)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    H = build_bethe_hessian(A, R_OPT)
    vals_h, vecs_h = eigsh(H, k=3, which="SA")
    order = np.argsort(vals_h); vals_h = vals_h[order]; vecs_h = vecs_h[:, order]
    n_neg = np.sum(vals_h < 0)
    u2_bh = vecs_h[:, 1] if n_neg >= 2 else vecs_h[:, 0]
    mu2   = vals_h[1]    if n_neg >= 2 else vals_h[0]

    # (a) D^{-1} A applied once to u2_BH
    Au2 = A @ u2_bh
    step_vec = d_inv * Au2
    nmi_reweight_step.append(best_nmi(step_vec))

    # (b) D^{-1/2} A D^{-1/2} second eigvec (via LinearOperator)
    from scipy.sparse.linalg import LinearOperator
    n = A.shape[0]
    def _mv(x, _A=A, _d=d_inv_sqrt): return _d * (_A @ (_d * x))
    A_norm_op = LinearOperator((n, n), matvec=_mv, dtype=float)
    ev, evec = eigsh(A_norm_op, k=2, which="LA", tol=1e-6, maxiter=3000)
    idx = np.argsort(ev)[::-1]
    v2_dnorm = evec[:, idx[1]]
    nmi_dnorm.append(best_nmi(v2_dnorm))

    # (c) For each lam_eff, compute (A - lam_eff * D) u_BH and classify sign
    for li, lam_eff in enumerate(lam_eff_values):
        u_gen = Au2 - lam_eff * deg * u2_bh   # (A - lam_eff D) u2_BH
        nmi_gen_grid[s, li] = best_nmi(u_gen)

# Best lam_eff per sample
nmi_gen_best = nmi_gen_grid.max(axis=1)
best_lam_idx = nmi_gen_grid.mean(axis=0).argmax()
best_lam_eff = lam_eff_values[best_lam_idx]
nmi_gen_at_best = nmi_gen_grid[:, best_lam_idx]

print(f"  D^{{-1}} A u2_BH (one step)     mean NMI = {np.mean(nmi_reweight_step):.4f}  std={np.std(nmi_reweight_step):.4f}")
print(f"  D^{{-1/2}} A D^{{-1/2}} 2nd vec  mean NMI = {np.mean(nmi_dnorm):.4f}  std={np.std(nmi_dnorm):.4f}")
print(f"  (A - lam*D) u2_BH  oracle best mean NMI = {np.mean(nmi_gen_best):.4f}  std={np.std(nmi_gen_best):.4f}")
print(f"  (A - lam*D) u2_BH  lam={best_lam_eff:.3f}  mean NMI = {np.mean(nmi_gen_at_best):.4f}  std={np.std(nmi_gen_at_best):.4f}")
print()
print("  Lam_eff sweep (mean NMI across samples):")
print(f"  {'lam_eff':>9}  {'mean NMI':>10}")
print("  " + "-" * 24)
for li, lam_eff in enumerate(lam_eff_values):
    marker = " <-- best" if li == best_lam_idx else ""
    print(f"  {lam_eff:>9.3f}  {nmi_gen_grid[:, li].mean():>10.4f}{marker}")
print()


# =============================================================================
# APPROACH 3: BH eigvec decomposition in adjacency eigvec basis
#             + degree mixing matrix analysis
# =============================================================================
print("=" * 70)
print("APPROACH 3: BH EIGVEC DECOMPOSITION IN ADJ EIGVEC BASIS")
print("  + Degree mixing matrix (how D mixes adjacency eigvecs)")
print("=" * 70)

K = 10   # number of adjacency eigvecs available

# Accumulate decomposition statistics across samples
proj_sq_mean   = np.zeros(K)   # mean |c_k|^2 / ||u_BH||^2
residual_proj  = np.zeros(K)   # residual f = u_BH - c_2*v_2 decomposed in remaining vecs
cross_matrix   = np.zeros((K, K))  # D-mixing: A_cross[j,k] = a_{jk}

nmi_v2_component   = []   # NMI of c_2 * v_2 (v2 component of u_BH)
nmi_residual_f     = []   # NMI of residual f = u_BH - c_2*v_2
nmi_reconstruction = []   # NMI of full reconstruction from adj eigvecs
nmi_best_linear    = []   # NMI of best single adj eigvec

for s in range(N_SAMPLES):
    A = load_network(s)
    vals_adj, V = load_eigvecs(s)   # V: (N, K), sorted by decreasing eigenvalue

    H = build_bethe_hessian(A, R_OPT)
    vals_h, vecs_h = eigsh(H, k=3, which="SA")
    order = np.argsort(vals_h); vals_h = vals_h[order]; vecs_h = vecs_h[:, order]
    n_neg = np.sum(vals_h < 0)
    u2_bh = vecs_h[:, 1] if n_neg >= 2 else vecs_h[:, 0]
    mu2   = vals_h[1]    if n_neg >= 2 else vals_h[0]

    # 4. Project u_BH onto adjacency eigvecs
    # c_k = dot(u_BH, v_k)  (V columns are already orthonormal from eigsh)
    c = V.T @ u2_bh   # (K,)
    norm_u2_sq = np.dot(u2_bh, u2_bh)
    proj_frac = (c**2) / norm_u2_sq
    proj_sq_mean += proj_frac

    # 5. Residual: f = u_BH - c_2 * v_2  (index 1 = second adj eigvec)
    v2_adj = V[:, 1]
    c2 = c[1]
    f_residual = u2_bh - c2 * v2_adj
    f_proj_frac = ((V.T @ f_residual)**2) / max(np.dot(f_residual, f_residual), 1e-30)
    residual_proj += f_proj_frac

    nmi_v2_component.append(best_nmi(c2 * v2_adj))
    nmi_residual_f.append(best_nmi(f_residual))

    # Reconstruction of u_BH from all K adj eigvecs
    u_reconstructed = V @ c
    nmi_reconstruction.append(best_nmi(u_reconstructed))

    # Best single adj eigvec
    nmis_single = [best_nmi(V[:, k]) for k in range(K)]
    nmi_best_linear.append(max(nmis_single))

    # 6. Degree mixing matrix: for each v_k, decompose D @ v_k in adj eigvec basis
    #    a_{jk} = dot(v_j, D @ v_k)  -> cross_matrix[j, k]
    deg = np.array(A.sum(axis=1)).ravel()
    for k in range(K):
        Dv_k = deg * V[:, k]
        a_k  = V.T @ Dv_k   # (K,)
        cross_matrix[:, k] += a_k

proj_sq_mean  /= N_SAMPLES
residual_proj /= N_SAMPLES
cross_matrix  /= N_SAMPLES

print()
print("4. Fraction of ||u_BH||^2 explained by each adj eigvec (mean over 30 samples):")
print(f"   {'k':>4}  {'|c_k|^2 / ||u||^2':>20}  {'cumulative':>12}")
cumsum = 0.0
for k in range(K):
    cumsum += proj_sq_mean[k]
    print(f"   {k+1:>4}  {proj_sq_mean[k]:>20.4f}  {cumsum:>12.4f}")

total_explained = proj_sq_mean.sum()
print(f"   Total variance explained by top-{K} adj eigvecs: {total_explained:.4f}")
print()

print("5. Residual f = u_BH - c_2*v_2:  projection onto adj eigvecs:")
print(f"   NMI of c_2*v_2 component        : {np.mean(nmi_v2_component):.4f} ± {np.std(nmi_v2_component):.4f}")
print(f"   NMI of residual f               : {np.mean(nmi_residual_f):.4f} ± {np.std(nmi_residual_f):.4f}")
print(f"   NMI of full reconstruction V@c  : {np.mean(nmi_reconstruction):.4f} ± {np.std(nmi_reconstruction):.4f}")
print(f"   NMI of best single adj eigvec   : {np.mean(nmi_best_linear):.4f} ± {np.std(nmi_best_linear):.4f}")
print()

print("   Projection of residual f onto adj eigvecs (mean |proj|^2 / ||f||^2):")
print(f"   {'k':>4}  {'fraction':>12}")
for k in range(K):
    print(f"   {k+1:>4}  {residual_proj[k]:>12.4f}")
print()

print("6. Degree mixing matrix A_cross[j,k] = <v_j | D | v_k>  (mean over samples):")
print("   (rows = output eigvec index, cols = input eigvec index)")
print("   " + "".join(f"{k+1:>8}" for k in range(K)))
for j in range(K):
    row_str = "".join(f"{cross_matrix[j, k]:>8.3f}" for k in range(K))
    print(f"   v{j+1:<2} {row_str}")
print()

# Relative off-diagonal power (how much D mixes vs keeps eigvecs)
diag_power    = np.sum(np.diag(cross_matrix)**2)
offdiag_power = np.sum(cross_matrix**2) - diag_power
total_power   = np.sum(cross_matrix**2)
print(f"   Diagonal power   : {diag_power:.4f}  ({100*diag_power/total_power:.1f}%)")
print(f"   Off-diagonal power: {offdiag_power:.4f}  ({100*offdiag_power/total_power:.1f}%)")
print()


# =============================================================================
# APPROACH 3b: Reconstruct BH eigvec from adj eigvecs via degree mixing
#
# We have: H(r) u = mu * u
# In adj eigvec basis (c = V^T u):
#   (r^2-1) c_k - r * lam_k * c_k + sum_j A_cross[k,j] * c_j = mu * c_k
# => [diag(r^2-1-r*lam) + A_cross] c = mu * c
# => This is an eigenvalue problem in the REDUCED K-dimensional space!
#
# Solve: find the eigvec of M = diag(r^2-1-r*lam) + A_cross with smallest eigenvalue
# Then reconstruct: u_approx = V @ c_approx
# =============================================================================
print("=" * 70)
print("APPROACH 3b: RECONSTRUCT BH EIGVEC FROM ADJ EIGVEC BASIS")
print("  Solve K×K eigenvalue problem: M c = mu c")
print("  M = diag(r^2 - 1 - r*lam) + A_cross  (A_cross = deg-mixing matrix)")
print("=" * 70)

nmi_reconstructed_bh = []
nmi_reconstructed_bh_ratio = []

for s in range(N_SAMPLES):
    A = load_network(s)
    vals_adj, V = load_eigvecs(s)

    # Build per-sample A_cross
    deg = np.array(A.sum(axis=1)).ravel()
    A_cross_s = np.zeros((K, K))
    for k in range(K):
        Dv_k = deg * V[:, k]
        A_cross_s[:, k] = V.T @ Dv_k

    # Diagonal part from adj eigenvalues
    diag_vals = (R_OPT**2 - 1) - R_OPT * vals_adj[:K]   # shape (K,)
    M = np.diag(diag_vals) + A_cross_s

    # Smallest eigenvalue of M
    evals_M, evecs_M = np.linalg.eigh(M)
    idx_min = np.argmin(evals_M)
    c_approx = evecs_M[:, idx_min]
    u_approx = V @ c_approx

    nmi_reconstructed_bh.append(best_nmi(u_approx))

    # Also try ratio: u_approx / v1_adj (SCORE-like on reconstructed BH)
    v1_adj = V[:, 0]
    eps = 1e-12
    ratio_approx = u_approx / (np.abs(v1_adj) + eps)
    nmi_reconstructed_bh_ratio.append(best_nmi(ratio_approx))

print(f"  Reconstructed BH eigvec (sign)   mean NMI = {np.mean(nmi_reconstructed_bh):.4f}  std={np.std(nmi_reconstructed_bh):.4f}")
print(f"  Reconstructed BH eigvec / v1_adj mean NMI = {np.mean(nmi_reconstructed_bh_ratio):.4f}  std={np.std(nmi_reconstructed_bh_ratio):.4f}")
print()


# =============================================================================
# APPROACH 4: Combined — SCORE ratio applied after degree-mixing correction
#             u_corrected = u2_BH - (c2/c1) * v1_adj
# =============================================================================
print("=" * 70)
print("APPROACH 4: COMBINED CORRECTIONS")
print("  (a) u_corrected = u2_BH - beta * v1_adj  (subtract degree component)")
print("      beta = dot(u2_BH, v1_adj)  (projection coefficient)")
print("  (b) u_BH then divided by v1_adj (SCORE-like on BH vec)")
print("  (c) u_BH weighted: u2_BH * v1_BH / v1_adj")
print("=" * 70)

nmi_corrected_a    = []
nmi_corrected_b    = []
nmi_corrected_c    = []
nmi_corrected_d    = []   # D^{-1} u_BH direct

for s in range(N_SAMPLES):
    A = load_network(s)
    _, V = load_eigvecs(s)
    v1_adj = V[:, 0]

    H = build_bethe_hessian(A, R_OPT)
    vals_h, vecs_h = eigsh(H, k=3, which="SA")
    order = np.argsort(vals_h); vals_h = vals_h[order]; vecs_h = vecs_h[:, order]
    n_neg = np.sum(vals_h < 0)
    u1_bh = vecs_h[:, 0]
    u2_bh = vecs_h[:, 1] if n_neg >= 2 else vecs_h[:, 0]

    deg = np.array(A.sum(axis=1)).ravel()
    d_inv = np.where(deg > 0, 1.0 / deg, 0.0)

    # (a) subtract degree component
    beta = np.dot(u2_bh, v1_adj)
    u_corr_a = u2_bh - beta * v1_adj
    nmi_corrected_a.append(best_nmi(u_corr_a))

    # (b) u2_BH / |v1_adj|  (already done as nmi_ratio_cross above)
    eps = 1e-12
    u_corr_b = u2_bh / (np.abs(v1_adj) + eps)
    nmi_corrected_b.append(best_nmi(u_corr_b))

    # (c) u2_BH * u1_BH / v1_adj
    u_corr_c = u2_bh * u1_bh / (np.abs(v1_adj) + eps)
    nmi_corrected_c.append(best_nmi(u_corr_c))

    # (d) degree-inverse of u2_BH directly
    u_corr_d = d_inv * u2_bh
    nmi_corrected_d.append(best_nmi(u_corr_d))

print(f"  (a) u2_BH - beta*v1_adj (proj removal) mean NMI = {np.mean(nmi_corrected_a):.4f}  std={np.std(nmi_corrected_a):.4f}")
print(f"  (b) u2_BH / |v1_adj|  (cross ratio)    mean NMI = {np.mean(nmi_corrected_b):.4f}  std={np.std(nmi_corrected_b):.4f}")
print(f"  (c) u2_BH * u1_BH / |v1_adj|           mean NMI = {np.mean(nmi_corrected_c):.4f}  std={np.std(nmi_corrected_c):.4f}")
print(f"  (d) D^{{-1}} u2_BH                       mean NMI = {np.mean(nmi_corrected_d):.4f}  std={np.std(nmi_corrected_d):.4f}")
print()


# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("FINAL SUMMARY  (BP target = {:.4f})".format(bp_mean))
print("=" * 70)

methods = [
    ("Baseline: sign(v2_adj)",                 np.mean(baseline_sign_v2),     np.std(baseline_sign_v2)),
    ("Baseline: BH sign(u2)  [sanity check]",  np.mean(nmi_bh_sign2),         np.std(nmi_bh_sign2)),
    ("Baseline: Belief Propagation  [TARGET]", bp_mean,                        np.std(baseline_bp)),
    ("---", None, None),
    ("App1a: sign(v2/v1) adj ratio",           np.mean(nmi_ratio_adj),        np.std(nmi_ratio_adj)),
    ("App1a: K-means on v2/v1",                np.mean(nmi_ratio_km_adj),     np.std(nmi_ratio_km_adj)),
    ("App1b: sign(u2/u1) BH ratio",            np.mean(nmi_ratio_bh),         np.std(nmi_ratio_bh)),
    ("App1b: K-means on u2/u1",                np.mean(nmi_ratio_km_bh),      np.std(nmi_ratio_km_bh)),
    ("App1c: sign(u2_BH / v1_adj)",            np.mean(nmi_ratio_cross),      np.std(nmi_ratio_cross)),
    ("---", None, None),
    ("App2a: D^{-1} A u2_BH (one step)",       np.mean(nmi_reweight_step),    np.std(nmi_reweight_step)),
    ("App2b: D^{-1/2} A D^{-1/2} 2nd eigvec", np.mean(nmi_dnorm),            np.std(nmi_dnorm)),
    (f"App2c: (A-{best_lam_eff:.3f}*D) u2_BH  [best lam]", np.mean(nmi_gen_at_best), np.std(nmi_gen_at_best)),
    ("App2c: oracle best-lam per sample",       np.mean(nmi_gen_best),         np.std(nmi_gen_best)),
    ("---", None, None),
    ("App3:  V@c reconstruction of u_BH",       np.mean(nmi_reconstruction),   np.std(nmi_reconstruction)),
    ("App3b: reduced K×K eigsolver",            np.mean(nmi_reconstructed_bh), np.std(nmi_reconstructed_bh)),
    ("App3b: reduced K×K / v1_adj ratio",       np.mean(nmi_reconstructed_bh_ratio), np.std(nmi_reconstructed_bh_ratio)),
    ("---", None, None),
    ("App4a: u2_BH - beta*v1_adj (projection)", np.mean(nmi_corrected_a),     np.std(nmi_corrected_a)),
    ("App4b: u2_BH / |v1_adj|",                np.mean(nmi_corrected_b),     np.std(nmi_corrected_b)),
    ("App4c: u2_BH * u1_BH / |v1_adj|",        np.mean(nmi_corrected_c),     np.std(nmi_corrected_c)),
    ("App4d: D^{-1} u2_BH",                    np.mean(nmi_corrected_d),     np.std(nmi_corrected_d)),
]

print(f"\n  {'Method':<48}  {'Mean NMI':>10}  {'Std':>7}  {'vs BP':>8}")
print("  " + "-" * 82)
for name, mean, std in methods:
    if name == "---":
        print("  " + "-" * 82)
        continue
    flag = " > BP!" if mean > bp_mean else ""
    print(f"  {name:<48}  {mean:>10.4f}  {std:>7.4f}  {mean-bp_mean:>+8.4f}{flag}")

print()
print("Done.")
