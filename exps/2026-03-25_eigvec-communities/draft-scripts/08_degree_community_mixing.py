"""
08_degree_community_mixing.py
==============================
Hypothesis: near the detectability limit, the adjacency v2 is "contaminated"
by degree structure, which hurts community detection.  The Bethe Hessian
corrects for this.

Setup:
  N=2000, 2 communities (0-999=comm0, 1000-1999=comm1)
  cave=5.0, mu=0.5, 30 samples
"""

import numpy as np
import os
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score as nmi_score

DATA_DIR = "/home/skojaku/projects/detlim-stability/exps/2026-03-25_eigvec-communities/data"
N_SAMPLES = 30
N_NODES = 2000

true_labels = np.array([0] * 1000 + [1] * 1000)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_network(s):
    return sp.load_npz(os.path.join(DATA_DIR, f"net_sample_{s:03d}.npz"))


def load_eigvecs(s):
    d = np.load(os.path.join(DATA_DIR, f"eigvecs_sample_{s:03d}.npz"))
    return d["vals"], d["V"]          # eigenvalues and 2000×10 matrix


def nmi(pred, true=true_labels):
    return nmi_score(true, pred, average_method="arithmetic")


def classify_sign(v):
    return (v >= 0).astype(int)


def partial_corr_residual(v, deg):
    """Remove linear effect of degree from v; return residual."""
    coeff = np.polyfit(deg, v, 1)
    return v - np.polyval(coeff, deg)


def r_squared(y, x):
    """R² of regressing y on x (+ intercept)."""
    coeff = np.polyfit(x, y, 1)
    y_hat = np.polyval(coeff, x)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def r_squared_multi(y, x1, x2):
    """R² of regressing y on x1 and x2 jointly (+ intercept)."""
    X = np.column_stack([x1, x2, np.ones(len(y))])
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeff
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ── accumulators ─────────────────────────────────────────────────────────────

# Section 1
spearman_v2_deg      = []
nmi_plain            = []
nmi_after_deg_remove = []

# Section 2
nmi_normed_sqrt      = []
nmi_normed_linear    = []

# Section 3
frac_signal          = []
corr_noise_deg       = []

# Section 4
corr_boundary_abs_v2 = []
corr_boundary_sign   = []

# Section 5
r2_deg_only          = []
r2_comm_only         = []
r2_both              = []


# ── main loop ────────────────────────────────────────────────────────────────

for s in range(N_SAMPLES):
    A = load_network(s)
    _, V = load_eigvecs(s)

    # v2 = second eigenvector (index 1, since sorted by *decreasing* eigenvalue)
    v2 = V[:, 1]

    deg = np.array(A.sum(axis=1)).flatten().astype(float)

    # avoid divide-by-zero for isolated nodes
    safe_deg = np.where(deg > 0, deg, 1.0)

    # community indicator as ±1 for regressions
    comm_indicator = np.where(true_labels == 0, +1.0, -1.0)

    # ── 1. Spearman r(v2, degree) and NMI after degree removal ───────────────
    rho, _ = spearmanr(v2, deg)
    spearman_v2_deg.append(rho)

    pred_plain = classify_sign(v2)
    nmi_plain.append(nmi(pred_plain))

    v2_resid = partial_corr_residual(v2, deg)
    pred_resid = classify_sign(v2_resid)
    nmi_after_deg_remove.append(nmi(pred_resid))

    # ── 2. Degree-normalised v2 ───────────────────────────────────────────────
    v2_norm_sqrt   = v2 / np.sqrt(safe_deg)
    v2_norm_linear = v2 / safe_deg

    nmi_normed_sqrt.append(nmi(classify_sign(v2_norm_sqrt)))
    nmi_normed_linear.append(nmi(classify_sign(v2_norm_linear)))

    # ── 3. Signal vs noise decomposition ─────────────────────────────────────
    # signal_i = mean(v2[community_of_i])
    mean_c0 = v2[true_labels == 0].mean()
    mean_c1 = v2[true_labels == 1].mean()
    signal = np.where(true_labels == 0, mean_c0, mean_c1)
    noise  = v2 - signal

    norm_v2_sq     = np.dot(v2, v2)
    norm_signal_sq = np.dot(signal, signal)
    frac_signal.append(norm_signal_sq / norm_v2_sq if norm_v2_sq > 0 else 0.0)

    rho_noise, _ = spearmanr(noise, deg)
    corr_noise_deg.append(rho_noise)

    # ── 4. Boundary fraction ──────────────────────────────────────────────────
    # A is sparse; inter-community edges for each node
    # For node i in community c, boundary edges = edges to other community
    A_csc = A.tocsr()
    boundary_frac = np.zeros(N_NODES)
    for i in range(N_NODES):
        row = A_csc.getrow(i)
        if deg[i] == 0:
            boundary_frac[i] = 0.0
            continue
        neighbors = row.indices
        same_comm = np.sum(true_labels[neighbors] == true_labels[i])
        boundary_frac[i] = 1.0 - same_comm / deg[i]

    abs_v2 = np.abs(v2)
    rho_bf_abs, _ = spearmanr(boundary_frac, abs_v2)
    corr_boundary_abs_v2.append(rho_bf_abs)

    # Does high boundary fraction hurt sign accuracy?
    # measure: for each node, 1 if sign(v2) matches community, 0 otherwise
    # community 0 → expect v2 > 0 if mean_c0 > mean_c1, else < 0
    expected_pos = (mean_c0 > mean_c1)
    if expected_pos:
        sign_correct = ((v2 >= 0) == (true_labels == 0)).astype(float)
    else:
        sign_correct = ((v2 < 0) == (true_labels == 0)).astype(float)
    rho_bf_sign, _ = spearmanr(boundary_frac, sign_correct)
    corr_boundary_sign.append(rho_bf_sign)

    # ── 5. Variance decomposition (R²) ───────────────────────────────────────
    r2_deg_only.append(r_squared(v2, deg))
    r2_comm_only.append(r_squared(v2, comm_indicator))
    r2_both.append(r_squared_multi(v2, deg, comm_indicator))


# ── print results ─────────────────────────────────────────────────────────────

sep = "=" * 68

print(sep)
print("SECTION 1: v2 VALUE vs DEGREE AND COMMUNITY")
print(sep)
print(f"  Mean Spearman r(v2, degree)        = {np.mean(spearman_v2_deg):+.4f}  "
      f"(std {np.std(spearman_v2_deg):.4f})")
print(f"  Fraction of samples with |r| > 0.1 = "
      f"{np.mean(np.abs(spearman_v2_deg) > 0.1):.2f}")
print()
print(f"  Mean NMI  plain sign(v2)            = {np.mean(nmi_plain):.4f}  "
      f"(std {np.std(nmi_plain):.4f})")
print(f"  Mean NMI  after removing degree     = {np.mean(nmi_after_deg_remove):.4f}  "
      f"(std {np.std(nmi_after_deg_remove):.4f})")
delta = np.mean(nmi_after_deg_remove) - np.mean(nmi_plain)
print(f"  Delta NMI (degree removal)          = {delta:+.4f}")
print()

print(sep)
print("SECTION 2: DEGREE-NORMALISED v2")
print(sep)
print(f"  Mean NMI  plain sign(v2)            = {np.mean(nmi_plain):.4f}  "
      f"(std {np.std(nmi_plain):.4f})")
print(f"  Mean NMI  sign(v2 / sqrt(degree))   = {np.mean(nmi_normed_sqrt):.4f}  "
      f"(std {np.std(nmi_normed_sqrt):.4f})")
print(f"  Mean NMI  sign(v2 / degree)         = {np.mean(nmi_normed_linear):.4f}  "
      f"(std {np.std(nmi_normed_linear):.4f})")
print()

print(sep)
print("SECTION 3: SIGNAL vs NOISE DECOMPOSITION IN v2")
print(sep)
print(f"  Mean ||signal||² / ||v2||²          = {np.mean(frac_signal):.4f}  "
      f"(std {np.std(frac_signal):.4f})")
print(f"  → Community structure explains this fraction of v2 energy")
print()
print(f"  Mean Spearman r(noise_v2, degree)   = {np.mean(corr_noise_deg):+.4f}  "
      f"(std {np.std(corr_noise_deg):.4f})")
print(f"  → Positive: residual noise in v2 aligns with high-degree nodes")
print()

print(sep)
print("SECTION 4: BOUNDARY FRACTION vs v2")
print(sep)
print(f"  Mean Spearman r(boundary_frac, |v2|)    = {np.mean(corr_boundary_abs_v2):+.4f}  "
      f"(std {np.std(corr_boundary_abs_v2):.4f})")
print(f"  Mean Spearman r(boundary_frac, correct) = {np.mean(corr_boundary_sign):+.4f}  "
      f"(std {np.std(corr_boundary_sign):.4f})")
print(f"  → Negative r: more cross-community edges → weaker v2 signal / worse accuracy")
print()

print(sep)
print("SECTION 5: VARIANCE DECOMPOSITION  (R² of v2)")
print(sep)
print(f"  {'Source':<30}  {'Mean R²':>9}  {'Std':>7}")
print(f"  {'-'*50}")
print(f"  {'Degree alone':<30}  {np.mean(r2_deg_only):>9.4f}  {np.std(r2_deg_only):>7.4f}")
print(f"  {'Community alone':<30}  {np.mean(r2_comm_only):>9.4f}  {np.std(r2_comm_only):>7.4f}")
print(f"  {'Degree + community (joint)':<30}  {np.mean(r2_both):>9.4f}  {np.std(r2_both):>7.4f}")
print()
mean_deg  = np.mean(r2_deg_only)
mean_comm = np.mean(r2_comm_only)
mean_both = np.mean(r2_both)
unique_deg  = mean_both - mean_comm
unique_comm = mean_both - mean_deg
shared      = mean_both - unique_deg - unique_comm
print("  Unique contribution of degree      = "
      f"{unique_deg:.4f}  ({100*unique_deg/max(mean_both,1e-9):.1f}% of joint R²)")
print("  Unique contribution of community   = "
      f"{unique_comm:.4f}  ({100*unique_comm/max(mean_both,1e-9):.1f}% of joint R²)")
print("  Shared / interaction               = "
      f"{shared:.4f}  ({100*shared/max(mean_both,1e-9):.1f}% of joint R²)")
print()

print(sep)
print("KEY TAKE-AWAYS")
print(sep)
print(f"  1. v2 ↔ degree Spearman r = {np.mean(spearman_v2_deg):+.4f}  "
      f"→ {'substantial' if abs(np.mean(spearman_v2_deg)) > 0.15 else 'moderate' if abs(np.mean(spearman_v2_deg)) > 0.07 else 'weak'} "
      f"degree contamination")
print(f"  2. Removing linear degree effect "
      f"{'improves' if delta > 0 else 'degrades'} NMI by {delta:+.4f}  "
      f"({'helps' if delta > 0.002 else 'marginal effect'})")
print(f"  3. v2 / sqrt(d) normalisation changes NMI by "
      f"{np.mean(nmi_normed_sqrt) - np.mean(nmi_plain):+.4f}")
print(f"  4. Community signal in v2 energy: {100*np.mean(frac_signal):.1f}%  "
      f"(Bethe Hessian should push this closer to 100%)")
print(f"  5. Noise in v2 correlates with degree: r = {np.mean(corr_noise_deg):+.4f}  "
      f"→ degree structure leaks into v2 residual")
print(f"  6. Boundary fraction correlates with |v2|: r = {np.mean(corr_boundary_abs_v2):+.4f}  "
      f"(boundary nodes have weaker signal)")
print(f"  7. Degree alone explains {100*mean_deg:.1f}% of v2 variance; "
      f"community alone {100*mean_comm:.1f}%; both together {100*mean_both:.1f}%")
print()
print("Done.")
