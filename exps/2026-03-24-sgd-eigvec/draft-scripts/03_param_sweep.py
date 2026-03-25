# %% Parameter sweep: NMI vs mu for mini-batch power iteration vs spectral vs BP
#
# Insight from script 02:
#   - Symmetric factorization: Z collapses to near-zero (gradient ∝ Z → no escape from init)
#   - Rayleigh quotient SGD: effective lr was ~0 due to double-division bug
#
# Correct SGD formulation = mini-batch power iteration:
#   Z ← E[scaled_batch_AZ]   (unbiased estimate of A@Z)
#   Z ← QR(Z)                (re-orthonormalize)
#
# Key question: does varying batch_size (= noise level) change the NMI-vs-mu curve?
# Specifically: is mini-batch power iteration MORE ROBUST near the KS threshold?

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'libs', 'BeliefPropagation'))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import igraph as ig
import belief_propagation as bpmod

sns.set_style("white")
sns.set_context("talk", font_scale=1.2)

# -- params --
n, cave = 1000, 5.0
N = 2 * n
mu_c = 1.0 - 1.0 / np.sqrt(cave)
mu_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
n_samples = 15   # fewer samples for speed across many mu values


def make_sbm(n, cave, mu, rng_state):
    """Generate one SBM instance."""
    N = 2 * n
    c_out = mu * cave
    c_in  = 2 * cave - c_out
    p_in, p_out = c_in / N, c_out / N
    np.random.seed(rng_state)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
    edges = g.get_edgelist()
    if not edges:
        return sp.csr_matrix((N, N))
    rows, cols = zip(*edges)
    A = sp.csr_matrix(
        (np.ones(len(rows)), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )
    A = A + A.T; A.data[:] = 1.0
    return A


def mini_batch_power_iter(A, k=2, n_epochs=50, batch_frac=1.0, seed=0):
    """
    Mini-batch power iteration.
    batch_frac = fraction of edges to use per step (1.0 = full batch = deterministic).
    Each epoch:
      AZ_est = (n_edges / batch_size) * sum_{(i,j) in batch} z_j^T (unbiased A@Z estimate)
      Z ← QR(AZ_est)
    """
    local_rng = np.random.default_rng(seed)
    N = A.shape[0]
    A_coo = A.tocoo()
    ei, ej = A_coo.row.astype(np.int32), A_coo.col.astype(np.int32)
    n_edges = len(ei)
    batch_size = max(1, int(n_edges * batch_frac))

    Z = local_rng.normal(0, 1.0, (N, k))
    Z, _ = np.linalg.qr(Z)

    for _ in range(n_epochs):
        if batch_size < n_edges:
            idx = local_rng.choice(n_edges, batch_size, replace=False)
            bi, bj = ei[idx], ej[idx]
            scale = n_edges / batch_size
        else:
            bi, bj = ei, ej
            scale = 1.0

        AZ = np.zeros((N, k))
        np.add.at(AZ, bi, Z[bj])
        np.add.at(AZ, bj, Z[bi])
        AZ *= scale

        Z, _ = np.linalg.qr(AZ)

    return Z


def spectral_embedding(A, k=2):
    """Top-k eigenvectors of A via ARPACK (Lanczos)."""
    vals, V = eigsh(A.astype(float), k=k + 1, which="LA")
    idx = np.argsort(vals)[::-1]
    return V[:, idx][:, 1:k+1]   # skip trivial eigvec


def run_bp(A, membership):
    labels = bpmod.detect(A.copy(), q=2, init_memberships=membership)
    return normalized_mutual_info_score(membership, labels)


def kmeans2(Z, seed=0):
    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    return km.fit_predict(Z)


# =============================================================================
# Main sweep
# =============================================================================
batch_fracs = {
    "Full batch (1.0)":    1.0,
    "Batch 50%":           0.5,
    "Batch 10%":           0.1,
    "Batch 2%":            0.02,
}

# Results storage: methods × mu_values
results_spectral = np.zeros((len(mu_values), n_samples))
results_bp       = np.zeros((len(mu_values), n_samples))
results_mbpi     = {name: np.zeros((len(mu_values), n_samples)) for name in batch_fracs}

membership = np.array([0] * n + [1] * n)

for mi, mu in enumerate(mu_values):
    print(f"\n--- mu={mu:.2f} (mu_c={mu_c:.3f}) ---")
    for s in range(n_samples):
        A = make_sbm(n, cave, mu, rng_state=mi * 1000 + s)

        # Spectral
        V = spectral_embedding(A, k=2)
        nmi_sp = normalized_mutual_info_score(membership, kmeans2(V, seed=s))
        results_spectral[mi, s] = nmi_sp

        # BP
        results_bp[mi, s] = run_bp(A, membership)

        # Mini-batch power iteration at each batch fraction
        for name, bf in batch_fracs.items():
            Z = mini_batch_power_iter(A, k=2, n_epochs=50, batch_frac=bf, seed=s)
            nmi_mbpi = normalized_mutual_info_score(membership, kmeans2(Z, seed=s))
            results_mbpi[name][mi, s] = nmi_mbpi

    print(
        f"  spectral={results_spectral[mi].mean():.3f}  "
        f"bp={results_bp[mi].mean():.3f}  " +
        "  ".join(f"{n}={results_mbpi[n][mi].mean():.3f}" for n in batch_fracs)
    )

# =============================================================================
# Results table
# =============================================================================
print("\n=== NMI mean ± std across mu values ===")
header = f"{'mu':>6}  {'spectral':>10}  {'BP':>10}"
for name in batch_fracs:
    header += f"  {name[:12]:>12}"
print(header)
for mi, mu in enumerate(mu_values):
    row = f"{mu:6.2f}  {results_spectral[mi].mean():8.4f}±{results_spectral[mi].std():.3f}"
    row += f"  {results_bp[mi].mean():8.4f}±{results_bp[mi].std():.3f}"
    for name in batch_fracs:
        row += f"  {results_mbpi[name][mi].mean():10.4f}±{results_mbpi[name][mi].std():.3f}"
    print(row)

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mus = np.array(mu_values)
colors = sns.color_palette("Set2", len(batch_fracs))

# Left: all mini-batch variants + spectral
ax = axes[0]
ax.plot(mus, results_spectral.mean(1), 'k-o', lw=2, label="Spectral (eigsh)", zorder=5)
for (name, _), color in zip(batch_fracs.items(), colors):
    means = results_mbpi[name].mean(1)
    stds  = results_mbpi[name].std(1)
    ax.plot(mus, means, 'o--', color=color, lw=1.5, label=name)
    ax.fill_between(mus, means - stds, means + stds, alpha=0.12, color=color)
ax.axvline(mu_c, color="red", ls=":", lw=1.5, label=f"μ_c={mu_c:.3f}")
ax.set_xlabel("μ (mixing parameter)")
ax.set_ylabel("NMI")
ax.set_title("Mini-batch Power Iter vs Spectral")
ax.legend(fontsize=9, loc="upper left")
sns.despine(ax=ax)

# Right: best mini-batch vs spectral vs BP
ax = axes[1]
best_name = "Full batch (1.0)"
ax.plot(mus, results_spectral.mean(1), 'k-o', lw=2, label="Spectral")
ax.plot(mus, results_bp.mean(1), 's-', color="crimson", lw=2, label="BP")
ax.plot(mus, results_mbpi[best_name].mean(1), 'D--',
        color=colors[0], lw=2, label=f"MBPI {best_name}")
for name, color in zip(list(batch_fracs.keys())[1:], colors[1:]):
    ax.plot(mus, results_mbpi[name].mean(1), '^:', color=color, lw=1.5, alpha=0.8, label=f"MBPI {name}")
ax.axvline(mu_c, color="red", ls=":", lw=1.5, label=f"μ_c")
ax.set_xlabel("μ (mixing parameter)")
ax.set_ylabel("NMI")
ax.set_title("Comparison vs BP (N=2000, cave=5)")
ax.legend(fontsize=9, loc="upper right")
sns.despine(ax=ax)

fig.suptitle(f"Does Mini-batch Noise Help Near Detection Threshold? (cave={cave})")
fig.tight_layout()

print("\nFigure (left): NMI vs mu for mini-batch power iteration at 4 batch fractions + spectral.")
print("Figure (right): Best comparison — spectral, BP, MBPI variants.")
print("Red dashed vertical line = KS detectability threshold mu_c.")
print("Key: if mini-batch methods show higher NMI below mu_c, the hypothesis holds.")
plt.show()
