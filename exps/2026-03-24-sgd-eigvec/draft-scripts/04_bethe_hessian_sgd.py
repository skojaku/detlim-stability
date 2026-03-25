# %% Bethe-Hessian MBPI: does the right operator bring SGD close to BP?
#
# Prior scripts showed:
#   - Mini-batch noise hurts near the threshold (slower convergence)
#   - Full-batch power iteration outperforms eigsh at strong signal (mu<0.3)
#     but loses near threshold because Lanczos converges faster with tiny spectral gap
#
# Key insight: the INPUT MATRIX matters more than the optimizer.
# Bethe-Hessian: H(r) = (r²-1)I - rA + D,  r = sqrt(mean_degree)
#   → noise bulk: all eigenvalues > 0
#   → community signal: k-1 negative eigenvalues
#   → provably reaches KS threshold for spectral methods (Saade et al. 2014)
#
# Plan:
#   1. BH spectral (eigsh on H(r), smallest algebraic eigenvalues)
#   2. BH MBPI full batch (power iter on -H(r))
#   3. BH MBPI mini-batch (50%, 10% of edges per step)
#   4. Compare with spectral(A), BP

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'libs', 'BeliefPropagation'))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
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
mu_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
n_samples = 15


def make_sbm(n, cave, mu, seed):
    N = 2 * n
    c_out = mu * cave; c_in = 2 * cave - c_out
    p_in, p_out = c_in / N, c_out / N
    np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
    edges = g.get_edgelist()
    if not edges:
        return sp.csr_matrix((N, N))
    rows, cols = zip(*edges)
    A = sp.csr_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(N, N))
    A = A + A.T; A.data[:] = 1.0
    return A


def bethe_hessian(A):
    """Return H(r), deg, r where H(r) = (r²-1)I - rA + D."""
    deg = np.asarray(A.sum(axis=1)).flatten()
    r   = np.sqrt(deg.mean())
    h_diag = (r**2 - 1.0) + deg   # diagonal of H
    return A, h_diag, r


def bh_spectral(A, k=2):
    """
    Find k eigenvectors of H(r) with smallest algebraic eigenvalues (most negative).
    Uses shift-invert or SA mode in eigsh.
    """
    _, h_diag, r = bethe_hessian(A)
    N = A.shape[0]

    def matvec(x):
        return h_diag * x - r * (A @ x)

    H_op = LinearOperator((N, N), matvec=matvec, dtype=float)
    # 'SA' = smallest algebraic eigenvalues
    vals, V = eigsh(H_op, k=k + 2, which="SA", tol=1e-4, maxiter=3000)
    idx = np.argsort(vals)
    vals, V = vals[idx], V[:, idx]
    # Keep only negative eigenvalues (community signal)
    neg_mask = vals < 0
    if neg_mask.sum() == 0:
        return V[:, :k]   # fallback: return smallest even if positive
    return V[:, neg_mask][:, :k]


def bh_mbpi(A, k=2, n_epochs=100, batch_frac=1.0, seed=0):
    """
    Mini-batch power iteration on -H(r): find smallest eigenvectors of H(r).
    H(r)z = h_diag * z - r * A @ z
    -H(r)z = -h_diag * z + r * A @ z
    Each epoch: Z ← QR(-H(r)Z) via mini-batch A estimate.
    """
    local_rng = np.random.default_rng(seed)
    _, h_diag, r = bethe_hessian(A)
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

        # -H(r)Z = r*AZ - h_diag * Z
        neg_HZ = r * AZ - h_diag[:, None] * Z
        Z, _ = np.linalg.qr(neg_HZ)

    return Z


def spectral_A(A, k=2):
    """K-means on top-k eigenvectors of A (skip trivial v1)."""
    vals, V = eigsh(A.astype(float), k=k + 1, which="LA")
    idx = np.argsort(vals)[::-1]
    return V[:, idx][:, 1:k+1]


def kmeans2(Z, seed=0):
    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    return km.fit_predict(Z)


def run_bp(A, membership):
    labels = bpmod.detect(A.copy(), q=2, init_memberships=membership)
    return normalized_mutual_info_score(membership, labels)


# =============================================================================
# Main sweep
# =============================================================================
membership = np.array([0] * n + [1] * n)

methods = {
    "Spectral A":           lambda A, s: spectral_A(A),
    "BH spectral":          lambda A, s: bh_spectral(A),
    "BH MBPI full":         lambda A, s: bh_mbpi(A, n_epochs=100, batch_frac=1.0, seed=s),
    "BH MBPI 50%":          lambda A, s: bh_mbpi(A, n_epochs=100, batch_frac=0.5, seed=s),
    "BH MBPI 10%":          lambda A, s: bh_mbpi(A, n_epochs=100, batch_frac=0.1, seed=s),
}

results = {name: np.zeros((len(mu_values), n_samples)) for name in methods}
results_bp = np.zeros((len(mu_values), n_samples))

for mi, mu in enumerate(mu_values):
    print(f"\n--- mu={mu:.2f} (mu_c={mu_c:.3f}) ---")
    for s in range(n_samples):
        A = make_sbm(n, cave, mu, seed=mi * 1000 + s)
        for name, fn in methods.items():
            Z = fn(A, s)
            nmi = normalized_mutual_info_score(membership, kmeans2(Z, seed=s))
            results[name][mi, s] = nmi
        results_bp[mi, s] = run_bp(A, membership)

    row = "  " + "  ".join(f"{n}={results[n][mi].mean():.3f}" for n in methods)
    print(f"  bp={results_bp[mi].mean():.3f}" + row)

# =============================================================================
# Print summary
# =============================================================================
print("\n=== NMI (mean ± std) ===")
header = f"{'mu':>5}  {'BP':>8}"
for name in methods:
    header += f"  {name[:14]:>14}"
print(header)
for mi, mu in enumerate(mu_values):
    row = f"{mu:5.2f}  {results_bp[mi].mean():.4f}±{results_bp[mi].std():.3f}"
    for name in methods:
        row += f"  {results[name][mi].mean():.4f}±{results[name][mi].std():.3f}"
    print(row)

# =============================================================================
# Plot
# =============================================================================
mus    = np.array(mu_values)
colors = sns.color_palette("Set2", len(methods))
styles = ['-o', '-s', '--D', '--^', '--v']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, title, show_bp in [(axes[0], "All methods", True), (axes[1], "Near threshold (zoom)", True)]:
    ax.plot(mus, results_bp.mean(1), 'k-*', lw=2.5, ms=9, zorder=6, label="Belief propagation")
    for (name, _), color, style in zip(methods.items(), colors, styles):
        m = results[name].mean(1)
        s = results[name].std(1)
        ax.plot(mus, m, style, color=color, lw=2, label=name)
        ax.fill_between(mus, m - s, m + s, alpha=0.1, color=color)
    ax.axvline(mu_c, color="red", ls=":", lw=1.5)
    ax.set_xlabel("μ")
    ax.set_ylabel("NMI")
    ax.legend(fontsize=9, loc="upper right")
    sns.despine(ax=ax)

axes[0].set_title(f"Bethe-Hessian SGD vs Spectral vs BP (cave={cave})")
axes[1].set_xlim(0.3, 0.7)
axes[1].set_title("Zoom near KS threshold")

fig.tight_layout()
print("\nFigure: NMI vs mu. Key question: does BH-MBPI close the gap to BP near mu_c?")
plt.show()
