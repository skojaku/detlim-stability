# %% DeepWalk (random walk SGD) vs BH spectral vs regular spectral vs BP
#
# Diagnosis from scripts 02-04:
#   - Symmetric factorization: gradient collapse (Z≈0 init, gradient ∝ Z → stuck)
#   - Oja's rule: too slow near threshold (spectral gap ≈0, need >>200 epochs)
#   - BH power iteration: bulk modes (|λ|≈23) dominate community modes (λ≈+2.94)
#   - Shifted BH power iteration: converges but very slowly (ratio λ3/λ2 ≈ 0.991)
#   - BH spectral (Lanczos eigsh): NMI≈0.10-0.19, close to BP
#
# Theory says DeepWalk/node2vec ≈ normalized Laplacian eigenvectors (Qiu et al. 2019)
# and node2vec reaches the KS threshold for not-too-sparse SBMs (Kojaku 2023).
# This IS an SGD method (skip-gram with negative sampling).
#
# This script:
#   1. Implements DeepWalk: random walks + skip-gram SGD
#   2. Compares with normalized Laplacian eigvecs (DeepWalk's theoretical limit)
#   3. Compares with BH spectral, regular spectral, BP
#   4. Sweeps mu to find where each method breaks down

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


def make_sbm(mu, seed):
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


# =============================================================================
# DeepWalk: random walks + skip-gram SGD
# =============================================================================

def generate_walks(A_csr, walk_length, n_walks, rng):
    """
    Vectorized random walk generation.
    Returns walks: array of shape (N * n_walks, walk_length).
    """
    N = A_csr.shape[0]
    indptr  = A_csr.indptr
    indices = A_csr.indices
    degrees = np.diff(indptr)  # (N,) degree per node

    starts = np.tile(np.arange(N), n_walks)        # (N*n_walks,)
    walks  = np.empty((len(starts), walk_length), dtype=np.int32)
    walks[:, 0] = starts
    current = starts.copy()

    for step in range(1, walk_length):
        degs     = degrees[current]                  # degree of current nodes
        has_nbr  = degs > 0
        rand_off = (rng.random(len(current)) * degs).astype(np.int32)
        rand_off = np.minimum(rand_off, np.maximum(degs - 1, 0))
        next_v   = current.copy()                    # default: stay
        if has_nbr.any():
            next_v[has_nbr] = indices[indptr[current[has_nbr]] + rand_off[has_nbr]]
        current = next_v
        walks[:, step] = current

    return walks


def deepwalk_sgd(A, embed_dim=2, walk_length=10, n_walks=5, window=3,
                 n_neg=5, lr=0.025, n_epochs=1, seed=0):
    """
    DeepWalk: vectorized random walk generation + skip-gram SGD.
    One epoch = one set of n_walks walks per node, processed as mini-batches.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    Z_center  = rng.normal(0, 0.01, (N, embed_dim))
    Z_context = np.zeros((N, embed_dim))
    A_csr = A.tocsr()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    for _ in range(n_epochs):
        walks = generate_walks(A_csr, walk_length, n_walks, rng)
        # Collect all (center, context) pairs — vectorized
        center_list, context_list = [], []
        for off in range(1, window + 1):
            # Positive offset
            c = walks[:, :-off].reshape(-1)
            t = walks[:, off:].reshape(-1)
            center_list.append(c); context_list.append(t)
            # Negative offset (symmetric)
            center_list.append(t); context_list.append(c)
        centers_all  = np.concatenate(center_list)
        contexts_all = np.concatenate(context_list)
        # Filter out invalid (negative) entries from short walks
        valid = (centers_all >= 0) & (contexts_all >= 0)
        pairs = np.stack([centers_all[valid], contexts_all[valid]], axis=1)
        rng.shuffle(pairs)

        # Mini-batch SGD over all pairs
        batch_size = 512
        for start in range(0, len(pairs), batch_size):
            bp_batch = pairs[start:start+batch_size]
            centers  = bp_batch[:, 0]
            contexts = bp_batch[:, 1]
            bs = len(centers)

            # Positive pairs
            zc = Z_center[centers]    # (bs, d)
            zt = Z_context[contexts]  # (bs, d)
            scores = (zc * zt).sum(1)  # (bs,)
            g = 1.0 - sigmoid(scores)  # (bs,)

            dZc = np.zeros_like(Z_center)
            dZt = np.zeros_like(Z_context)
            np.add.at(dZc, centers,  g[:, None] * zt)
            np.add.at(dZt, contexts, g[:, None] * zc)

            # Negative samples
            negs = rng.integers(0, N, bs * n_neg)
            c_rep = np.repeat(centers, n_neg)
            zc_rep = Z_center[c_rep]
            zn = Z_context[negs]
            scores_neg = (zc_rep * zn).sum(1)
            g_neg = -sigmoid(scores_neg)
            np.add.at(dZc, c_rep, g_neg[:, None] * zn)
            np.add.at(dZt, negs, g_neg[:, None] * zc_rep)

            Z_center  += lr * dZc
            Z_context += lr * dZt

    return Z_center + Z_context


def normalized_laplacian_eigvec(A, k=2):
    """Top-k eigenvectors of D^{-1/2} A D^{-1/2}."""
    deg = np.asarray(A.sum(axis=1)).flatten()
    tau = np.sqrt(deg.mean())   # regularization
    d_inv = 1.0 / np.sqrt(deg + tau)
    def mv(x): return d_inv * (A @ (d_inv * x))
    L_op = LinearOperator((A.shape[0], A.shape[0]), matvec=mv, dtype=float)
    vals, V = eigsh(L_op, k=k+1, which="LA", tol=1e-4, maxiter=3000)
    idx = np.argsort(vals)[::-1]
    return V[:, idx][:, 1:k+1]   # skip trivial eigvec


def bh_spectral(A, k=2):
    """BH spectral: k most-negative eigenvectors of H(r)."""
    deg = np.asarray(A.sum(axis=1)).flatten()
    r   = np.sqrt(deg.mean())
    h_diag = (r**2 - 1.0) + deg
    def mv(x): return h_diag * x - r * (A @ x)
    H_op = LinearOperator((A.shape[0], A.shape[0]), matvec=mv, dtype=float)
    vals, V = eigsh(H_op, k=k+2, which="SA", tol=1e-4, maxiter=5000)
    idx = np.argsort(vals)
    vals, V = vals[idx], V[:, idx]
    neg = vals < 0
    return V[:, neg][:, :k] if neg.sum() > 0 else V[:, :k]


def spectral_A(A, k=2):
    vals, V = eigsh(A.astype(float), k=k+1, which="LA")
    idx = np.argsort(vals)[::-1]
    return V[:, idx][:, 1:k+1]


def km2(Z, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(Z)


def run_bp(A, membership):
    labels = bpmod.detect(A.copy(), q=2, init_memberships=membership)
    return normalized_mutual_info_score(membership, labels)


# =============================================================================
# Main sweep
# =============================================================================
membership = np.array([0] * n + [1] * n)

methods = {
    "Spectral (A)":     lambda A, s: spectral_A(A),
    "NL eigvecs":       lambda A, s: normalized_laplacian_eigvec(A),
    "BH spectral":      lambda A, s: bh_spectral(A),
    "DeepWalk SGD":     lambda A, s: deepwalk_sgd(A, embed_dim=2, walk_length=10,
                                                    n_walks=10, n_epochs=3, seed=s),
}

results = {m: np.zeros((len(mu_values), n_samples)) for m in methods}
res_bp  = np.zeros((len(mu_values), n_samples))

for mi, mu in enumerate(mu_values):
    print(f"\n--- mu={mu:.2f} ---")
    for s in range(n_samples):
        A = make_sbm(mu, seed=mi*1000+s)
        for name, fn in methods.items():
            Z = fn(A, s)
            results[name][mi, s] = normalized_mutual_info_score(membership, km2(Z, s))
        res_bp[mi, s] = run_bp(A, membership)

    row = f"  bp={res_bp[mi].mean():.3f}  " + "  ".join(
        f"{n.split()[0]}={results[n][mi].mean():.3f}" for n in methods)
    print(row)

# =============================================================================
# Print table
# =============================================================================
print("\n=== NMI mean ± std ===")
hdr = f"{'mu':>5}  {'BP':>8}" + "".join(f"  {m[:12]:>12}" for m in methods)
print(hdr)
for mi, mu in enumerate(mu_values):
    row = f"{mu:5.2f}  {res_bp[mi].mean():.4f}±{res_bp[mi].std():.3f}"
    for m in methods:
        row += f"  {results[m][mi].mean():.4f}±{results[m][mi].std():.3f}"
    print(row)

# =============================================================================
# Plot
# =============================================================================
mus    = np.array(mu_values)
colors = sns.color_palette("Set2", len(methods))
styles = ['-o', '-s', '-D', '--^']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, xlim, title in [(axes[0], (0, 0.7), "Full range"),
                         (axes[1], (0.3, 0.7), "Zoom near threshold")]:
    ax.plot(mus, res_bp.mean(1), 'k-*', lw=2.5, ms=9, zorder=6, label="Belief propagation")
    for (name, _), color, style in zip(methods.items(), colors, styles):
        m = results[name].mean(1)
        s = results[name].std(1)
        ax.plot(mus, m, style, color=color, lw=2, label=name)
        ax.fill_between(mus, m - s, m + s, alpha=0.12, color=color)
    ax.axvline(mu_c, color="red", ls=":", lw=1.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("μ (mixing parameter)")
    ax.set_ylabel("NMI")
    ax.set_title(title)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

fig.suptitle(f"SGD (DeepWalk) vs BH Spectral vs BP (cave={cave})")
fig.tight_layout()
print("\nFigure: NMI vs mu. Methods: BH spectral (Lanczos), DeepWalk SGD, NL eigvecs, regular spectral, BP.")
plt.show()
