# %% SGD-based eigenvector solver vs spectral vs belief propagation
# Hypothesis: SGD implicit regularization is more robust near the detection limit
# Methods compared:
#   1. Spectral (sign of v2)
#   2. Spectral (K-means on top eigvecs)
#   3. Oja's rule  — online stochastic PCA on rows of A
#   4. SGD symmetric factorization of A  (ZZ^T ≈ A)
#   5. SGD symmetric factorization of D^{-1/2} A D^{-1/2}
#   6. Belief propagation

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
import belief_propagation as bp

sns.set_style("white")
sns.set_context("talk", font_scale=1.2)

# -- params --
n         = 1000
N         = 2 * n
cave      = 5.0
mu        = 0.5
n_samples = 30
embed_dim = 2
rng       = np.random.default_rng(42)

c_out = mu * cave
c_in  = 2 * cave - c_out
p_in  = c_in / N
p_out = c_out / N
membership = np.array([0] * n + [1] * n)

mu_c = 1.0 - 1.0 / np.sqrt(cave)
print(f"SBM: N={N}, cave={cave}, mu={mu}, mu_c={mu_c:.3f} (mu {'<' if mu < mu_c else '>='} mu_c)")


# =============================================================================
# SGD solvers
# =============================================================================

def degree_normalize(A):
    """D^{-1/2} A D^{-1/2} with regularization tau = sqrt(mean_degree)."""
    deg   = np.asarray(A.sum(axis=1)).flatten()
    tau   = np.sqrt(deg.mean())  # Bethe-Hessian-style regularization
    d_inv = 1.0 / np.sqrt(deg + tau)
    D_inv = sp.diags(d_inv)
    return D_inv @ A @ D_inv


def sgd_factorization(A, embed_dim=2, n_epochs=100, lr=0.005,
                       batch_size=512, neg_ratio=1, seed=0):
    """
    Symmetric SGD matrix factorization: minimize ||M - ZZ^T||_F^2
    using mini-batches of positive edges + random negative samples.

    At convergence Z ≈ V sqrt(Λ_+) where A = VΛV^T (top positive part).
    Negative sampling implicitly regularizes by penalizing large off-diagonal scores.
    """
    local_rng = np.random.default_rng(seed)
    N = A.shape[0]
    A_coo = A.tocoo()
    pos_i = A_coo.row.astype(np.int32)
    pos_j = A_coo.col.astype(np.int32)
    vals  = A_coo.data.astype(np.float32)
    n_edges = len(pos_i)

    Z = local_rng.normal(0, 0.01, (N, embed_dim)).astype(np.float64)

    for epoch in range(n_epochs):
        perm = local_rng.permutation(n_edges)
        pi, pj, pv = pos_i[perm], pos_j[perm], vals[perm]

        for start in range(0, n_edges, batch_size):
            end = min(start + batch_size, n_edges)
            bi, bj, bv = pi[start:end], pj[start:end], pv[start:end]
            bs = end - start

            # --- positive pairs (A_ij = 1) ---
            zi, zj   = Z[bi], Z[bj]
            scores   = (zi * zj).sum(axis=1)         # (bs,)
            err_pos  = bv - scores                    # target = A_ij value

            dZ = np.zeros_like(Z)
            np.add.at(dZ, bi, err_pos[:, None] * zj)
            np.add.at(dZ, bj, err_pos[:, None] * zi)

            # --- negative samples (A_ij ≈ 0) ---
            n_neg = bs * neg_ratio
            ni = local_rng.integers(0, N, n_neg)
            nj = local_rng.integers(0, N, n_neg)
            zni, znj  = Z[ni], Z[nj]
            err_neg   = -(zni * znj).sum(axis=1)     # target = 0

            np.add.at(dZ, ni, err_neg[:, None] * znj)
            np.add.at(dZ, nj, err_neg[:, None] * zni)

            Z += (lr / bs) * dZ

    return Z


def ojas_rule(A, embed_dim=2, n_epochs=20, lr=0.005, seed=0):
    """
    Oja's rule (online stochastic PCA) on rows of the adjacency matrix.
    Finds top-d eigenvectors of A^T A = A^2 (same directions as eigvecs of A).
    Returns Z = A @ W  (N x d), which clusters like the standard spectral embedding.
    """
    local_rng = np.random.default_rng(seed)
    N = A.shape[0]
    W = local_rng.normal(0, 0.01, (N, embed_dim))
    # Orthonormalize columns
    W, _ = np.linalg.qr(W)
    W = W[:, :embed_dim]

    for epoch in range(n_epochs):
        perm = local_rng.permutation(N)
        for i in perm:
            x = A[i].toarray().flatten()    # row i of A
            for k in range(embed_dim):
                y = W[:, k] @ x
                W[:, k] += lr * (y * x - y**2 * W[:, k])
                # Gram-Schmidt deflation
                for j in range(k):
                    W[:, k] -= (W[:, k] @ W[:, j]) * W[:, j]
                norm = np.linalg.norm(W[:, k])
                if norm > 1e-10:
                    W[:, k] /= norm

    Z = A @ W   # N x d
    return Z


def kmeans2(Z, seed=0):
    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    return km.fit_predict(Z)


# =============================================================================
# Run experiment
# =============================================================================
methods = ["spectral_sign", "spectral_km", "ojas", "sgd_A", "sgd_normed", "bp"]
nmi = {m: np.zeros(n_samples) for m in methods}

for s in range(n_samples):
    # Generate network
    g     = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
    edges = g.get_edgelist()
    if edges:
        rows, cols = zip(*edges)
        A = sp.csr_matrix(
            (np.ones(len(rows)), (np.array(rows), np.array(cols))),
            shape=(N, N),
        )
        A = A + A.T
        A.data[:] = 1.0
    else:
        A = sp.csr_matrix((N, N))

    # 1. Standard spectral — sign of v2
    vals, V = eigsh(A.astype(float), k=3, which="LA")
    idx = np.argsort(vals)[::-1]
    V   = V[:, idx]
    v2  = V[:, 1]
    nmi["spectral_sign"][s] = normalized_mutual_info_score(membership, (v2 >= 0).astype(int))

    # 2. Spectral K-means on top eigvecs
    nmi["spectral_km"][s] = normalized_mutual_info_score(membership, kmeans2(V[:, :2], seed=s))

    # 3. Oja's rule
    Z_ojas = ojas_rule(A, embed_dim=embed_dim, n_epochs=20, lr=0.005, seed=s)
    nmi["ojas"][s] = normalized_mutual_info_score(membership, kmeans2(Z_ojas, seed=s))

    # 4. SGD symmetric factorization of A
    Z_sgd = sgd_factorization(A, embed_dim=embed_dim, n_epochs=80, lr=0.005,
                               batch_size=512, neg_ratio=2, seed=s)
    nmi["sgd_A"][s] = normalized_mutual_info_score(membership, kmeans2(Z_sgd, seed=s))

    # 5. SGD on degree-normalized A
    A_norm = degree_normalize(A)
    Z_sgd_n = sgd_factorization(A_norm, embed_dim=embed_dim, n_epochs=80, lr=0.005,
                                 batch_size=512, neg_ratio=2, seed=s)
    nmi["sgd_normed"][s] = normalized_mutual_info_score(membership, kmeans2(Z_sgd_n, seed=s))

    # 6. Belief propagation
    labels_bp = bp.detect(A.copy(), q=2, init_memberships=membership)
    nmi["bp"][s] = normalized_mutual_info_score(membership, labels_bp)

    print(
        f"[{s+1:2d}/{n_samples}] "
        f"spec_sign={nmi['spectral_sign'][s]:.3f}  "
        f"spec_km={nmi['spectral_km'][s]:.3f}  "
        f"ojas={nmi['ojas'][s]:.3f}  "
        f"sgd_A={nmi['sgd_A'][s]:.3f}  "
        f"sgd_norm={nmi['sgd_normed'][s]:.3f}  "
        f"bp={nmi['bp'][s]:.3f}"
    )


# =============================================================================
# Results
# =============================================================================
labels = {
    "spectral_sign": "Spectral (sign v₂)",
    "spectral_km":   "Spectral (K-means)",
    "ojas":          "Oja's rule",
    "sgd_A":         "SGD factorize A",
    "sgd_normed":    "SGD factorize D⁻½AD⁻½",
    "bp":            "Belief propagation",
}

print("\n--- NMI Results ---")
for m in methods:
    arr = nmi[m]
    print(f"  {labels[m]:30s}  {arr.mean():.4f} ± {arr.std():.4f}")

# -- plot --
fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(methods))
means  = [nmi[m].mean() for m in methods]
stds   = [nmi[m].std()  for m in methods]
colors = sns.color_palette("Set2", len(methods))

bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in methods], rotation=25, ha="right")
ax.set_ylabel("NMI")
ax.set_title(f"SGD Autoencoder vs Spectral vs BP\n(N={N}, cave={cave}, μ={mu}, {n_samples} samples)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
sns.despine()
fig.tight_layout()

print(f"\nFigure: Bar chart of NMI ± std for {len(methods)} methods.")
print("x-axis: method, y-axis: NMI, error bars = 1 std over 30 samples.")

# -- 2D embedding comparison for one sample (last) --
fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
sample_data = {
    "Spectral v₂ vs v₃":       V[:, 1:3],
    "SGD factorize A":          Z_sgd,
    "SGD factorize D⁻½AD⁻½":   Z_sgd_n,
}
colors2 = ["#E74C3C", "#3498DB"]
for ax2, (title, Z2) in zip(axes, sample_data.items()):
    for c in range(2):
        mask = membership == c
        ax2.scatter(Z2[mask, 0], Z2[mask, 1], s=8, alpha=0.4,
                    color=colors2[c], label=f"community {c}")
    ax2.set_title(title)
    ax2.set_xlabel("dim 1")
    ax2.set_ylabel("dim 2")
    sns.despine(ax=ax2)
axes[0].legend(markerscale=3, fontsize=10)
fig2.suptitle(f"2D embeddings (last sample, μ={mu})")
fig2.tight_layout()

print("Figure 2: 2D scatter of node embeddings for three methods (last sample).")
print("Red = community 0, Blue = community 1.")

plt.show()
