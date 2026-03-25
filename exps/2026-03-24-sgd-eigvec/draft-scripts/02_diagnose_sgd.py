# %% Diagnose why SGD fails + test Rayleigh quotient objective
#
# Previous run showed SGD (Oja, symmetric factorization) far worse than spectral.
# Diagnoses:
#   A) Convergence curves for Oja's rule vs epochs (is 20 enough?)
#   B) Z norm collapse in symmetric factorization (trivial solution?)
#   C) Rayleigh quotient SGD: directly maximize z^T A z / ||z||^2
#   D) Effect of degree normalization on Rayleigh quotient SGD

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

sns.set_style("white")
sns.set_context("talk", font_scale=1.2)

# -- params --
n, N, cave, mu = 1000, 2000, 5.0, 0.5
c_out = mu * cave
c_in  = 2 * cave - c_out
p_in, p_out = c_in / N, c_out / N
membership = np.array([0] * n + [1] * n)
rng = np.random.default_rng(0)

# Generate one fixed network for diagnosis
g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
edges = g.get_edgelist()
rows, cols = zip(*edges)
A = sp.csr_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(N, N))
A = A + A.T; A.data[:] = 1.0

# Ground truth spectral
vals, V = eigsh(A.astype(float), k=3, which="LA")
idx = np.argsort(vals)[::-1]; vals = vals[idx]; V = V[:, idx]
v2  = V[:, 1]
nmi_spectral = normalized_mutual_info_score(membership, (v2 >= 0).astype(int))
print(f"Spectral (sign v2) NMI = {nmi_spectral:.4f}")
print(f"Eigenvalues: {vals}")

# =============================================================================
# A) Oja's rule convergence vs epochs
# =============================================================================
print("\n--- A) Oja's rule convergence ---")

def ojas_embedding(A, embed_dim=2, n_epochs=1, W_init=None, lr=0.005, seed=0):
    """One or more epochs of Oja's rule; returns (W, Z) after each epoch."""
    local_rng = np.random.default_rng(seed)
    N = A.shape[0]
    if W_init is None:
        W = local_rng.normal(0, 0.01, (N, embed_dim))
        W, _ = np.linalg.qr(W); W = W[:, :embed_dim]
    else:
        W = W_init.copy()

    for _ in range(n_epochs):
        perm = local_rng.permutation(N)
        for i in perm:
            x = A[i].toarray().flatten()
            for k in range(embed_dim):
                y = W[:, k] @ x
                W[:, k] += lr * (y * x - y**2 * W[:, k])
                for j in range(k):
                    W[:, k] -= (W[:, k] @ W[:, j]) * W[:, j]
                nm = np.linalg.norm(W[:, k])
                if nm > 1e-10: W[:, k] /= nm
    Z = A @ W
    return W, Z

epoch_list = [1, 5, 10, 20, 50, 100, 200]
nmi_ojas_epochs = []
W = None
prev = 0
for ep in epoch_list:
    n_extra = ep - prev
    W, Z = ojas_embedding(A, embed_dim=2, n_epochs=n_extra, W_init=W, lr=0.005, seed=0)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Z)
    nmi_ep = normalized_mutual_info_score(membership, km)
    nmi_ojas_epochs.append(nmi_ep)
    prev = ep
    print(f"  Epochs={ep:4d}:  NMI={nmi_ep:.4f}")

# =============================================================================
# B) Z-norm collapse in symmetric factorization
# =============================================================================
print("\n--- B) Symmetric factorization Z-norm over epochs ---")

A_coo = A.tocoo()
pos_i = A_coo.row.astype(np.int32)
pos_j = A_coo.col.astype(np.int32)
n_edges = len(pos_i)

Z_sgd = rng.normal(0, 0.01, (N, 2))
lr_sgd = 0.005
norms, losses_pos, nmi_sgd_epochs = [], [], []
epoch_check = [1, 5, 10, 20, 50, 80, 100, 200]

for epoch in range(1, max(epoch_check)+1):
    perm = rng.permutation(n_edges)
    pi, pj = pos_i[perm], pos_j[perm]
    epoch_loss = 0
    bs = 512
    for start in range(0, n_edges, bs):
        bi, bj = pi[start:start+bs], pj[start:start+bs]
        bs_ = len(bi)
        zi, zj = Z_sgd[bi], Z_sgd[bj]
        scores = (zi * zj).sum(axis=1)
        err_pos = 1.0 - scores
        dZ = np.zeros_like(Z_sgd)
        np.add.at(dZ, bi, err_pos[:, None] * zj)
        np.add.at(dZ, bj, err_pos[:, None] * zi)
        # negative samples
        ni = rng.integers(0, N, bs_ * 2)
        nj = rng.integers(0, N, bs_ * 2)
        zni, znj = Z_sgd[ni], Z_sgd[nj]
        err_neg = -(zni * znj).sum(axis=1)
        np.add.at(dZ, ni, err_neg[:, None] * znj)
        np.add.at(dZ, nj, err_neg[:, None] * zni)
        Z_sgd += (lr_sgd / bs_) * dZ
        epoch_loss += ((err_pos**2).sum() + (err_neg**2).sum()) / bs_

    if epoch in epoch_check:
        norm_ = np.linalg.norm(Z_sgd, axis=1).mean()
        norms.append(norm_)
        km = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(Z_sgd)
        nmi_ep = normalized_mutual_info_score(membership, km)
        nmi_sgd_epochs.append(nmi_ep)
        print(f"  Epoch={epoch:4d}:  mean|z|={norm_:.4f}  NMI={nmi_ep:.4f}  loss={epoch_loss:.4f}")

# =============================================================================
# C) Rayleigh quotient SGD: maximize z^T A z / ||z||^2
# =============================================================================
print("\n--- C) Rayleigh quotient SGD ---")

def rayleigh_sgd(A, embed_dim=2, n_epochs=100, lr=0.01, batch_size=512, seed=0):
    """
    Mini-batch Rayleigh quotient maximization.
    For a single eigenvector z: maximize z^T A z / z^T z
    Extended to d components via sequential deflation.

    Mini-batch gradient: estimate z^T A z via sample of edges.
    Gradient: g = 2*Az_batch / ||z||^2 - 2*(z^T A z_batch / ||z||^4) * z
    Simplified (normalize z after each step):
      z <- z + lr * (mini-batch estimate of Az)
      then orthogonalize and normalize
    """
    local_rng = np.random.default_rng(seed)
    N = A.shape[0]
    A_coo = A.tocoo()
    ei, ej = A_coo.row, A_coo.col
    n_edges = len(ei)

    # Initialize: random orthonormal columns
    Z = local_rng.normal(0, 1.0, (N, embed_dim))
    Z, _ = np.linalg.qr(Z)

    nmi_curve = []
    for epoch in range(n_epochs):
        perm = local_rng.permutation(n_edges)
        bi_all, bj_all = ei[perm], ej[perm]

        for start in range(0, n_edges, batch_size):
            bi = bi_all[start:start+batch_size]
            bj = bj_all[start:start+batch_size]
            bs_ = len(bi)

            # Estimate gradient = A @ Z restricted to this edge batch
            # (A @ Z)[i] = sum_j A[i,j] * Z[j] ≈ (N/bs) * mean over sampled neighbors
            # Scale: each edge (i,j) contributes Z[j] to gradient of i and Z[i] to j
            dZ = np.zeros_like(Z)
            np.add.at(dZ, bi, Z[bj])    # dZ[i] += Z[j] for edge (i,j)
            np.add.at(dZ, bj, Z[bi])    # dZ[j] += Z[i] for edge (i,j) (symmetric)
            # Scale to estimate full A@Z
            scale = n_edges / bs_
            dZ *= scale

            # Rayleigh quotient gradient: g = 2*(A@Z - (z^T A z / z^T z) Z) / ||z||^2
            # Since we'll normalize, just use: Z <- Z + lr * A@Z, then QR
            Z += (lr / n_edges) * dZ

        # Re-orthonormalize (QR decomposition)
        Z, _ = np.linalg.qr(Z)

        if (epoch + 1) % 10 == 0:
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(Z)
            nmi_ep = normalized_mutual_info_score(membership, km)
            nmi_curve.append((epoch+1, nmi_ep))
            print(f"  Epoch={epoch+1:4d}:  NMI={nmi_ep:.4f}")

    return Z, nmi_curve

Z_rq, nmi_rq_curve = rayleigh_sgd(A, embed_dim=2, n_epochs=100, lr=0.01, batch_size=512)
print(f"Rayleigh SGD final NMI: {nmi_rq_curve[-1][1]:.4f}")

# =============================================================================
# D) Degree-normalized A + Rayleigh quotient SGD
# =============================================================================
print("\n--- D) Rayleigh quotient SGD on degree-normalized A ---")
deg   = np.asarray(A.sum(axis=1)).flatten()
tau   = np.sqrt(deg.mean())
d_inv = 1.0 / np.sqrt(deg + tau)
A_norm = sp.diags(d_inv) @ A @ sp.diags(d_inv)

Z_rq_n, nmi_rq_n_curve = rayleigh_sgd(A_norm, embed_dim=2, n_epochs=100, lr=0.01, batch_size=512)
print(f"Rayleigh SGD (norm) final NMI: {nmi_rq_n_curve[-1][1]:.4f}")

# =============================================================================
# Summary plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# A: Oja convergence
ax = axes[0]
ax.plot(epoch_list, nmi_ojas_epochs, 'o-', color=sns.color_palette("Set2")[0], lw=2)
ax.axhline(nmi_spectral, color='gray', ls='--', lw=1.5, label=f'Spectral {nmi_spectral:.3f}')
ax.set_xlabel("Epochs (Oja's rule)")
ax.set_ylabel("NMI")
ax.set_title("Oja's Rule Convergence")
ax.legend(fontsize=10)
sns.despine(ax=ax)

# B: Z-norm and NMI for factorization
ax = axes[1]
ax2_twin = ax.twinx()
ax.plot(epoch_check[:len(nmi_sgd_epochs)], nmi_sgd_epochs, 'o-',
        color=sns.color_palette("Set2")[1], lw=2, label='NMI')
ax2_twin.plot(epoch_check[:len(norms)], norms, 's--',
              color=sns.color_palette("Set2")[2], lw=2, label='mean |z|')
ax.axhline(nmi_spectral, color='gray', ls='--', lw=1.5)
ax.set_xlabel("Epochs (SGD factorize A)")
ax.set_ylabel("NMI", color=sns.color_palette("Set2")[1])
ax2_twin.set_ylabel("mean ||z||", color=sns.color_palette("Set2")[2])
ax.set_title("SGD Factorization: NMI & Z norm")
sns.despine(ax=ax)

# C+D: Rayleigh quotient
ax = axes[2]
ep_rq  = [e for e, _ in nmi_rq_curve]
nmi_rq = [nmi for _, nmi in nmi_rq_curve]
ep_rqn  = [e for e, _ in nmi_rq_n_curve]
nmi_rqn = [nmi for _, nmi in nmi_rq_n_curve]
ax.plot(ep_rq, nmi_rq, 'o-', color=sns.color_palette("Set2")[3], lw=2, label='Rayleigh SGD (A)')
ax.plot(ep_rqn, nmi_rqn, 's-', color=sns.color_palette("Set2")[4], lw=2, label='Rayleigh SGD (norm)')
ax.axhline(nmi_spectral, color='gray', ls='--', lw=1.5, label=f'Spectral {nmi_spectral:.3f}')
ax.set_xlabel("Epochs (Rayleigh quotient SGD)")
ax.set_ylabel("NMI")
ax.set_title("Rayleigh Quotient SGD")
ax.legend(fontsize=10)
sns.despine(ax=ax)

fig.suptitle(f"SGD Convergence Diagnostics (N={N}, cave={cave}, μ={mu})")
fig.tight_layout()

print("\nFigure: 3 panels showing convergence curves for Oja's rule, SGD factorization,")
print("and Rayleigh quotient SGD. Dashed gray = spectral baseline.")
print("Panel 1: Oja's rule NMI vs epochs.")
print("Panel 2: SGD factorization NMI (left axis) + Z-norm (right axis) vs epochs.")
print("Panel 3: Rayleigh quotient SGD NMI vs epochs, raw A and degree-normalized A.")

plt.show()
