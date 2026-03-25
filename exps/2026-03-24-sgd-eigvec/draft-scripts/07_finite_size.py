# %% Finite-size scaling: does BH spectral → BP as N → ∞?
# Theory predicts both reach the KS threshold at the same mu_c.
# For finite N, BH spectral is ~91% of BP at mu=0.5, N=2000.
# Question: does the gap shrink with N?

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'libs', 'BeliefPropagation'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
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

cave  = 5.0
mu    = 0.5          # just below mu_c ≈ 0.553
n_sizes = [250, 500, 1000, 2000, 4000]   # n per community → N = 2n
n_samples = 20


def make_sbm(n, cave, mu, seed):
    N = 2 * n
    c_out = mu * cave; c_in = 2 * cave - c_out
    p_in, p_out = c_in / N, c_out / N
    np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
    edges = g.get_edgelist()
    if not edges: return sp.csr_matrix((N, N))
    rows, cols = zip(*edges)
    A = sp.csr_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(N, N))
    A = A + A.T; A.data[:] = 1.0
    return A


def bh_spectral(A):
    deg = np.asarray(A.sum(axis=1)).flatten()
    r   = np.sqrt(deg.mean())
    h   = (r**2 - 1.0) + deg
    def mv(x): return h * x - r * (A @ x)
    N   = A.shape[0]
    H   = LinearOperator((N, N), matvec=mv, dtype=float)
    vals, V = eigsh(H, k=4, which="SA", tol=1e-5, maxiter=5000)
    idx = np.argsort(vals); vals, V = vals[idx], V[:, idx]
    neg = vals < 0
    return V[:, neg][:, :2] if neg.sum() > 0 else V[:, :2]


def km2(Z, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(Z)


results_bh = np.zeros((len(n_sizes), n_samples))
results_bp = np.zeros((len(n_sizes), n_samples))

for ni, n in enumerate(n_sizes):
    N = 2 * n
    membership = np.array([0]*n + [1]*n)
    for s in range(n_samples):
        A = make_sbm(n, cave, mu, seed=ni*1000+s)
        Z = bh_spectral(A)
        results_bh[ni, s] = normalized_mutual_info_score(membership, km2(Z, s))
        results_bp[ni, s] = normalized_mutual_info_score(
            membership, bpmod.detect(A.copy(), q=2, init_memberships=membership))
    print(f"N={N:5d}: BH={results_bh[ni].mean():.4f}±{results_bh[ni].std():.3f}  "
          f"BP={results_bp[ni].mean():.4f}±{results_bp[ni].std():.3f}  "
          f"ratio={results_bh[ni].mean()/max(results_bp[ni].mean(),1e-6):.3f}")

# -- plot --
Ns = np.array([2*n for n in n_sizes])
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(Ns, results_bp.mean(1), yerr=results_bp.std(1),
            fmt='k-*', ms=9, lw=2, capsize=4, label="BP")
ax.errorbar(Ns, results_bh.mean(1), yerr=results_bh.std(1),
            fmt='D--', color="#4CAF50", ms=8, lw=2, capsize=4, label="BH spectral")
ax.set_xscale("log")
ax.set_xlabel("N (total nodes, log scale)")
ax.set_ylabel("NMI")
ax.set_title(f"Finite-size scaling at mu={mu}, cave={cave}\n(mu_c={1-1/np.sqrt(cave):.3f})")
ax.legend()
sns.despine()
fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), '..', 'figs', 'finite_size_scaling.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {outpath}")
print("Figure: NMI vs N (log scale) for BH spectral and BP at fixed mu=0.5.")
print("If BH → BP as N→∞, the curves should converge at large N.")
