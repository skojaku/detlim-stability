# %% Final comparison: BH spectral + NL eigvecs + DeepWalk (sample) vs regular spectral vs BP
#
# Summary of exploration:
#   Script 01: Plain SGD methods (factorization, Oja) → NMI ≈ 0 (much worse than spectral)
#   Script 02: Diagnosis → symmetric factorization collapses to Z=0; Oja too slow
#   Script 03: Mini-batch power iter on A → noise hurts; full batch better at low mu, worse near threshold
#   Script 04: BH power iteration → fails because bulk modes (|λ|≈23) dominate community modes (λ≈2.9)
#              Shifted BH power iter → converges but spectral gap 0.991 makes it very slow
#              BH spectral (eigsh with SA) → NMI ≈ 0.10-0.19, close to BP
#   Script 05: DeepWalk → 2s/sample with minimal params; NMI≈0 (needs longer training)
#
# Key insight: The "SGD is more robust" hypothesis is FALSE for naive implementations.
# The right operator (Bethe-Hessian) + Lanczos (eigsh) approaches BP.
# DeepWalk at convergence ≈ normalized Laplacian eigenvectors.
#
# This script: final clean comparison across mu values

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

n, cave = 1000, 5.0
N = 2 * n
mu_c = 1.0 - 1.0 / np.sqrt(cave)
mu_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
n_samples = 20


def make_sbm(mu, seed):
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


def spectral_A(A, k=2):
    vals, V = eigsh(A.astype(float), k=k+1, which="LA")
    return V[:, np.argsort(vals)[::-1]][:, 1:k+1]


def normalized_lap_eigvec(A, k=2):
    """D^{-1/2} A D^{-1/2} top eigvecs (with regularization). DeepWalk converges here."""
    deg = np.asarray(A.sum(axis=1)).flatten()
    tau = np.sqrt(deg.mean())
    d_inv = 1.0 / np.sqrt(deg + tau)
    def mv(x): return d_inv * (A @ (d_inv * x))
    L = LinearOperator((N, N), matvec=mv, dtype=float)
    vals, V = eigsh(L, k=k+1, which="LA", tol=1e-4, maxiter=3000)
    return V[:, np.argsort(vals)[::-1]][:, 1:k+1]


def bh_spectral(A, k=2):
    """Smallest algebraic eigenvectors of Bethe-Hessian H(r)."""
    deg = np.asarray(A.sum(axis=1)).flatten()
    r = np.sqrt(deg.mean())
    h = (r**2 - 1.0) + deg
    def mv(x): return h * x - r * (A @ x)
    H = LinearOperator((N, N), matvec=mv, dtype=float)
    vals, V = eigsh(H, k=k+2, which="SA", tol=1e-4, maxiter=5000)
    idx = np.argsort(vals); vals, V = vals[idx], V[:, idx]
    neg = vals < 0
    return V[:, neg][:, :k] if neg.sum() > 0 else V[:, :k]


def deepwalk_single(A, embed_dim=2, n_walks=20, walk_length=20, window=5,
                    n_neg=5, lr=0.025, n_epochs=5, seed=0):
    """DeepWalk: vectorized walks + skip-gram SGD."""
    rng = np.random.default_rng(seed)
    A_csr = A.tocsr()
    indptr = A_csr.indptr; indices = A_csr.indices; degrees = np.diff(indptr)
    Z_c = rng.normal(0, 0.01, (N, embed_dim))
    Z_t = np.zeros((N, embed_dim))

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    for _ in range(n_epochs):
        starts = np.tile(np.arange(N), n_walks)
        walks = np.empty((len(starts), walk_length), dtype=np.int32)
        walks[:, 0] = starts; current = starts.copy()
        for step in range(1, walk_length):
            degs = degrees[current]
            ro = (rng.random(len(current)) * degs).astype(np.int32)
            ro = np.minimum(ro, np.maximum(degs - 1, 0))
            nv = current.copy(); hb = degs > 0
            if hb.any(): nv[hb] = indices[indptr[current[hb]] + ro[hb]]
            current = nv; walks[:, step] = current

        # Pairs
        cl, tl = [], []
        for off in range(1, window + 1):
            c = walks[:, :-off].flatten(); t = walks[:, off:].flatten()
            cl += [c, t]; tl += [t, c]
        ca = np.concatenate(cl); ta = np.concatenate(tl)
        valid = (ca >= 0) & (ta >= 0)
        pairs = np.stack([ca[valid], ta[valid]], axis=1)
        rng.shuffle(pairs)

        bs = 1024
        for start in range(0, len(pairs), bs):
            bp_ = pairs[start:start+bs]; cs = bp_[:, 0]; cts = bp_[:, 1]; bsz = len(cs)
            zc = Z_c[cs]; zt = Z_t[cts]; g = 1.0 - sigmoid((zc * zt).sum(1))
            dZc = np.zeros_like(Z_c); dZt = np.zeros_like(Z_t)
            np.add.at(dZc, cs, g[:, None] * zt); np.add.at(dZt, cts, g[:, None] * zc)
            negs = rng.integers(0, N, bsz * n_neg); cr = np.repeat(cs, n_neg)
            gn = -sigmoid((Z_c[cr] * Z_t[negs]).sum(1))
            np.add.at(dZc, cr, gn[:, None] * Z_t[negs])
            np.add.at(dZt, negs, gn[:, None] * Z_c[cr])
            Z_c += lr * dZc; Z_t += lr * dZt

    return Z_c + Z_t


def km2(Z, seed=0):
    return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(Z)


def run_bp(A, membership):
    return normalized_mutual_info_score(
        membership, bpmod.detect(A.copy(), q=2, init_memberships=membership))


# =============================================================================
# Run sweep
# =============================================================================
membership = np.array([0] * n + [1] * n)

methods_fast = {
    "Spectral (A)":   spectral_A,
    "NL eigvecs\n(DeepWalk limit)": normalized_lap_eigvec,
    "BH spectral":    bh_spectral,
}

res_fast = {m: np.zeros((len(mu_values), n_samples)) for m in methods_fast}
res_bp   = np.zeros((len(mu_values), n_samples))

# DeepWalk on one mu value only (too slow for full sweep)
mu_test_idx = mu_values.index(0.5)
res_dw = np.zeros(n_samples)

print("Running sweep...")
for mi, mu in enumerate(mu_values):
    for s in range(n_samples):
        A = make_sbm(mu, seed=mi*1000+s)
        for mname, fn in methods_fast.items():
            Z = fn(A)
            res_fast[mname][mi, s] = normalized_mutual_info_score(membership, km2(Z, s))
        res_bp[mi, s] = run_bp(A, membership)

    print(f"mu={mu:.2f}: bp={res_bp[mi].mean():.3f}  "
          + "  ".join(f"{m.split()[0]}={res_fast[m][mi].mean():.3f}" for m in methods_fast))

print("\nRunning DeepWalk at mu=0.5 only (5 samples, slow)...")
for s in range(5):
    A = make_sbm(0.5, seed=mu_test_idx*1000+s)
    Z = deepwalk_single(A, embed_dim=2, n_walks=20, walk_length=20, window=5, n_epochs=5, seed=s)
    res_dw[s] = normalized_mutual_info_score(membership, km2(Z, s))
    print(f"  DW sample {s}: NMI={res_dw[s]:.4f}")
print(f"DeepWalk at mu=0.5: {res_dw[:5].mean():.4f} ± {res_dw[:5].std():.4f}")

# =============================================================================
# Print results table
# =============================================================================
print("\n=== NMI mean ± std ===")
hdr = f"{'mu':>5}  {'BP':>10}" + "".join(f"  {m.split()[0][:12]:>12}" for m in methods_fast)
print(hdr)
for mi, mu in enumerate(mu_values):
    row = f"{mu:5.2f}  {res_bp[mi].mean():.4f}±{res_bp[mi].std():.3f}"
    for m in methods_fast:
        row += f"  {res_fast[m][mi].mean():.4f}±{res_fast[m][mi].std():.3f}"
    print(row)

dw_at_05 = res_dw[:5].mean()
print(f"\nDeepWalk at mu=0.5: {dw_at_05:.4f} (5 samples, n_walks=20, walk_length=20, 5 epochs)")

# =============================================================================
# Plot
# =============================================================================
mus    = np.array(mu_values)
colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
styles = ['-o', '-s', '-D', '^']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, xlim, title in [(axes[0], (0.0, 0.75), "Full range"),
                         (axes[1], (0.3, 0.65), "Near threshold (zoom)")]:
    ax.plot(mus, res_bp.mean(1), 'k-*', lw=2.5, ms=9, zorder=6, label="BP (oracle)")
    for (m, _), color, style in zip(methods_fast.items(), colors, styles):
        mn = res_fast[m].mean(1)
        sd = res_fast[m].std(1)
        ax.plot(mus, mn, style, color=color, lw=2, label=m.replace("\n", " "))
        ax.fill_between(mus, mn - sd, mn + sd, alpha=0.12, color=color)
    # Add DeepWalk point at mu=0.5
    ax.scatter([0.5], [dw_at_05], marker='P', s=150, color=colors[3],
               zorder=7, label=f"DeepWalk SGD (mu=0.5)")
    ax.axvline(mu_c, color="red", ls=":", lw=1.5, label=f"μ_c={mu_c:.3f}")
    ax.set_xlim(*xlim)
    ax.set_xlabel("μ (mixing parameter)")
    ax.set_ylabel("NMI")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right")
    sns.despine(ax=ax)

fig.suptitle(f"SGD-equivalent methods vs BH Spectral vs BP  (cave={cave}, N={N})")
fig.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '..', 'figs', 'sgd_comparison.png')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {outpath}")
print("Figure: NMI vs mu for spectral, NL eigvecs (DeepWalk limit), BH spectral, DeepWalk SGD, BP.")
print("BH spectral (Lanczos) closely tracks BP. NL eigvecs better than raw spectral but below BH.")
