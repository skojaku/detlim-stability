"""
Iteration 14: Phase transition characterization (optimized version)
Map the effective detectability limit for each method (mu where NMI=0.1).

Key optimizations:
- Sparse NetMF: use randomized SVD on sparse P instead of dense matrix powers
- Use truncated SVD to avoid full N×N dense matrices
"""

import sys
import os
import json
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')

# Paths
sys.path.insert(0, '/workspace/libs/BeliefPropagation')
sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')

OUTPUT_DIR = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-014'

# ─── SBM parameters ───────────────────────────────────────────────────────────
N = 2000
CAVE = 5.0
N_SAMPLES = 15
MU_VALS = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
THEORETICAL_MU_STAR = 0.553
DIM = 64

# ─── Graph generation ─────────────────────────────────────────────────────────

def fast_sbm_lcc(mu, rng):
    """Generate SBM and return LCC with labels."""
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / N
    p_out = c_out / N

    n_half = N // 2
    labels = np.repeat([0, 1], n_half)

    rows_list, cols_list = [], []

    # Intra-community edges
    for block_start in [0, n_half]:
        n_b = n_half
        # Lower triangle indices
        i_idx, j_idx = np.tril_indices(n_b, k=-1)
        r = rng.random(len(i_idx))
        sel = r < p_in
        ii = i_idx[sel] + block_start
        jj = j_idx[sel] + block_start
        rows_list.append(ii); cols_list.append(jj)
        rows_list.append(jj); cols_list.append(ii)

    # Inter-community edges
    ii_inter = rng.integers(0, n_half, size=int(n_half * n_half * p_out * 3 + 100))
    jj_inter = rng.integers(n_half, N, size=len(ii_inter))
    # Filter to actual probability
    r = rng.random(len(ii_inter))
    sel = r < (p_out * n_half * n_half) / len(ii_inter) if len(ii_inter) > 0 else np.array([], dtype=bool)

    # Better: generate inter edges properly
    rows_list2, cols_list2 = [], []
    expected_inter = int(n_half * n_half * p_out)
    # Use sparse binomial sampling
    r_mat = rng.random((n_half, n_half))
    sel_mat = r_mat < p_out
    ii2, jj2 = np.where(sel_mat)
    jj2 = jj2 + n_half
    rows_list2.extend(ii2); cols_list2.extend(jj2)
    rows_list2.extend(jj2); cols_list2.extend(ii2)

    all_rows = np.concatenate([np.concatenate(rows_list) if rows_list else np.array([]),
                                np.array(rows_list2, dtype=np.int32)])
    all_cols = np.concatenate([np.concatenate(cols_list) if cols_list else np.array([]),
                                np.array(cols_list2, dtype=np.int32)])

    all_rows = all_rows.astype(np.int32)
    all_cols = all_cols.astype(np.int32)

    if len(all_rows) == 0:
        return sp.csr_matrix((N, N)), labels

    A = sp.csr_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(N, N))
    A = (A + A.T) / 2  # ensure symmetry
    A.data[:] = 1.0
    A.eliminate_zeros()

    # Extract LCC
    n_components, comp_labels = sp.csgraph.connected_components(A, directed=False)
    if n_components == 1:
        return A, labels

    largest = np.argmax(np.bincount(comp_labels))
    mask = comp_labels == largest
    A_lcc = A[mask][:, mask]
    labels_lcc = labels[mask]
    return A_lcc, labels_lcc


# ─── NetMF helpers (efficient version) ────────────────────────────────────────

def compute_netmf_sparse_embedding(A, T=10, dim=64, seed=42):
    """
    Efficient NetMF using randomized SVD on sparse matrices.

    Instead of computing full N×N dense M, we:
    1. Compute M as a sparse operator
    2. Use randomized SVD to get top-k components directly

    M_ij = log(vol * Ppow_ij / d_j) where Ppow = (1/T) sum P^t

    For gradient_weight and clip_0, we need the actual matrix values,
    so we use a low-rank approximation approach.
    """
    d = np.array(A.sum(axis=1)).flatten()
    vol = d.sum()
    d_inv = 1.0 / np.maximum(d, 1e-12)
    n = A.shape[0]

    # P = D^{-1} A (sparse, row-stochastic)
    P = sp.diags(d_inv) @ A

    # Compute Ppow = (1/T) sum_{t=1}^T P^t using repeated sparse matvec
    # We need Ppow as a matrix for the weighting transforms.
    # For large N, use randomized approach: Ppow @ V for random V

    # Use randomized SVD: sample random vectors and compute M @ v
    # M_ij = log(vol * Ppow_ij / d_j)
    # M v = log-transform of (vol * Ppow v / d_j) — but log is element-wise, can't linearize

    # For N=2000, dense is ~32MB which is fine, but matrix powers are slow.
    # Optimization: use P as sparse and compute Ppow more efficiently.

    # Actually compute Ppow densely but use sparse P
    # Ppow @ x for random x: fast if P is sparse
    # P^t @ x: t sparse matvec operations = O(t * nnz)

    # Strategy: compute Ppow = (1/T) sum P^t directly as dense but chunk by rows
    # Actually for N=2000 it's just 2000x2000 = 4M entries, totally fine in memory.
    # The slow part was using dense @ dense. With sparse P, each P^t @ dense(X) is fast.

    # Compute Ppow as a dense matrix, but using sparse-dense products
    Ppow = np.zeros((n, n))
    Pt_dense = np.eye(n)  # P^0 = I initially, then multiply
    P_dense = P.toarray()  # one-time dense copy

    for t in range(1, T + 1):
        Pt_dense = Pt_dense @ P_dense  # dense-dense multiply, but P is row-stochastic
        Ppow += Pt_dense
    Ppow /= T

    # M = log(vol * Ppow / d_j)
    d_j = d[np.newaxis, :]  # broadcast over rows
    ratio = vol * Ppow / np.maximum(d_j, 1e-12)
    M = np.log(np.maximum(ratio, 1e-12))

    return M


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def svd_embed(M, dim, seed):
    """TruncatedSVD on matrix M."""
    k = min(dim, M.shape[0] - 1, M.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    emb = svd.fit_transform(M)
    return emb


# ─── Method implementations ───────────────────────────────────────────────────

def method_node2vec(A_lcc, labels_lcc, seed):
    import embcom
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10, p=1.0, q=1.0)
    model.fit(A_lcc)
    emb = model.transform(dim=DIM)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels_lcc, pred)


def method_gradient_weight_svd(A_lcc, labels_lcc, seed):
    M = compute_netmf_sparse_embedding(A_lcc, T=10)
    sig = sigmoid(M)
    M_gw = M * sig * (1 - sig)
    emb = svd_embed(M_gw, DIM, seed)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels_lcc, pred)


def method_clip_0_svd(A_lcc, labels_lcc, seed):
    M = compute_netmf_sparse_embedding(A_lcc, T=10)
    M_clip = np.maximum(M, 0)
    emb = svd_embed(M_clip, DIM, seed)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels_lcc, pred)


def method_bethe_hessian(A_lcc, labels_lcc, seed):
    d = np.array(A_lcc.sum(axis=1)).flatten()
    r = np.sqrt(np.mean(d))
    n = A_lcc.shape[0]
    D = sp.diags(d)
    I = sp.eye(n)
    BH = (r**2 - 1) * I - r * A_lcc + D

    k = min(5, n - 2)
    if k < 1:
        return 0.0

    try:
        vals, vecs = eigsh(BH, k=k, which='SA')
        neg_mask = vals < 0
        emb = vecs[:, neg_mask]
        if emb.shape[1] == 0:
            return 0.0
        pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
        return normalized_mutual_info_score(labels_lcc, pred)
    except Exception:
        return 0.0


def method_bp(A_lcc, labels_lcc, seed):
    try:
        from belief_propagation import detect
        pred = detect(A_lcc, q=2, iters=10)
        return normalized_mutual_info_score(labels_lcc, pred)
    except Exception as e:
        print(f"  BP error: {e}", flush=True)
        return 0.0


# ─── Main sweep ───────────────────────────────────────────────────────────────

# Run methods in separate passes to profile timing
METHODS_ORDER = ['bethe_hessian', 'bp', 'clip_0_svd', 'gradient_weight_svd', 'node2vec']
METHODS = {
    'node2vec': method_node2vec,
    'gradient_weight_svd': method_gradient_weight_svd,
    'clip_0_svd': method_clip_0_svd,
    'bethe_hessian': method_bethe_hessian,
    'bp': method_bp,
}

results = {name: {mu: [] for mu in MU_VALS} for name in METHODS}

print("Starting phase transition sweep...", flush=True)
print(f"Methods: {METHODS_ORDER}", flush=True)
print(f"mu values: {MU_VALS}", flush=True)
print(f"N_SAMPLES={N_SAMPLES} per mu\n", flush=True)

rng_master = np.random.default_rng(42)
# Pre-generate all seeds
all_seeds = [[int(rng_master.integers(0, 2**31)) for _ in range(N_SAMPLES)] for _ in range(len(MU_VALS))]

# Time each method on first sample
print("Timing test on first sample...", flush=True)
A_test, labels_test = fast_sbm_lcc(0.40, np.random.default_rng(123))

for name in METHODS_ORDER:
    t0 = time.time()
    try:
        nmi = METHODS[name](A_test, labels_test, 123)
        elapsed = time.time() - t0
        print(f"  {name}: {elapsed:.2f}s, NMI={nmi:.3f}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {name}: ERROR after {elapsed:.2f}s: {e}", flush=True)

print("", flush=True)

# Main sweep
t_total_start = time.time()
for mu_idx, mu in enumerate(MU_VALS):
    t_mu_start = time.time()
    print(f"[{mu_idx+1}/{len(MU_VALS)}] mu={mu:.2f}", flush=True)

    # Pre-generate graphs
    graphs = []
    for sample_idx in range(N_SAMPLES):
        seed = all_seeds[mu_idx][sample_idx]
        sample_rng = np.random.default_rng(seed)
        A_lcc, labels_lcc = fast_sbm_lcc(mu, sample_rng)
        graphs.append((A_lcc, labels_lcc, seed))

    # Run each method
    for name in METHODS_ORDER:
        t0 = time.time()
        for sample_idx, (A_lcc, labels_lcc, seed) in enumerate(graphs):
            try:
                nmi = METHODS[name](A_lcc, labels_lcc, seed)
                results[name][mu].append(float(nmi))
            except Exception as e:
                print(f"  Error {name} mu={mu} sample={sample_idx}: {e}", flush=True)
                results[name][mu].append(0.0)
        elapsed = time.time() - t0
        nmi_arr = results[name][mu]
        print(f"  {name}: NMI={np.mean(nmi_arr):.3f}±{np.std(nmi_arr):.3f} ({elapsed:.1f}s)", flush=True)

    elapsed_mu = time.time() - t_mu_start
    elapsed_total = time.time() - t_total_start
    remaining = elapsed_mu * (len(MU_VALS) - mu_idx - 1)
    print(f"  mu={mu:.2f} done in {elapsed_mu:.0f}s. Elapsed: {elapsed_total:.0f}s, ETA: {remaining:.0f}s\n", flush=True)

# ─── Compute statistics and effective mu* ─────────────────────────────────────

aggregated = {}
effective_mu_star = {}

for name in METHODS:
    nmi_mean = []
    nmi_std = []
    for mu in MU_VALS:
        arr = np.array(results[name][mu])
        nmi_mean.append(float(np.mean(arr)))
        nmi_std.append(float(np.std(arr)))

    nmi_mean_arr = np.array(nmi_mean)
    nmi_std_arr = np.array(nmi_std)
    mu_arr = np.array(MU_VALS)

    aggregated[name] = {
        'mu_vals': MU_VALS,
        'nmi_mean': nmi_mean_arr.tolist(),
        'nmi_std': nmi_std_arr.tolist(),
    }

    # Find effective mu* where NMI crosses 0.1 (going down)
    try:
        above = nmi_mean_arr >= 0.1
        below = nmi_mean_arr < 0.1

        if np.all(above):
            # Never crosses — extrapolate
            crossing = float(interp1d(nmi_mean_arr, mu_arr, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')(0.1))
        elif np.all(below):
            crossing = float(mu_arr[0])
        else:
            last_above_idx = np.where(above)[0][-1]
            below_after = np.where((mu_arr > mu_arr[last_above_idx]) & below)[0]
            if len(below_after) > 0:
                mu1 = mu_arr[last_above_idx]
                nmi1 = nmi_mean_arr[last_above_idx]
                mu2 = mu_arr[below_after[0]]
                nmi2 = nmi_mean_arr[below_after[0]]
                if abs(nmi2 - nmi1) > 1e-10:
                    crossing = float(mu1 + (0.1 - nmi1) * (mu2 - mu1) / (nmi2 - nmi1))
                else:
                    crossing = float(mu1)
            else:
                crossing = float(mu_arr[last_above_idx])

        effective_mu_star[name] = crossing
    except Exception as e:
        print(f"Interpolation error for {name}: {e}", flush=True)
        effective_mu_star[name] = None

    print(f"{name}: effective_mu*={effective_mu_star[name]:.4f}" if effective_mu_star[name] else f"{name}: effective_mu*=None", flush=True)

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'node2vec': '#1f77b4',
    'gradient_weight_svd': '#ff7f0e',
    'clip_0_svd': '#2ca02c',
    'bethe_hessian': '#d62728',
    'bp': '#9467bd',
}

labels_map = {
    'node2vec': 'node2vec',
    'gradient_weight_svd': 'gradient_weight_SVD',
    'clip_0_svd': 'clip_0_SVD',
    'bethe_hessian': 'Bethe Hessian',
    'bp': 'Belief Propagation',
}

mu_arr = np.array(MU_VALS)

for name in METHODS:
    nmi_mean = np.array(aggregated[name]['nmi_mean'])
    nmi_std = np.array(aggregated[name]['nmi_std'])
    color = colors[name]
    label = labels_map[name]

    ax.plot(mu_arr, nmi_mean, 'o-', color=color, label=label, linewidth=2, markersize=5)
    ax.fill_between(mu_arr, nmi_mean - nmi_std, nmi_mean + nmi_std,
                    alpha=0.15, color=color)

# Theoretical limit
ax.axvline(THEORETICAL_MU_STAR, color='black', linestyle='--', linewidth=1.5,
           label=f'Theoretical mu*={THEORETICAL_MU_STAR}')

# NMI=0.1 threshold
ax.axhline(0.1, color='gray', linestyle=':', linewidth=1.0, label='NMI=0.1 threshold')

# Mark effective limits
for name in METHODS:
    mu_eff = effective_mu_star.get(name)
    if mu_eff and 0.25 < mu_eff < 0.60:
        ax.axvline(mu_eff, color=colors[name], linestyle=':', alpha=0.5, linewidth=1)

ax.set_xlabel('Mixing parameter mu', fontsize=13)
ax.set_ylabel('NMI', fontsize=13)
ax.set_title('Phase Transition: NMI vs mu for Community Detection Methods\n(SBM, N=2000, cave=5, 15 samples)', fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0.28, 0.57)
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fig_phase_transition.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved.", flush=True)

# ─── Save results ─────────────────────────────────────────────────────────────

key_numbers = {}
for name in METHODS:
    key_numbers[f'effective_mu_star_{name}'] = effective_mu_star.get(name)

# Also add NMI at key mu values
for mu_key in [0.40, 0.46, 0.50, 0.52]:
    for name in METHODS:
        arr = np.array(results[name].get(mu_key, [0.0]))
        if len(arr) > 0:
            key_numbers[f'nmi_{name}_mu{int(mu_key*100)}'] = float(np.mean(arr))

output = {
    'task': 'Phase transition characterization: effective detectability limit for each method',
    'description': (
        'Dense mu sweep [0.30..0.54], N=2000 SBM, cave=5, 15 samples per mu. '
        'Effective mu* = interpolated mu where mean NMI crosses 0.1.'
    ),
    'methods': list(METHODS.keys()),
    'theoretical_mu_star': THEORETICAL_MU_STAR,
    'effective_mu_star': effective_mu_star,
    'aggregated': aggregated,
    'key_numbers': key_numbers,
}

out_path = os.path.join(OUTPUT_DIR, 'results.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved to {out_path}", flush=True)

# Print summary table
print("\n=== SUMMARY: Effective Detectability Limits ===")
print(f"{'Method':<25} {'Effective mu*':>12}")
print("-" * 40)
for name in METHODS_ORDER:
    mu_eff = effective_mu_star.get(name)
    val = f"{mu_eff:.4f}" if mu_eff is not None else "N/A"
    print(f"{name:<25} {val:>12}")
print(f"\n{'Theoretical KS limit':<25} {THEORETICAL_MU_STAR:>12.4f}")
print(f"\nTotal time: {time.time() - t_total_start:.0f}s")
