"""
Iteration 15: sigmoid(M) factorization at T=1 and T=10

Compare NMI across 4 matrix operators:
1. sigmoid_T1: sigmoid(M^{T=1}) — H_{ij} = p/(p+p_i*p_j)
2. sigmoid_T10: sigmoid(M^{T=10}) — same but 10-hop
3. gradient_weight_T10: M * sigmoid(M) * (1 - sigmoid(M)) — best prior method
4. clip_0_T10: max(M, 0) — simple hard clip

Also: verify edge weight formula H_{ij} ≈ vol/(vol + d_i*d_j) for T=1 edges.
"""

import sys
import os
import json
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')

OUTPUT_DIR = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-015'

# ─── Parameters ───────────────────────────────────────────────────────────────
N = 2000
CAVE = 5.0
N_SAMPLES = 20
MU_VALS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.52]
DIM = 64
T1 = 1
T10 = 10


# ─── Graph generation ─────────────────────────────────────────────────────────

def fast_sbm_lcc(mu, rng):
    """Generate SBM and return LCC adjacency + labels."""
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / N
    p_out = c_out / N

    labels = np.array([0] * (N // 2) + [1] * (N // 2))

    # Vectorized edge generation
    half = N // 2

    # Within-community edges
    def gen_block(size, p):
        n_possible = size * (size - 1) // 2
        edges_mask = rng.random(n_possible) < p
        idx = np.where(edges_mask)[0]
        rows, cols = [], []
        for k in idx:
            # Convert linear index to (i, j) with i < j
            # Use the formula: row = floor((2*size-1-sqrt((2*size-1)^2-8*k))/2)
            i = int(np.floor((2 * size - 1 - np.sqrt((2 * size - 1) ** 2 - 8 * k)) / 2))
            j = k - i * (2 * size - 1 - i) // 2 + i + 1
            rows.append(i)
            cols.append(j)
        return np.array(rows), np.array(cols)

    # Block-based vectorized approach
    all_rows, all_cols = [], []

    # In-community block 1 (nodes 0..half-1)
    n1 = half
    n_pairs1 = n1 * (n1 - 1) // 2
    mask1 = rng.random(n_pairs1) < p_in
    if mask1.any():
        pair_idx = np.where(mask1)[0]
        # Convert pair indices to (i, j)
        # Use cumulative counts approach
        counts = np.arange(n1 - 1, 0, -1)
        cumcounts = np.concatenate([[0], np.cumsum(counts)])
        i_idx = np.searchsorted(cumcounts, pair_idx, side='right') - 1
        j_idx = pair_idx - cumcounts[i_idx] + i_idx + 1
        all_rows.extend(i_idx.tolist())
        all_cols.extend(j_idx.tolist())

    # In-community block 2 (nodes half..N-1)
    n2 = N - half
    n_pairs2 = n2 * (n2 - 1) // 2
    mask2 = rng.random(n_pairs2) < p_in
    if mask2.any():
        pair_idx = np.where(mask2)[0]
        counts = np.arange(n2 - 1, 0, -1)
        cumcounts = np.concatenate([[0], np.cumsum(counts)])
        i_idx = np.searchsorted(cumcounts, pair_idx, side='right') - 1
        j_idx = pair_idx - cumcounts[i_idx] + i_idx + 1
        all_rows.extend((i_idx + half).tolist())
        all_cols.extend((j_idx + half).tolist())

    # Between-community edges (block 1 x block 2)
    n_cross = n1 * n2
    mask_cross = rng.random(n_cross) < p_out
    if mask_cross.any():
        cross_idx = np.where(mask_cross)[0]
        i_idx = cross_idx // n2
        j_idx = cross_idx % n2 + half
        all_rows.extend(i_idx.tolist())
        all_cols.extend(j_idx.tolist())

    if len(all_rows) == 0:
        # Fallback: return a trivial graph
        return sp.eye(N, format='csr'), labels

    rows = np.array(all_rows)
    cols = np.array(all_cols)

    # Symmetrize
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data = np.ones(len(rows_sym))

    A = sp.csr_matrix((data, (rows_sym, cols_sym)), shape=(N, N))

    # Get LCC
    n_comp, comp_labels_arr = sp.csgraph.connected_components(A, directed=False)
    if n_comp == 1:
        return A, labels

    # Find largest component
    comp_sizes = np.bincount(comp_labels_arr)
    largest_comp = np.argmax(comp_sizes)
    lcc_mask = comp_labels_arr == largest_comp
    lcc_idx = np.where(lcc_mask)[0]

    A_lcc = A[np.ix_(lcc_idx, lcc_idx)]
    labels_lcc = labels[lcc_idx]
    return A_lcc, labels_lcc


# ─── Log-PMI computation ──────────────────────────────────────────────────────

def compute_logpmi(A, T=10):
    """
    Compute log-PMI matrix M where M_{ij} = log(vol * P^T_{ij} / d_j).
    Uses dense matrix powers (manageable for N~2000 with LCC).
    """
    d = np.array(A.sum(axis=1)).flatten()
    vol = d.sum()
    D_inv = sp.diags(1.0 / np.maximum(d, 1e-12))
    P = D_inv @ A
    P_dense = P.toarray()

    if T == 1:
        Ppow = P_dense.copy()
    else:
        Ppow = P_dense.copy()
        Pt = P_dense.copy()
        for t in range(2, T + 1):
            Pt = Pt @ P_dense
            Ppow += Pt
        Ppow /= T

    d_j = d[np.newaxis, :]
    M = np.log(np.maximum(vol * Ppow / d_j, 1e-300))
    return M, d, vol


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─── Embedding & scoring ──────────────────────────────────────────────────────

def embed_and_score(H, labels, dim=64, seed=0):
    n = H.shape[0]
    actual_dim = min(dim, n - 1)
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    emb = svd.fit_transform(H)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment():
    results = {method: {mu: [] for mu in MU_VALS} for method in
               ['sigmoid_T1', 'sigmoid_T10', 'gradient_weight_T10', 'clip_0_T10']}

    # For edge weight statistics
    edge_weight_stats = {'sigmoid_T1': {'mean': [], 'std': [], 'formula_mean': [], 'formula_std': []}}

    total_tasks = len(MU_VALS) * N_SAMPLES
    done = 0
    t_start = time.time()

    print(f"Running {total_tasks} (mu, sample) pairs...")

    for mu in MU_VALS:
        mu_edge_weights = []
        mu_formula_weights = []

        for sample_idx in range(N_SAMPLES):
            rng = np.random.default_rng(seed=sample_idx * 1000 + int(mu * 1000))
            A, labels = fast_sbm_lcc(mu, rng)

            n_nodes = A.shape[0]
            if n_nodes < DIM + 2:
                done += 1
                continue

            # Compute PMI matrices for T=1 and T=10
            M_T1, d_T1, vol_T1 = compute_logpmi(A, T=T1)
            M_T10, d_T10, vol_T10 = compute_logpmi(A, T=T10)

            # ── Method 1: sigmoid_T1 ──────────────────────────────────────────
            H_sig_T1 = sigmoid(M_T1)
            nmi1 = embed_and_score(H_sig_T1, labels, dim=DIM, seed=sample_idx)
            results['sigmoid_T1'][mu].append(nmi1)

            # Edge weight statistics for T=1
            A_dense = A.toarray().astype(bool)
            edge_mask = A_dense  # upper triangle or full?
            # Use upper triangle to avoid double-counting
            iu = np.triu_indices(n_nodes, k=1)
            edge_in_upper = A_dense[iu]
            sig_vals = H_sig_T1[iu]
            edge_sig_vals = sig_vals[edge_in_upper]
            if len(edge_sig_vals) > 0:
                mu_edge_weights.extend(edge_sig_vals.tolist())

            # Formula prediction: vol/(vol + d_i*d_j)
            di = d_T1[iu[0]]
            dj = d_T1[iu[1]]
            formula_vals = vol_T1 / (vol_T1 + di * dj)
            formula_edge_vals = formula_vals[edge_in_upper]
            if len(formula_edge_vals) > 0:
                mu_formula_weights.extend(formula_edge_vals.tolist())

            # ── Method 2: sigmoid_T10 ────────────────────────────────────────
            H_sig_T10 = sigmoid(M_T10)
            nmi2 = embed_and_score(H_sig_T10, labels, dim=DIM, seed=sample_idx)
            results['sigmoid_T10'][mu].append(nmi2)

            # ── Method 3: gradient_weight_T10 ────────────────────────────────
            sig_M10 = sigmoid(M_T10)
            H_gw = M_T10 * sig_M10 * (1 - sig_M10)
            nmi3 = embed_and_score(H_gw, labels, dim=DIM, seed=sample_idx)
            results['gradient_weight_T10'][mu].append(nmi3)

            # ── Method 4: clip_0_T10 ─────────────────────────────────────────
            H_clip = np.maximum(M_T10, 0)
            nmi4 = embed_and_score(H_clip, labels, dim=DIM, seed=sample_idx)
            results['clip_0_T10'][mu].append(nmi4)

            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                remaining = (total_tasks - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total_tasks}] mu={mu:.2f} sample={sample_idx} | "
                      f"NMI: sig_T1={nmi1:.3f} sig_T10={nmi2:.3f} gw={nmi3:.3f} clip={nmi4:.3f} | "
                      f"ETA {remaining:.0f}s")

        # Aggregate edge weight stats for this mu
        if mu_edge_weights:
            edge_weight_stats['sigmoid_T1']['mean'].append(float(np.mean(mu_edge_weights)))
            edge_weight_stats['sigmoid_T1']['std'].append(float(np.std(mu_edge_weights)))
        if mu_formula_weights:
            edge_weight_stats['sigmoid_T1']['formula_mean'].append(float(np.mean(mu_formula_weights)))
            edge_weight_stats['sigmoid_T1']['formula_std'].append(float(np.std(mu_formula_weights)))

        print(f"\nmu={mu:.2f} done:")
        for method in ['sigmoid_T1', 'sigmoid_T10', 'gradient_weight_T10', 'clip_0_T10']:
            vals = results[method][mu]
            if vals:
                print(f"  {method}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return results, edge_weight_stats


def compute_mu_star(mu_vals, nmi_means, threshold=0.1):
    """Interpolate mu where NMI crosses threshold from above."""
    from scipy.interpolate import interp1d
    # Find crossing (NMI decreasing)
    mu_arr = np.array(mu_vals)
    nmi_arr = np.array(nmi_means)

    # Find where NMI goes below threshold
    above = nmi_arr >= threshold
    if not above.any():
        return mu_arr[0]  # All below
    if above.all():
        return mu_arr[-1]  # All above

    # Find last index where above
    last_above = np.where(above)[0][-1]
    if last_above + 1 >= len(mu_arr):
        return mu_arr[-1]

    # Linear interpolate
    mu1, mu2 = mu_arr[last_above], mu_arr[last_above + 1]
    nmi1, nmi2 = nmi_arr[last_above], nmi_arr[last_above + 1]
    if nmi2 == nmi1:
        return mu1
    mu_star = mu1 + (threshold - nmi1) * (mu2 - mu1) / (nmi2 - nmi1)
    return float(mu_star)


def main():
    print("=" * 70)
    print("Iteration 15: sigmoid(M) factorization")
    print("=" * 70)

    results, edge_weight_stats = run_experiment()

    # ─── Aggregate ────────────────────────────────────────────────────────────
    methods = ['sigmoid_T1', 'sigmoid_T10', 'gradient_weight_T10', 'clip_0_T10']
    aggregated = {}
    mu_star = {}

    for method in methods:
        nmi_means = []
        nmi_stds = []
        for mu in MU_VALS:
            vals = results[method][mu]
            if vals:
                nmi_means.append(float(np.mean(vals)))
                nmi_stds.append(float(np.std(vals)))
            else:
                nmi_means.append(0.0)
                nmi_stds.append(0.0)
        aggregated[method] = {
            'mu_vals': MU_VALS,
            'nmi_mean': nmi_means,
            'nmi_std': nmi_stds
        }
        mu_star[method] = compute_mu_star(MU_VALS, nmi_means)

    # ─── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'mu':>6} | " + " | ".join(f"{m:>20}" for m in methods))
    print("-" * (6 + 3 + len(methods) * 23))
    for i, mu in enumerate(MU_VALS):
        row = f"{mu:>6.2f} | "
        for method in methods:
            mean = aggregated[method]['nmi_mean'][i]
            std = aggregated[method]['nmi_std'][i]
            row += f"{mean:.3f}±{std:.3f}         | "
        print(row)

    print("\nEffective mu* (NMI=0.1 crossing):")
    for method in methods:
        print(f"  {method:30s}: {mu_star[method]:.4f}")

    print("\nEdge weight statistics for sigmoid_T1:")
    print(f"  Mean edge weights (sigmoid): {edge_weight_stats['sigmoid_T1']['mean']}")
    print(f"  Mean edge weights (formula vol/(vol+d_i*d_j)): {edge_weight_stats['sigmoid_T1']['formula_mean']}")

    # ─── Load iter-014 node2vec for comparison ────────────────────────────────
    node2vec_data = None
    iter14_path = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-014/results.json'
    if os.path.exists(iter14_path):
        with open(iter14_path) as f:
            iter14 = json.load(f)
        if 'aggregated' in iter14 and 'node2vec' in iter14['aggregated']:
            node2vec_data = iter14['aggregated']['node2vec']
            print(f"\nLoaded node2vec reference from iter-014")

    # ─── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {
        'sigmoid_T1': '#e41a1c',
        'sigmoid_T10': '#ff7f00',
        'gradient_weight_T10': '#4daf4a',
        'clip_0_T10': '#377eb8',
        'node2vec': '#984ea3',
    }
    labels_map = {
        'sigmoid_T1': 'sigmoid(M) T=1 [H formula]',
        'sigmoid_T10': 'sigmoid(M) T=10',
        'gradient_weight_T10': 'M·σ(M)·(1-σ(M)) T=10 [best prior]',
        'clip_0_T10': 'max(M,0) T=10',
        'node2vec': 'node2vec (iter-014 ref)',
    }

    for method in methods:
        mu_vals = aggregated[method]['mu_vals']
        nmi_mean = aggregated[method]['nmi_mean']
        nmi_std = aggregated[method]['nmi_std']
        ax.plot(mu_vals, nmi_mean, 'o-', color=colors[method], label=labels_map[method], linewidth=2)
        ax.fill_between(mu_vals,
                        np.array(nmi_mean) - np.array(nmi_std),
                        np.array(nmi_mean) + np.array(nmi_std),
                        alpha=0.15, color=colors[method])

    # Add node2vec reference if available
    if node2vec_data is not None:
        # Filter to matching mu values
        nv_mu = node2vec_data['mu_vals']
        nv_mean = node2vec_data['nmi_mean']
        nv_std = node2vec_data['nmi_std']
        # Only plot mu values in our range
        mask = [m in MU_VALS or (m >= min(MU_VALS) and m <= max(MU_VALS)) for m in nv_mu]
        nv_mu_f = [m for m, mk in zip(nv_mu, mask) if mk]
        nv_mean_f = [v for v, mk in zip(nv_mean, mask) if mk]
        nv_std_f = [v for v, mk in zip(nv_std, mask) if mk]
        ax.plot(nv_mu_f, nv_mean_f, 's--', color=colors['node2vec'],
                label=labels_map['node2vec'], linewidth=1.5, alpha=0.7)

    # Threshold line
    ax.axhline(0.1, color='k', linestyle=':', linewidth=1, label='NMI=0.1 threshold')
    ax.axvline(0.553, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Theory μ*=0.553')

    ax.set_xlabel('μ (mixing parameter)', fontsize=12)
    ax.set_ylabel('NMI', fontsize=12)
    ax.set_title('Iter-015: sigmoid(M) factorization vs baselines\n'
                 f'N=2000, cave=5, {N_SAMPLES} samples/μ, TruncatedSVD(64)+KMeans', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(-0.02, 0.75)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig_sigmoid_methods.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # ─── Save JSON ────────────────────────────────────────────────────────────
    output = {
        'task': 'Iteration 15: sigmoid(M) factorization at T=1 and T=10',
        'description': (
            'Compare 4 matrix operators for spectral community detection: '
            'sigmoid_T1 (H=p/(p+p_i*p_j)), sigmoid_T10, gradient_weight_T10 (best prior), clip_0_T10. '
            'N=2000, cave=5, 20 samples/mu.'
        ),
        'methods': methods,
        'theoretical_mu_star': 0.553,
        'effective_mu_star': mu_star,
        'edge_weight_verification': {
            'description': 'For T=1 edges: mean sigmoid(M) vs formula vol/(vol+d_i*d_j)',
            'mu_vals': MU_VALS,
            'sigmoid_mean': edge_weight_stats['sigmoid_T1']['mean'],
            'formula_mean': edge_weight_stats['sigmoid_T1']['formula_mean'],
            'sigmoid_std': edge_weight_stats['sigmoid_T1']['std'],
            'formula_std': edge_weight_stats['sigmoid_T1']['formula_std'],
        },
        'aggregated': aggregated,
        'params': {'N': N, 'CAVE': CAVE, 'N_SAMPLES': N_SAMPLES, 'DIM': DIM, 'T1': T1, 'T10': T10}
    }

    json_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {json_path}")

    return output


if __name__ == '__main__':
    main()
