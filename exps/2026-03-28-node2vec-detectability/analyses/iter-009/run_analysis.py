"""
iter-009: Generalizability of node2vec principles across graph operators.
Tests Principle 1 (multi-hop) and Principle 2 (sigmoid/clipping) on:
- Adjacency A, Normalized adjacency M_na, Modularity B, Random walk P
"""

import sys, warnings, json, os
import numpy as np
import scipy.sparse as sp
import igraph as ig
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
from embcom import utils as embcom_utils

# ---- SBM generation ----

def make_sbm_lcc(mu, N=2000, cave=5.0, seed=0):
    n_each = N // 2
    c_out, c_in = mu * cave, 2 * cave - mu * cave
    p_in, p_out = c_in / N, c_out / N
    np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
    labels = np.array([0] * n_each + [1] * n_each)
    A = sp.csr_matrix(g.get_adjacency_sparse(), dtype=float)
    comps = g.connected_components(mode="weak")
    lcc_idx = sorted(max(comps, key=len))
    return A[np.ix_(lcc_idx, lcc_idx)], labels[lcc_idx]


# ---- Matrix constructions ----

def build_operators(A_lcc):
    d = np.array(A_lcc.sum(1)).flatten()
    m = d.sum() / 2

    # Adjacency
    A_dense = A_lcc.toarray().astype(np.float64)

    # Normalized adjacency
    D_isqrt = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    M_na = (D_isqrt @ A_lcc @ D_isqrt).toarray().astype(np.float64)

    # Modularity
    B = A_dense - np.outer(d, d) / (2 * m)

    # Random walk
    D_inv = sp.diags(1.0 / np.maximum(d, 1e-12))
    P = (D_inv @ A_lcc).toarray().astype(np.float64)

    return {
        'adj': A_dense,
        'norm_adj': M_na,
        'modularity': B,
        'randwalk': P,
    }


# ---- Multi-hop ----

def multihop(M_dense, T=10):
    """Compute (1/T) * sum_{t=1}^T M^t"""
    n = M_dense.shape[0]
    result = np.zeros((n, n))
    Mt = np.eye(n)
    for t in range(T):
        Mt = Mt @ M_dense
        result += Mt
    return result / T


# ---- Sigmoid ----

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---- Apply transforms ----

def apply_transform(M, transform, T=10):
    if transform == '1hop':
        return M
    elif transform == '10hop':
        return multihop(M, T=T)
    elif transform == '10hop_clip0':
        return np.maximum(multihop(M, T=T), 0)
    elif transform == '10hop_sigmoid':
        Mh = multihop(M, T=T)
        return sigmoid(Mh)
    elif transform == '10hop_gradweight':
        Mh = multihop(M, T=T)
        s = sigmoid(Mh)
        return Mh * s * (1 - s)
    else:
        raise ValueError(f"Unknown transform: {transform}")


# ---- NetMF reference ----

def compute_netmf_clipped(A_lcc, window=10):
    P = embcom_utils.to_trans_mat(A_lcc).toarray().astype(np.float64)
    Ppow = np.zeros_like(P)
    Pt = np.eye(len(P))
    for _ in range(window):
        Pt = Pt @ P
        Ppow += Pt
    Ppow /= window
    d = np.array(A_lcc.sum(1)).flatten()
    vol = d.sum()
    M = Ppow @ np.diag(vol / np.maximum(d, 1e-12))
    return np.maximum(np.log(np.maximum(M, 1e-300)), 0)


# ---- Embed and NMI ----

def embed_and_nmi(M, labels, dim=64, seed=42):
    n = M.shape[0]
    svd = TruncatedSVD(n_components=min(dim, n - 1), random_state=seed)
    emb = svd.fit_transform(M)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred), svd.singular_values_[:5]


# ---- Main experiment ----

MU_VALUES = [0.30, 0.40, 0.45, 0.50]
N_SAMPLES = 10
N = 2000
T = 10
OPERATORS = ['adj', 'norm_adj', 'modularity', 'randwalk']
TRANSFORMS = ['1hop', '10hop', '10hop_clip0', '10hop_sigmoid', '10hop_gradweight']

results = {}
diagnostics = {}

print("Starting iter-009 analysis...")

for mu in MU_VALUES:
    print(f"\n=== mu = {mu} ===")
    nmi_by_key = {}

    # collect diagnostics at first sample
    diag_collected = set()

    for sample_idx in range(N_SAMPLES):
        seed = sample_idx
        A_lcc, labels = make_sbm_lcc(mu, N=N, seed=seed)
        ops = build_operators(A_lcc)

        # Precompute multihop for each operator (expensive, do once)
        multihop_cache = {}
        for op_name, M in ops.items():
            multihop_cache[op_name] = multihop(M, T=T)

        # Evaluate each operator × transform
        for op_name in OPERATORS:
            M_base = ops[op_name]
            M_mh = multihop_cache[op_name]

            for transform in TRANSFORMS:
                key = f"{op_name}_{transform}"
                if key not in nmi_by_key:
                    nmi_by_key[key] = []

                if transform == '1hop':
                    M_transformed = M_base
                elif transform == '10hop':
                    M_transformed = M_mh
                elif transform == '10hop_clip0':
                    M_transformed = np.maximum(M_mh, 0)
                elif transform == '10hop_sigmoid':
                    M_transformed = sigmoid(M_mh)
                elif transform == '10hop_gradweight':
                    s = sigmoid(M_mh)
                    M_transformed = M_mh * s * (1 - s)

                nmi, svs = embed_and_nmi(M_transformed, labels)
                nmi_by_key[key].append(nmi)

                # diagnostics at first sample, mu=0.40
                if mu == 0.40 and sample_idx == 0 and op_name not in diag_collected and transform == '10hop':
                    sv_ratio = float(svs[0] / svs[1]) if len(svs) >= 2 and svs[1] > 0 else None
                    frac_pos = float(np.mean(np.maximum(M_mh, 0) > 0))
                    diagnostics[op_name] = {
                        'sv_ratio_sv1_sv2': sv_ratio,
                        'top5_sv': svs.tolist(),
                        'fraction_positive_after_clip': frac_pos,
                    }
                    diag_collected.add(op_name)

        # NetMF reference
        key = 'netmf_clip0'
        if key not in nmi_by_key:
            nmi_by_key[key] = []
        M_netmf = compute_netmf_clipped(A_lcc, window=T)
        nmi_netmf, _ = embed_and_nmi(M_netmf, labels)
        nmi_by_key[key].append(nmi_netmf)

        print(f"  sample {sample_idx+1}/{N_SAMPLES} done (mu={mu})")

    # Store results
    for key, nmis in nmi_by_key.items():
        if key not in results:
            results[key] = {}
        results[key][str(mu)] = {
            'nmi_mean': float(np.mean(nmis)),
            'nmi_std': float(np.std(nmis)),
        }

print("\nDone! Computing key numbers...")

# ---- Key numbers ----
def principle1_verdict(op):
    """Does multi-hop help? Compare adj_1hop vs adj_10hop at mu=0.40"""
    k1 = f"{op}_1hop"
    k10 = f"{op}_10hop"
    nmi1 = results[k1]['0.4']['nmi_mean']
    nmi10 = results[k10]['0.4']['nmi_mean']
    delta = nmi10 - nmi1
    if delta > 0.05:
        return f"yes (delta={delta:.3f})"
    elif delta > 0.01:
        return f"partially (delta={delta:.3f})"
    else:
        return f"no (delta={delta:.3f})"

def principle2_verdict(op):
    """Does clip0 help on top of multihop? Compare adj_10hop vs adj_10hop_clip0 at mu=0.40"""
    k10 = f"{op}_10hop"
    kc = f"{op}_10hop_clip0"
    nmi10 = results[k10]['0.4']['nmi_mean']
    nmic = results[kc]['0.4']['nmi_mean']
    delta = nmic - nmi10
    if delta > 0.05:
        return f"yes (delta={delta:.3f})"
    elif delta > 0.01:
        return f"partially (delta={delta:.3f})"
    else:
        return f"no (delta={delta:.3f})"

key_numbers = {
    'principle1_multihop_helps': {op: principle1_verdict(op) for op in OPERATORS},
    'principle2_clip_helps_on_top_of_multihop': {op: principle2_verdict(op) for op in OPERATORS},
    'reference_netmf_clip0_nmi_at_mu040': results['netmf_clip0']['0.4']['nmi_mean'],
}

# ---- Save results ----
os.makedirs('/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-009', exist_ok=True)

output = {
    'task': 'Test generalizability of node2vec detectability principles (multi-hop and sigmoid/clipping) across four graph operators: adjacency A, normalized adjacency M_na, modularity B, and random walk P.',
    'results': results,
    'diagnostics': diagnostics,
    'key_numbers': key_numbers,
}

with open('/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-009/results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("results.json written.")

# ---- Figure ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

op_labels = {
    'adj': 'Adjacency A',
    'norm_adj': 'Normalized Adj M_na',
    'modularity': 'Modularity B',
    'randwalk': 'Random Walk P',
}

transform_styles = {
    '1hop': ('b', '-', '1-hop'),
    '10hop': ('g', '-', '10-hop'),
    '10hop_clip0': ('r', '-', '10-hop clip0'),
    '10hop_sigmoid': ('purple', '-', '10-hop sigmoid'),
    '10hop_gradweight': ('orange', '-', '10-hop gradweight'),
}

for ax, op_name in zip(axes, OPERATORS):
    mu_vals = [0.30, 0.40, 0.45, 0.50]

    for transform, (color, ls, label) in transform_styles.items():
        key = f"{op_name}_{transform}"
        means = [results[key][str(mu)]['nmi_mean'] for mu in mu_vals]
        stds = [results[key][str(mu)]['nmi_std'] for mu in mu_vals]
        ax.plot(mu_vals, means, color=color, ls=ls, label=label, marker='o', markersize=4)
        ax.fill_between(mu_vals,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.1)

    # NetMF reference (dashed)
    netmf_means = [results['netmf_clip0'][str(mu)]['nmi_mean'] for mu in mu_vals]
    netmf_stds = [results['netmf_clip0'][str(mu)]['nmi_std'] for mu in mu_vals]
    ax.plot(mu_vals, netmf_means, 'k--', label='NetMF clip0 (ref)', linewidth=2)
    ax.fill_between(mu_vals,
                    [m - s for m, s in zip(netmf_means, netmf_stds)],
                    [m + s for m, s in zip(netmf_means, netmf_stds)],
                    color='black', alpha=0.05)

    ax.axvline(0.553, color='gray', ls=':', alpha=0.5, label='mu* = 0.553')
    ax.set_title(op_labels[op_name])
    ax.set_xlabel('mu (mixing parameter)')
    ax.set_ylabel('NMI')
    ax.set_xlim(0.28, 0.52)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Generalizability of node2vec principles across graph operators\n(N=2000, 10 samples per mu, cave=5)', fontsize=12)
plt.tight_layout()
plt.savefig('/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-009/fig_generalizability.png', dpi=150, bbox_inches='tight')
print("Figure saved.")
print("All done!")
