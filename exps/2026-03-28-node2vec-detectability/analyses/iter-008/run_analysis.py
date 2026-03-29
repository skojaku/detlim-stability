import sys, warnings, json, os
import numpy as np
import scipy.sparse as sp
import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
from embcom import utils as embcom_utils
import embcom

warnings.filterwarnings('ignore')

# ============================================================
# Setup
# ============================================================
N = 2000
MU_VALUES = [0.35, 0.40, 0.45, 0.50]
N_SAMPLES = 15
DIM = 64

def make_sbm_lcc(mu, N=2000, cave=5.0, seed=0):
    n_each = N // 2
    c_out = mu * cave
    c_in = 2 * cave - c_out
    p_in = c_in / N
    p_out = c_out / N
    np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
    labels = np.array([0] * n_each + [1] * n_each)
    A = sp.csr_matrix(g.get_adjacency_sparse(), dtype=float)
    comps = g.connected_components(mode="weak")
    lcc_idx = sorted(max(comps, key=len))
    return A[np.ix_(lcc_idx, lcc_idx)], labels[lcc_idx]

def compute_netmf_raw(A_lcc, window=10):
    """Returns raw log PMI matrix."""
    P = embcom_utils.to_trans_mat(A_lcc).toarray().astype(np.float64)
    Ppow = np.zeros_like(P)
    Pt = np.eye(len(P))
    for _ in range(window):
        Pt = Pt @ P
        Ppow += Pt
    Ppow /= window
    d = np.array(A_lcc.sum(1)).flatten()
    vol = d.sum()
    M_raw = Ppow @ np.diag(vol / np.maximum(d, 1e-12))
    return np.log(np.maximum(M_raw, 1e-300))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def embed_matrix(M, dim=64):
    n = M.shape[0]
    svd = TruncatedSVD(n_components=min(dim, n - 1), random_state=42)
    return svd.fit_transform(M)

def nmi_kmeans(emb, labels, seed=42):
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)

def get_matrix_variants(M_raw):
    """Return dict of matrix variants from raw log PMI."""
    clip_0 = np.maximum(M_raw, 0)
    sig = sigmoid(M_raw)
    sig_centered = sig - 0.5
    grad_weight = sig * (1 - sig) * M_raw
    sig_minus_half_clipped = np.maximum(sig - 0.5, 0)
    return {
        'clip_0': clip_0,
        'sigmoid_M': sig,
        'sigmoid_centered': sig_centered,
        'gradient_weight': grad_weight,
        'sigmoid_minus_half_clipped': sig_minus_half_clipped,
    }

# ============================================================
# Main sweep (matrix methods)
# ============================================================
print("Running matrix method sweep...")
nmi_results = {m: {mu: [] for mu in MU_VALUES}
               for m in ['clip_0', 'sigmoid_M', 'sigmoid_centered',
                         'gradient_weight', 'sigmoid_minus_half_clipped']}

for mu in MU_VALUES:
    print(f"  mu={mu}")
    for seed in range(N_SAMPLES):
        A, labels = make_sbm_lcc(mu, N=N, seed=seed)
        M_raw = compute_netmf_raw(A)
        variants = get_matrix_variants(M_raw)
        for mname, M in variants.items():
            emb = embed_matrix(M, dim=DIM)
            nmi = nmi_kmeans(emb, labels)
            nmi_results[mname][mu].append(nmi)

# ============================================================
# node2vec sweep
# ============================================================
print("Running node2vec sweep...")
nmi_results['node2vec'] = {mu: [] for mu in MU_VALUES}
for mu in MU_VALUES:
    print(f"  node2vec mu={mu}")
    for seed in range(N_SAMPLES):
        A, labels = make_sbm_lcc(mu, N=N, seed=seed)
        model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10)
        model.fit(A)
        emb = model.transform(dim=DIM)
        nmi = nmi_kmeans(emb, labels)
        nmi_results['node2vec'][mu].append(nmi)

# ============================================================
# Compile NMI stats
# ============================================================
nmi_by_method_mu = {}
for mname in nmi_results:
    nmi_by_method_mu[mname] = {}
    for mu in MU_VALUES:
        vals = nmi_results[mname][mu]
        nmi_by_method_mu[mname][str(mu)] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals))
        }

# ============================================================
# Singular value / eigenvec analysis at mu=0.40, seed=0
# ============================================================
print("Computing singular value analysis at mu=0.40...")
A40, labels40 = make_sbm_lcc(0.40, N=N, seed=0)
M_raw40 = compute_netmf_raw(A40)
variants40 = get_matrix_variants(M_raw40)

# Also add full M_raw for reference
variants40['full_M'] = M_raw40

singular_values_mu40 = {}
eigvec_corr_mu40 = {}
sparsity_mu40 = {}
n_top = 5

for mname, M in variants40.items():
    n = M.shape[0]
    svd = TruncatedSVD(n_components=min(n_top, n - 1), random_state=42)
    U = svd.fit_transform(M)
    svals = svd.singular_values_.tolist()
    # Pad if needed
    while len(svals) < n_top:
        svals.append(0.0)
    singular_values_mu40[mname] = svals[:n_top]

    # Correlation of each left singular vector with community labels (binary 0/1)
    corrs = []
    for k in range(min(n_top, U.shape[1])):
        r = abs(float(np.corrcoef(U[:, k], labels40)[0, 1]))
        corrs.append(r)
    while len(corrs) < n_top:
        corrs.append(0.0)
    eigvec_corr_mu40[mname] = corrs

    # Fraction of entries > 0
    frac_pos = float(np.mean(M > 0))
    sparsity_mu40[mname] = frac_pos

# Check sigmoid non-edge property
M_raw_sample = M_raw40
very_neg_mask = M_raw_sample < -10  # approximate "non-edge" region
sig_at_nonedge = sigmoid(M_raw_sample[very_neg_mask]).mean() if very_neg_mask.any() else 0.0
print(f"  sigmoid mean at very negative PMI (<-10): {sig_at_nonedge:.6f}")

# ============================================================
# Figure
# ============================================================
print("Making figure...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

methods_ordered = ['node2vec', 'clip_0', 'sigmoid_M', 'sigmoid_centered',
                   'gradient_weight', 'sigmoid_minus_half_clipped']
colors = ['black', 'steelblue', 'darkorange', 'green', 'red', 'purple']
mu_arr = np.array(MU_VALUES)

ax = axes[0]
for mname, color in zip(methods_ordered, colors):
    means = [nmi_by_method_mu[mname][str(mu)]['mean'] for mu in MU_VALUES]
    stds = [nmi_by_method_mu[mname][str(mu)]['std'] for mu in MU_VALUES]
    means = np.array(means)
    stds = np.array(stds)
    ax.plot(mu_arr, means, 'o-', color=color, label=mname)
    ax.fill_between(mu_arr, means - stds, means + stds, alpha=0.15, color=color)
ax.axvline(x=0.553, linestyle='--', color='gray', alpha=0.5, label='mu*=0.553')
ax.set_xlabel('mu (mixing parameter)')
ax.set_ylabel('NMI')
ax.set_title('NMI vs mu (mean ± std, N=2000, 15 samples)')
ax.legend(fontsize=8)
ax.set_ylim([-0.02, 1.05])

# Panel 2: singular values
ax2 = axes[1]
variants_plot = ['full_M', 'clip_0', 'sigmoid_M', 'sigmoid_centered',
                 'gradient_weight', 'sigmoid_minus_half_clipped']
colors2 = ['gray', 'steelblue', 'darkorange', 'green', 'red', 'purple']
x_pos = np.arange(1, n_top + 1)
for mname, color in zip(variants_plot, colors2):
    svals = singular_values_mu40[mname]
    ax2.plot(x_pos, svals, 'o-', color=color, label=mname)
ax2.set_yscale('log')
ax2.set_xlabel('Singular value rank')
ax2.set_ylabel('Singular value (log scale)')
ax2.set_title('Top-5 singular values at mu=0.40')
ax2.legend(fontsize=8)
ax2.set_xticks(x_pos)

fig.tight_layout()
out_fig = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-008/fig_sigmoid_comparison.png'
fig.savefig(out_fig, dpi=150)
plt.close()
print(f"Saved figure to {out_fig}")

# ============================================================
# Write results.json
# ============================================================
results = {
    "task": "Compare sigmoid-transformed NetMF matrix variants against clip(M,0) and node2vec for community detection in SBM near detectability limit",
    "method": "SVD of matrix variants + KMeans NMI, plus node2vec SGNS baseline; N=2000, mu=[0.35,0.40,0.45,0.50], 15 samples each",
    "result_summary": (
        f"clip_0 NMI at mu=0.40: {nmi_by_method_mu['clip_0']['0.4']['mean']:.3f}±{nmi_by_method_mu['clip_0']['0.4']['std']:.3f}; "
        f"node2vec: {nmi_by_method_mu['node2vec']['0.4']['mean']:.3f}±{nmi_by_method_mu['node2vec']['0.4']['std']:.3f}; "
        f"sigmoid_M: {nmi_by_method_mu['sigmoid_M']['0.4']['mean']:.3f}±{nmi_by_method_mu['sigmoid_M']['0.4']['std']:.3f}; "
        f"sigmoid_minus_half_clipped: {nmi_by_method_mu['sigmoid_minus_half_clipped']['0.4']['mean']:.3f}±{nmi_by_method_mu['sigmoid_minus_half_clipped']['0.4']['std']:.3f}."
    ),
    "key_numbers": {
        "nmi_clip0_mu040": nmi_by_method_mu['clip_0']['0.4']['mean'],
        "nmi_node2vec_mu040": nmi_by_method_mu['node2vec']['0.4']['mean'],
        "nmi_sigmoid_M_mu040": nmi_by_method_mu['sigmoid_M']['0.4']['mean'],
        "nmi_sigmoid_centered_mu040": nmi_by_method_mu['sigmoid_centered']['0.4']['mean'],
        "nmi_gradient_weight_mu040": nmi_by_method_mu['gradient_weight']['0.4']['mean'],
        "nmi_sigmoid_minus_half_clipped_mu040": nmi_by_method_mu['sigmoid_minus_half_clipped']['0.4']['mean'],
        "sigmoid_mean_at_very_neg_pmi": float(sig_at_nonedge),
    },
    "nmi_by_method_mu": nmi_by_method_mu,
    "singular_values_mu40": singular_values_mu40,
    "eigvec_corr_mu40": eigvec_corr_mu40,
    "sparsity_mu40": sparsity_mu40,
    "figures_created": ["analyses/iter-008/fig_sigmoid_comparison.png"],
    "code_used": """
def compute_netmf_raw(A_lcc, window=10):
    P = embcom_utils.to_trans_mat(A_lcc).toarray().astype(np.float64)
    Ppow = np.zeros_like(P); Pt = np.eye(len(P))
    for _ in range(window): Pt = Pt @ P; Ppow += Pt
    Ppow /= window
    d = np.array(A_lcc.sum(1)).flatten(); vol = d.sum()
    M_raw = Ppow @ np.diag(vol / np.maximum(d, 1e-12))
    return np.log(np.maximum(M_raw, 1e-300))

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

variants = {
    'clip_0': np.maximum(M_raw, 0),
    'sigmoid_M': sigmoid(M_raw),
    'sigmoid_centered': sigmoid(M_raw) - 0.5,
    'gradient_weight': sigmoid(M_raw) * (1 - sigmoid(M_raw)) * M_raw,
    'sigmoid_minus_half_clipped': np.maximum(sigmoid(M_raw) - 0.5, 0),
}
""",
    "shared_components": [
        {
            "name": "make_sbm_lcc",
            "description": "SBM graph generation (N=2000, cave=5, correct parameterization) with LCC extraction",
            "output_path": "analyses/iter-008/run_analysis.py",
            "why_shared": "Same graph generation used across all methods and iterations"
        },
        {
            "name": "compute_netmf_raw",
            "description": "Raw log PMI NetMF matrix computation (window=10, N=2000)",
            "output_path": "analyses/iter-008/run_analysis.py",
            "why_shared": "Same matrix used as input for all SVD-based variants"
        }
    ],
    "next_questions": [
        "Does gradient_weight outperform clip_0 at any mu? The gradient-weighting focuses on PMI≈0 pairs which carry maximum signal per the SGNS gradient argument.",
        "Is sigmoid_minus_half_clipped equivalent to clip_0 in practice? Both zero out negative PMI entries but sigmoid version smooths the boundary.",
        "At what threshold t does sigmoid_M - t begin to match node2vec? This would reveal the implicit 'effective threshold' in SGNS.",
        "Does the top singular vector alignment (eigvec_corr) predict NMI more reliably than raw NMI across methods?",
        "Would a free_netmf (gradient descent factorization, no torch) match node2vec better than SVD variants?"
    ],
    "failed_attempts": []
}

out_path = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-008/results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Wrote results to {out_path}")
print("\nDone.")
