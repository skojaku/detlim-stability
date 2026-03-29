import sys, warnings, json, numpy as np, scipy.sparse as sp
import igraph as ig
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
from embcom import utils as embcom_utils
warnings.filterwarnings('ignore')

def make_sbm_lcc(mu, N=2000, cave=5.0, seed=0):
    n_each = N // 2
    c_out = mu * cave
    c_in = 2 * cave - c_out
    np.random.seed(seed)
    g = ig.Graph.SBM([[c_in/N, c_out/N], [c_out/N, c_in/N]], [n_each, n_each], directed=False)
    labels = np.array([0]*n_each + [1]*n_each)
    A = sp.csr_matrix(g.get_adjacency_sparse(), dtype=float)
    comps = g.connected_components(mode="weak")
    lcc_idx = sorted(max(comps, key=len))
    return A[np.ix_(lcc_idx, lcc_idx)], labels[lcc_idx]

def multihop(M, T=10):
    result = np.zeros_like(M)
    Mt = np.eye(len(M))
    for _ in range(T):
        Mt = Mt @ M
        result += Mt
    return result / T

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def embed_nmi(M, labels, dim=64, seed=42):
    k = min(dim, M.shape[0]-1)
    emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(M)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)

def sv_ratio_k(M, k=5):
    """Top-k singular values and eigvec-label correlations."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return s[:k], U[:, :k]

# ---- Sub-analysis A: degree-norm ablation at mu=0.40 ----
print("Running Sub-analysis A: degree-norm ablation at mu=0.40...")
mu_A = 0.40
n_samples_A = 10
methods_A = ['P_multihop_log_clip', 'Msym_multihop_log_clip', 'A_multihop_log_clip',
             'A_degnorm_then_log_clip', 'A_row_then_log_clip', 'Dsym_A_log_clip']
nmi_ablation = {m: [] for m in methods_A}
sv_ratios_A = {m: [] for m in methods_A}

for seed in range(n_samples_A):
    A_lcc, labels = make_sbm_lcc(mu_A, seed=seed)
    n = A_lcc.shape[0]
    d = np.array(A_lcc.sum(1)).flatten()
    vol = d.sum()
    A_dense = A_lcc.toarray().astype(np.float64)
    D_isqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    D_inv = np.diag(1.0 / np.maximum(d, 1e-12))

    # Method 1: P_multihop_log_clip  (standard NetMF)
    P = (D_inv @ A_dense)
    P_mh = multihop(P, T=10)
    # Apply the standard NetMF log+clip: M_netmf = log(P_mh * vol/d_j) clipped at 0
    M1 = P_mh * (vol / np.maximum(d, 1e-12))[np.newaxis, :]  # broadcast row: multiply col j by vol/dj
    M1 = np.maximum(np.log(np.maximum(M1, 1e-300)), 0)
    nmi_ablation['P_multihop_log_clip'].append(embed_nmi(M1, labels))
    sv1 = np.linalg.svd(M1, full_matrices=False, compute_uv=False)
    sv_ratios_A['P_multihop_log_clip'].append(sv1[0]/sv1[1] if sv1[1] > 0 else np.inf)

    # Method 2: Msym_multihop_log_clip (norm_adj variant)
    Msym = D_isqrt @ A_dense @ D_isqrt
    Msym_mh = multihop(Msym, T=10)
    M2 = Msym_mh * (vol / np.maximum(d, 1e-12))[:, np.newaxis]  # row-wise: each row i × vol/d_i
    M2 = np.maximum(np.log(np.maximum(M2, 1e-300)), 0)
    nmi_ablation['Msym_multihop_log_clip'].append(embed_nmi(M2, labels))
    sv2 = np.linalg.svd(M2, full_matrices=False, compute_uv=False)
    sv_ratios_A['Msym_multihop_log_clip'].append(sv2[0]/sv2[1] if sv2[1] > 0 else np.inf)

    # Method 3: A_multihop_log_clip (raw A, no degree norm) -- expected to fail
    A_mh = multihop(A_dense, T=10)
    M3 = A_mh * (vol / np.maximum(d, 1e-12))[np.newaxis, :]
    M3 = np.maximum(np.log(np.maximum(M3, 1e-300)), 0)
    nmi_ablation['A_multihop_log_clip'].append(embed_nmi(M3, labels))
    sv3 = np.linalg.svd(M3, full_matrices=False, compute_uv=False)
    sv_ratios_A['A_multihop_log_clip'].append(sv3[0]/sv3[1] if sv3[1] > 0 else np.inf)

    # Method 4: A_degnorm_then_log_clip (post-hoc symmetric degree norm on A^10hop)
    A_mh_sym_norm = A_mh / np.maximum(np.sqrt(np.outer(d, d)), 1e-12)
    M4 = np.maximum(np.log(np.maximum(A_mh_sym_norm, 1e-300)), 0)
    nmi_ablation['A_degnorm_then_log_clip'].append(embed_nmi(M4, labels))
    sv4 = np.linalg.svd(M4, full_matrices=False, compute_uv=False)
    sv_ratios_A['A_degnorm_then_log_clip'].append(sv4[0]/sv4[1] if sv4[1] > 0 else np.inf)

    # Method 5: A_row_then_log_clip (row-normalize A^10hop)
    row_sums = A_mh.sum(axis=1, keepdims=True)
    A_mh_row = A_mh / np.maximum(row_sums, 1e-12)
    M5 = A_mh_row * (vol / np.maximum(d, 1e-12))[np.newaxis, :]
    M5 = np.maximum(np.log(np.maximum(M5, 1e-300)), 0)
    nmi_ablation['A_row_then_log_clip'].append(embed_nmi(M5, labels))
    sv5 = np.linalg.svd(M5, full_matrices=False, compute_uv=False)
    sv_ratios_A['A_row_then_log_clip'].append(sv5[0]/sv5[1] if sv5[1] > 0 else np.inf)

    # Method 6: Dsym_A_log_clip (symmetric degree-normalize A^10hop)
    A_mh_dsym = D_isqrt @ A_mh @ D_isqrt
    M6 = np.maximum(np.log(np.maximum(A_mh_dsym, 1e-300)), 0)
    nmi_ablation['Dsym_A_log_clip'].append(embed_nmi(M6, labels))
    sv6 = np.linalg.svd(M6, full_matrices=False, compute_uv=False)
    sv_ratios_A['Dsym_A_log_clip'].append(sv6[0]/sv6[1] if sv6[1] > 0 else np.inf)

    if seed % 3 == 0:
        print(f"  seed={seed} done")

nmi_ablation_mu40 = {m: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for m, v in nmi_ablation.items()}
sv_ratios_mean = {m: float(np.mean(v)) for m, v in sv_ratios_A.items()}
print("Sub-A done.")
print("NMI ablation:", {m: f"{v['mean']:.3f}±{v['std']:.3f}" for m, v in nmi_ablation_mu40.items()})

# ---- Sub-analysis B: best methods full mu sweep ----
print("\nRunning Sub-analysis B: full mu sweep with 20 samples...")
mu_values = [0.30, 0.35, 0.40, 0.45, 0.50]
n_samples_B = 20
methods_B = ['netmf_clip0', 'norm_adj_netmf_clip0', 'norm_adj_netmf_gradweight', 'mod_10hop_sigmoid']
nmi_comparison = {m: {} for m in methods_B}

for mu in mu_values:
    nmis = {m: [] for m in methods_B}
    for seed in range(n_samples_B):
        A_lcc, labels = make_sbm_lcc(mu, seed=seed)
        n = A_lcc.shape[0]
        d = np.array(A_lcc.sum(1)).flatten()
        vol = d.sum()
        A_dense = A_lcc.toarray().astype(np.float64)
        D_isqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
        D_inv = np.diag(1.0 / np.maximum(d, 1e-12))

        # netmf_clip0: standard NetMF
        P = D_inv @ A_dense
        P_mh = multihop(P, T=10)
        M = P_mh * (vol / np.maximum(d, 1e-12))[np.newaxis, :]
        M = np.maximum(np.log(np.maximum(M, 1e-300)), 0)
        nmis['netmf_clip0'].append(embed_nmi(M, labels))

        # norm_adj_netmf_clip0: symmetric normalize then same recipe
        Msym = D_isqrt @ A_dense @ D_isqrt
        Msym_mh = multihop(Msym, T=10)
        M2 = Msym_mh * (vol / np.maximum(d, 1e-12))[:, np.newaxis]
        M2 = np.maximum(np.log(np.maximum(M2, 1e-300)), 0)
        nmis['norm_adj_netmf_clip0'].append(embed_nmi(M2, labels))

        # norm_adj_netmf_gradweight: M_na → 10-hop → log → gradient-weight σ(M)(1-σ(M))·M
        M3_raw = Msym_mh * (vol / np.maximum(d, 1e-12))[:, np.newaxis]
        M3_log = np.log(np.maximum(M3_raw, 1e-300))
        sig = sigmoid(M3_log)
        M3 = sig * (1 - sig) * M3_log
        nmis['norm_adj_netmf_gradweight'].append(embed_nmi(M3, labels))

        # mod_10hop_sigmoid: modularity 10-hop + sigmoid
        # Modularity B = A - dd^T/vol
        B = A_dense - np.outer(d, d) / vol
        B_mh = multihop(B, T=10)
        M4 = sigmoid(B_mh)
        nmis['mod_10hop_sigmoid'].append(embed_nmi(M4, labels))

    for m in methods_B:
        nmi_comparison[m][str(mu)] = {'mean': float(np.mean(nmis[m])), 'std': float(np.std(nmis[m]))}
    print(f"  mu={mu}: " + ", ".join(f"{m}={np.mean(nmis[m]):.3f}" for m in methods_B))

print("Sub-B done.")

# ---- Sub-analysis C: WHY raw A fails ----
print("\nRunning Sub-analysis C: singular value analysis...")
mu_C = 0.40
A_lcc, labels = make_sbm_lcc(mu_C, seed=0)
n = A_lcc.shape[0]
d = np.array(A_lcc.sum(1)).flatten()
vol = d.sum()
A_dense = A_lcc.toarray().astype(np.float64)
D_inv = np.diag(1.0 / np.maximum(d, 1e-12))

# A_10hop
A_mh = multihop(A_dense, T=10)
# P_10hop
P = D_inv @ A_dense
P_mh = multihop(P, T=10)

# Singular values
U_A, s_A, Vt_A = np.linalg.svd(A_mh, full_matrices=False)
U_P, s_P, Vt_P = np.linalg.svd(P_mh, full_matrices=False)

k = 5
top5_sv_A = s_A[:k].tolist()
top5_sv_P = s_P[:k].tolist()

# Correlation of left singular vectors with community labels (binarized)
labels_arr = labels.astype(float)
labels_centered = labels_arr - labels_arr.mean()
corr_A = [abs(np.corrcoef(U_A[:, i], labels_centered)[0, 1]) for i in range(k)]
corr_P = [abs(np.corrcoef(U_P[:, i], labels_centered)[0, 1]) for i in range(k)]

# SV ratio for all ablation methods (at mu=0.40, seed=0)
# Recompute to get their sv1/sv2 ratios for scatter plot
A_lcc0, labels0 = make_sbm_lcc(0.40, seed=0)
d0 = np.array(A_lcc0.sum(1)).flatten()
vol0 = d0.sum()
A0 = A_lcc0.toarray().astype(np.float64)
D_isqrt0 = np.diag(1.0 / np.sqrt(np.maximum(d0, 1e-12)))
D_inv0 = np.diag(1.0 / np.maximum(d0, 1e-12))

A_mh0 = multihop(A0, T=10)
P0 = D_inv0 @ A0
P_mh0 = multihop(P0, T=10)
Msym0 = D_isqrt0 @ A0 @ D_isqrt0
Msym_mh0 = multihop(Msym0, T=10)

# Build matrices for SV analysis
M_ref = {}
M_ref['P_multihop_log_clip'] = np.maximum(np.log(np.maximum(P_mh0 * (vol0 / np.maximum(d0, 1e-12))[np.newaxis, :], 1e-300)), 0)
M_ref['Msym_multihop_log_clip'] = np.maximum(np.log(np.maximum(Msym_mh0 * (vol0 / np.maximum(d0, 1e-12))[:, np.newaxis], 1e-300)), 0)
M_ref['A_multihop_log_clip'] = np.maximum(np.log(np.maximum(A_mh0 * (vol0 / np.maximum(d0, 1e-12))[np.newaxis, :], 1e-300)), 0)
A_sym_norm0 = A_mh0 / np.maximum(np.sqrt(np.outer(d0, d0)), 1e-12)
M_ref['A_degnorm_then_log_clip'] = np.maximum(np.log(np.maximum(A_sym_norm0, 1e-300)), 0)
row_sums0 = A_mh0.sum(axis=1, keepdims=True)
A_row0 = A_mh0 / np.maximum(row_sums0, 1e-12)
M_ref['A_row_then_log_clip'] = np.maximum(np.log(np.maximum(A_row0 * (vol0 / np.maximum(d0, 1e-12))[np.newaxis, :], 1e-300)), 0)
A_dsym0 = D_isqrt0 @ A_mh0 @ D_isqrt0
M_ref['Dsym_A_log_clip'] = np.maximum(np.log(np.maximum(A_dsym0, 1e-300)), 0)

sv_for_scatter = {}
for name, M in M_ref.items():
    sv = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    sv_for_scatter[name] = {'sv1': float(sv[0]), 'sv2': float(sv[1]), 'ratio': float(sv[0]/sv[1]) if sv[1] > 0 else float('inf')}

print("Sub-C done.")
print("top5 sv A_mh:", top5_sv_A[:5])
print("top5 sv P_mh:", top5_sv_P[:5])
print("corr with labels A:", corr_A)
print("corr with labels P:", corr_P)

# ---- Figures ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: NMI bar chart
method_labels = ['P_mh_log', 'Msym_mh_log', 'A_mh_log', 'A_deg_post', 'A_row_post', 'Dsym_post']
means = [nmi_ablation_mu40[m]['mean'] for m in methods_A]
stds = [nmi_ablation_mu40[m]['std'] for m in methods_A]
colors = ['steelblue', 'darkorange', 'crimson', 'green', 'purple', 'teal']
axes[0].bar(range(len(methods_A)), means, yerr=stds, color=colors, alpha=0.8, capsize=5)
axes[0].set_xticks(range(len(methods_A)))
axes[0].set_xticklabels(method_labels, rotation=30, ha='right', fontsize=9)
axes[0].set_ylabel('NMI')
axes[0].set_title('Degree-norm ablation NMI at mu=0.40 (10 samples)')
axes[0].set_ylim(0, 0.55)
axes[0].axhline(0, color='k', linewidth=0.5)
for i, (m, s) in enumerate(zip(means, stds)):
    axes[0].text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=8)

# Panel 2: SV1/SV2 vs NMI scatter
ratios = [sv_ratios_mean[m] for m in methods_A]
nmis_scatter = [nmi_ablation_mu40[m]['mean'] for m in methods_A]
axes[1].scatter(ratios, nmis_scatter, c=colors, s=100, zorder=5)
for i, name in enumerate(method_labels):
    axes[1].annotate(name, (ratios[i], nmis_scatter[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)
axes[1].set_xlabel('SV1/SV2 ratio (mean over 10 samples)')
axes[1].set_ylabel('NMI (mean)')
axes[1].set_title('SV1/SV2 ratio vs NMI at mu=0.40')
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-011/fig_degree_norm_ablation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved.")

# ---- Compile results ----
results = {
    "task": "Test whether degree normalization is THE essential transferable ingredient, and find the cleanest unified formulation. Sub-A: degree-norm ablation at mu=0.40 (10 samples). Sub-B: best 4 methods full mu sweep (20 samples). Sub-C: SV analysis explaining why raw A fails.",
    "method": "SBM N=2000 cave=5.0, 10-hop multi-hop averaging, TruncatedSVD+KMeans for NMI, full SVD for singular value analysis",
    "result_summary": (
        f"Sub-A: P_mh_log_clip NMI={nmi_ablation_mu40['P_multihop_log_clip']['mean']:.3f}, "
        f"Msym_mh_log_clip={nmi_ablation_mu40['Msym_multihop_log_clip']['mean']:.3f}, "
        f"A_mh_log_clip={nmi_ablation_mu40['A_multihop_log_clip']['mean']:.3f} (fails), "
        f"A_degnorm_post={nmi_ablation_mu40['A_degnorm_then_log_clip']['mean']:.3f}, "
        f"A_row_post={nmi_ablation_mu40['A_row_then_log_clip']['mean']:.3f}, "
        f"Dsym_post={nmi_ablation_mu40['Dsym_A_log_clip']['mean']:.3f}. "
        f"Top-5 SVs A_mh={[round(x,1) for x in top5_sv_A]}, P_mh={[round(x,3) for x in top5_sv_P]}. "
        f"Corr labels A_mh={[round(x,3) for x in corr_A]}, P_mh={[round(x,3) for x in corr_P]}."
    ),
    "key_numbers": {
        "nmi_ablation_mu40": nmi_ablation_mu40,
        "sv_ratios_mean_mu40": sv_ratios_mean,
        "sv_for_scatter_seed0": sv_for_scatter,
        "nmi_comparison_full": nmi_comparison,
        "sv_analysis": {
            "A_mh": {
                "top5_sv": [float(x) for x in top5_sv_A],
                "top5_eigcorr_with_labels": [float(x) for x in corr_A]
            },
            "P_mh": {
                "top5_sv": [float(x) for x in top5_sv_P],
                "top5_eigcorr_with_labels": [float(x) for x in corr_P]
            }
        }
    },
    "figures_created": ["analyses/iter-011/fig_degree_norm_ablation.png"],
    "code_used": """
# Key compute logic (see run_analysis.py for full)
P = D_inv @ A_dense; P_mh = multihop(P, T=10)
Msym = D_isqrt @ A_dense @ D_isqrt; Msym_mh = multihop(Msym, T=10)
A_mh = multihop(A_dense, T=10)
# NetMF log+clip:
M = np.maximum(np.log(np.maximum(P_mh * (vol/d_j), 1e-300)), 0)
# Post-hoc sym-norm then log:
A_mh_sym_norm = A_mh / sqrt(outer(d,d)); M = max(log(A_mh_sym_norm), 0)
# Row-norm then log:
A_mh_row = A_mh / row_sums; M = max(log(A_mh_row * vol/d_j), 0)
""",
    "shared_components": [
        {
            "name": "make_sbm_lcc",
            "description": "SBM N=2000 cave=5.0 with LCC extraction, seeds 0..19",
            "output_path": "analyses/iter-011/run_analysis.py",
            "why_shared": "Same graph generation function used in all prior iters"
        }
    ],
    "next_questions": [
        "Does post-hoc symmetric degree normalization (A_degnorm_then_log_clip) recover any community structure, or is it identically NMI~0 like raw A?",
        "If row-normalization rescues A^10hop, does it collapse to an equivalent of P=D^{-1}A (since P^T row-sums=1 by construction)?",
        "What is the SV1/SV2 threshold below which community detection consistently works? All working methods appear to have sv1/sv2 < ~2, while failing raw A has sv1/sv2~163",
        "Is norm_adj_netmf_clip0 or gradient-weight the better performing variant for a unified formulation?",
        "Can we explain why mod_10hop_sigmoid works — is it also degree-normalizing implicitly via the B=A-dd^T/vol subtraction?"
    ],
    "failed_attempts": []
}

out_path = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-011/results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {out_path}")
print("\n=== SUMMARY ===")
print("Sub-A NMI at mu=0.40:")
for m in methods_A:
    v = nmi_ablation_mu40[m]
    print(f"  {m}: {v['mean']:.3f} ± {v['std']:.3f}  (sv1/sv2={sv_ratios_mean[m]:.1f})")
print("\nSub-B NMI comparison (mu=0.40):")
for m in methods_B:
    v = nmi_comparison[m].get('0.4', nmi_comparison[m].get('0.40', {}))
    print(f"  {m}: {v.get('mean', 'N/A'):.3f} ± {v.get('std', 'N/A'):.3f}")
print("\nSub-C SV analysis:")
print(f"  A_mh top5 sv: {[round(x,1) for x in top5_sv_A]}")
print(f"  P_mh top5 sv: {[round(x,3) for x in top5_sv_P]}")
print(f"  Corr with labels (A_mh): {[round(x,3) for x in corr_A]}")
print(f"  Corr with labels (P_mh): {[round(x,3) for x in corr_P]}")
