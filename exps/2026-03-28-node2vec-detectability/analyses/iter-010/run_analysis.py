import sys, warnings, numpy as np, scipy.sparse as sp
import igraph as ig
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
from embcom import utils as embcom_utils
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-010'

def make_sbm_lcc(mu, N=2000, cave=5.0, seed=0):
    n_each = N // 2
    c_out, c_in = mu*cave, 2*cave - mu*cave
    p_in, p_out = c_in/N, c_out/N
    np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
    labels = np.array([0]*n_each + [1]*n_each)
    A = sp.csr_matrix(g.get_adjacency_sparse(), dtype=float)
    comps = g.connected_components(mode="weak")
    lcc_idx = sorted(max(comps, key=len))
    return A[np.ix_(lcc_idx, lcc_idx)], labels[lcc_idx]

def multihop(M_dense, T=10):
    n = M_dense.shape[0]
    result = np.zeros((n, n))
    Mt = np.eye(n)
    for _ in range(T):
        Mt = Mt @ M_dense
        result += Mt
    return result / T

def degree_normalize_log_clip(M_mh, d, vol):
    """Degree-normalize (by d_j), then log+clip."""
    M_norm = M_mh * (vol / np.maximum(d, 1e-12))[np.newaxis, :]
    return np.maximum(np.log(np.maximum(M_norm, 1e-300)), 0.0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def embed_nmi(M, labels, dim=64, seed=42):
    svd = TruncatedSVD(n_components=min(dim, M.shape[0]-1), random_state=seed)
    emb = svd.fit_transform(M)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)

def sv_ratio(M, dim=64):
    """Return SV1/SV2 ratio."""
    svd = TruncatedSVD(n_components=min(dim, M.shape[0]-1), random_state=42)
    svd.fit(M)
    sv = svd.singular_values_
    return sv[0] / sv[1] if sv[1] > 0 else float('inf')

def compute_methods(A_lcc, return_matrices=False):
    """Compute all matrices for one graph."""
    d = np.array(A_lcc.sum(1)).flatten()
    vol = d.sum()
    A_dense = A_lcc.toarray().astype(np.float64)

    # Random walk P = D^{-1} A
    P = (A_dense.T / np.maximum(d, 1e-12)).T

    # Normalized adjacency M_na = D^{-1/2} A D^{-1/2}
    d_inv_sqrt = 1.0 / np.maximum(np.sqrt(d), 1e-12)
    M_na = d_inv_sqrt[:, None] * A_dense * d_inv_sqrt[None, :]

    # Modularity matrix B = A - d*d^T / vol
    B = A_dense - np.outer(d, d) / vol

    # --- Group A: Unified recipe (multi-hop + degree-norm + log + clip) ---

    # 1. adj_netmf_clip: A -> 10-hop -> degree-normalize -> log -> clip0
    A_mh = multihop(A_dense, T=10)
    adj_netmf_clip = degree_normalize_log_clip(A_mh, d, vol)

    # 2. randwalk_netmf_clip: P -> 10-hop -> degree-normalize -> log -> clip0 (= standard NetMF)
    P_mh = multihop(P, T=10)
    randwalk_netmf_clip = degree_normalize_log_clip(P_mh, d, vol)

    # 3. norm_adj_netmf_clip: M_na -> 10-hop -> degree-normalize -> log -> clip0
    Mna_mh = multihop(M_na, T=10)
    norm_adj_netmf_clip = degree_normalize_log_clip(Mna_mh, d, vol)

    # 4. modularity_netmf_clip: B -> 10-hop -> offset to positive -> log -> clip0
    # B has negative entries; offset so min=0 before log
    B_mh = multihop(B, T=10)
    B_mh_pos = B_mh - B_mh.min()  # shift to all-positive
    # Apply same degree-normalize + log + clip (skip degree-norm since B already normalizes)
    mod_netmf_clip = np.maximum(np.log(np.maximum(B_mh_pos, 1e-300)), 0.0)

    # --- Group B: Modularity-specific recipes ---

    # 5. mod_10hop_raw: SVD(B_10hop) - raw
    mod_10hop_raw = B_mh

    # 6. mod_10hop_clip0: SVD(max(B_10hop, 0))
    mod_10hop_clip0 = np.maximum(B_mh, 0.0)

    # 7. mod_10hop_sigmoid: SVD(sigmoid(B_10hop))
    mod_10hop_sigmoid = sigmoid(B_mh)

    # 8. mod_10hop_gradweight: SVD(B_10hop * sig * (1-sig))
    sig_B = sigmoid(B_mh)
    mod_10hop_gradweight = B_mh * sig_B * (1.0 - sig_B)

    matrices = {
        'adj_netmf_clip': adj_netmf_clip,
        'randwalk_netmf_clip': randwalk_netmf_clip,
        'norm_adj_netmf_clip': norm_adj_netmf_clip,
        'mod_netmf_clip': mod_netmf_clip,
        'mod_10hop_raw': mod_10hop_raw,
        'mod_10hop_clip0': mod_10hop_clip0,
        'mod_10hop_sigmoid': mod_10hop_sigmoid,
        'mod_10hop_gradweight': mod_10hop_gradweight,
    }
    return matrices

# Check if adj_netmf_clip == randwalk_netmf_clip (sanity check)
print("Running sanity check...")
A_test, lab_test = make_sbm_lcc(0.30, N=500, seed=0)
mats = compute_methods(A_test)
diff = np.abs(mats['adj_netmf_clip'] - mats['randwalk_netmf_clip'])
print(f"  adj_netmf_clip vs randwalk_netmf_clip: max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}")

# Full sweep
MU_VALUES = [0.30, 0.40, 0.45, 0.50]
N_SAMPLES = 10
N = 2000

method_names = [
    'adj_netmf_clip',
    'randwalk_netmf_clip',
    'norm_adj_netmf_clip',
    'mod_netmf_clip',
    'mod_10hop_raw',
    'mod_10hop_clip0',
    'mod_10hop_sigmoid',
    'mod_10hop_gradweight',
]

results = {m: {mu: [] for mu in MU_VALUES} for m in method_names}
sv_ratios_mu40 = {m: [] for m in method_names}

total = len(MU_VALUES) * N_SAMPLES
done = 0
for mu in MU_VALUES:
    for seed in range(N_SAMPLES):
        A_lcc, labels = make_sbm_lcc(mu, N=N, seed=seed)
        mats = compute_methods(A_lcc)
        for m in method_names:
            nmi = embed_nmi(mats[m], labels)
            results[m][mu].append(nmi)
            if mu == 0.40:
                sv_ratios_mu40[m].append(sv_ratio(mats[m]))
        done += 1
        if done % 10 == 0:
            print(f"  Progress: {done}/{total}")

print("Computing summary statistics...")
summary = {}
for m in method_names:
    summary[m] = {}
    for mu in MU_VALUES:
        vals = results[m][mu]
        summary[m][mu] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

sv_summary = {m: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for m, v in sv_ratios_mu40.items()}

# Print table
print("\n=== NMI Results ===")
print(f"{'Method':<30}", end="")
for mu in MU_VALUES:
    print(f"  mu={mu:.2f}", end="")
print()
for m in method_names:
    print(f"{m:<30}", end="")
    for mu in MU_VALUES:
        s = summary[m][mu]
        print(f"  {s['mean']:.3f}±{s['std']:.3f}", end="")
    print()

print("\n=== SV1/SV2 Ratios at mu=0.40 ===")
for m in method_names:
    print(f"  {m:<30}: {sv_summary[m]['mean']:.2f} ± {sv_summary[m]['std']:.2f}")

# ===== FIGURE =====
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Group A (unified recipe)
ax1 = axes[0]
group_a = ['adj_netmf_clip', 'randwalk_netmf_clip', 'norm_adj_netmf_clip', 'mod_netmf_clip']
colors_a = ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple']
labels_a = ['adj_netmf_clip (A)', 'randwalk_netmf_clip (P=NetMF)', 'norm_adj_netmf_clip (M_na)', 'mod_netmf_clip (B)']
for m, c, lbl in zip(group_a, colors_a, labels_a):
    means = [summary[m][mu]['mean'] for mu in MU_VALUES]
    stds = [summary[m][mu]['std'] for mu in MU_VALUES]
    ax1.errorbar(MU_VALUES, means, yerr=stds, label=lbl, color=c, marker='o', capsize=4)
ax1.set_xlabel('mu')
ax1.set_ylabel('NMI')
ax1.set_title('Group A: Unified Recipe (Multi-hop + log + clip)\non each operator')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.25, 0.55)

# Panel 2: Group B (modularity variants)
ax2 = axes[1]
group_b = ['mod_10hop_raw', 'mod_10hop_clip0', 'mod_10hop_sigmoid', 'mod_10hop_gradweight', 'randwalk_netmf_clip']
colors_b = ['tab:gray', 'tab:cyan', 'tab:red', 'tab:olive', 'tab:blue']
labels_b = ['mod_10hop_raw', 'mod_10hop_clip0', 'mod_10hop_sigmoid (iter-009)', 'mod_10hop_gradweight', 'randwalk_netmf_clip (NetMF ref)']
for m, c, lbl in zip(group_b, colors_b, labels_b):
    means = [summary[m][mu]['mean'] for mu in MU_VALUES]
    stds = [summary[m][mu]['std'] for mu in MU_VALUES]
    ls = '--' if m == 'randwalk_netmf_clip' else '-'
    ax2.errorbar(MU_VALUES, means, yerr=stds, label=lbl, color=c, marker='o', capsize=4, linestyle=ls)
ax2.set_xlabel('mu')
ax2.set_ylabel('NMI')
ax2.set_title('Group B: Modularity Variants vs NetMF')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.25, 0.55)

# Panel 3: SV1/SV2 bar chart at mu=0.40
ax3 = axes[2]
sv_means = [sv_summary[m]['mean'] for m in method_names]
sv_stds = [sv_summary[m]['std'] for m in method_names]
x = np.arange(len(method_names))
bars = ax3.bar(x, sv_means, yerr=sv_stds, capsize=4, color='steelblue', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(method_names, rotation=45, ha='right', fontsize=7)
ax3.set_ylabel('SV1/SV2 ratio')
ax3.set_title('SV1/SV2 Ratio at mu=0.40\n(lower = less degree-dominated)')
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='ideal (=1)')
ax3.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig_unified_framework.png')
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

# ===== WRITE results.json =====
key_numbers = {}
for m in method_names:
    for mu in MU_VALUES:
        key = f"nmi_{m}_mu{int(mu*100)}"
        key_numbers[key] = round(summary[m][mu]['mean'], 4)
for m in method_names:
    key_numbers[f"sv_ratio_{m}"] = round(sv_summary[m]['mean'], 3)

# Check if adj==randwalk (sanity)
key_numbers['adj_vs_randwalk_max_diff'] = round(float(diff.max()), 4)

# Best method at mu=0.40
nmi_mu40 = {m: summary[m][0.40]['mean'] for m in method_names}
best_method = max(nmi_mu40, key=nmi_mu40.__getitem__)
key_numbers['best_method_mu40'] = best_method
key_numbers['best_nmi_mu40'] = round(nmi_mu40[best_method], 4)
key_numbers['netmf_nmi_mu40'] = round(nmi_mu40['randwalk_netmf_clip'], 4)

result_summary = (
    f"randwalk_netmf_clip (standard NetMF) achieves NMI={nmi_mu40['randwalk_netmf_clip']:.3f} at mu=0.40. "
    f"adj_netmf_clip (same recipe on A instead of P) achieves {nmi_mu40['adj_netmf_clip']:.3f}. "
    f"Best method is {best_method} with NMI={nmi_mu40[best_method]:.3f}. "
    f"adj vs randwalk max_diff={diff.max():.4f} (sanity check for recipe equivalence). "
    f"mod_10hop_sigmoid reaches {nmi_mu40['mod_10hop_sigmoid']:.3f}."
)

results_json = {
    "task": "Test unified 3-step recipe (multi-hop + degree-normalize + log+clip) applied to each base operator (A, M_na, B, P) and compare to modularity-specific recipes and NetMF reference.",
    "method": "SVD+KMeans on 8 matrix variants: Group A (unified PMI recipe on each operator) and Group B (modularity B multi-hop with raw/clip/sigmoid/gradweight). N=2000 SBM, mu=[0.30,0.40,0.45,0.50], 10 samples each.",
    "result_summary": result_summary,
    "key_numbers": key_numbers,
    "figures_created": ["analyses/iter-010/fig_unified_framework.png"],
    "code_used": """
# Core per-graph computation
def compute_methods(A_lcc):
    d = np.array(A_lcc.sum(1)).flatten(); vol = d.sum()
    A_dense = A_lcc.toarray().astype(float)
    P = (A_dense.T / np.maximum(d, 1e-12)).T
    d_inv_sqrt = 1.0 / np.maximum(np.sqrt(d), 1e-12)
    M_na = d_inv_sqrt[:, None] * A_dense * d_inv_sqrt[None, :]
    B = A_dense - np.outer(d, d) / vol
    A_mh = multihop(A_dense, T=10)
    adj_netmf_clip = degree_normalize_log_clip(A_mh, d, vol)
    P_mh = multihop(P, T=10)
    randwalk_netmf_clip = degree_normalize_log_clip(P_mh, d, vol)  # = standard NetMF
    Mna_mh = multihop(M_na, T=10)
    norm_adj_netmf_clip = degree_normalize_log_clip(Mna_mh, d, vol)
    B_mh = multihop(B, T=10)
    B_mh_pos = B_mh - B_mh.min()
    mod_netmf_clip = np.maximum(np.log(np.maximum(B_mh_pos, 1e-300)), 0.0)
    mod_10hop_sigmoid = sigmoid(B_mh)
    ...
""",
    "shared_components": [
        {
            "name": "make_sbm_lcc",
            "description": "SBM graph generator with LCC extraction, N=2000, cave=5.0, corrected parameterization",
            "output_path": "analyses/iter-010/run_analysis.py",
            "why_shared": "Same SBM generation used across all iterations"
        }
    ],
    "next_questions": [
        "Does adj_netmf_clip == randwalk_netmf_clip? (sanity check: if not, the PMI recipe is NOT operator-invariant)",
        "If the unified recipe on A and P give different results, what is the mathematical reason?",
        "Can we combine the best modularity recipe with degree-normalization to close the gap to NetMF?",
        "Is there a theoretical reason why the random walk P is the 'correct' operator for the log-PMI recipe?"
    ],
    "failed_attempts": []
}

results_path = os.path.join(OUTPUT_DIR, 'results.json')
with open(results_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Results written: {results_path}")
print("\nDone.")
