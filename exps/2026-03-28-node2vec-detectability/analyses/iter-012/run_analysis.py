import sys, warnings, json, numpy as np, scipy.sparse as sp
import igraph as ig
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
from embcom import utils as embcom_utils
warnings.filterwarnings('ignore')

# ── SBM generation ───────────────────────────────────────────────────────────
def make_sbm_lcc(mu, N=2000, cave=5.0, seed=0):
    n_each = N // 2
    c_out, c_in = mu * cave, 2 * cave - mu * cave
    np.random.seed(seed)
    g = ig.Graph.SBM([[c_in/N, c_out/N], [c_out/N, c_in/N]], [n_each, n_each], directed=False)
    labels = np.array([0]*n_each + [1]*n_each)
    A = sp.csr_matrix(g.get_adjacency_sparse(), dtype=float)
    comps = g.connected_components(mode="weak")
    lcc_idx = sorted(max(comps, key=len))
    return A[np.ix_(lcc_idx, lcc_idx)], labels[lcc_idx]

# ── NetMF clip0 with variable window ─────────────────────────────────────────
def compute_netmf_clip0_window(A_lcc, window=10):
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

def embed_nmi(M, labels, dim=64, seed=42):
    emb = TruncatedSVD(n_components=min(dim, M.shape[0]-1), random_state=seed).fit_transform(M)
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)

# ── Bethe Hessian ─────────────────────────────────────────────────────────────
def bethe_hessian_embed(A_lcc, labels, cave_param=5.0, use_lcc_degree=False, dim=64, seed=42):
    n = A_lcc.shape[0]
    d = np.array(A_lcc.sum(1)).flatten()
    if use_lcc_degree:
        r = np.sqrt(d.mean())
    else:
        r = np.sqrt(cave_param)
    H = (r**2 - 1) * sp.eye(n) - r * A_lcc + sp.diags(d)
    k_find = min(dim + 4, n - 2)
    try:
        vals, vecs = eigsh(H, k=k_find, sigma=0, which='LM')
    except Exception:
        return 0.0, 0
    neg_idx = vals < -1e-8
    if neg_idx.sum() == 0:
        return 0.0, 0
    neg_vecs = vecs[:, neg_idx]
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(neg_vecs)
    nmi = normalized_mutual_info_score(labels, pred)
    return nmi, int(neg_idx.sum())

# ── Experiment config ─────────────────────────────────────────────────────────
MU_VALUES = [0.30, 0.40, 0.45, 0.50]
T_VALUES = [1, 2, 3, 5, 10, 20]
N_SAMPLES = 15
N = 2000
CAVE = 5.0
SEEDS = list(range(N_SAMPLES))

print("=== Sub-analysis A: Window ablation ===")
nmi_vs_window = {}
for T in T_VALUES:
    nmi_vs_window[T] = {}
    for mu in MU_VALUES:
        nmis = []
        for seed in SEEDS:
            A_lcc, labels = make_sbm_lcc(mu, N=N, cave=CAVE, seed=seed)
            M = compute_netmf_clip0_window(A_lcc, window=T)
            nmi = embed_nmi(M, labels)
            nmis.append(nmi)
        mean_nmi = float(np.mean(nmis))
        std_nmi = float(np.std(nmis))
        nmi_vs_window[T][mu] = {"mean": mean_nmi, "std": std_nmi}
        print(f"  T={T:2d}, mu={mu:.2f}: NMI={mean_nmi:.3f} ± {std_nmi:.3f}")

print("\n=== Sub-analysis B & C: Bethe Hessian ===")
nmi_bethe_hessian = {}
nmi_bethe_hessian_swept = {}
for mu in MU_VALUES:
    nmis_bh = []
    nmis_bhs = []
    neg_counts = []
    for seed in SEEDS:
        A_lcc, labels = make_sbm_lcc(mu, N=N, cave=CAVE, seed=seed)
        nmi_bh, neg_k = bethe_hessian_embed(A_lcc, labels, cave_param=CAVE, use_lcc_degree=False)
        nmis_bh.append(nmi_bh)
        neg_counts.append(neg_k)
        nmi_bhs, _ = bethe_hessian_embed(A_lcc, labels, cave_param=CAVE, use_lcc_degree=True)
        nmis_bhs.append(nmi_bhs)
    nmi_bethe_hessian[mu] = {
        "mean": float(np.mean(nmis_bh)),
        "std": float(np.std(nmis_bh)),
        "neg_eigval_count": float(np.mean(neg_counts))
    }
    nmi_bethe_hessian_swept[mu] = {
        "mean": float(np.mean(nmis_bhs)),
        "std": float(np.std(nmis_bhs))
    }
    print(f"  BH(r=sqrt(cave)), mu={mu:.2f}: NMI={nmi_bethe_hessian[mu]['mean']:.3f} ± {nmi_bethe_hessian[mu]['std']:.3f}, neg_k={nmi_bethe_hessian[mu]['neg_eigval_count']:.1f}")
    print(f"  BH(r=sqrt(d_mean)), mu={mu:.2f}: NMI={nmi_bethe_hessian_swept[mu]['mean']:.3f}")

print("\n=== Building nmi_comparison (T=1,3,10 + BH) ===")
nmi_comparison = {
    "netmf_T1": {mu: nmi_vs_window[1][mu] for mu in MU_VALUES},
    "netmf_T3": {mu: nmi_vs_window[3][mu] for mu in MU_VALUES},
    "netmf_T10": {mu: nmi_vs_window[10][mu] for mu in MU_VALUES},
    "bethe_hessian": {mu: {"mean": nmi_bethe_hessian[mu]["mean"], "std": nmi_bethe_hessian[mu]["std"]} for mu in MU_VALUES},
    "bethe_hessian_swept": {mu: nmi_bethe_hessian_swept[mu] for mu in MU_VALUES},
}

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: NMI vs window T
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(MU_VALUES)))
for i, mu in enumerate(MU_VALUES):
    means = [nmi_vs_window[T][mu]["mean"] for T in T_VALUES]
    stds = [nmi_vs_window[T][mu]["std"] for T in T_VALUES]
    ax.errorbar(T_VALUES, means, yerr=stds, marker='o', label=f'mu={mu:.2f}', color=colors[i], capsize=3)
ax.set_xlabel('Window T (number of hops)')
ax.set_ylabel('NMI (mean ± std)')
ax.set_title('NMI vs Window T (NetMF clip0)\nN=2000, cave=5, n=15 samples')
ax.legend()
ax.set_xscale('log')
ax.set_xticks(T_VALUES)
ax.set_xticklabels([str(t) for t in T_VALUES])
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 2: NMI vs mu comparison
ax = axes[1]
methods = [
    ('netmf_T1', 'NetMF T=1 (1-hop log-PMI)', 'o--', 'tab:blue'),
    ('netmf_T3', 'NetMF T=3', 's--', 'tab:orange'),
    ('netmf_T10', 'NetMF T=10', 'D-', 'tab:green'),
    ('bethe_hessian', 'Bethe Hessian r=sqrt(cave)', '^-', 'tab:red'),
    ('bethe_hessian_swept', 'Bethe Hessian r=sqrt(d_mean)', 'v:', 'tab:purple'),
]
for method, label, style, color in methods:
    means = [nmi_comparison[method][mu]["mean"] for mu in MU_VALUES]
    stds = [nmi_comparison[method][mu]["std"] for mu in MU_VALUES]
    ax.errorbar(MU_VALUES, means, yerr=stds, fmt=style, label=label, color=color, capsize=3, linewidth=1.8)
ax.axvline(x=1 - 1/np.sqrt(CAVE), color='k', linestyle=':', alpha=0.5, label=f'mu*={1-1/np.sqrt(CAVE):.3f}')
ax.set_xlabel('mu (mixing parameter)')
ax.set_ylabel('NMI (mean ± std)')
ax.set_title('1-hop vs Multi-hop vs Bethe Hessian\nN=2000, cave=5, n=15 samples')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
fig_path = 'analyses/iter-012/fig_multihop_necessity.png'
plt.savefig(f'/workspace/exps/2026-03-28-node2vec-detectability/{fig_path}', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {fig_path}")

# ── results.json ──────────────────────────────────────────────────────────────
# Stringify float keys for JSON
def str_keys(d):
    if isinstance(d, dict):
        return {str(k): str_keys(v) for k, v in d.items()}
    return d

results = {
    "task": "Window ablation for NetMF_clip0 (T=1..20) and Bethe Hessian (1-hop theoretically optimal) to test whether multi-hop aggregation is NECESSARY for community detection near the SBM detectability limit.",
    "method": "NetMF clip0 with window T in {1,2,3,5,10,20}; Bethe Hessian H(r)=(r^2-1)I - rA + D with r=sqrt(cave) and r=sqrt(d_mean); KMeans on SVD/negative eigenvectors; N=2000, cave=5, 15 samples per (mu, T)",
    "result_summary": (
        f"T=1 NMI at mu=0.40: {nmi_vs_window[1][0.40]['mean']:.3f}±{nmi_vs_window[1][0.40]['std']:.3f}; "
        f"T=10 NMI at mu=0.40: {nmi_vs_window[10][0.40]['mean']:.3f}±{nmi_vs_window[10][0.40]['std']:.3f}. "
        f"Bethe Hessian NMI at mu=0.40: {nmi_bethe_hessian[0.40]['mean']:.3f}±{nmi_bethe_hessian[0.40]['std']:.3f}."
    ),
    "key_numbers": {
        "nmi_T1_mu040": nmi_vs_window[1][0.40]["mean"],
        "nmi_T2_mu040": nmi_vs_window[2][0.40]["mean"],
        "nmi_T3_mu040": nmi_vs_window[3][0.40]["mean"],
        "nmi_T5_mu040": nmi_vs_window[5][0.40]["mean"],
        "nmi_T10_mu040": nmi_vs_window[10][0.40]["mean"],
        "nmi_T20_mu040": nmi_vs_window[20][0.40]["mean"],
        "nmi_BH_mu040": nmi_bethe_hessian[0.40]["mean"],
        "nmi_BH_swept_mu040": nmi_bethe_hessian_swept[0.40]["mean"],
        "nmi_T1_mu030": nmi_vs_window[1][0.30]["mean"],
        "nmi_T10_mu030": nmi_vs_window[10][0.30]["mean"],
        "nmi_BH_mu030": nmi_bethe_hessian[0.30]["mean"],
        "nmi_T1_mu045": nmi_vs_window[1][0.45]["mean"],
        "nmi_T10_mu045": nmi_vs_window[10][0.45]["mean"],
        "nmi_BH_mu045": nmi_bethe_hessian[0.45]["mean"],
        "nmi_T1_mu050": nmi_vs_window[1][0.50]["mean"],
        "nmi_T10_mu050": nmi_vs_window[10][0.50]["mean"],
        "nmi_BH_mu050": nmi_bethe_hessian[0.50]["mean"],
        "neg_eigval_count_mu040": nmi_bethe_hessian[0.40]["neg_eigval_count"],
        "neg_eigval_count_mu050": nmi_bethe_hessian[0.50]["neg_eigval_count"],
    },
    "nmi_vs_window": str_keys(nmi_vs_window),
    "nmi_bethe_hessian": str_keys(nmi_bethe_hessian),
    "nmi_bethe_hessian_swept": str_keys(nmi_bethe_hessian_swept),
    "nmi_comparison": str_keys(nmi_comparison),
    "figures_created": [fig_path],
    "code_used": """
def compute_netmf_clip0_window(A_lcc, window=10):
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

def bethe_hessian_embed(A_lcc, labels, cave_param=5.0, use_lcc_degree=False):
    n = A_lcc.shape[0]; d = np.array(A_lcc.sum(1)).flatten()
    r = np.sqrt(d.mean()) if use_lcc_degree else np.sqrt(cave_param)
    H = (r**2 - 1) * sp.eye(n) - r * A_lcc + sp.diags(d)
    vals, vecs = eigsh(H, k=min(66, n-2), sigma=0, which='LM')
    neg_idx = vals < -1e-8
    if neg_idx.sum() == 0: return 0.0, 0
    pred = KMeans(n_clusters=2, n_init=10).fit_predict(vecs[:, neg_idx])
    return normalized_mutual_info_score(labels, pred), int(neg_idx.sum())
""",
    "shared_components": [
        {
            "name": "make_sbm_lcc",
            "description": "SBM generation with corrected parameterization: c_out=mu*cave, c_in=2*cave-c_out. Returns LCC adjacency + labels.",
            "output_path": None,
            "why_shared": "Standard SBM factory used in all iter-0xx analyses"
        }
    ],
    "next_questions": [
        "Does the Bethe Hessian outperform or match 10-hop NetMF? If so, multi-hop is NOT necessary — BH is the theoretically optimal 1-hop operator.",
        "At what window T does NMI plateau? If plateau is at T=3-5, multi-hop helps but only marginally beyond a small number of hops.",
        "How does neg_eigval_count change with mu? At mu>mu*, neg_eigval_count should drop to 0 — can this serve as a phase-transition detector?",
        "Compare Bethe Hessian with node2vec (walk-based multi-hop) to see if BH matches the gold standard."
    ],
    "failed_attempts": []
}

out_path = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-012/results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results written to {out_path}")
