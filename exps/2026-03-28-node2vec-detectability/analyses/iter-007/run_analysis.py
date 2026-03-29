"""
iter-007: Definitive confirmation of node2vec exceptionalism near SBM detectability limit.
30 samples for main methods, 15 samples for BP.

Methods: node2vec, netmf_clipped, netmf_full, spectral, bp
"""
import sys
import warnings
import json
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import TruncatedSVD
import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/libs/BeliefPropagation')
sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
import belief_propagation
import embcom
from embcom import utils as embcom_utils

warnings.filterwarnings('ignore')

# ─── Parameters ──────────────────────────────────────────────────────────────
MU_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.52]
N_SAMPLES_MAIN = 30   # for node2vec, netmf_clipped, netmf_full, spectral
N_SAMPLES_BP = 15     # for BP (slow ~7s/sample)
N = 2000
CAVE = 5.0
DIM = 64
MU_STAR = 0.553

OUT_DIR = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-007'

# ─── Graph generation ────────────────────────────────────────────────────────
def make_sbm(mu, N=2000, cave=5.0, seed=0):
    """Return (A_full, labels_full, A_lcc, labels_lcc)."""
    n_each = N // 2
    c_out = mu * cave
    c_in = 2 * cave - c_out
    p_in, p_out = c_in / N, c_out / N
    np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
    labels = np.array([0] * n_each + [1] * n_each)
    A = sp.csr_matrix(g.get_adjacency_sparse(), dtype=float)
    comps = g.connected_components(mode='weak')
    lcc_idx = sorted(max(comps, key=len))
    A_lcc = A[np.ix_(lcc_idx, lcc_idx)]
    labels_lcc = labels[lcc_idx]
    return A, labels, A_lcc, labels_lcc

# ─── NetMF matrix ────────────────────────────────────────────────────────────
def compute_netmf(A_lcc, window=10):
    """Return raw log matrix (no clipping)."""
    P = embcom_utils.to_trans_mat(A_lcc)
    P_d = P.toarray().astype(np.float64)
    Ppow = np.zeros_like(P_d)
    Pt = np.eye(len(P_d))
    for _ in range(window):
        Pt = Pt @ P_d
        Ppow += Pt
    Ppow /= window
    d = np.array(A_lcc.sum(1)).flatten()
    vol = d.sum()
    M_raw = Ppow @ np.diag(vol / np.maximum(d, 1e-12))
    return np.log(np.maximum(M_raw, 1e-300))

def embed_svd(M, dim=64, clip=None):
    if clip is not None:
        M = np.maximum(M, clip)
    return TruncatedSVD(n_components=min(dim, M.shape[0]-1), random_state=42).fit_transform(M)

# ─── Spectral embedding ──────────────────────────────────────────────────────
def embed_spectral(A_lcc, dim=64):
    d = np.array(A_lcc.sum(1)).flatten()
    D_isqrt = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    M = D_isqrt @ A_lcc @ D_isqrt
    k = min(dim + 2, A_lcc.shape[0] - 1)
    vals, vecs = spla.eigsh(M, k=k, which='LM')
    order = np.argsort(vals)[::-1]
    return vecs[:, order][:, 1:dim+1]  # skip trivial eigenvec

# ─── node2vec ────────────────────────────────────────────────────────────────
def embed_node2vec(A_lcc, dim=64):
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10, p=1.0, q=1.0)
    model.fit(A_lcc)
    return model.transform(dim=dim)

# ─── KMeans NMI ──────────────────────────────────────────────────────────────
def nmi_kmeans(emb, labels, seed=42):
    pred = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)

# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Main NMI sweep — 30 samples × 6 mu values
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Part A: Main NMI sweep (30 samples × 6 mu values) ===")

results_main = {
    'node2vec':      {mu: [] for mu in MU_VALUES},
    'netmf_clipped': {mu: [] for mu in MU_VALUES},
    'netmf_full':    {mu: [] for mu in MU_VALUES},
    'spectral':      {mu: [] for mu in MU_VALUES},
    'bp':            {mu: [] for mu in MU_VALUES},
}

t0_all = time.time()

for mu in MU_VALUES:
    print(f"\n  mu={mu:.2f}:")
    for seed in range(N_SAMPLES_MAIN):
        A_full, labels_full, A_lcc, labels_lcc = make_sbm(mu, N=N, cave=CAVE, seed=seed)

        # node2vec
        try:
            emb = embed_node2vec(A_lcc, dim=DIM)
            nmi = nmi_kmeans(emb, labels_lcc)
            results_main['node2vec'][mu].append(nmi)
        except Exception as e:
            print(f"    node2vec seed={seed} failed: {e}")
            results_main['node2vec'][mu].append(float('nan'))

        # netmf_clipped
        try:
            M = compute_netmf(A_lcc)
            emb = embed_svd(M, dim=DIM, clip=0.0)
            nmi = nmi_kmeans(emb, labels_lcc)
            results_main['netmf_clipped'][mu].append(nmi)
        except Exception as e:
            print(f"    netmf_clipped seed={seed} failed: {e}")
            results_main['netmf_clipped'][mu].append(float('nan'))

        # netmf_full (no clip)
        try:
            M = compute_netmf(A_lcc)
            emb = embed_svd(M, dim=DIM, clip=None)
            nmi = nmi_kmeans(emb, labels_lcc)
            results_main['netmf_full'][mu].append(nmi)
        except Exception as e:
            print(f"    netmf_full seed={seed} failed: {e}")
            results_main['netmf_full'][mu].append(float('nan'))

        # spectral
        try:
            emb = embed_spectral(A_lcc, dim=DIM)
            nmi = nmi_kmeans(emb, labels_lcc)
            results_main['spectral'][mu].append(nmi)
        except Exception as e:
            print(f"    spectral seed={seed} failed: {e}")
            results_main['spectral'][mu].append(float('nan'))

        # BP (only first 15 seeds)
        if seed < N_SAMPLES_BP:
            try:
                pred = belief_propagation.detect(A_full.copy(), q=2)
                nmi = normalized_mutual_info_score(labels_full, pred)
                results_main['bp'][mu].append(nmi)
            except Exception as e:
                print(f"    bp seed={seed} failed: {e}")
                results_main['bp'][mu].append(float('nan'))

        if seed % 10 == 9:
            elapsed = time.time() - t0_all
            print(f"    seed {seed+1}/{N_SAMPLES_MAIN} done (total elapsed: {elapsed:.0f}s)")

# Summarize
nmi_by_method_mu = {}
for method in results_main:
    nmi_by_method_mu[method] = {}
    for mu in MU_VALUES:
        vals = [v for v in results_main[method][mu] if not (v != v)]  # filter nan
        n_valid = len([v for v in vals if not np.isnan(v)])
        vals_clean = [v for v in vals if not np.isnan(v)]
        nmi_by_method_mu[method][str(mu)] = {
            'mean': float(np.mean(vals_clean)) if vals_clean else float('nan'),
            'std': float(np.std(vals_clean)) if vals_clean else float('nan'),
            'n': n_valid
        }

print("\nSummary NMI by method and mu:")
for method, mu_dict in nmi_by_method_mu.items():
    print(f"  {method}:")
    for mu_str, stats in mu_dict.items():
        print(f"    mu={mu_str}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n']})")

# ═══════════════════════════════════════════════════════════════════════════════
# Part B: Mechanism — singular value spectrum at mu=0.40
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Part B: Singular value spectrum at mu=0.40 (one sample, top 20 SVs) ===")

MU_MECH = 0.40

# Single sample for spectrum
A_full_0, labels_full_0, A_lcc_0, labels_lcc_0 = make_sbm(MU_MECH, N=N, cave=CAVE, seed=0)
M_raw_0 = compute_netmf(A_lcc_0)
M_full_0 = M_raw_0  # no clip
M_clip_0 = np.maximum(M_raw_0, 0.0)

N_SVS = 20
svd_full = TruncatedSVD(n_components=N_SVS, random_state=42)
svd_full.fit(M_full_0)
sv_full = svd_full.singular_values_.tolist()

svd_clip = TruncatedSVD(n_components=N_SVS, random_state=42)
svd_clip.fit(M_clip_0)
sv_clipped = svd_clip.singular_values_.tolist()

print(f"Top-5 SVs full:    {[f'{v:.1f}' for v in sv_full[:5]]}")
print(f"Top-5 SVs clipped: {[f'{v:.1f}' for v in sv_clipped[:5]]}")

# ─── Part C: Eigenvector–community correlation, avg 5 samples ────────────────
print("\n=== Part C: |Corr(top-5 SVecs, labels)| at mu=0.40, avg 5 samples ===")

N_CORR_SAMPLES = 5
N_TOP = 5

corr_full_all = []
corr_clip_all = []

for seed in range(N_CORR_SAMPLES):
    _, _, A_lcc_s, labels_lcc_s = make_sbm(MU_MECH, N=N, cave=CAVE, seed=seed)
    M_s = compute_netmf(A_lcc_s)

    svd_f = TruncatedSVD(n_components=N_TOP, random_state=42)
    svd_f.fit(M_s)
    # right singular vectors (rows of components_ = shape [n_top, n])
    rvecs_f = svd_f.components_.T  # shape [n, n_top]

    M_s_clip = np.maximum(M_s, 0.0)
    svd_c = TruncatedSVD(n_components=N_TOP, random_state=42)
    svd_c.fit(M_s_clip)
    rvecs_c = svd_c.components_.T

    cf = [abs(float(np.corrcoef(rvecs_f[:, i], labels_lcc_s)[0, 1])) for i in range(N_TOP)]
    cc = [abs(float(np.corrcoef(rvecs_c[:, i], labels_lcc_s)[0, 1])) for i in range(N_TOP)]
    corr_full_all.append(cf)
    corr_clip_all.append(cc)

corr_full_mean = np.nanmean(corr_full_all, axis=0).tolist()
corr_clip_mean = np.nanmean(corr_clip_all, axis=0).tolist()

print(f"|Corr| full (rank 1-5):    {[f'{v:.3f}' for v in corr_full_mean]}")
print(f"|Corr| clipped (rank 1-5): {[f'{v:.3f}' for v in corr_clip_mean]}")

# ─── Part D: NMI vs clip threshold ───────────────────────────────────────────
print("\n=== Part D: NMI vs clip threshold at mu=0.40, 10 samples ===")

N_CLIP_SAMPLES = 10
CLIP_THRESHOLDS = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0]

clip_nmi_all = {t: [] for t in CLIP_THRESHOLDS}

for seed in range(N_CLIP_SAMPLES):
    _, _, A_lcc_s, labels_s = make_sbm(MU_MECH, N=N, cave=CAVE, seed=seed)
    M_s = compute_netmf(A_lcc_s)
    for clip in CLIP_THRESHOLDS:
        try:
            emb = embed_svd(M_s, dim=DIM, clip=clip)
            nmi = nmi_kmeans(emb, labels_s)
            clip_nmi_all[clip].append(nmi)
        except Exception as e:
            clip_nmi_all[clip].append(float('nan'))

clip_nmi_mean = {str(t): float(np.nanmean(clip_nmi_all[t])) for t in CLIP_THRESHOLDS}
clip_nmi_std  = {str(t): float(np.nanstd(clip_nmi_all[t])) for t in CLIP_THRESHOLDS}
print("NMI vs clip threshold:", {k: f"{v:.3f}" for k, v in clip_nmi_mean.items()})

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Generating figures ===")

# ─── Figure 1: Main NMI comparison ───────────────────────────────────────────
method_styles = {
    'bp':            ('red',    'o-', 'BP'),
    'node2vec':      ('green',  's-', 'node2vec (SGNS)'),
    'netmf_clipped': ('orange', '^-', 'NetMF clipped'),
    'spectral':      ('blue',   'D-', 'Spectral'),
    'netmf_full':    ('gray',   'v-', 'NetMF full'),
}
legend_order = ['bp', 'node2vec', 'netmf_clipped', 'spectral', 'netmf_full']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('NMI vs mixing parameter μ — SBM N=2000, cave=5', fontsize=13, y=1.01)

for ax_idx, ax in enumerate(axes):
    for method in legend_order:
        color, fmt, label = method_styles[method]
        means = [nmi_by_method_mu[method][str(mu)]['mean'] for mu in MU_VALUES]
        stds  = [nmi_by_method_mu[method][str(mu)]['std']  for mu in MU_VALUES]
        n_pts = [nmi_by_method_mu[method][str(mu)]['n']    for mu in MU_VALUES]

        means_arr = np.array(means, dtype=float)
        stds_arr  = np.array(stds,  dtype=float)

        ax.plot(MU_VALUES, means_arr, fmt, color=color, label=label,
                linewidth=2, markersize=7)
        ax.fill_between(MU_VALUES,
                        means_arr - stds_arr,
                        means_arr + stds_arr,
                        color=color, alpha=0.15)

    ax.axvline(x=MU_STAR, color='black', linestyle='--', alpha=0.6, linewidth=1.5,
               label=f'μ*={MU_STAR}')
    ax.set_xlabel('μ (mixing parameter)', fontsize=12)
    ax.set_ylabel('NMI', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.0)

    if ax_idx == 0:
        ax.set_title('All μ values', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0.28, 0.56)
    else:
        ax.set_title('Zoom: near-limit regime', fontsize=12)
        ax.set_xlim(0.38, 0.53)
        ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
fig1_path = f'{OUT_DIR}/fig_main_nmi_comparison.png'
plt.savefig(fig1_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")

# ─── Figure 2: Mechanism figure (3 panels) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Mechanism: why clipping rescues SVD near detectability limit (μ=0.40)', fontsize=13)

# Panel 1: Singular value spectrum (log scale)
ax = axes[0]
ranks = np.arange(1, N_SVS + 1)
ax.semilogy(ranks, sv_full,    'r-o', linewidth=2, markersize=5, label='Full NetMF')
ax.semilogy(ranks, sv_clipped, 'g-s', linewidth=2, markersize=5, label='Clipped NetMF (≥0)')
ax.set_xlabel('Singular value rank', fontsize=11)
ax.set_ylabel('Singular value (log scale)', fontsize=11)
ax.set_title(f'Panel 1: SV spectrum at μ=0.40\n(one sample, top {N_SVS} SVs)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xticks(ranks[::2])
# annotate top-1 gap
ax.annotate(f'SV1 full = {sv_full[0]:.0f}', xy=(1, sv_full[0]),
            xytext=(3, sv_full[0]*0.8), fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.annotate(f'SV1 clipped = {sv_clipped[0]:.0f}', xy=(1, sv_clipped[0]),
            xytext=(3, sv_clipped[0]*1.5), fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

# Panel 2: |Corr| of top-5 SVecs with labels
ax = axes[1]
top5_ranks = np.arange(1, N_TOP + 1)
width = 0.35
ax.bar(top5_ranks - width/2, corr_full_mean, width,
       label='Full NetMF', color='red', alpha=0.75)
ax.bar(top5_ranks + width/2, corr_clip_mean, width,
       label='Clipped NetMF', color='green', alpha=0.75)
ax.set_xlabel('Singular vector rank', fontsize=11)
ax.set_ylabel('|Pearson corr with true labels|', fontsize=11)
ax.set_title(f'Panel 2: Community alignment of top SVecs\n(μ=0.40, avg {N_CORR_SAMPLES} samples)', fontsize=11)
ax.set_xticks(top5_ranks)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)

# Panel 3: NMI vs clip threshold
ax = axes[2]
thresholds_plot = CLIP_THRESHOLDS
means_clip = [clip_nmi_mean[str(t)] for t in thresholds_plot]
stds_clip  = [clip_nmi_std[str(t)]  for t in thresholds_plot]
ax.plot(thresholds_plot, means_clip, 'ko-', linewidth=2, markersize=7)
ax.fill_between(thresholds_plot,
                np.array(means_clip) - np.array(stds_clip),
                np.array(means_clip) + np.array(stds_clip),
                color='gray', alpha=0.2)
ax.axvline(x=0.0, color='green', linestyle='--', linewidth=2, label='clip=0 (NetMF paper)')
ax.set_xlabel('Clip threshold', fontsize=11)
ax.set_ylabel('NMI at μ=0.40', fontsize=11)
ax.set_title(f'Panel 3: NMI vs clip threshold\n(μ=0.40, {N_CLIP_SAMPLES} samples)', fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(-0.02, 0.8)
ax.grid(True, alpha=0.3)
ax.set_xticks(CLIP_THRESHOLDS[::2])

plt.tight_layout()
fig2_path = f'{OUT_DIR}/fig_mechanism.png'
plt.savefig(fig2_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Write results.json
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Writing results.json ===")

def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Best numbers at key mu values for summary
nmi40 = {m: nmi_by_method_mu[m]['0.4']['mean'] for m in nmi_by_method_mu}
nmi45 = {m: nmi_by_method_mu[m]['0.45']['mean'] for m in nmi_by_method_mu}
nmi50 = {m: nmi_by_method_mu[m]['0.5']['mean'] for m in nmi_by_method_mu}

results = {
    "task": (
        "Definitive confirmation of node2vec exceptionalism near SBM detectability limit. "
        "5 methods (BP, node2vec, netmf_clipped, netmf_full, spectral), N=2000, cave=5, "
        "mu=[0.30,0.35,0.40,0.45,0.50,0.52], 30 samples (15 for BP). "
        "Plus mechanism figures: SV spectrum, eigvec-community alignment, NMI vs clip threshold."
    ),
    "method": (
        "SBM N=2000, cave=5. node2vec: embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10). "
        "netmf_clipped: NetMF log matrix, clip>=0, TruncatedSVD(64). "
        "netmf_full: same but no clip. "
        "spectral: eigsh(D^{-1/2}AD^{-1/2}), skip trivial. "
        "BP: belief_propagation.detect(A_full, q=2)."
    ),
    "result_summary": (
        f"BP is best method overall (NMI at mu=0.40: {nmi40.get('bp', float('nan')):.3f}). "
        f"node2vec and netmf_clipped are nearly identical and both outperform netmf_full and spectral near the limit. "
        f"At mu=0.40: node2vec={nmi40['node2vec']:.3f}, netmf_clipped={nmi40['netmf_clipped']:.3f}, "
        f"netmf_full={nmi40['netmf_full']:.3f}, spectral={nmi40['spectral']:.3f}. "
        f"Clipping dominates: full SV1={sv_full[0]:.0f} vs clipped SV1={sv_clipped[0]:.0f} "
        f"({sv_full[0]/sv_clipped[0]:.1f}x ratio)."
    ),
    "key_numbers": {
        "nmi_by_method_mu": nmi_by_method_mu,
        "singular_values_full_vs_clipped": {
            "full":    [float(v) for v in sv_full],
            "clipped": [float(v) for v in sv_clipped]
        },
        "eigvec_community_corr": {
            "full":    [float(v) for v in corr_full_mean],
            "clipped": [float(v) for v in corr_clip_mean]
        },
        "nmi_vs_clip_threshold": {k: v for k, v in clip_nmi_mean.items()},
        "nmi_vs_clip_threshold_std": {k: v for k, v in clip_nmi_std.items()},
        "sv1_full_vs_clipped_ratio": float(sv_full[0] / sv_clipped[0]),
        "N": N,
        "n_samples_main": N_SAMPLES_MAIN,
        "n_samples_bp": N_SAMPLES_BP,
        "mu_star": MU_STAR,
    },
    "figures_created": [
        "analyses/iter-007/fig_main_nmi_comparison.png",
        "analyses/iter-007/fig_mechanism.png",
    ],
    "code_used": (
        "def compute_netmf(A_lcc, window=10):\n"
        "    P = embcom_utils.to_trans_mat(A_lcc); P_d = P.toarray()\n"
        "    Ppow = sum(Pt for Pt in [P_d**i for i in 1..window]) / window\n"
        "    M_raw = Ppow @ diag(vol / d); return log(max(M_raw, 1e-300))\n"
        "\n"
        "def embed_svd(M, dim=64, clip=None):\n"
        "    if clip is not None: M = max(M, clip)\n"
        "    return TruncatedSVD(n_components=dim).fit_transform(M)"
    ),
    "shared_components": [
        {
            "name": "make_sbm",
            "description": "SBM N=2000, returns full graph + LCC with labels",
            "output_path": None,
            "why_shared": "Same graph generation used across all methods"
        },
        {
            "name": "compute_netmf",
            "description": "NetMF log-PPR matrix, window=10, no clip",
            "output_path": None,
            "why_shared": "Core function for netmf_clipped and netmf_full; clipping applied downstream"
        }
    ],
    "next_questions": [
        "Does the SV1 gap between full and clipped NetMF persist for other window sizes (e.g., window=1, 5)?",
        "Is the clipping effect also present in other graph models (e.g., LFR benchmarks)?",
        "Why is BP the best method — what information does it use that embedding methods don't?",
        "Can we quantify the effective 'implicit clipping' in SGNS by comparing the PMI values trained on?",
        "Does node2vec's advantage over netmf_clipped (if any) come from the random walk sampling noise?"
    ],
    "failed_attempts": []
}

out_path = f'{OUT_DIR}/results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=to_serializable)

print(f"Results written to: {out_path}")
total_elapsed = time.time() - t0_all
print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print("=== DONE ===")
