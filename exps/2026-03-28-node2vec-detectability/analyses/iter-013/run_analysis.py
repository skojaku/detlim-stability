"""
Iter-013: Embedding geometry analysis
Test whether node2vec dimensions are near-parallel (low effective rank),
and compare effective dimensionality across methods.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import networkx as nx
import sys
sys.path.insert(0, '/workspace/libs/embcom_repo')
import embcom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = '/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-013'

# ── SBM factory ──────────────────────────────────────────────────────────────
def make_sbm_lcc(N, mu, cave=5.0, seed=None):
    rng = np.random.default_rng(seed)
    c_out = mu * cave
    c_in  = 2 * cave - c_out
    p_in  = c_in  / N
    p_out = c_out / N
    half  = N // 2
    sizes = [half, N - half]
    labels_full = np.array([0]*sizes[0] + [1]*sizes[1])
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            p = p_in if labels_full[i] == labels_full[j] else p_out
            if rng.random() < p:
                A[i,j] = A[j,i] = 1.0
    A_sp = sp.csr_matrix(A)
    # LCC
    G = nx.from_scipy_sparse_array(A_sp)
    lcc_nodes = sorted(max(nx.connected_components(G), key=len))
    lcc_nodes = np.array(lcc_nodes)
    A_lcc = A_sp[lcc_nodes][:, lcc_nodes]
    labels_lcc = labels_full[lcc_nodes]
    return A_lcc, labels_lcc

# faster SBM using networkx stochastic_block_model
def make_sbm_lcc_fast(N, mu, cave=5.0, seed=None):
    rng = np.random.default_rng(seed)
    c_out = mu * cave
    c_in  = 2 * cave - c_out
    p_in  = c_in  / N
    p_out = c_out / N
    half  = N // 2
    sizes = [half, N - half]
    # Use networkx SBM
    pmat = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, pmat, seed=int(rng.integers(1e9)))
    A_sp = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
    labels_full = np.array([G.nodes[n]['block'] for n in G.nodes()])
    # LCC
    lcc_nodes = sorted(max(nx.connected_components(G), key=len))
    lcc_nodes = np.array(lcc_nodes)
    A_lcc = A_sp[lcc_nodes][:, lcc_nodes]
    labels_lcc = labels_full[lcc_nodes]
    return A_lcc, labels_lcc

# ── Embedding methods ─────────────────────────────────────────────────────────
def embed_node2vec(A_lcc, dim=64):
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10, p=1.0, q=1.0)
    model.fit(A_lcc)
    emb = model.transform(dim=dim)
    return emb

def compute_netmf_logpmi(A_lcc, T=10):
    d = np.array(A_lcc.sum(axis=1)).flatten()
    D_inv = sp.diags(1.0 / np.maximum(d, 1e-12))
    P = D_inv @ A_lcc

    # Multi-hop averaging: Ppow = (1/T) * sum_{t=1}^{T} P^t
    Ppow = P.copy().toarray().astype(np.float64)
    Pt = Ppow.copy()
    P_dense = P.toarray().astype(np.float64)
    for t in range(2, T+1):
        Pt = Pt @ P_dense
        Ppow += Pt
    Ppow /= T

    vol = d.sum()
    d_j = d[np.newaxis, :]
    M = np.log(np.maximum(vol * Ppow / np.maximum(d_j, 1e-12), 1e-300))
    return M

def embed_gradient_weight_svd(A_lcc, dim=64, T=10):
    M = compute_netmf_logpmi(A_lcc, T=T)
    sigmoid = 1.0 / (1.0 + np.exp(-M))
    M_gw = M * sigmoid * (1.0 - sigmoid)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    emb = svd.fit_transform(M_gw)
    return emb

def embed_clip0_svd(A_lcc, dim=64, T=10):
    M = compute_netmf_logpmi(A_lcc, T=T)
    M_clip = np.maximum(M, 0.0)
    M_sp = sp.csr_matrix(M_clip)
    svd = TruncatedSVD(n_components=dim, random_state=42)
    emb = svd.fit_transform(M_sp)
    return emb

def embed_spectral(A_lcc, dim=64):
    d = np.array(A_lcc.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.maximum(np.sqrt(d), 1e-12))
    L_sym = D_inv_sqrt @ A_lcc @ D_inv_sqrt
    k = min(dim + 2, A_lcc.shape[0] - 2)
    vals, vecs = eigsh(L_sym, k=k, which='LM')
    # Sort descending and skip trivial eigenvector (all-ones, largest eigenvalue)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    # Skip trivial (first)
    vecs = vecs[:, 1:dim+1]
    return vecs

# ── Geometry metrics ──────────────────────────────────────────────────────────
def compute_geometry(emb):
    """
    Given N×d embedding matrix, compute:
    - Gram matrix G = U_norm^T @ U_norm where U_norm is column-normalized
    - Eigenvalues of G
    - Participation ratio PR = (sum lambda)^2 / sum(lambda^2)
    - Off-diagonal |cos| statistics
    """
    # L2-normalize each column
    U = emb.copy()
    col_norms = np.linalg.norm(U, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-12)
    U_norm = U / col_norms  # N × d

    # Gram matrix: d × d
    G = U_norm.T @ U_norm  # d × d

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = np.maximum(eigvals, 0)  # numerical stability

    # Participation ratio
    sum_lam = eigvals.sum()
    sum_lam2 = (eigvals**2).sum()
    PR = (sum_lam**2) / (sum_lam2 + 1e-300)

    # Off-diagonal |cos| of embedding dimensions
    # Normalize G by diagonal: G_ij / sqrt(G_ii * G_jj) = cos(theta_ij) for dimension pairs
    d = G.shape[0]
    diag = np.diag(G)
    diag_sqrt = np.sqrt(np.maximum(diag, 1e-12))
    cos_mat = G / (diag_sqrt[:, None] * diag_sqrt[None, :])
    # Get upper triangle off-diagonal
    upper_idx = np.triu_indices(d, k=1)
    off_diag_cos = np.abs(cos_mat[upper_idx])

    mean_offdiag_cos = off_diag_cos.mean()
    frac_gt05 = (off_diag_cos > 0.5).mean()
    frac_gt09 = (off_diag_cos > 0.9).mean()

    return {
        'PR': float(PR),
        'eigvals': eigvals.tolist(),
        'mean_offdiag_cos': float(mean_offdiag_cos),
        'frac_gt05': float(frac_gt05),
        'frac_gt09': float(frac_gt09),
        'offdiag_cos_hist': off_diag_cos.tolist(),
        'Gram_diag': diag.tolist(),
    }

# ── Main experiment ───────────────────────────────────────────────────────────
def run_experiment(mu_values, n_samples=10, N=2000, dim=64):
    results = {}
    methods = ['node2vec', 'gradient_weight_svd', 'clip_0_svd', 'spectral']

    for mu in mu_values:
        print(f"\n=== mu={mu} ===")
        results[mu] = {m: [] for m in methods}

        for s in range(n_samples):
            print(f"  sample {s+1}/{n_samples}", end='', flush=True)
            A_lcc, labels = make_sbm_lcc_fast(N, mu, cave=5.0, seed=s*100 + int(mu*1000))

            for method in methods:
                print(f" [{method}]", end='', flush=True)
                try:
                    if method == 'node2vec':
                        emb = embed_node2vec(A_lcc, dim=dim)
                    elif method == 'gradient_weight_svd':
                        emb = embed_gradient_weight_svd(A_lcc, dim=dim)
                    elif method == 'clip_0_svd':
                        emb = embed_clip0_svd(A_lcc, dim=dim)
                    elif method == 'spectral':
                        emb = embed_spectral(A_lcc, dim=dim)

                    geom = compute_geometry(emb)

                    # Also compute NMI for reference
                    emb_norm = normalize(emb, norm='l2')
                    km = KMeans(n_clusters=2, n_init=10, random_state=42)
                    pred = km.fit_predict(emb_norm)
                    nmi = normalized_mutual_info_score(labels, pred)

                    results[mu][method].append({
                        'PR': geom['PR'],
                        'mean_offdiag_cos': geom['mean_offdiag_cos'],
                        'frac_gt05': geom['frac_gt05'],
                        'frac_gt09': geom['frac_gt09'],
                        'eigvals': geom['eigvals'],
                        'nmi': nmi,
                    })
                except Exception as e:
                    print(f" ERROR({e})", end='')
                    results[mu][method].append({'error': str(e)})

            print()  # newline after each sample

    return results

# ── Aggregate results ─────────────────────────────────────────────────────────
def aggregate(results, mu, method):
    runs = [r for r in results[mu][method] if 'error' not in r]
    if not runs:
        return {}
    def mean_std(key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))
    pr_mean, pr_std = mean_std('PR')
    cos_mean, cos_std = mean_std('mean_offdiag_cos')
    frac05_mean, _ = mean_std('frac_gt05')
    frac09_mean, _ = mean_std('frac_gt09')
    nmi_mean, nmi_std = mean_std('nmi')
    # Average eigenvalue spectrum
    avg_eigvals = np.mean([r['eigvals'] for r in runs], axis=0).tolist()
    return {
        'PR_mean': pr_mean, 'PR_std': pr_std,
        'cos_mean': cos_mean, 'cos_std': cos_std,
        'frac_gt05': frac05_mean,
        'frac_gt09': frac09_mean,
        'nmi_mean': nmi_mean, 'nmi_std': nmi_std,
        'avg_eigvals': avg_eigvals,
        'n_runs': len(runs),
    }

# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plots(results, mu_values):
    methods = ['node2vec', 'gradient_weight_svd', 'clip_0_svd', 'spectral']
    colors = {'node2vec': 'tab:blue', 'gradient_weight_svd': 'tab:orange',
              'clip_0_svd': 'tab:green', 'spectral': 'tab:red'}
    labels_map = {'node2vec': 'Node2Vec', 'gradient_weight_svd': 'GradWeight-SVD',
                  'clip_0_svd': 'Clip0-SVD', 'spectral': 'Spectral'}

    fig, axes = plt.subplots(2, len(mu_values), figsize=(5*len(mu_values), 9))
    if len(mu_values) == 1:
        axes = axes[:, np.newaxis]

    for col, mu in enumerate(mu_values):
        # Top row: eigenvalue decay
        ax = axes[0, col]
        for method in methods:
            agg = aggregate(results, mu, method)
            if agg:
                ev = np.array(agg['avg_eigvals'])
                ax.semilogy(range(1, len(ev)+1), ev, label=labels_map[method],
                           color=colors[method], linewidth=2)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Eigenvalue (Gram)')
        ax.set_title(f'mu={mu}: Gram eigenvalue decay')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: PR bar chart
        ax = axes[1, col]
        PRs = []
        PRstds = []
        method_labels = []
        for method in methods:
            agg = aggregate(results, mu, method)
            if agg:
                PRs.append(agg['PR_mean'])
                PRstds.append(agg['PR_std'])
                method_labels.append(labels_map[method])
        x = np.arange(len(method_labels))
        bars = ax.bar(x, PRs, yerr=PRstds, capsize=5,
                     color=[colors[m] for m in methods[:len(method_labels)]])
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Participation Ratio (PR)')
        ax.set_title(f'mu={mu}: Effective dimensionality')
        ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='PR=1 (all parallel)')
        ax.axhline(64, color='gray', linestyle=':', alpha=0.5, label='PR=64 (orthogonal)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = f'{OUTPUT_DIR}/fig_embedding_geometry.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Additional figure: off-diagonal cosine histograms for mu=0.40
    mu_target = 0.40
    if mu_target in mu_values:
        fig2, axes2 = plt.subplots(1, len(methods), figsize=(5*len(methods), 4))
        for ax, method in zip(axes2, methods):
            runs = [r for r in results[mu_target][method] if 'error' not in r]
            if runs:
                # Plot all runs' off-diag cosines pooled... but we didn't store histogram
                # Use mean_offdiag_cos as proxy, show scatter
                vals = [r['mean_offdiag_cos'] for r in runs]
                ax.hist(vals, bins=10, color=colors[method], alpha=0.7)
                ax.set_title(f'{labels_map[method]}\nMean|cos|={np.mean(vals):.3f}')
                ax.set_xlabel('Mean off-diagonal |cos|')
                ax.set_ylabel('Count')
        plt.suptitle(f'Distribution of mean off-diagonal cosine similarity (mu={mu_target})')
        plt.tight_layout()
        fig2_path = f'{OUTPUT_DIR}/fig_cos_similarity_dist.png'
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig2_path}")

    return [fig_path, f'{OUTPUT_DIR}/fig_cos_similarity_dist.png']


if __name__ == '__main__':
    mu_values = [0.30, 0.40]  # start with these two, add 0.50 if time permits
    n_samples = 10
    N = 2000
    dim = 64

    print("Running embedding geometry analysis...")
    print(f"N={N}, dim={dim}, samples={n_samples}, mu_values={mu_values}")
    results = run_experiment(mu_values, n_samples=n_samples, N=N, dim=dim)

    # Aggregate
    agg_results = {}
    for mu in mu_values:
        agg_results[str(mu)] = {}
        for method in ['node2vec', 'gradient_weight_svd', 'clip_0_svd', 'spectral']:
            agg_results[str(mu)][method] = aggregate(results, mu, method)

    # Print summary
    print("\n=== SUMMARY ===")
    for mu in mu_values:
        print(f"\nmu={mu}:")
        for method in ['node2vec', 'gradient_weight_svd', 'clip_0_svd', 'spectral']:
            agg = agg_results[str(mu)][method]
            if agg:
                print(f"  {method}: PR={agg['PR_mean']:.2f}±{agg['PR_std']:.2f}, "
                      f"mean_cos={agg['cos_mean']:.4f}±{agg['cos_std']:.4f}, "
                      f"frac>0.5={agg['frac_gt05']:.4f}, "
                      f"NMI={agg['nmi_mean']:.3f}±{agg['nmi_std']:.3f}")

    # Make plots
    figs = make_plots(results, mu_values)

    # Build key numbers
    key_numbers = {}
    for mu in mu_values:
        mu_str = str(mu).replace('.', '')
        for method in ['node2vec', 'gradient_weight_svd', 'clip_0_svd', 'spectral']:
            agg = agg_results[str(mu)].get(method, {})
            if agg:
                short = method.replace('_', '_').replace('gradient_weight_svd', 'gw')
                short = {'node2vec': 'n2v', 'gradient_weight_svd': 'gw',
                         'clip_0_svd': 'clip0', 'spectral': 'spec'}[method]
                key_numbers[f'PR_{short}_mu{mu_str}'] = agg['PR_mean']
                key_numbers[f'cos_{short}_mu{mu_str}'] = agg['cos_mean']
                key_numbers[f'nmi_{short}_mu{mu_str}'] = agg['nmi_mean']

    # Find the result summary
    mu040_n2v = agg_results.get('0.4', {}).get('node2vec', {})
    mu040_clip = agg_results.get('0.4', {}).get('clip_0_svd', {})
    mu040_spec = agg_results.get('0.4', {}).get('spectral', {})
    n2v_PR = mu040_n2v.get('PR_mean', 'N/A')
    clip_PR = mu040_clip.get('PR_mean', 'N/A')
    spec_PR = mu040_spec.get('PR_mean', 'N/A')

    result_summary = (
        f"At mu=0.40: Node2Vec PR={n2v_PR:.2f}, Clip0-SVD PR={clip_PR:.2f}, Spectral PR={spec_PR:.2f}. "
        f"{'Node2Vec has lower PR (more near-parallel dimensions)' if isinstance(n2v_PR, float) and isinstance(clip_PR, float) and n2v_PR < clip_PR else 'Hypothesis not confirmed — PR differences were different than expected'}; "
        f"node2vec NMI={mu040_n2v.get('nmi_mean', 0):.3f} vs spectral NMI={mu040_spec.get('nmi_mean', 0):.3f}."
    )

    out = {
        "task": "Embedding geometry analysis: test whether node2vec dimensions are near-parallel (low effective rank PR) compared to SVD-based methods",
        "method": "Participation ratio PR = (Σλ)²/Σλ² of Gram matrix; off-diagonal cosine similarity; N=2000 SBM, 10 samples, dim=64",
        "result_summary": result_summary,
        "key_numbers": key_numbers,
        "aggregated_results": agg_results,
        "figures_created": figs,
        "code_used": "see run_analysis.py in iter-013/",
        "shared_components": [
            {
                "name": "make_sbm_lcc_fast",
                "description": "Fast SBM via networkx stochastic_block_model; corrected parameterization c_out=mu*cave, c_in=2*cave-c_out"
            },
            {
                "name": "compute_geometry",
                "description": "Compute Gram matrix, participation ratio, and off-diagonal cosine statistics from an N×d embedding"
            }
        ],
        "next_questions": [
            "If node2vec PR is lower: does the near-parallel geometry arise from the SGNS gradient weighting or the multi-hop walk structure?",
            "Does gradient_weight_svd also show low PR (paralleling its NMI performance)?",
            "Is the low PR a property of the learned representation or a property of the community structure?",
            "Can we artificially rotate node2vec embeddings to be orthogonal without changing NMI — testing whether near-parallelism is causal?",
        ],
        "failed_attempts": []
    }

    out_path = f'{OUTPUT_DIR}/results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")
