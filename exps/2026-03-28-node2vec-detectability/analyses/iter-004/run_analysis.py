import sys, warnings, time, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, '/workspace/libs/BeliefPropagation')
sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
import embcom
from embcom import utils as embcom_utils
warnings.filterwarnings('ignore')

# ─── Parameters ───────────────────────────────────────────────────────────────
N = 2000
cave = 5.0
dim = 64
n_samples = 10
mu_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.52]

# ─── SBM generation ───────────────────────────────────────────────────────────
def gen_sbm(N, cave, mu, seed):
    rng = np.random.default_rng(seed)
    c_out = mu * cave
    c_in = 2 * cave - c_out
    p_in = c_in / N
    p_out = c_out / N
    labels = np.array([0] * (N // 2) + [1] * (N // 2))
    rows, cols = [], []
    for i in range(N):
        for j in range(i+1, N):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, labels

def get_lcc(A, labels):
    G = nx.from_scipy_sparse_array(A)
    lcc_nodes = sorted(max(nx.connected_components(G), key=len))
    lcc_nodes = sorted(lcc_nodes)
    A_lcc = A[np.ix_(lcc_nodes, lcc_nodes)]
    labels_lcc = labels[lcc_nodes]
    return A_lcc, labels_lcc, lcc_nodes

# ─── Embeddings ───────────────────────────────────────────────────────────────
def embed_node2vec(A_lcc, dim):
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10, p=1.0, q=1.0)
    model.fit(A_lcc)
    return model.transform(dim=dim)

def embed_n2vec_mf(A_lcc, dim):
    model = embcom.Node2VecMatrixFactorization(window_length=10, num_blocks=500)
    model.fit(A_lcc)
    return model.transform(dim=dim)

def embed_spectral(A_lcc, dim):
    d = np.array(A_lcc.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.maximum(np.sqrt(d), 1e-12))
    M = d_inv_sqrt @ A_lcc @ d_inv_sqrt
    k = min(dim + 1, M.shape[0] - 1)
    vals, vecs = eigsh(M, k=k, which='LM')
    # sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]; vecs = vecs[:, idx]
    # skip trivial (largest) eigenvector
    emb = vecs[:, 1:dim+1]
    return emb

def compute_netmf_matrix(A_lcc, T=10):
    P = embcom_utils.to_trans_mat(A_lcc)
    P_dense = P.toarray().astype(np.float64)
    Ppow = np.zeros_like(P_dense)
    Pt = np.eye(P_dense.shape[0])
    for _ in range(T):
        Pt = Pt @ P_dense
        Ppow += Pt
    Ppow /= T
    d = np.array(A_lcc.sum(axis=1)).flatten()
    vol = d.sum()
    M = Ppow @ np.diag(vol / np.maximum(d, 1e-12))
    M_netmf = np.log(np.maximum(M, 1e-12))
    return M_netmf

def embed_free_netmf(A_lcc, dim, steps=2000, lr=0.01, seed=0):
    import torch
    M_netmf = compute_netmf_matrix(A_lcc, T=10)
    torch.manual_seed(seed)
    N = M_netmf.shape[0]
    U = torch.randn(N, dim, requires_grad=True)
    V = torch.randn(N, dim, requires_grad=True)
    M_t = torch.tensor(M_netmf, dtype=torch.float32)
    opt = torch.optim.Adam([U, V], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = ((U @ V.T - M_t) ** 2).mean()
        loss.backward()
        opt.step()
    return U.detach().numpy()

def cluster_and_nmi(emb, labels, k=2, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    pred = km.fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)

# ─── Main loop ────────────────────────────────────────────────────────────────
results = {
    "node2vec": {mu: [] for mu in mu_values},
    "n2vec_mf":  {mu: [] for mu in mu_values},
    "free_netmf":{mu: [] for mu in mu_values},
    "spectral":  {mu: [] for mu in mu_values},
}

t0 = time.time()
for mu in mu_values:
    print(f"\n=== mu={mu} ===")
    for s in range(n_samples):
        seed = s * 1000 + int(mu * 1000)
        A, labels = gen_sbm(N, cave, mu, seed)
        A_lcc, labels_lcc, _ = get_lcc(A, labels)
        print(f"  sample {s}: LCC size={A_lcc.shape[0]}")

        # spectral (fast, always run)
        try:
            emb = embed_spectral(A_lcc, dim)
            nmi = cluster_and_nmi(emb, labels_lcc)
            results["spectral"][mu].append(nmi)
        except Exception as e:
            print(f"  spectral failed: {e}")
            results["spectral"][mu].append(np.nan)

        # n2vec_mf
        try:
            emb = embed_n2vec_mf(A_lcc, dim)
            nmi = cluster_and_nmi(emb, labels_lcc)
            results["n2vec_mf"][mu].append(nmi)
        except Exception as e:
            print(f"  n2vec_mf failed: {e}")
            results["n2vec_mf"][mu].append(np.nan)

        # free_netmf
        try:
            emb = embed_free_netmf(A_lcc, dim)
            nmi = cluster_and_nmi(emb, labels_lcc)
            results["free_netmf"][mu].append(nmi)
        except Exception as e:
            print(f"  free_netmf failed: {e}")
            results["free_netmf"][mu].append(np.nan)

        # node2vec
        try:
            emb = embed_node2vec(A_lcc, dim)
            nmi = cluster_and_nmi(emb, labels_lcc)
            results["node2vec"][mu].append(nmi)
        except Exception as e:
            print(f"  node2vec failed: {e}")
            results["node2vec"][mu].append(np.nan)

        elapsed = time.time() - t0
        print(f"  -> spectral={results['spectral'][mu][-1]:.3f}, "
              f"n2vec_mf={results['n2vec_mf'][mu][-1]:.3f}, "
              f"free_netmf={results['free_netmf'][mu][-1]:.3f}, "
              f"node2vec={results['node2vec'][mu][-1]:.3f}  "
              f"[{elapsed:.0f}s elapsed]")

# ─── Summarize ────────────────────────────────────────────────────────────────
summary = {}
for method in results:
    summary[method] = {}
    for mu in mu_values:
        vals = [v for v in results[method][mu] if not np.isnan(v)]
        summary[method][mu] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std":  float(np.std(vals))  if vals else None,
            "n":    len(vals),
        }

# Gap: node2vec - n2vec_mf
gaps = {}
for mu in mu_values:
    n2v = summary["node2vec"][mu]["mean"]
    mf  = summary["n2vec_mf"][mu]["mean"]
    if n2v is not None and mf is not None:
        gaps[mu] = round(n2v - mf, 4)
    else:
        gaps[mu] = None

# free_netmf vs n2vec_mf
free_vs_svd = {}
for mu in mu_values:
    free = summary["free_netmf"][mu]["mean"]
    mf   = summary["n2vec_mf"][mu]["mean"]
    if free is not None and mf is not None:
        free_vs_svd[mu] = round(free - mf, 4)
    else:
        free_vs_svd[mu] = None

# Collect key numbers
key_numbers = {}
for method in summary:
    for mu in mu_values:
        key_numbers[f"{method}_nmi_mu{mu}"] = summary[method][mu]["mean"]
        key_numbers[f"{method}_std_mu{mu}"]  = summary[method][mu]["std"]
for mu in mu_values:
    key_numbers[f"gap_n2v_minus_mf_mu{mu}"] = gaps[mu]
    key_numbers[f"gap_free_minus_svd_mu{mu}"] = free_vs_svd[mu]

total_time = time.time() - t0

print("\n\n=== SUMMARY ===")
print(f"Total time: {total_time:.0f}s")
for mu in mu_values:
    print(f"mu={mu}: node2vec={summary['node2vec'][mu]['mean']:.3f}±{summary['node2vec'][mu]['std']:.3f}, "
          f"n2vec_mf={summary['n2vec_mf'][mu]['mean']:.3f}±{summary['n2vec_mf'][mu]['std']:.3f}, "
          f"free_netmf={summary['free_netmf'][mu]['mean']:.3f}±{summary['free_netmf'][mu]['std']:.3f}, "
          f"spectral={summary['spectral'][mu]['mean']:.3f}±{summary['spectral'][mu]['std']:.3f}, "
          f"gap={gaps[mu]:.3f}")

# Determine if node2vec outperforms n2vec_mf (H1 evidence)
avg_gap = np.nanmean([gaps[mu] for mu in mu_values if gaps[mu] is not None])
h1_supported = bool(avg_gap > 0.01)

# Determine if free_netmf > n2vec_mf
avg_free_vs_svd = np.nanmean([free_vs_svd[mu] for mu in mu_values if free_vs_svd[mu] is not None])
free_beats_svd = bool(avg_free_vs_svd > 0.01)

out = {
    "task": "Direct node2vec vs n2vec_mf comparison at matched N=2000, same networks. Also test free_netmf (unconstrained Adam) vs both.",
    "method": "SBM N=2000, cave=5, mu sweep [0.3-0.52], 10 samples/mu. Methods: node2vec (walk+SGNS), n2vec_mf (SVD NetMF), free_netmf (Adam NetMF), spectral (eigsh).",
    "result_summary": f"node2vec mean NMI gap over n2vec_mf: {avg_gap:.4f} (H1={'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}). free_netmf vs n2vec_mf: {avg_free_vs_svd:.4f} ({'free beats SVD' if free_beats_svd else 'SVD wins or tie'}).",
    "key_numbers": key_numbers,
    "gaps_n2v_minus_mf": {str(k): v for k, v in gaps.items()},
    "gaps_free_minus_svd": {str(k): v for k, v in free_vs_svd.items()},
    "per_method_summary": {
        method: {str(mu): {"mean": summary[method][mu]["mean"], "std": summary[method][mu]["std"], "n": summary[method][mu]["n"]}
                 for mu in mu_values}
        for method in summary
    },
    "H1_node2vec_walk_dynamics_real": h1_supported,
    "free_netmf_beats_svd": free_beats_svd,
    "total_time_seconds": round(total_time, 1),
    "figures_created": [],
    "code_used": "See run_analysis.py in analyses/iter-004/",
    "shared_components": [
        {
            "name": "SBM generator",
            "description": "gen_sbm(N=2000, cave=5, mu, seed) with corrected parameterization",
            "output_path": None,
            "why_shared": "Same SBM instances used by all 4 methods via same seed"
        }
    ],
    "next_questions": [
        "If H1 is supported (walk dynamics matter), why does SGNS outperform SVD of the same target — is it the noise injection, negative sampling, or implicit regularization?",
        "Does free_netmf (Adam, no ortho) bridge the gap, or does it underperform SVD — testing whether orthogonality constraint in SVD helps or hurts?",
        "If free_netmf > n2vec_mf, the orthogonality constraint of SVD is the bottleneck, not the factorization method per se."
    ],
    "failed_attempts": []
}

out_path = "/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-004/results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults written to {out_path}")
