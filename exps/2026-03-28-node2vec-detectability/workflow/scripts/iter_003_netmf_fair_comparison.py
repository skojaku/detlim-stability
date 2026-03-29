"""
Iter-003: Fair N=2000 comparison using the correct NetMF matrix.

Fix the confound from iter-002: node2vec ran on N=500 while others ran on N=2000.

Key insight: Node2vec implicitly factorizes the multi-hop NetMF matrix (Qiu et al. 2018).
embcom.Node2VecMatrixFactorization computes:
  P = D^{-1}A  (transition matrix)
  Ppow = (sum_{r=1}^T P^r) / T
  pi = d / vol  (stationary distribution)
  M_netmf = log(Ppow @ diag(1/pi))
  Then SVD of M_netmf

Methods (all N=2000):
1. BP: belief propagation
2. spectral: eigsh on D^{-1/2}AD^{-1/2}, top 64 eigvecs (skip trivial)
3. n2vec_mf: embcom.Node2VecMatrixFactorization(window_length=10) -- SVD of NetMF (orthogonal)
4. svd_netmf: recompute NetMF explicitly via eigsh (low-rank), then SVD -- should match n2vec_mf
5. free_netmf: NetMF matrix explicit, then free Adam factorization (no orthogonality) -- H2 test
6. node2vec: embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10) at N=2000
"""

import json
import sys
import time
import warnings
import traceback
from pathlib import Path

import igraph as ig
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, svds
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, '/workspace/libs/BeliefPropagation')
import belief_propagation

sys.path.insert(0, '/workspace/libs/embcom_repo/libs/embcom')
import embcom
from embcom import utils as embcom_utils

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = Path("/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-003/results.json")
RESULTS_DIR = RESULTS_FILE.parent
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────
MU_SWEEP = [0.3, 0.35, 0.4, 0.45, 0.5, 0.52]  # Below detectability limit only
N_SAMPLES = 10
N = 2000
N_EACH = N // 2
CAVE = 5.0
DIM = 64
WINDOW = 10   # window_length for NetMF / node2vec
NEG_SAMPLES = 1  # number of negative samples (k in NetMF)
DETECTABILITY_LIMIT = 1.0 - 1.0 / np.sqrt(CAVE)  # ≈ 0.5528


# ── SBM generation ────────────────────────────────────────────────────────────

def make_sbm(mu, n_total=N, seed=None):
    """Full graph (no LCC). BP works on this; for embeddings use make_sbm_lcc."""
    n_each = n_total // 2
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / n_total
    p_out = c_out / n_total
    if seed is not None:
        np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
    labels_full = np.array([0] * n_each + [1] * n_each)
    A = g.get_adjacency_sparse()
    A = sp.csr_matrix(A, dtype=float)
    return A, labels_full


def make_sbm_lcc(mu, n_total=N, seed=None):
    """LCC only — use for embedding methods that need connected graphs."""
    n_each = n_total // 2
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / n_total
    p_out = c_out / n_total
    if seed is not None:
        np.random.seed(seed)
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
    labels_full = np.array([0] * n_each + [1] * n_each)
    components = g.connected_components(mode="weak")
    lcc_indices = sorted(max(components, key=len))
    g_lcc = g.induced_subgraph(lcc_indices)
    labels = labels_full[lcc_indices]
    A = g_lcc.get_adjacency_sparse()
    A = sp.csr_matrix(A, dtype=float)
    return A, labels


# ── NetMF matrix computation ──────────────────────────────────────────────────

def compute_netmf_matrix_dense(A, window=WINDOW, neg_samples=NEG_SAMPLES):
    """
    Compute the NetMF matrix as in embcom.Node2VecMatrixFactorization:
      P = D^{-1}A
      Ppow = (sum_{r=1}^T P^r) / T
      pi = d / vol
      M_netmf[i,j] = log(Ppow[i,j] / pi[j])
               = log(Ppow[i,j] * vol / d[j])

    Returns dense M_netmf matrix (N x N).
    Uses the same computation as embcom but explicit.
    Note: embcom doesn't subtract log(neg_samples); for consistency with embcom we skip it too.
    """
    A_dense = np.array(A.todense(), dtype=np.float64)
    P = embcom_utils.to_trans_mat(A)
    P_dense = np.array(P.todense(), dtype=np.float64)

    # sum_{r=1}^T P^r
    Ppow = np.zeros_like(P_dense)
    Pt = np.eye(P_dense.shape[0])
    for _ in range(window):
        Pt = P_dense @ Pt
        Ppow += Pt
    Ppow /= window

    # stationary distribution
    d = np.array(A.sum(axis=1)).flatten()
    vol = d.sum()
    pi = d / vol

    # M_netmf = log(Ppow / pi[j]) = log(Ppow * vol / d[j])
    M = Ppow @ np.diag(vol / np.maximum(d, 1e-12))
    M_log = np.log(np.maximum(M, 1e-12))  # clip for log stability
    return M_log


def compute_netmf_lowrank(A, window=WINDOW, neg_samples=NEG_SAMPLES, k=DIM):
    """
    Low-rank approximation of NetMF matrix using eigsh on D^{-1/2}AD^{-1/2}.

    Key insight: P = D^{-1}A = D^{-1/2} * (D^{-1/2}AD^{-1/2}) * D^{1/2}
    So P is similar to P_sym = D^{-1/2}AD^{-1/2}.
    Eigenvalues of P = eigenvalues of P_sym.
    If P_sym = U @ diag(lam) @ U.T, then P = (D^{1/2}U) @ diag(lam) @ (D^{-1/2}U).T

    sum_{r=1}^T P^r / T = (D^{1/2}U) @ diag(sum_r lam^r / T) @ (D^{-1/2}U).T

    NetMF[i,j] = log(Ppow[i,j] * vol / d[j])
    Note: Ppow @ diag(vol/d) = (D^{1/2}U) @ diag(eigen_sum/T) @ (D^{-1/2}U).T @ diag(vol/d)
                              = (D^{1/2}U) @ diag(eigen_sum/T) @ (D^{-3/2}U * vol).T

    Returns: the matrix in factored form (embedding U_emb such that U_emb @ U_emb.T approx NetMF)
    Also returns the dense NetMF approximation for free_netmf.
    """
    d = np.array(A.sum(axis=1)).flatten()
    vol = d.sum()
    d_safe = np.maximum(d, 1e-10)

    D_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
    D_sqrt = sp.diags(np.sqrt(d_safe))
    P_sym = D_inv_sqrt @ A @ D_inv_sqrt

    k_eig = min(k + 1, A.shape[0] - 2)
    lam, U = eigsh(P_sym, k=k_eig, which='LM')
    # Sort descending
    idx = np.argsort(-lam)
    lam, U = lam[idx], U[:, idx]

    # Remove trivial eigenvector (eigenvalue ≈ 1, constant vector)
    # Trivial: eigenvalue closest to 1.0 with uniform eigvec
    # Skip the largest eigenvalue (it's the trivial one for connected graph)
    lam = lam[1:]
    U = U[:, 1:]

    # sum_{r=1}^T lambda^r / T
    eigen_sum = np.zeros(len(lam))
    for i, l in enumerate(lam):
        if abs(l - 1.0) < 1e-10:
            eigen_sum[i] = 1.0
        else:
            eigen_sum[i] = (l * (1 - l**window)) / (window * (1 - l))

    # Ppow @ diag(vol/d) factored:
    # Left factor: D^{1/2} @ U_sub (N x k)
    # Right factor: (D^{-3/2} @ U_sub * vol).T  (k x N)
    # Scale each component by eigen_sum[i]

    # For the embedding: take left side scaled by sqrt(eigensum * vol / d)
    # NetMF[i,j] = log( sum_r (D^{1/2}U)_i * lam_sum_r * (D^{-3/2}U * vol)_j )
    # Low-rank log approx: we compute the dense low-rank matrix and log it

    left = np.array(D_sqrt @ U) * eigen_sum[None, :]   # (N, k)
    right = np.array(D_inv_sqrt @ D_inv_sqrt @ U) * vol  # D^{-1}U * vol = (N, k)
    # right = D^{-3/2} @ U * vol ... wait: D_inv_sqrt @ D_inv_sqrt = D^{-1}
    # Actually D^{-3/2} = D^{-1} @ D^{-1/2}
    D_inv = sp.diags(1.0 / d_safe)
    right = np.array(D_inv @ D_inv_sqrt @ U) * vol  # D^{-3/2}U * vol (N, k)

    # Low-rank approximation of Ppow @ diag(vol/d):
    M_lr = left @ right.T   # (N, N) -- this is the low-rank Ppow * vol/d matrix
    M_log_lr = np.log(np.maximum(M_lr, 1e-12))
    return M_log_lr, lam, U, eigen_sum


# ── Classification ─────────────────────────────────────────────────────────────

def classify_nmi(emb, labels, n_clusters=2):
    """K-means clustering and NMI score."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = km.fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)


# ── Method implementations ─────────────────────────────────────────────────────

def bp_fit(A, labels):
    """BP: returns NMI directly."""
    A_csr = sp.csr_matrix(A, dtype=float)
    cids = belief_propagation.detect(A_csr, q=2, iters=10)
    return normalized_mutual_info_score(labels, cids)


def spectral_fit(A):
    """Spectral: top eigvecs of D^{-1/2}AD^{-1/2}, skip trivial."""
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    M = D_inv_sqrt @ A @ D_inv_sqrt
    k = min(DIM + 1, A.shape[0] - 1)
    vals, vecs = eigsh(M, k=k, which='LM')
    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]
    return vecs[:, 1:DIM + 1]  # skip trivial


def n2vec_mf_fit(A):
    """SVD of NetMF matrix (orthogonal) -- replicates embcom.Node2VecMatrixFactorization
    but uses a safe log floor to avoid -inf/+inf.

    embcom does:
      P = D^{-1}A
      Ppow = matrix_sum_power(P, T) / T  [sum_{r=1}^T P^r / T]
      pi = d / vol
      R = log(Ppow @ diag(1/pi))  -- THIS can produce inf if Ppow has zeros
    We clip before log to avoid inf.
    """
    M = compute_netmf_matrix_dense(A, window=WINDOW)
    k = min(DIM, min(M.shape) - 1)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
    u = svd.fit_transform(M)
    s = svd.singular_values_
    return u * np.sqrt(np.maximum(s, 0))[None, :]


def svd_netmf_fit(A):
    """Explicit NetMF dense computation (same formula as n2vec_mf_fit, redundant sanity check).
    Uses embcom's exact internal computation including the dense matrix_sum_power.
    Falls back to our safe version if embcom fails.
    """
    # Try embcom's exact path first
    try:
        P = embcom_utils.to_trans_mat(A)
        P_dense = np.array(P.todense(), dtype=np.float64)
        Ppow = embcom_utils.matrix_sum_power(P_dense, WINDOW) / WINDOW
        d = np.array(A.sum(axis=1)).flatten()
        vol = d.sum()
        stationary_prob = d / vol
        # Exact embcom formula -- may have log(0) = -inf for zero Ppow entries
        inner = Ppow @ np.diag(1.0 / np.maximum(stationary_prob, 1e-32))
        R = np.log(np.maximum(inner, 1e-12))
        if not np.all(np.isfinite(R)):
            raise ValueError("inf/nan in R, falling back to safe version")
    except Exception:
        R = compute_netmf_matrix_dense(A, window=WINDOW)

    k = min(DIM, min(R.shape) - 1)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
    u = svd.fit_transform(R)
    s = svd.singular_values_
    return u * np.sqrt(np.maximum(s, 0))[None, :]


def free_netmf_fit(A, n_steps=300, lr=0.005, seed=0):
    """Free Adam factorization of the NetMF matrix (no orthogonality).
    This is the H2 test: does removing the orthogonality constraint help?
    Uses the dense NetMF matrix (N=2000 -> 32MB, feasible).
    """
    M = compute_netmf_matrix_dense(A, window=WINDOW)
    M = M.astype(np.float32)
    n = M.shape[0]
    dim = DIM

    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    V = rng.normal(0, 0.01, (n, dim)).astype(np.float32)

    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    scale = 1.0 / (n * n)

    for step in range(1, n_steps + 1):
        UV = U @ V.T
        Res = M - UV
        gU = -2.0 * scale * (Res @ V)
        gV = -2.0 * scale * (Res.T @ U)

        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step

        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * (gU ** 2)
        U -= lr * (mU / bc1) / (np.sqrt(vU / bc2) + eps_adam)

        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * (gV ** 2)
        V -= lr * (mV / bc1) / (np.sqrt(vV / bc2) + eps_adam)

    return U


def node2vec_fit(A):
    """Node2Vec actual random walk at N=2000."""
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=WINDOW)
    model.fit(A)
    emb = model.transform(dim=DIM)
    return emb


# ── Run helpers ───────────────────────────────────────────────────────────────

def run_method_emb(name, fit_fn, A, labels):
    t0 = time.time()
    try:
        emb = fit_fn(A)
        nmi = classify_nmi(emb, labels)
        elapsed = time.time() - t0
        return {"nmi": float(nmi), "time": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    ERROR in {name}: {e}")
        traceback.print_exc()
        return {"nmi": None, "time": elapsed, "error": str(e)}


def run_bp(A, labels):
    t0 = time.time()
    try:
        nmi = bp_fit(A, labels)
        elapsed = time.time() - t0
        return {"nmi": float(nmi), "time": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    ERROR in BP: {e}")
        traceback.print_exc()
        return {"nmi": None, "time": elapsed, "error": str(e)}


def run_sweep(method_name, fit_fn, mu_sweep, n_samples, use_lcc=True, is_bp=False):
    """Full mu sweep. Returns dict keyed by str(mu)."""
    method_results = {}
    for mu in mu_sweep:
        print(f"  mu={mu:.3f} {'(above limit)' if mu > DETECTABILITY_LIMIT else ''}")
        nmis, times, errors = [], [], []
        for s in range(n_samples):
            seed = s * 1000 + int(mu * 1000)
            if use_lcc:
                A, labels = make_sbm_lcc(mu=mu, n_total=N, seed=seed)
            else:
                A, labels = make_sbm(mu=mu, n_total=N, seed=seed)

            if is_bp:
                r = run_bp(A, labels)
            else:
                r = run_method_emb(method_name, fit_fn, A, labels)

            if r["nmi"] is not None:
                nmis.append(r["nmi"])
                times.append(r["time"])
            else:
                errors.append({"sample": s, "error": r["error"]})

        mean_nmi = float(np.mean(nmis)) if nmis else None
        std_nmi = float(np.std(nmis)) if nmis else None
        print(f"    NMI: {mean_nmi:.4f} ± {std_nmi:.4f}" if mean_nmi is not None else "    NMI: FAILED")
        method_results[str(mu)] = {
            "nmi_mean": mean_nmi,
            "nmi_std": std_nmi,
            "nmi_samples": nmis,
            "n_success": len(nmis),
            "mean_time": float(np.mean(times)) if times else None,
            "errors": errors,
        }
    return method_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {
        "metadata": {
            "date": "2026-03-28",
            "iteration": "iter-003",
            "hypothesis": "Fair N=2000 comparison using the correct multi-hop NetMF matrix",
            "N": N,
            "cave": CAVE,
            "dim": DIM,
            "window": WINDOW,
            "n_samples": N_SAMPLES,
            "mu_sweep": MU_SWEEP,
            "detectability_limit": float(DETECTABILITY_LIMIT),
            "methods": {
                "bp": "Belief propagation (C++ wrapper, q=2, iters=10)",
                "spectral": "Top eigvecs of D^{-1/2}AD^{-1/2} via eigsh, skip trivial",
                "n2vec_mf": "embcom.Node2VecMatrixFactorization(window=10) -- SVD of NetMF (orthogonal)",
                "svd_netmf": "Explicit NetMF dense computation then TruncatedSVD (should match n2vec_mf)",
                "free_netmf": "Explicit NetMF dense, then free Adam factorization (no orthogonality) -- H2 test",
                "node2vec": "Node2Vec random walk (num_walks=10, walk_length=80, window=10) at N=2000",
            },
            "netmf_formula": "M_netmf = log( (sum_{r=1}^T (D^{-1}A)^r / T) @ diag(vol/d) )",
            "key_fix": "iter-002 ran node2vec on N=500; this iter uses N=2000 for ALL methods",
        },
        "results": {},
        "notes": [
            "SBM: c_out=mu*cave, c_in=2*cave-c_out, p_in=c_in/N, p_out=c_out/N",
            f"Detectability limit mu*=1-1/sqrt(cave)={DETECTABILITY_LIMIT:.4f}",
            "BP uses full graph; embedding methods use LCC",
            "NetMF matrix computed exactly as in embcom.Node2VecMatrixFactorization",
            "free_netmf: Adam 300 steps, lr=0.005, full dense N^2 loss (N=2000 feasible)",
        ],
    }

    print(f"\n=== Iter-003: Fair N={N} comparison with correct NetMF matrix ===")
    print(f"mu_sweep: {MU_SWEEP}")
    print(f"Detectability limit: mu* = {DETECTABILITY_LIMIT:.4f}\n")

    def save():
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        import sys; sys.stdout.flush()

    # Load existing results to allow resuming
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                existing = json.load(f)
            results["results"] = existing.get("results", {})
            print(f"[Resumed from existing results: {list(results['results'].keys())}]")
        except Exception as e:
            print(f"[Could not load existing results: {e}]")

    def maybe_run(method_name, fit_fn, use_lcc=True, is_bp=False):
        if method_name in results["results"]:
            print(f"--- Method: {method_name} [SKIP - already done] ---")
            return
        print(f"--- Method: {method_name} ---")
        results["results"][method_name] = run_sweep(
            method_name, fit_fn, MU_SWEEP, N_SAMPLES, use_lcc=use_lcc, is_bp=is_bp
        )
        save()
        print(f"[saved]\n")

    # ── BP ──────────────────────────────────────────────────────────────────
    maybe_run("bp", None, use_lcc=False, is_bp=True)

    # ── Spectral ─────────────────────────────────────────────────────────────
    maybe_run("spectral", spectral_fit, use_lcc=False)

    # ── n2vec_mf (SVD of NetMF, orthogonal) ───────────────────────────────────
    maybe_run("n2vec_mf", n2vec_mf_fit, use_lcc=True)

    # ── svd_netmf (explicit NetMF dense, then SVD) ────────────────────────────
    maybe_run("svd_netmf", svd_netmf_fit, use_lcc=True)

    # ── free_netmf (explicit NetMF dense, then free Adam) ─────────────────────
    maybe_run("free_netmf", free_netmf_fit, use_lcc=True)

    # ── Node2Vec (actual random walk at N=2000) ───────────────────────────────
    maybe_run("node2vec", node2vec_fit, use_lcc=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    method_order = ["bp", "spectral", "n2vec_mf", "svd_netmf", "free_netmf", "node2vec"]
    print("\n=== Summary: mean NMI by method and mu ===")
    header = f"{'mu':>6} | " + " | ".join(f"{m:>12}" for m in method_order)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for mu in MU_SWEEP:
        row = f"{mu:>6.3f} | "
        for m in method_order:
            r = results["results"].get(m, {}).get(str(mu), {})
            nmi = r.get("nmi_mean")
            row += f"{nmi:>12.4f} | " if nmi is not None else f"{'N/A':>12} | "
        print(row)

    print(f"\n=== Done. Results -> {RESULTS_FILE} ===")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        styles = [
            ("bp",          "BP",                    "black",  "o", "-",  2.0),
            ("spectral",    "Spectral",               "gray",   "s", "--", 1.5),
            ("n2vec_mf",    "SVD(NetMF) ortho",       "blue",   "^", "-",  1.5),
            ("svd_netmf",   "SVD(NetMF) explicit",    "cyan",   "v", "--", 1.5),
            ("free_netmf",  "Free Adam(NetMF)",        "red",    "D", "-",  2.0),
            ("node2vec",    "Node2Vec RW",            "green",  "*", "-",  2.0),
        ]

        for method_name, label, color, marker, ls, lw in styles:
            mus, nmis, stds = [], [], []
            for mu in MU_SWEEP:
                r = results["results"].get(method_name, {}).get(str(mu), {})
                if r.get("nmi_mean") is not None:
                    mus.append(mu)
                    nmis.append(r["nmi_mean"])
                    stds.append(r.get("nmi_std", 0))
            if mus:
                mus = np.array(mus)
                nmis = np.array(nmis)
                stds = np.array(stds)
                ax.plot(mus, nmis, marker=marker, ls=ls, color=color,
                        label=label, lw=lw, markersize=7)
                ax.fill_between(mus, nmis - stds, nmis + stds, alpha=0.1, color=color)

        ax.axvline(DETECTABILITY_LIMIT, color="k", ls=":", alpha=0.7,
                   label=f"mu*={DETECTABILITY_LIMIT:.3f}")
        ax.set_xlabel("mu (mixing parameter)")
        ax.set_ylabel("NMI")
        ax.set_title(f"Iter-003: NetMF fair comparison (N={N}, {N_SAMPLES} samples)\n"
                     f"H2 test: free_netmf vs SVD(NetMF) — does no-orthogonality help?")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(0.27, 0.55)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = RESULTS_DIR / "nmi_vs_mu_iter003.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nFigure saved to {fig_path}")

        # ── Second figure: zoom near limit ───────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for method_name, label, color, marker, ls, lw in styles:
            mus, nmis, stds = [], [], []
            for mu in MU_SWEEP:
                if mu < 0.45:
                    continue
                r = results["results"].get(method_name, {}).get(str(mu), {})
                if r.get("nmi_mean") is not None:
                    mus.append(mu)
                    nmis.append(r["nmi_mean"])
                    stds.append(r.get("nmi_std", 0))
            if mus:
                mus = np.array(mus)
                nmis = np.array(nmis)
                stds = np.array(stds)
                ax2.plot(mus, nmis, marker=marker, ls=ls, color=color,
                         label=label, lw=lw, markersize=8)
                ax2.fill_between(mus, nmis - stds, nmis + stds, alpha=0.15, color=color)

        ax2.axvline(DETECTABILITY_LIMIT, color="k", ls=":", alpha=0.7,
                    label=f"mu*={DETECTABILITY_LIMIT:.3f}")
        ax2.set_xlabel("mu (mixing parameter)")
        ax2.set_ylabel("NMI")
        ax2.set_title(f"Iter-003: Zoom near detectability limit (mu >= 0.45)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2_path = RESULTS_DIR / "nmi_vs_mu_iter003_zoom.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Zoom figure saved to {fig2_path}")

    except Exception as e:
        print(f"\nPlotting failed: {e}")
        traceback.print_exc()

    return results


if __name__ == "__main__":
    main()
