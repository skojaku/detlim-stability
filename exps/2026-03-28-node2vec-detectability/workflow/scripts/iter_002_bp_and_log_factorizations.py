"""
Iter-002: BP baseline + log/linear × orthogonal/free factorization on R matrix + node2vec.

Hypothesis: Does node2vec's superiority near the SBM detectability limit arise from:
  (a) lack of orthogonality constraints (free factorization)
  (b) log-nonlinearity (log PMI matrix)
  (c) implicit SGD regularization (random walk sampling)

Methods:
  1. BP (belief propagation) -- critical baseline to verify SBM setup
  2. Spectral (eigsh on M=D^{-1/2}AD^{-1/2}) -- reference
  3. SVD of R (linear + orthogonal)
  4. Unconstrained MSE of R (linear + free)
  5. SVD of log(R+) (log + orthogonal)
  6. Unconstrained MSE of log(R+) (log + free)
  7. Node2Vec actual (N=500, random walk baseline)

R matrix (PMI): R = (2m) * D_inv @ A @ D_inv
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

import embcom

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = Path("/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-002/results.json")
RESULTS_DIR = RESULTS_FILE.parent
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────
MU_SWEEP = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
N_SAMPLES = 10
N = 2000
N_EACH = N // 2
CAVE = 5.0
DIM = 64
DETECTABILITY_LIMIT = 1.0 - 1.0 / np.sqrt(CAVE)  # ≈ 0.5528

# Node2vec uses smaller graph for speed
N_N2V = 500
N_EACH_N2V = N_N2V // 2
N_SAMPLES_N2V = 5

LOG_EPS = 1e-10  # floor for log(R) to avoid -inf

# ── SBM generation ────────────────────────────────────────────────────────────

def make_sbm(mu, n_total=None, seed=None):
    """Generate SBM with 2 equal communities, cave=5.0.
    Parameterization: c_out = mu*cave, c_in = 2*cave - c_out
    => detectability limit mu* = 1 - 1/sqrt(cave) ≈ 0.553 for cave=5.
    Returns (A, labels) for the full graph (no LCC filter -- BP handles isolated nodes).
    LCC is only applied when needed for embcom methods.
    """
    if n_total is None:
        n_total = N
    n_each = n_total // 2
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / n_total
    p_out = c_out / n_total
    pref_matrix = [[p_in, p_out], [p_out, p_in]]
    block_sizes = [n_each, n_each]

    rng_seed = seed if seed is not None else 42
    np.random.seed(rng_seed)  # igraph uses numpy random state

    g = ig.Graph.SBM(pref_matrix, block_sizes, directed=False)
    labels_full = np.array([0] * n_each + [1] * n_each)

    A = g.get_adjacency_sparse()
    A = sp.csr_matrix(A, dtype=float)
    return A, labels_full


def make_sbm_lcc(mu, n_total=None, seed=None):
    """Same as make_sbm but returns LCC to avoid isolated nodes in embcom."""
    if n_total is None:
        n_total = N
    n_each = n_total // 2
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / n_total
    p_out = c_out / n_total
    pref_matrix = [[p_in, p_out], [p_out, p_in]]
    block_sizes = [n_each, n_each]

    rng_seed = seed if seed is not None else 42
    np.random.seed(rng_seed)

    g = ig.Graph.SBM(pref_matrix, block_sizes, directed=False)
    labels_full = np.array([0] * n_each + [1] * n_each)

    components = g.connected_components(mode="weak")
    lcc_indices = sorted(max(components, key=len))
    g_lcc = g.induced_subgraph(lcc_indices)
    labels = labels_full[lcc_indices]

    A = g_lcc.get_adjacency_sparse()
    A = sp.csr_matrix(A, dtype=float)
    return A, labels


# ── PMI R matrix ──────────────────────────────────────────────────────────────

def compute_R(A):
    """Compute PMI matrix R = (2m) * D_inv @ A @ D_inv (sparse)."""
    d = np.array(A.sum(axis=1)).flatten()
    m = d.sum() / 2.0
    d_safe = np.maximum(d, 1.0)
    D_inv = sp.diags(1.0 / d_safe)
    R = (2 * m) * D_inv @ A @ D_inv  # sparse, same sparsity as A
    return R


def compute_log_R(A):
    """Compute log(R) for nonzero entries only (sparse log-PMI matrix).
    Only nonzero positions are kept; zeros treated as missing (implicit negative PMI).
    Uses LOG_EPS floor for numerical stability.
    """
    R = compute_R(A)
    R = R.tocsr().astype(float)
    # Apply log elementwise to nonzero data
    R.data = np.log(np.maximum(R.data, LOG_EPS))
    return R


# ── Classification ─────────────────────────────────────────────────────────────

def classify_nmi(emb, labels, n_clusters=2):
    """K-means clustering and NMI score."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = km.fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)


# ── Method implementations ─────────────────────────────────────────────────────

def bp_fit(A, labels):
    """Belief propagation via C++ wrapper. Returns NMI directly (not embeddings).
    Uses iters=10 to reduce probability of trivial fixed point convergence.
    """
    A_csr = sp.csr_matrix(A, dtype=float)
    cids = belief_propagation.detect(A_csr, q=2, iters=5)
    return normalized_mutual_info_score(labels, cids)


def spectral_fit(A):
    """Top eigenvectors of M=D^{-1/2}AD^{-1/2}, skip trivial eigenvector."""
    degrees = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    M = D_inv_sqrt @ A @ D_inv_sqrt

    k = min(DIM + 1, A.shape[0] - 1)
    vals, vecs = eigsh(M, k=k, which='LM')
    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]
    return vecs[:, 1:DIM + 1]  # skip trivial eigvec


def svd_R_fit(A):
    """SVD of R (linear + orthogonal). Uses sparse R."""
    R = compute_R(A)
    k = min(DIM, min(R.shape) - 1)
    U, s, Vt = svds(R, k=k)
    # Sort by descending singular value
    idx = np.argsort(-s)
    U = U[:, idx]
    s = s[idx]
    return U * np.sqrt(s)


def unconstrained_R_fit(A, n_steps=500, lr=0.01, seed=0):
    """Unconstrained MSE factorization of R (linear + free). Adam optimizer.
    R is sparse; compute gradient only over nonzero entries for memory efficiency.
    Actually since N=2000 and R has ~cave*N nonzeros, we can use sparse operations.
    """
    R = compute_R(A)
    R_dense = R.toarray().astype(np.float32)
    n = R_dense.shape[0]
    dim = DIM

    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    V = rng.normal(0, 0.01, (n, dim)).astype(np.float32)

    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for step in range(1, n_steps + 1):
        UV = U @ V.T
        Res = R_dense - UV
        scale = 1.0 / (n * n)
        gU = -2.0 * scale * (Res @ V)
        gV = -2.0 * scale * (Res.T @ U)

        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * (gU ** 2)
        U -= lr * (mU / (1 - beta1 ** step)) / (np.sqrt(vU / (1 - beta2 ** step)) + eps_adam)

        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * (gV ** 2)
        V -= lr * (mV / (1 - beta1 ** step)) / (np.sqrt(vV / (1 - beta2 ** step)) + eps_adam)

    return U


def svd_logR_fit(A):
    """SVD of log(R+) over nonzero entries (log + orthogonal)."""
    log_R = compute_log_R(A)
    k = min(DIM, min(log_R.shape) - 1)
    U, s, Vt = svds(log_R, k=k)
    idx = np.argsort(-s)
    U = U[:, idx]
    s = s[idx]
    return U * np.sqrt(np.maximum(s, 0))


def unconstrained_logR_fit(A, n_steps=500, lr=0.01, seed=0):
    """Unconstrained MSE factorization of log(R+) over nonzero entries only (log + free).
    This is the closest batch approximation to the node2vec SGNS objective (Levy & Goldberg).
    Only reconstructs nonzero positions of log(R); treats zeros as missing.
    """
    log_R = compute_log_R(A)
    log_R_csr = log_R.tocsr().astype(np.float32)

    n = log_R_csr.shape[0]
    dim = DIM
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    V = rng.normal(0, 0.01, (n, dim)).astype(np.float32)

    # Get nonzero indices for efficient sparse gradient
    rows, cols = log_R_csr.nonzero()
    vals = np.array(log_R_csr[rows, cols]).flatten()
    nnz = len(rows)

    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for step in range(1, n_steps + 1):
        # Predicted values at nonzero positions
        pred = np.sum(U[rows] * V[cols], axis=1)  # (nnz,)
        residuals = vals - pred  # (nnz,)

        scale = 1.0 / nnz
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        weighted_res = -2.0 * scale * residuals
        np.add.at(gU, rows, weighted_res[:, None] * V[cols])
        np.add.at(gV, cols, weighted_res[:, None] * U[rows])

        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * (gU ** 2)
        U -= lr * (mU / (1 - beta1 ** step)) / (np.sqrt(vU / (1 - beta2 ** step)) + eps_adam)

        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * (gV ** 2)
        V -= lr * (mV / (1 - beta1 ** step)) / (np.sqrt(vV / (1 - beta2 ** step)) + eps_adam)

        if step % 100 == 0:
            loss = np.mean(residuals ** 2)
            print(f"    Step {step:4d}: MSE loss = {loss:.6f}")

    return U


def node2vec_fit(A):
    """Node2Vec actual random walk (for N=500 graphs)."""
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10)
    model.fit(A)
    return model.transform(dim=DIM)


# ── Run experiment ─────────────────────────────────────────────────────────────

def run_method_emb(name, fit_fn, A, labels):
    """Run embedding method, return NMI and timing."""
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
    """Run BP, return NMI and timing."""
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


def run_sweep(method_name, fit_fn, mu_sweep, n_samples, n_total, use_lcc=False,
              is_bp=False):
    """Run a full mu sweep for one method. Returns dict keyed by str(mu)."""
    method_results = {}
    for mu in mu_sweep:
        label = f"{'above' if mu > DETECTABILITY_LIMIT else 'below'} limit"
        print(f"  mu={mu:.3f} ({label})")
        nmis, times, errors = [], [], []
        for s in range(n_samples):
            seed = s * 1000 + int(mu * 1000)
            if use_lcc:
                A, labels = make_sbm_lcc(mu=mu, n_total=n_total, seed=seed)
            else:
                A, labels = make_sbm(mu=mu, n_total=n_total, seed=seed)

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
            "iteration": "iter-002",
            "hypothesis": "Does node2vec power near SBM limit come from (a) no ortho, (b) log, or (c) SGD?",
            "N_main": N,
            "N_node2vec": N_N2V,
            "cave": CAVE,
            "dim": DIM,
            "n_samples_main": N_SAMPLES,
            "n_samples_node2vec": N_SAMPLES_N2V,
            "mu_sweep": MU_SWEEP,
            "detectability_limit": float(DETECTABILITY_LIMIT),
            "methods": {
                "bp": "Belief propagation (C++ wrapper, q=2). Critical SBM sanity check.",
                "spectral": "Top eigvecs of M=D^{-1/2}AD^{-1/2} via eigsh, skip trivial eigvec",
                "svd_R": "SVD of R=(2m)*D_inv@A@D_inv (linear + orthogonal)",
                "free_R": "Unconstrained MSE factorization of R (linear + free, Adam 500 steps)",
                "svd_logR": "SVD of log(R+) over nonzero entries (log + orthogonal)",
                "free_logR": "Unconstrained MSE of log(R+) over nonzero entries (log + free). Closest to node2vec SGNS.",
                "node2vec": f"Node2Vec actual (N={N_N2V}, walks=10, length=80, window=10). Separate sweep.",
            },
        },
        "results": {},
        "notes": [
            "SBM parameterization: c_out=mu*cave, c_in=2*cave-c_out, p_in=c_in/N, p_out=c_out/N",
            f"Detectability limit mu*=1-1/sqrt(cave)={DETECTABILITY_LIMIT:.4f} for cave={CAVE}",
            "BP uses full graph (no LCC). Embedding methods use LCC to avoid embcom issues.",
            "log(R+): only nonzero entries kept, floor at 1e-10 before log",
            "free_logR factorizes only nonzero positions (implicit negative PMI = missing)",
        ],
    }

    # ── Section 1: Main methods on N=2000 ────────────────────────────────────
    print(f"\n=== Section 1: Main methods (N={N}, {N_SAMPLES} samples per mu) ===\n")
    print(f"Detectability limit: mu* = {DETECTABILITY_LIMIT:.4f}\n")

    # 1. BP
    print("--- Method: BP ---")
    results["results"]["bp"] = run_sweep(
        "bp", None, MU_SWEEP, N_SAMPLES, n_total=N, use_lcc=False, is_bp=True
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after BP]\n")

    # 2. Spectral
    print("--- Method: spectral ---")
    results["results"]["spectral"] = run_sweep(
        "spectral", spectral_fit, MU_SWEEP, N_SAMPLES, n_total=N, use_lcc=False
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after spectral]\n")

    # 3. SVD of R (linear + orthogonal)
    print("--- Method: svd_R ---")
    results["results"]["svd_R"] = run_sweep(
        "svd_R", svd_R_fit, MU_SWEEP, N_SAMPLES, n_total=N, use_lcc=False
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after svd_R]\n")

    # 4. Unconstrained MSE of R (linear + free)
    print("--- Method: free_R ---")
    results["results"]["free_R"] = run_sweep(
        "free_R", unconstrained_R_fit, MU_SWEEP, N_SAMPLES, n_total=N, use_lcc=False
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after free_R]\n")

    # 5. SVD of log(R+) (log + orthogonal)
    print("--- Method: svd_logR ---")
    results["results"]["svd_logR"] = run_sweep(
        "svd_logR", svd_logR_fit, MU_SWEEP, N_SAMPLES, n_total=N, use_lcc=False
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after svd_logR]\n")

    # 6. Unconstrained MSE of log(R+) (log + free)
    print("--- Method: free_logR ---")
    results["results"]["free_logR"] = run_sweep(
        "free_logR", unconstrained_logR_fit, MU_SWEEP, N_SAMPLES, n_total=N, use_lcc=False
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after free_logR]\n")

    # ── Section 2: Node2Vec on N=500 ─────────────────────────────────────────
    print(f"\n=== Section 2: Node2Vec actual (N={N_N2V}, {N_SAMPLES_N2V} samples per mu) ===\n")
    print("--- Method: node2vec ---")
    results["results"]["node2vec"] = run_sweep(
        "node2vec", node2vec_fit, MU_SWEEP, N_SAMPLES_N2V, n_total=N_N2V, use_lcc=True
    )
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved after node2vec]\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    method_order = ["bp", "spectral", "svd_R", "free_R", "svd_logR", "free_logR", "node2vec"]
    print("\n=== Summary: mean NMI by method and mu ===")
    header = f"{'mu':>6} | " + " | ".join(f"{m:>12}" for m in method_order)
    print(header)
    print("-" * len(header))
    for mu in MU_SWEEP:
        row = f"{mu:>6.3f} | "
        for method_name in method_order:
            r = results["results"].get(method_name, {}).get(str(mu), {})
            nmi = r.get("nmi_mean")
            row += f"{nmi:>12.4f} | " if nmi is not None else f"{'N/A':>12} | "
        print(row)

    print(f"\n=== Done. Results saved to {RESULTS_FILE} ===")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: main methods (N=2000)
        ax = axes[0]
        main_methods = [
            ("bp", "BP", "black", "o", "-"),
            ("spectral", "Spectral", "gray", "s", "--"),
            ("svd_R", "SVD(R) lin+ortho", "blue", "^", "-"),
            ("free_R", "Free(R) lin+free", "blue", "v", "--"),
            ("svd_logR", "SVD(logR) log+ortho", "red", "^", "-"),
            ("free_logR", "Free(logR) log+free", "red", "v", "--"),
        ]
        for method_name, label, color, marker, ls in main_methods:
            mus, nmis, stds = [], [], []
            for mu in MU_SWEEP:
                r = results["results"].get(method_name, {}).get(str(mu), {})
                if r.get("nmi_mean") is not None:
                    mus.append(mu)
                    nmis.append(r["nmi_mean"])
                    stds.append(r.get("nmi_std", 0))
            if mus:
                mus, nmis, stds = np.array(mus), np.array(nmis), np.array(stds)
                ax.plot(mus, nmis, marker=marker, ls=ls, color=color, label=label)
                ax.fill_between(mus, nmis - stds, nmis + stds, alpha=0.15, color=color)

        ax.axvline(DETECTABILITY_LIMIT, color="k", ls=":", alpha=0.7, label=f"mu*={DETECTABILITY_LIMIT:.3f}")
        ax.set_xlabel("mu (mixing parameter)")
        ax.set_ylabel("NMI")
        ax.set_title(f"Community detection NMI vs mu (N={N}, {N_SAMPLES} samples)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        # Right panel: node2vec N=500
        ax2 = axes[1]
        r_n2v = results["results"].get("node2vec", {})
        mus_n2v, nmis_n2v, stds_n2v = [], [], []
        for mu in MU_SWEEP:
            r = r_n2v.get(str(mu), {})
            if r.get("nmi_mean") is not None:
                mus_n2v.append(mu)
                nmis_n2v.append(r["nmi_mean"])
                stds_n2v.append(r.get("nmi_std", 0))
        if mus_n2v:
            mus_n2v = np.array(mus_n2v)
            nmis_n2v = np.array(nmis_n2v)
            stds_n2v = np.array(stds_n2v)
            ax2.plot(mus_n2v, nmis_n2v, "go-", label=f"Node2Vec (N={N_N2V})")
            ax2.fill_between(mus_n2v, nmis_n2v - stds_n2v, nmis_n2v + stds_n2v,
                             alpha=0.15, color="green")

        # Also overlay free_logR from N=2000 for comparison
        mus_fl, nmis_fl, stds_fl = [], [], []
        for mu in MU_SWEEP:
            r = results["results"].get("free_logR", {}).get(str(mu), {})
            if r.get("nmi_mean") is not None:
                mus_fl.append(mu)
                nmis_fl.append(r["nmi_mean"])
                stds_fl.append(r.get("nmi_std", 0))
        if mus_fl:
            ax2.plot(mus_fl, nmis_fl, "r^--", label=f"Free(logR) (N={N})", alpha=0.7)

        ax2.axvline(DETECTABILITY_LIMIT, color="k", ls=":", alpha=0.7, label=f"mu*={DETECTABILITY_LIMIT:.3f}")
        ax2.set_xlabel("mu (mixing parameter)")
        ax2.set_ylabel("NMI")
        ax2.set_title(f"Node2Vec vs Free(logR): (N={N_N2V} vs N={N})")
        ax2.legend(fontsize=9)
        ax2.set_xlim(0.25, 0.75)
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = RESULTS_DIR / "nmi_vs_mu.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nFigure saved to {fig_path}")

    except Exception as e:
        print(f"\nPlotting failed: {e}")
        traceback.print_exc()

    return results


if __name__ == "__main__":
    main()
