"""
Iter-001: Unconstrained MSE factorization of normalized adjacency vs baselines.

Hypothesis H2: Does unconstrained (non-orthogonal) matrix factorization of the
normalized adjacency M = D^{-1/2} A D^{-1/2} improve detectability near the SBM limit?

Compares:
  1. Spectral (LaplacianEigenMap)
  2. Node2VecMatrixFactorization
  3. Unconstrained MSE factorization of M (Adam, ~500 steps)
  4. Node2Vec (if time permits)
"""

import json
import time
import warnings
import traceback
from pathlib import Path

import igraph as ig
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

import embcom

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
if "snakemake" in dir():
    RESULTS_FILE = Path(snakemake.output["output_file"])
else:
    RESULTS_FILE = Path("/workspace/exps/2026-03-28-node2vec-detectability/analyses/iter-001/results.json")

RESULTS_DIR = RESULTS_FILE.parent
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment parameters ─────────────────────────────────────────────────────
MU_SWEEP = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
N_SAMPLES = 10
N = 2000
N_EACH = N // 2
CAVE = 5.0
DIM = 64
DETECTABILITY_LIMIT = 0.553

# ── SBM generation ────────────────────────────────────────────────────────────

def make_sbm(mu, seed=None):
    """Generate SBM with N=2000, 2 equal communities, cave=5.0.
    Parameterization: c_out = mu*cave, c_in = 2*cave - c_out
    => detectability limit at mu* = 1 - 1/sqrt(cave) ≈ 0.553 for cave=5
    Returns (A, labels) restricted to the largest connected component (LCC)
    to avoid isolated node issues with embcom methods.
    """
    c_out = mu * CAVE
    c_in = 2 * CAVE - c_out
    p_in = c_in / N
    p_out = c_out / N
    pref_matrix = [[p_in, p_out], [p_out, p_in]]
    block_sizes = [N_EACH, N_EACH]
    g = ig.Graph.SBM(pref_matrix, block_sizes, directed=False)
    labels_full = np.array([0] * N_EACH + [1] * N_EACH)

    # Extract LCC to avoid isolated nodes causing NaN in embcom methods
    components = g.connected_components(mode="weak")
    lcc_indices = np.array(components.giant().vs["_nx_name"]
                           if "_nx_name" in g.vertex_attributes()
                           else max(components, key=len))
    lcc_indices = sorted(lcc_indices)
    g_lcc = g.induced_subgraph(lcc_indices)
    labels = labels_full[lcc_indices]

    A = g_lcc.get_adjacency_sparse()
    A = sp.csr_matrix(A, dtype=float)
    return A, labels


def normalized_adjacency(A):
    """Compute M = D^{-1/2} A D^{-1/2}."""
    degrees = np.array(A.sum(axis=1)).flatten()
    # Handle isolated nodes
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    M = D_inv_sqrt @ A @ D_inv_sqrt
    return M


# ── Unconstrained MSE factorization ──────────────────────────────────────────

def unconstrained_mf(M_dense, dim=64, n_steps=500, lr=0.01, seed=0):
    """
    Factorize M ≈ U V^T minimizing ||M - U V^T||_F^2
    using full-batch Adam gradient descent.
    No orthogonality constraint.
    """
    rng = np.random.default_rng(seed)
    n = M_dense.shape[0]

    # Initialize
    U = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    V = rng.normal(0, 0.01, (n, dim)).astype(np.float32)
    M = M_dense.astype(np.float32)

    # Adam state
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)

    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, n_steps + 1):
        # Forward: residual = M - U V^T
        UV = U @ V.T  # (n, n)
        R = M - UV    # (n, n)

        # Gradients of ||R||_F^2 / n^2 (normalized for stability)
        scale = 1.0 / (n * n)
        gU = -2.0 * scale * (R @ V)   # (n, dim)
        gV = -2.0 * scale * (R.T @ U) # (n, dim)

        # Adam update for U
        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * (gU ** 2)
        mU_hat = mU / (1 - beta1 ** step)
        vU_hat = vU / (1 - beta2 ** step)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)

        # Adam update for V
        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * (gV ** 2)
        mV_hat = mV / (1 - beta1 ** step)
        vV_hat = vV / (1 - beta2 ** step)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)

        if step % 100 == 0:
            loss = np.mean(R ** 2)
            print(f"  Step {step:4d}: MSE loss = {loss:.6f}")

    return U


# ── Classification ─────────────────────────────────────────────────────────────

def classify_nmi(emb, labels, n_clusters=2):
    """K-means clustering and NMI score."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = km.fit_predict(emb)
    return normalized_mutual_info_score(labels, pred)


# ── Run experiment ─────────────────────────────────────────────────────────────

def run_method(name, fit_fn, A, labels):
    """Run a single method, return NMI and timing."""
    t0 = time.time()
    try:
        emb = fit_fn(A)
        nmi = classify_nmi(emb, labels)
        elapsed = time.time() - t0
        return {"nmi": float(nmi), "time": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR in {name}: {e}")
        traceback.print_exc()
        return {"nmi": None, "time": elapsed, "error": str(e)}


def spectral_fit(A):
    """Direct spectral embedding using top eigenvectors of normalized adjacency M=D^{-1/2}AD^{-1/2}.
    Skips the trivial (constant) eigenvector (eigenvalue=1) and returns dim eigenvectors.
    Uses embcom.LaplacianEigenMap as a wrapper but falls back to direct eigsh if embcom fails.
    """
    from scipy.sparse.linalg import eigsh
    degrees = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    M = D_inv_sqrt @ A @ D_inv_sqrt

    # Get dim+1 eigenvectors, skip the first (trivial, eigenvalue=1)
    k = min(DIM + 1, A.shape[0] - 1)
    vals, vecs = eigsh(M, k=k, which='LM')
    # Sort descending and skip the first eigenvector (eigenvalue ~1)
    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]
    # Return eigenvectors 1..DIM (skip trivial eigvec 0)
    return vecs[:, 1:DIM+1]


def n2vec_mf_fit(A):
    """Node2VecMatrixFactorization with safe log to handle zero Ppow entries."""
    from embcom import utils as embcom_utils
    from sklearn.decomposition import TruncatedSVD
    from scipy import sparse

    model = embcom.Node2VecMatrixFactorization(window_length=10)
    model.fit(A)

    # Replicate update_embedding with safe log (clips zero entries before log)
    P = embcom_utils.to_trans_mat(model.A)
    Ppow = embcom_utils.matrix_sum_power(P, model.window_length) / model.window_length
    stationary_prob = model.deg / np.sum(model.deg)
    tmp = Ppow @ np.diag(1.0 / np.maximum(stationary_prob, 1e-12))
    R = np.log(np.maximum(tmp, 1e-12))  # safe log: avoid -inf

    svd = TruncatedSVD(n_components=DIM + 1, n_iter=7, random_state=42)
    u = svd.fit_transform(R)
    s = svd.singular_values_
    emb = u @ sparse.diags(np.sqrt(s)).toarray()
    return emb


def node2vec_fit(A):
    model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10)
    model.fit(A)
    return model.transform(dim=DIM)


def unconstrained_mf_fit(A):
    M = normalized_adjacency(A)
    # Convert sparse to dense (N=2000 is manageable: 2000x2000 = 4M floats ~ 32MB)
    M_dense = M.toarray()
    return unconstrained_mf(M_dense, dim=DIM, n_steps=500, lr=0.01)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {
        "metadata": {
            "date": "2026-03-28",
            "iteration": "iter-001",
            "hypothesis": "H2 - unconstrained MSE factorization of normalized adjacency (I - L_norm)",
            "N": N,
            "cave": CAVE,
            "dim": DIM,
            "n_samples": N_SAMPLES,
            "mu_sweep": MU_SWEEP,
            "detectability_limit": DETECTABILITY_LIMIT,
            "methods": {
                "spectral": "Direct spectral: top eigenvectors of M=D^{-1/2}AD^{-1/2} via eigsh, skip trivial eigvec",
                "n2vec_mf": "Node2VecMatrixFactorization (embcom, window=10)",
                "unconstrained_mf": "MSE factorization of M=D^{-1/2}AD^{-1/2} (=I-L_norm), Adam 500 steps lr=0.01, no ortho constraint",
                "node2vec": "Node2Vec (embcom, walks=10, length=80, window=10) - may be skipped if slow",
            },
        },
        "results": {},
        "failed_attempts": [],
        "notes": [
            "SBM parameterization: c_out=mu*cave, c_in=2*cave-c_out, p_in=c_in/N, p_out=c_out/N",
            "This gives mu* = 1 - 1/sqrt(cave) = 0.553 for cave=5 (Decelle limit)",
            "Task spec code used p_total=cave*2/N which gives wrong detectability limit (mu*~0.276)",
            "LCC extraction used to avoid isolated nodes in embcom methods",
        ],
    }

    methods = [
        ("spectral", spectral_fit),
        ("n2vec_mf", n2vec_mf_fit),
        ("unconstrained_mf", unconstrained_mf_fit),
    ]

    # Try node2vec with a timeout probe on mu=0.3, sample=0
    print("\n--- Probing node2vec speed ---")
    A_probe, labels_probe = make_sbm(mu=0.3, seed=0)
    t_probe = time.time()
    try:
        emb_probe = node2vec_fit(A_probe)
        t_node2vec = time.time() - t_probe
        print(f"  node2vec probe: {t_node2vec:.1f}s")
        if t_node2vec * N_SAMPLES * len(MU_SWEEP) < 300:
            methods.append(("node2vec", node2vec_fit))
            print("  -> Adding node2vec to methods (estimated total time OK)")
        else:
            est = t_node2vec * N_SAMPLES * len(MU_SWEEP)
            print(f"  -> Skipping node2vec (estimated {est:.0f}s > 300s limit)")
            results["failed_attempts"].append({
                "method": "node2vec",
                "reason": f"Too slow: estimated {est:.0f}s for full sweep",
                "probe_time": t_node2vec,
            })
    except Exception as e:
        print(f"  -> node2vec probe failed: {e}")
        results["failed_attempts"].append({
            "method": "node2vec",
            "reason": f"Probe failed: {e}",
        })

    # Initialize results structure
    for method_name, _ in methods:
        results["results"][method_name] = {}

    print(f"\n=== Running {len(methods)} methods x {len(MU_SWEEP)} mu values x {N_SAMPLES} samples ===\n")

    for mu in MU_SWEEP:
        print(f"\n--- mu = {mu} {'(above limit)' if mu > DETECTABILITY_LIMIT else '(below limit)'} ---")
        for method_name, fit_fn in methods:
            nmis = []
            times = []
            errors = []
            print(f"  [{method_name}]")
            for s in range(N_SAMPLES):
                A, labels = make_sbm(mu=mu, seed=s * 1000 + int(mu * 1000))
                r = run_method(method_name, fit_fn, A, labels)
                if r["nmi"] is not None:
                    nmis.append(r["nmi"])
                    times.append(r["time"])
                else:
                    errors.append({"sample": s, "error": r["error"]})
            mean_nmi = float(np.mean(nmis)) if nmis else None
            std_nmi = float(np.std(nmis)) if nmis else None
            print(f"    NMI: {mean_nmi:.4f} ± {std_nmi:.4f}" if mean_nmi is not None else "    NMI: FAILED")
            results["results"][method_name][str(mu)] = {
                "nmi_mean": mean_nmi,
                "nmi_std": std_nmi,
                "nmi_samples": nmis,
                "n_success": len(nmis),
                "mean_time": float(np.mean(times)) if times else None,
                "errors": errors,
            }

        # Save intermediate results
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [saved intermediate results to {RESULTS_FILE}]")

    print(f"\n=== Done. Results saved to {RESULTS_FILE} ===")

    # Print summary table
    print("\n=== Summary: mean NMI by method and mu ===")
    header = f"{'mu':>6} | " + " | ".join(f"{m[0]:>16}" for m in methods)
    print(header)
    print("-" * len(header))
    for mu in MU_SWEEP:
        row = f"{mu:>6.3f} | "
        for method_name, _ in methods:
            r = results["results"][method_name].get(str(mu), {})
            nmi = r.get("nmi_mean")
            row += f"{nmi:>16.4f} | " if nmi is not None else f"{'N/A':>16} | "
        print(row)

    return results


if __name__ == "__main__":
    main()
