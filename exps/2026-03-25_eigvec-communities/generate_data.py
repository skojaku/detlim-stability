"""Generate SBM networks and eigenvectors near the detectability limit. Save to data/."""
import numpy as np
import igraph as ig
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import belief_propagation as bp
import os

# Parameters
n = 1000
N = 2 * n
cave = 5.0
mu = 0.5        # near detectability limit (~0.553 for cave=5)
n_samples = 30
k_eigvecs = 10  # compute more eigenvectors to study higher ones

c_out = mu * cave
c_in = 2 * cave - c_out
p_in = c_in / N
p_out = c_out / N
membership = np.array([0] * n + [1] * n)

outdir = "data"
os.makedirs(outdir, exist_ok=True)

nmi_sign = np.zeros(n_samples)
nmi_kmeans = np.zeros(n_samples)
nmi_bp = np.zeros(n_samples)

for s in range(n_samples):
    g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
    edges = g.get_edgelist()
    if len(edges) > 0:
        rows, cols = zip(*edges)
        A = sp.csr_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(N, N))
        A = A + A.T
        A.data[:] = 1
    else:
        A = sp.csr_matrix((N, N))

    # Save network
    sp.save_npz(f"{outdir}/net_sample_{s:03d}.npz", A)

    # Eigenvectors (top k by largest eigenvalue magnitude)
    vals, V = eigsh(A.astype(float), k=k_eigvecs, which="LA")
    vals = vals[::-1]
    V = V[:, ::-1]
    np.savez(f"{outdir}/eigvecs_sample_{s:03d}.npz", vals=vals, V=V)

    # Baseline: sign of 2nd eigenvector
    ev2 = V[:, 1].copy()
    if ev2[0] < 0:
        ev2 = -ev2
    nmi_sign[s] = normalized_mutual_info_score(membership, (ev2 >= 0).astype(int))

    # Baseline: standard K-means on top-2 eigenvectors
    km = KMeans(n_clusters=2, n_init=10, random_state=s)
    nmi_kmeans[s] = normalized_mutual_info_score(membership, km.fit_predict(V[:, :5]))

    # Baseline: belief propagation (reference target)
    labels_bp = bp.detect(A.copy(), q=2, init_memberships=membership)
    nmi_bp[s] = normalized_mutual_info_score(membership, labels_bp)

    if (s + 1) % 10 == 0:
        print(f"Sample {s + 1}/{n_samples} done")

# Save baselines
import pandas as pd
df = pd.DataFrame({
    "sample": np.arange(n_samples),
    "nmi_sign_v2": nmi_sign,
    "nmi_kmeans": nmi_kmeans,
    "nmi_bp": nmi_bp,
})
df.to_csv(f"{outdir}/baselines.csv", index=False)

# Save parameters
params = dict(n=n, N=N, cave=cave, mu=mu, n_samples=n_samples, k_eigvecs=k_eigvecs,
              c_in=c_in, c_out=c_out, p_in=p_in, p_out=p_out)
np.savez(f"{outdir}/params.npz", **params)

print(f"\nSaved to {outdir}/:")
print(f"  Networks: net_sample_000.npz .. net_sample_{n_samples-1:03d}.npz")
print(f"  Eigvecs:  eigvecs_sample_000.npz .. eigvecs_sample_{n_samples-1:03d}.npz")
print(f"  Baselines: baselines.csv")
print(f"\nBaseline results:")
print(f"  Sign of 2nd eigvec:  NMI = {nmi_sign.mean():.4f} ± {nmi_sign.std():.4f}")
print(f"  Standard K-means:    NMI = {nmi_kmeans.mean():.4f} ± {nmi_kmeans.std():.4f}")
print(f"  Belief propagation:  NMI = {nmi_bp.mean():.4f} ± {nmi_bp.std():.4f}")
