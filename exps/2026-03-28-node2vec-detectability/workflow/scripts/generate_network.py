"""Generate one SBM sample and save adjacency + labels to .npz."""
import sys
import numpy as np
import igraph as ig
import scipy.sparse as sp

if "snakemake" in sys.modules:
    n = int(snakemake.params["n"])
    cave = float(snakemake.params["cave"])
    mu = float(snakemake.params["mu"])
    sample = int(snakemake.params["sample"])
    output_file = snakemake.output["network_file"]
else:
    n = 2000
    cave = 5.0
    mu = 0.3
    sample = 0
    output_file = "../data/networks/n~2000_cave~5.0_mu~0.3/sample~000.npz"

import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

N = n
n_each = N // 2

# Correct parameterization: mu* = 1 - 1/sqrt(cave) for these parameters
c_out = mu * cave
c_in = 2 * cave - c_out
p_in = c_in / N
p_out = c_out / N

rng = np.random.default_rng(seed=sample * 1000 + int(mu * 1000))
np.random.seed(sample * 1000 + int(mu * 1000))

g = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n_each, n_each], directed=False)
A = g.get_adjacency_sparse()
A = sp.csr_matrix(A, dtype=float)
labels = np.array([0] * n_each + [1] * n_each)

np.savez(
    output_file,
    data=A.data,
    indices=A.indices,
    indptr=A.indptr,
    shape=np.array(A.shape),
    labels=labels,
    mu=np.float64(mu),
    cave=np.float64(cave),
    n=np.int64(N),
    sample=np.int64(sample),
)
print(f"Saved: {output_file}  (N={N}, mu={mu:.2f}, sample={sample}, edges={A.nnz//2})")
