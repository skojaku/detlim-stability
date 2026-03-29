"""Run spectral clustering (normalized Laplacian eigenvectors) on a saved network."""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import pandas as pd

if "snakemake" in sys.modules:
    network_file = snakemake.input["network_file"]
    output_file = snakemake.output["nmi_file"]
    mu = float(snakemake.params["mu"])
    sample = int(snakemake.params["sample"])
    dim = int(snakemake.params["dim"])
else:
    network_file = "../data/networks/n~2000_cave~5.0_mu~0.3/sample~000.npz"
    output_file = "../data/baselines/spectral/n~2000_cave~5.0_mu~0.3_sample~000.csv"
    mu = 0.3
    sample = 0
    dim = 64

import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

d = np.load(network_file)
A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
labels = d["labels"]

# Use LCC for embedding methods
import igraph as ig
g = ig.Graph.Adjacency((A > 0).toarray().tolist(), mode="undirected")
components = g.connected_components(mode="weak")
lcc_idx = sorted(max(components, key=len))
A_lcc = A[np.ix_(lcc_idx, lcc_idx)]
labels_lcc = labels[lcc_idx]

# Normalized adjacency: D^{-1/2} A D^{-1/2}  (= I - L_norm)
# Correct spectral: use eigsh directly (avoids embcom.LaplacianEigenMap bug)
deg = np.array(A_lcc.sum(axis=1)).flatten()
deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
D_inv_sqrt = sp.diags(deg_inv_sqrt)
M = D_inv_sqrt @ A_lcc @ D_inv_sqrt

k = min(dim + 2, A_lcc.shape[0] - 1)
vals, vecs = eigsh(M, k=k, which="LM")
# Sort descending; skip trivial eigenvector (largest, all-positive)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
# Skip the trivial (first) eigenvector
emb = vecs[:, 1:dim+1]

pred = KMeans(n_clusters=2, n_init=10, random_state=sample).fit_predict(emb)
nmi = normalized_mutual_info_score(labels_lcc, pred)

pd.DataFrame({
    "method": ["spectral"],
    "mu": [mu],
    "sample": [sample],
    "nmi": [nmi],
}).to_csv(output_file, index=False)

print(f"Spectral: mu={mu:.2f}, sample={sample}, NMI={nmi:.4f}")
