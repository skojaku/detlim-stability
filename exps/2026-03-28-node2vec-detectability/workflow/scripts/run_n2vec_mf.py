"""Run Node2VecMatrixFactorization (SVD of NetMF matrix) on a saved network."""
import sys
import numpy as np
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, "/workspace/libs/embcom_repo/libs/embcom")
import embcom

if "snakemake" in sys.modules:
    network_file = snakemake.input["network_file"]
    output_file = snakemake.output["nmi_file"]
    mu = float(snakemake.params["mu"])
    sample = int(snakemake.params["sample"])
    dim = int(snakemake.params["dim"])
else:
    network_file = "../data/networks/n~2000_cave~5.0_mu~0.3/sample~000.npz"
    output_file = "../data/baselines/n2vec_mf/n~2000_cave~5.0_mu~0.3_sample~000.csv"
    mu = 0.3
    sample = 0
    dim = 64

import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import igraph as ig

d = np.load(network_file)
A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
labels = d["labels"]

# LCC
g = ig.Graph.Adjacency((A > 0).toarray().tolist(), mode="undirected")
components = g.connected_components(mode="weak")
lcc_idx = sorted(max(components, key=len))
A_lcc = A[np.ix_(lcc_idx, lcc_idx)]
labels_lcc = labels[lcc_idx]

import warnings
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = embcom.Node2VecMatrixFactorization(window_length=10, num_blocks=500)
        model.fit(A_lcc)
        emb = model.transform(dim=dim)
    if not np.isfinite(emb).all():
        raise ValueError("Embedding contains non-finite values")
    pred = KMeans(n_clusters=2, n_init=10, random_state=sample).fit_predict(emb)
    nmi = normalized_mutual_info_score(labels_lcc, pred)
except Exception as e:
    print(f"n2vec_mf failed (mu={mu:.2f}, sample={sample}): {e}")
    nmi = float("nan")

pd.DataFrame({
    "method": ["n2vec_mf"],
    "mu": [mu],
    "sample": [sample],
    "nmi": [nmi],
}).to_csv(output_file, index=False)

print(f"n2vec_mf: mu={mu:.2f}, sample={sample}, NMI={nmi:.4f}")
