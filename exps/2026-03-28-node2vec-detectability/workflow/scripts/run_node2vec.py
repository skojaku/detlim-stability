"""Run node2vec (random walk + SGNS) on a saved network."""
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
    output_file = "../data/baselines/node2vec/n~2000_cave~5.0_mu~0.3_sample~000.csv"
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

model = embcom.Node2Vec(num_walks=10, walk_length=80, window_length=10, p=1.0, q=1.0)
model.fit(A_lcc)
emb = model.transform(dim=dim)

pred = KMeans(n_clusters=2, n_init=10, random_state=sample).fit_predict(emb)
nmi = normalized_mutual_info_score(labels_lcc, pred)

pd.DataFrame({
    "method": ["node2vec"],
    "mu": [mu],
    "sample": [sample],
    "nmi": [nmi],
}).to_csv(output_file, index=False)

print(f"node2vec: mu={mu:.2f}, sample={sample}, NMI={nmi:.4f}")
