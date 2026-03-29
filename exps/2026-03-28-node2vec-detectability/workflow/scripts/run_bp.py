"""Run Belief Propagation on a saved network. Output NMI as single-row CSV."""
import sys
import numpy as np
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, "/workspace/libs/BeliefPropagation")
import belief_propagation

if "snakemake" in sys.modules:
    network_file = snakemake.input["network_file"]
    output_file = snakemake.output["nmi_file"]
    mu = float(snakemake.params["mu"])
    sample = int(snakemake.params["sample"])
else:
    network_file = "../data/networks/n~2000_cave~5.0_mu~0.3/sample~000.npz"
    output_file = "../data/baselines/bp/n~2000_cave~5.0_mu~0.3_sample~000.csv"
    mu = 0.3
    sample = 0

import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

d = np.load(network_file)
A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
labels = d["labels"]

from sklearn.metrics import normalized_mutual_info_score

try:
    pred = belief_propagation.detect(A.copy(), q=2)
    nmi = normalized_mutual_info_score(labels, pred)
except Exception as e:
    print(f"BP failed: {e}")
    nmi = float("nan")

pd.DataFrame({
    "method": ["bp"],
    "mu": [mu],
    "sample": [sample],
    "nmi": [nmi],
}).to_csv(output_file, index=False)

print(f"BP: mu={mu:.2f}, sample={sample}, NMI={nmi:.4f}")
