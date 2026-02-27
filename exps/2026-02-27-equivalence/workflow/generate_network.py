"""Generate a two-community SBM network."""
import sys
import numpy as np
import igraph as ig
import scipy.sparse as sp

if "snakemake" in sys.modules:
    n = int(snakemake.wildcards["n"])
    cave = float(snakemake.wildcards["cave"])
    mu = float(snakemake.wildcards["mu"])
    sample = int(snakemake.wildcards["sample"])
    net_file = snakemake.output["net_file"]
    node_file = snakemake.output["node_file"]
else:
    n = 500
    cave = 10
    mu = 0.3
    sample = 0
    net_file = "data/preprocessed/net_n~500_cave~10_mu~0.3_sample~0.npz"
    node_file = "data/preprocessed/node_n~500_cave~10_mu~0.3_sample~0.npz"

# Compute edge probabilities
N = 2 * n  # total nodes
c_out = mu * cave
c_in = 2 * cave - c_out
p_in = c_in / N
p_out = c_out / N

# Generate SBM using igraph
pref_matrix = [[p_in, p_out], [p_out, p_in]]
block_sizes = [n, n]
g = ig.Graph.SBM(pref_matrix, block_sizes)

# Convert to scipy sparse adjacency matrix
edges = g.get_edgelist()
if len(edges) > 0:
    rows, cols = zip(*edges)
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = A + A.T  # make symmetric (igraph gives upper triangle)
    A.data[:] = 1  # ensure binary
else:
    A = sp.csr_matrix((N, N))

# Community membership: first n nodes → 0, last n nodes → 1
membership = np.array([0] * n + [1] * n)

# Save
sp.save_npz(net_file, A)
np.savez(node_file, membership=membership)
