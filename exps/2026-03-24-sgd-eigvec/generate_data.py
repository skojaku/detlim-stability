"""Generate SBM networks and save to data/. No analysis — just raw networks."""
import numpy as np
import igraph as ig
import scipy.sparse as sp
import os

# ------------------------------------------------------------------
# Parameters (set by Snakemake or run standalone)
# ------------------------------------------------------------------
try:
    n         = int(snakemake.params.n)
    cave      = float(snakemake.params.cave)
    mu        = float(snakemake.params.mu)
    n_samples = int(snakemake.params.n_samples)
    outfile   = snakemake.output[0]
except NameError:
    n, cave, mu, n_samples = 1000, 5.0, 0.5, 30
    outfile = "data/networks_n~1000_cave~5.0_mu~0.5.npz"

N     = 2 * n
c_out = mu * cave
c_in  = 2 * cave - c_out
p_in  = c_in / N
p_out = c_out / N

os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

nets = []
memberships = []
for s in range(n_samples):
    g     = ig.Graph.SBM([[p_in, p_out], [p_out, p_in]], [n, n])
    edges = g.get_edgelist()
    if edges:
        rows, cols = zip(*edges)
        A = sp.csr_matrix(
            (np.ones(len(rows)), (np.array(rows), np.array(cols))),
            shape=(N, N),
        )
        A = A + A.T
        A.data[:] = 1.0
    else:
        A = sp.csr_matrix((N, N))
    nets.append(A)

membership = np.array([0] * n + [1] * n)

np.savez(
    outfile,
    membership=membership,
    n_nodes=np.array([A.shape[0] for A in nets]),
    n=n, N=N, cave=cave, mu=mu, n_samples=n_samples,
)
# Save each network separately (npz can't hold multiple sparse matrices)
outdir = os.path.dirname(outfile)
for s, A in enumerate(nets):
    sp.save_npz(os.path.join(outdir, f"net_{s:03d}.npz"), A)

print(f"Saved {n_samples} networks to {outdir}/")
print(f"  N={N}, cave={cave}, mu={mu}")
print(f"  p_in={p_in:.5f}, p_out={p_out:.5f}")
