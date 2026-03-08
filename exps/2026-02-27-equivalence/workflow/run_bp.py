"""Belief propagation community detection."""
import sys
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score
import belief_propagation

sys.path.insert(0, "workflow")
from loglikelihood import compute_sbm_loglikelihood

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", required=True)
    parser.add_argument("--node-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--cave", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    args = parser.parse_args()
    net_file = args.net_file
    node_file = args.node_file
    output_file = args.output_file
    n_per_comm = args.n
    cave = args.cave
    mu = args.mu

    # Reconstruct ground-truth p and q from parameters
    N = 2 * n_per_comm
    c_out = mu * cave
    c_in = 2 * cave - c_out
    p_true = c_in / N
    q_true = c_out / N

    # Load data
    A = sp.load_npz(net_file)
    node_data = np.load(node_file)
    membership = node_data["membership"]

    # Belief propagation with q=2 communities, initialized with ground-truth membership
    labels = belief_propagation.detect(A, q=2, init_memberships=membership)
    nmi = normalized_mutual_info_score(membership, labels)

    # Log-likelihood
    s = 2.0 * labels - 1.0
    loglik = compute_sbm_loglikelihood(A, s, p_true, q_true)

    np.savez(output_file, nmi=nmi, loglik=loglik)
