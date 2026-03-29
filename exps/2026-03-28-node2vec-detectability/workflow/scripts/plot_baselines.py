"""Combine per-sample NMI CSVs and plot NMI vs mu for all baseline methods."""
import sys
import os
import numpy as np
import pandas as pd

if "snakemake" in sys.modules:
    import matplotlib
    matplotlib.use("Agg")
    nmi_files = list(snakemake.input["nmi_files"])
    output_csv = snakemake.output["combined_csv"]
    output_fig = snakemake.output["figure"]
else:
    import glob
    nmi_files = glob.glob("../data/baselines/*/*.csv")
    output_csv = "../data/baselines/nmi_all.csv"
    output_fig = "../data/figs/nmi_vs_mu.png"

import matplotlib.pyplot as plt

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
os.makedirs(os.path.dirname(output_fig), exist_ok=True)

df = pd.concat([pd.read_csv(f) for f in nmi_files], ignore_index=True)
df.to_csv(output_csv, index=False)

# NMI mean ± std per (method, mu)
summary = df.groupby(["method", "mu"])["nmi"].agg(["mean", "std", "count"]).reset_index()
summary.columns = ["method", "mu", "nmi_mean", "nmi_std", "n"]

METHOD_COLORS = {
    "bp":       ("#e15759", "Belief Propagation"),
    "spectral": ("#4e79a7", "Spectral (eigsh)"),
    "node2vec": ("#59a14f", "node2vec (SGNS)"),
    "n2vec_mf": ("#f28e2b", "Node2Vec MF (SVD of NetMF)"),
}

fig, ax = plt.subplots(figsize=(7, 4.5))

mu_star = 1.0 - 1.0 / np.sqrt(5.0)
ax.axvline(mu_star, color="gray", linestyle="--", linewidth=1, label=f"μ* = {mu_star:.3f}")

for method in ["bp", "spectral", "node2vec", "n2vec_mf"]:
    sub = summary[summary["method"] == method].sort_values("mu")
    if sub.empty:
        continue
    color, label = METHOD_COLORS.get(method, ("#aaa", method))
    ax.plot(sub["mu"], sub["nmi_mean"], marker="o", color=color, label=label, linewidth=2)
    ax.fill_between(
        sub["mu"],
        sub["nmi_mean"] - sub["nmi_std"],
        sub["nmi_mean"] + sub["nmi_std"],
        alpha=0.15, color=color,
    )

ax.set_xlabel("Mixing parameter μ")
ax.set_ylabel("NMI (mean ± std)")
ax.set_title("Community detection NMI vs μ\n2-community SBM (N=2000, cave=5, 30 samples)")
ax.legend(fontsize=9)
ax.set_xlim(0.25, 0.58)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_fig, dpi=150, bbox_inches="tight")
print(f"Saved figure: {output_fig}")
print(f"Saved combined CSV: {output_csv}")

# Print summary table
print("\nNMI summary (mean ± std):")
pivot = summary.pivot(index="mu", columns="method", values="nmi_mean")
print(pivot.round(3).to_string())
