"""Combine per-run NMI results into a single CSV."""
import sys
import re
import numpy as np
import pandas as pd

if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    input_files = []
    output_file = "data/derived/combined_nmi_results.csv"

rows = []
for fpath in input_files:
    fname = fpath.split("/")[-1]

    # Parse algorithm name: "nmi_{algo}_{params}.npz"
    # The algo name is between the first underscore and the first parameter (key~val)
    m = re.match(r"nmi_([a-zA-Z_]+?)_[a-zA-Z]+~", fname)
    algo = m.group(1) if m else "unknown"

    # Extract parameters from filename
    params = {}
    for match in re.finditer(r"([a-zA-Z]+)~([\d.]+)", fname):
        key, val = match.group(1), match.group(2)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        params[key] = val

    data = np.load(fpath)
    params["algo"] = algo
    params["nmi"] = float(data["nmi"])
    if "loglik" in data:
        params["loglik"] = float(data["loglik"])
    rows.append(params)

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)
