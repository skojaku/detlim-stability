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
    # Extract parameters from filename
    fname = fpath.split("/")[-1]
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
    params["nmi_spectral"] = float(data["nmi_spectral"])
    params["nmi_sbm"] = float(data["nmi_sbm"])
    rows.append(params)

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)
