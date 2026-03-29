rule run_bp:
    input:
        network_file = NETWORK_FILE,
    output:
        nmi_file = NMI_FILE_BP,
    params:
        mu     = lambda wc: float(wc.mu),
        sample = lambda wc: int(wc.sample),
    script:
        "../scripts/run_bp.py"


rule run_spectral:
    input:
        network_file = NETWORK_FILE,
    output:
        nmi_file = NMI_FILE_SPECTRAL,
    params:
        mu     = lambda wc: float(wc.mu),
        sample = lambda wc: int(wc.sample),
        dim    = DIM,
    script:
        "../scripts/run_spectral.py"


rule run_node2vec:
    input:
        network_file = NETWORK_FILE,
    output:
        nmi_file = NMI_FILE_NODE2VEC,
    params:
        mu     = lambda wc: float(wc.mu),
        sample = lambda wc: int(wc.sample),
        dim    = DIM,
    script:
        "../scripts/run_node2vec.py"


rule run_n2vec_mf:
    input:
        network_file = NETWORK_FILE,
    output:
        nmi_file = NMI_FILE_N2VEC_MF,
    params:
        mu     = lambda wc: float(wc.mu),
        sample = lambda wc: int(wc.sample),
        dim    = DIM,
    script:
        "../scripts/run_n2vec_mf.py"


rule plot_baselines:
    input:
        nmi_files = (
            expand(NMI_FILE_BP,       NET_PARAMS, sample=SAMPLE_IDS) +
            expand(NMI_FILE_SPECTRAL, NET_PARAMS, sample=SAMPLE_IDS) +
            expand(NMI_FILE_NODE2VEC, NET_PARAMS, sample=SAMPLE_IDS) +
            expand(NMI_FILE_N2VEC_MF, NET_PARAMS, sample=SAMPLE_IDS)
        ),
    output:
        combined_csv = BASELINES_COMBINED,
        figure       = BASELINES_FIG,
    script:
        "../scripts/plot_baselines.py"


rule baselines_all:
    input:
        rules.plot_baselines.output,
