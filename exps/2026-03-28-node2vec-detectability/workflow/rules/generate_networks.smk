rule generate_network:
    output:
        network_file = NETWORK_FILE,
    params:
        n    = lambda wc: int(wc.n),
        cave = lambda wc: float(wc.cave),
        mu   = lambda wc: float(wc.mu),
        sample = lambda wc: int(wc.sample),
    script:
        "../scripts/generate_network.py"


rule generate_networks_all:
    input:
        expand(NETWORK_FILE, NET_PARAMS, sample=SAMPLE_IDS),
