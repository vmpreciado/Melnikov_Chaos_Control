# Optimal Multiharmonic Forcing Design for Homoclinic Chaos Control via Convex Optimization

**Víctor M. Preciado** · University of Pennsylvania · Preprint, version 12 · November 19, 2026

[Read the preprint](https://vmpreciado.github.io/Melnikov_Chaos_Control/) · [Single-column PDF](docs/main_v12.pdf) · [Code and numerical data](code/) · [Online Resource 1 (ZIP)](https://github.com/vmpreciado/Melnikov_Chaos_Control/releases/download/v12/ESM_1.zip) · [BibTeX citation](CITATION.bib)

How much can periodic forcing change the separation of stable and unstable manifolds near a homoclinic orbit? This paper gives an exact semidefinite formulation for shaping the first-order Melnikov function under a power budget. In the Helmholtz example, the best fundamental-frequency control has a margin ceiling of 0.0355 at any power. Adding the second harmonic makes a margin of 0.08 attainable.

## Reproduce the results

The [version 12 reproduction guide](code/README.md) identifies the archived results for the Duffing, Josephson, and Helmholtz systems and gives the commands for the longer integrations. For the principal checks, use Python 3.12 and run these commands from the repository root on macOS or Linux.

```sh
cd code
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --output ../chaos-control-reproduced
```

Choose a new output directory for each run. The command verifies the distributed data, re-solves the optimization problems, and compares the resulting values with the paper.

## Code provenance

The top-level `data/`, `figures/`, and `scripts/` directories belong to an earlier implementation. The [version 12 code](code/) reproduces the public preprint. The original code credits Víctor M. Preciado and Ricardo Chacón and is distributed under the [MIT software license](code/LICENSE).
