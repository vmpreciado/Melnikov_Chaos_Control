# Online Resource 1. Numerical data and reproduction code

**Article**  
Optimal Multiharmonic Forcing Design for Homoclinic Chaos Control via Convex Optimization

**Journal**  
Nonlinear Dynamics

**Author and corresponding author**  
Víctor M. Preciado  
Department of Electrical and Systems Engineering, University of Pennsylvania, Philadelphia, PA, USA  
preciado@seas.upenn.edu

This supplement accompanies revision 12. It contains the semidefinite synthesis code, archived numerical results, plotted figures, dynamical diagnostics and numerical table inputs for the three case studies. All parameters, data and code required by the reproduction commands are included. The manuscript and its editable LaTeX source are supplied separately.

## Reproduce the principal results

Extract the archive to an empty directory. From that directory, use Python 3.12 to create an environment and install the pinned scientific libraries. These commands are for macOS and Linux.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --output ../chaos-control-reproduced
```

On Windows use `py -3.12 -m venv .venv` and replace `.venv/bin/python` with `.venv\Scripts\python.exe`. The output directory must be new. Omitting `--output` uses a temporary directory. The calculation begins by verifying the SHA-256 hashes of the distributed files.

The command re-solves the single- and two-branch design problems, peak-amplitude example and Lenci–Rega benchmark. It compares the optimization results with the archived values to one unit in the final reported decimal place, with a floating-point allowance. Reported solve times are machine dependent. The command also tests the analytic identities and Helmholtz ceiling, regenerates the graphical abstract and stress plot, writes LaTeX tables, and checks the archived dynamical and quadrature diagnostics.

At the Helmholtz budget of 0.06, the single-harmonic ceiling is 0.0355 and the two-harmonic margin is 0.0982. Two harmonics reach the prescribed margin of 0.08 at a minimum power of 0.0355. With all three disturbance harmonics represented, full cancellation requires power 0.1225 and attains the limiting margin 0.12. The two occurrences of 0.0355 denote different quantities rounded to four decimal places.

## Regenerate additional results

Each flag below extends the principal calculation. The flags can be combined. Pass a fresh output directory for each run.

| Flag | Computation |
|---|---|
| `--figures` | Re-solve the Duffing, Josephson and Helmholtz power sweeps and generate the original design, waveform and scaling plots. |
| `--diagnostics` | Repeat the two-precision kernel quadratures, forced-saddle manifold continuation and tracing, Duffing sections and Lyapunov exponents, and settled Josephson sections and Floquet calculations. |
| `--scans` | Repeat the Duffing perturbation-strength sweep, the forcing-stress scan and the Helmholtz finite-horizon survival grids. The original scan script also computes a short Josephson diagnostic. The manuscript's settled Josephson results come from `--diagnostics`. |

The dynamics flags run numerical integrations and can take substantially longer than the principal calculation. Finite-time quantities, chaotic trajectories and solve timings can vary with the numerical environment.

## Data and program guide

| Path | Contents |
|---|---|
| `scripts/compute_results.py`, `scripts/kernels.py`, `scripts/sdp_solver.py` | Model parameters, Fourier kernels and optimization constraints for the tabulated comparisons. |
| `results/duffing.json`, `results/josephson.json`, `results/helmholtz.json` | Fixed-budget margins, matched-target powers, branch comparisons and case-specific outputs. |
| `results/peak_demo.json`, `results/lenci_rega.json` | Peak-constrained design and comparison with the published variational benchmark. |
| `results/validation.json`, `scripts/validation.py` | Original perturbation-strength, stress and survival sweep summaries and their generator. Null Josephson transition entries denote a non-detection on the sampled range. |
| `results/v11/`, `scripts/v11_*.py` | Detailed diagnostics used in revision 12, including return-map samples, exponents, intersection residuals, traced manifold coordinates, quadrature checks, the analytic ceiling and stress-plot provenance. The suffix records the calculation's provenance. |
| `figures/`, `scripts/make_design_figures.py`, `scripts/v12/graphical_abstract.py` | Manuscript plots, the original design-plot generator and the standalone graphical abstract generator. |
| `numbers.tex`, `tab_*.tex`, `revision_v11/tab_kernel_verification.tex` | Numerical LaTeX inputs. The tables include the timing measurements from the archived runs. |
| `requirements.txt`, `LICENSE`, `sha256.json` | Tested scientific package versions, code license and checksums of the distributed files. |

Open `manifold_curves.npz` with `numpy.load(..., allow_pickle=False)`. Array names encode the uncontrolled or controlled case, stable or unstable branch, and seed direction. A suffix `_q` marks the parameter samples and `_xy` the corresponding section coordinates. All other numerical data files use JSON.
