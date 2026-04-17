# Optimal Multiharmonic Forcing Design for Homoclinic Chaos Control via Convex Optimization

**Authors:** Víctor M. Preciado (University of Pennsylvania) and Ricardo Chacón (Universidad de Extremadura)

**Paper:** *Nonlinear Dynamics* (Springer), 2026.

## Overview

This repository contains the optimization and simulation code for designing multiharmonic open-loop control signals that suppress homoclinic chaos in nonlinear oscillators. The framework bridges Melnikov theory and semidefinite programming (SDP) through three ingredients:

1. **Melnikov–Fourier representation.** Under multiharmonic forcing, the Melnikov function is a trigonometric polynomial whose coefficients are computable in closed form.
2. **Fejér–Riesz certificate.** Global nonnegativity of a univariate trigonometric polynomial admits an exact semidefinite certificate with no relaxation gap.
3. **Cartesian harmonic parametrization.** Amplitude–phase variables are replaced by Cartesian coefficients, rendering the synthesis problems (max-margin, min-power, chaos induction) as convex semidefinite or second-order cone programs.

Three case studies are included:

| System | Physical setting | Key result |
|---|---|---|
| **MEMS/Duffing** | Microelectromechanical resonator | 1.76× margin improvement, 27% power saving |
| **Josephson junction** | Superconducting circuit (damped pendulum) | 11% margin improvement, 8% power saving |
| **Ship roll** | Vessel roll in beam seas | 3.64× margin improvement, 62% power saving |

## Repository Structure

```
Melnikov_Chaos_Control/
├── data/
│   ├── parameters.json          # Physical parameters for all case studies
│   └── all_results.json         # Cached SDP results
├── scripts/
│   ├── generate_all_figures.py  # Main driver: SDPs + all figures
│   ├── sdp_solver.py            # CVXPy SDP formulation (max-margin, min-power)
│   ├── kernels.py               # Analytical response kernels (alpha_k, beta_k)
│   ├── dynamics.py              # ODE integration, Poincaré sections, Lyapunov exponents
│   └── rebuild_ship.py          # Ship-only regeneration script
├── figures/                     # Generated PDF figures (vector graphics)
└── README.md
```

## Installation

**Requirements:** Python 3.9+ with the following packages:

```bash
pip install numpy scipy matplotlib cvxpy
```

The SDP solver uses [SCS](https://github.com/cvxgrp/scs) (bundled with CVXPy).

## Usage

### Regenerate all figures and SDP results

```bash
cd Melnikov_Chaos_Control
python scripts/generate_all_figures.py
```

This runs all three case studies (MEMS/Duffing, Josephson, ship roll), solves the SDPs for N = 1, 2, 3, 5 harmonics, and generates the publication figures in `figures/`.

### Run a single case study

Each case study can be run independently. For example, to regenerate only the ship roll figures:

```bash
python scripts/rebuild_ship.py
```

### Modify parameters

Physical parameters are stored in `data/parameters.json`. For example, the MEMS/Duffing entry:

```json
{
  "duffing": {
    "delta": 0.80,
    "Omega": 1.0,
    "gamma_exist": [1.20, 0.60, 0.32],
    "eps": 0.25,
    "P_max": 2.40
  }
}
```

Modify these values and rerun `generate_all_figures.py` to explore different parameter regimes.

### Use the SDP solver directly

```python
from scripts.sdp_solver import solve_sdp

result = solve_sdp(
    c0=-1.0667,              # damping coefficient
    alpha=[0, 0, 0],         # response kernels (cosine)
    beta=[1.771, 0.767, 0.239],  # response kernels (sine)
    c_exist=[...],           # existing perturbation Fourier coefficients
    d_exist=[...],
    N=3,                     # number of control harmonics
    P_max=2.40,              # power budget
    objective='max_margin'   # or 'min_power'
)

print(f"Optimal margin: {result['margin']:.4f}")
print(f"Optimal power:  {result['power']:.4f}")
```

## Case Studies

### MEMS/Duffing oscillator

Hardening Duffing equation with three-harmonic cosine drive. The homoclinic orbit is $x_h(t) = \sqrt{2}\,\text{sech}(t)$, and the response kernels satisfy $\alpha_k = 0$, $\beta_k = \sqrt{2}\,\pi k\Omega / \cosh(\pi k\Omega/2)$. Under a power budget of $P_{\max} = 2.40$, multiharmonic control with $N = 3$ achieves the full cancellation margin $|c_0| = 1.0667$.

### Josephson junction

Damped driven pendulum (RCSJ model) with mixed cosine/sine microwave drive at three harmonics. The homoclinic orbit is $\phi_h(t) = 4\arctan(e^t) - \pi$, and the kernels satisfy $\alpha_k = 2\pi/\cosh(\pi k\Omega/2)$, $\beta_k = 0$. Under a tight power budget of $P_{\max} = 0.15$, multiharmonic control achieves an 11% margin improvement over the single-harmonic baseline.

### Ship roll in beam seas

Duffing-type oscillator with rescaled nonlinearity modeling vessel roll dynamics. Three-harmonic sine wave excitation at amplitudes $\gamma = [0.20, 0.15, 0.08]$. Under a tight fin authority budget of $P_{\max} = 0.05$, the single-harmonic design barely suppresses chaos ($\tau^* = 0.051$), while the multiharmonic design achieves a 3.64× larger margin.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{PreciadoChacon2026,
  author  = {Preciado, V{\'\i}ctor M. and Chac{\'o}n, Ricardo},
  title   = {Optimal Multiharmonic Forcing Design for Homoclinic
             Chaos Control via Convex Optimization},
  journal = {Nonlinear Dynamics},
  year    = {2026},
  note    = {To appear}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Víctor M. Preciado — [preciado@seas.upenn.edu](mailto:preciado@seas.upenn.edu)

Department of Electrical and Systems Engineering, University of Pennsylvania
