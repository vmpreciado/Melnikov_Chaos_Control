"""
Head-to-head benchmark against Lenci and Rega (J. Vibration and Control 9,
281-315, 2003), Table 1.

Their optimal control of homoclinic bifurcation in the Helmholtz oscillator
maximizes the minimum over phase of the normalized Melnikov oscillatory part
cos(theta) + sum_{k>=2} f_k cos(k theta), with the fundamental fixed at 1. The
reported gain is the ratio of critical excitation amplitudes, G_N = 1/|min_theta|.

This is exactly a Fejer-Riesz nonnegativity SDP: minimize t such that
t + cos(theta) + sum_{k>=2} f_k cos(k theta) >= 0 for all theta, with G_N = 1/t.
The SDP recovers their optimal gains and coefficients by construction.
"""

import os
import sys
import json
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdp_solver import _fejer_riesz_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lenci and Rega (2003), Table 1: reported optimal gains for N harmonics.
LR_GAIN = {2: 1.4142, 3: 1.6180, 4: 1.7321, 5: 1.8019, 6: 1.8476}


def sdp_gain(N):
    """Optimal gain G_N = 1/t from the Fejer-Riesz SDP."""
    f = cp.Variable(N - 1)                       # f_2, ..., f_N
    t = cp.Variable()
    A = [1.0] + [f[k] for k in range(N - 1)]     # fundamental fixed at 1
    B = [0.0] * N
    cons = _fejer_riesz_block(t, A, B, N)
    cp.Problem(cp.Minimize(t), cons).solve(solver=cp.CLARABEL)
    return 1.0 / float(t.value), np.asarray(f.value)


def main():
    out = {'reference': 'Lenci and Rega, J. Vib. Control 9 (2003) Table 1',
           'gain_infinite': 2.0, 'rows': []}
    print(f"{'N':>2} {'SDP gain':>9} {'L&R gain':>9}")
    for N in sorted(LR_GAIN):
        g, f = sdp_gain(N)
        out['rows'].append({'N': N, 'gain_sdp': round(g, 4),
                            'gain_lr': LR_GAIN[N],
                            'coeffs_sdp': [round(x, 4) for x in f.tolist()]})
        print(f"{N:>2} {g:9.4f} {LR_GAIN[N]:9.4f}")
    with open(os.path.join(ROOT, 'results', 'lenci_rega.json'), 'w') as fp:
        json.dump(out, fp, indent=2)
    print('wrote results/lenci_rega.json')


if __name__ == '__main__':
    main()
