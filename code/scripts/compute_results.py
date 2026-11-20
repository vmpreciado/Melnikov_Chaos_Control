"""Optimization calculations for the three submitted case studies."""

import json
import os
from pathlib import Path
import numpy as np
from kernels import (duffing_c0, duffing_kernels_analytical,
                     josephson_c0, josephson_kernels_analytical,
                     helmholtz_c0, helmholtz_kernels_analytical)
from sdp_solver import solve_suppression_sdp

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def duffing_system():
    delta, Omega = 0.80, 1.0
    gamma = [1.20, 0.60, 0.32]
    Nmax = 5
    alpha = np.zeros(Nmax)
    beta = np.zeros(Nmax)
    for k in range(Nmax):
        alpha[k], beta[k] = duffing_kernels_analytical(k + 1, Omega)
    c_exist = np.zeros(3)                       # cosine forcing, alpha_k = 0
    d_exist = np.array([gamma[k] * beta[k] for k in range(3)])
    return dict(name='duffing', c0=duffing_c0(delta), alpha=alpha, beta=beta,
                c_exist=c_exist, d_exist=d_exist, Pmax=2.40, Ns=[1, 2, 3, 5])


def josephson_system():
    alpha_d, Omega = 0.10, 1.0
    gc = [0.30, 0.25, 0.20]
    gs = [0.20, 0.15, 0.10]
    Nmax = 5
    alpha = np.zeros(Nmax)
    beta = np.zeros(Nmax)
    for k in range(Nmax):
        alpha[k], beta[k] = josephson_kernels_analytical(k + 1, Omega)
    c_exist = np.array([gc[k] * alpha[k] for k in range(3)])
    d_exist = np.array([gs[k] * alpha[k] for k in range(3)])
    return dict(name='josephson', c0=josephson_c0(alpha_d, 0.0), alpha=alpha,
                beta=beta, c_exist=c_exist, d_exist=d_exist, Pmax=0.15,
                Ns=[1, 3, 5])


def helmholtz_system():
    delta, Om = 0.10, 1.0
    gamma = [0.10, 0.30, 0.15]
    Nmax = 5
    alpha = np.zeros(Nmax)
    beta = np.zeros(Nmax)
    for k in range(Nmax):
        alpha[k], beta[k] = helmholtz_kernels_analytical(k + 1, Om)
    c_exist = np.array([-gamma[k] * beta[k] for k in range(3)])   # sine forcing
    d_exist = np.zeros(3)
    return dict(name='helmholtz', c0=helmholtz_c0(delta), alpha=alpha, beta=beta,
                c_exist=c_exist, d_exist=d_exist, Pmax=0.06, Ns=[1, 2, 3, 5],
                gamma=gamma)


def compute_helmholtz():
    """Single-branch escape oscillator: max-margin at P_max, min-power at the
    ambitious target tau_min = (2/3)|c0| (N=1 infeasible), and the
    single-harmonic supremum over all power levels."""
    s = helmholtz_system()
    Ns = s['Ns']
    tau_target = round((2.0 / 3.0) * abs(s['c0']), 4)      # 0.08

    def mm(N, Pmax):
        return solve_suppression_sdp(
            s['c0'], s['c_exist'], s['d_exist'], s['alpha'][:N], s['beta'][:N],
            N, P_max=Pmax, objective='max_margin', both_branches=False)

    def mp(N, tau, Pmax=np.inf):
        return solve_suppression_sdp(
            s['c0'], s['c_exist'], s['d_exist'], s['alpha'][:N], s['beta'][:N],
            N, P_max=Pmax, objective='min_power', tau_min=tau, both_branches=False)

    from sdp_solver import evaluate_melnikov
    theta = np.linspace(0.0, 2.0 * np.pi, 40001)
    M_unc = evaluate_melnikov(theta, s['c0'], s['c_exist'], s['d_exist'])

    out = {'name': 'helmholtz', 'c0': float(s['c0']), 'abs_c0': float(abs(s['c0'])),
           'Pmax': s['Pmax'], 'Ns': Ns, 'single_branch': True,
           'tau_target': tau_target,
           'full_cancel_power': round(float(sum(g ** 2 for g in s['gamma'])), 4),
           'gamma': s['gamma'],
           'beta': [round(float(b), 4) for b in s['beta'][:3]],
           'cos_coeffs': [round(float(c), 4) for c in s['c_exist']],
           'M_unc_range': [round(float(M_unc.min()), 4), round(float(M_unc.max()), 4)],
           'max_margin': {'tau': {}}, 'min_power': {'P': {}}}

    for N in Ns:
        out['max_margin']['tau'][str(N)] = round(float(mm(N, s['Pmax'])['margin']), 4)

    for N in Ns:
        r = mp(N, tau_target)
        if r['energy'] is None or r['status'].startswith('infeasible'):
            out['min_power']['P'][str(N)] = 'INFEASIBLE'
        else:
            out['min_power']['P'][str(N)] = round(float(r['energy']), 4)

    # Single-harmonic supremum over all power levels, and the power at which
    # it is reached.
    sup = mm(1, 100.0)
    sup_tau_raw = float(sup['margin'])
    out['sh_sup_tau'] = round(sup_tau_raw, 4)
    rp = mp(1, sup_tau_raw)
    out['sh_sup_power'] = round(float(rp['energy']), 4) if rp['energy'] is not None else None
    return out


def compute(system):
    s = system
    Ns = s['Ns']
    out = {'name': s['name'], 'c0': float(s['c0']), 'abs_c0': float(abs(s['c0'])),
           'Pmax': s['Pmax'], 'Ns': Ns,
           'one_branch': {'tau': {}, 'tau_mirror': {}, 'solve_max': {}},
           'two_branch': {'tau': {}, 'Pmin': {}, 'solve_max': {}, 'solve_min': {}}}

    def mm(N, both):
        return solve_suppression_sdp(
            s['c0'], s['c_exist'], s['d_exist'], s['alpha'][:N], s['beta'][:N],
            N, P_max=s['Pmax'], objective='max_margin', both_branches=both)

    tau2_raw = {}
    for N in Ns:
        r1 = mm(N, False)
        out['one_branch']['tau'][str(N)] = round(float(r1['margin']), 4)
        out['one_branch']['tau_mirror'][str(N)] = round(float(r1['margin_L']), 4)
        out['one_branch']['solve_max'][str(N)] = round(float(r1['solve_time']), 3)
        r2 = mm(N, True)
        # Certified two-branch margin is the smaller of the two branch margins
        # (they coincide for the symmetric twin-well and pendulum cases).
        tau_both = min(float(r2['margin']), float(r2['margin_L']))
        tau2_raw[N] = tau_both
        out['two_branch']['tau'][str(N)] = round(tau_both, 4)
        out['two_branch']['solve_max'][str(N)] = round(float(r2['solve_time']), 3)

    tau_min = tau2_raw[Ns[0]]                            # unrounded tau*_{N=1}
    out['tau_min_matched'] = round(tau_min, 4)
    for N in Ns:
        rp = solve_suppression_sdp(
            s['c0'], s['c_exist'], s['d_exist'], s['alpha'][:N], s['beta'][:N],
            N, P_max=np.inf, objective='min_power', tau_min=tau_min,
            both_branches=True)
        if rp['energy'] is None:
            print(f"  [warn] {s['name']} min-power N={N}: status={rp['status']}")
            out['two_branch']['Pmin'][str(N)] = None
        else:
            out['two_branch']['Pmin'][str(N)] = round(float(rp['energy']), 4)
        out['two_branch']['solve_min'][str(N)] = round(float(rp['solve_time']), 3)
    return out


def compute_peak_demo():
    """MEMS peak-amplitude demonstration: max-margin at P_max with and without
    a peak constraint |u(t)| <= u_max (a normalized illustrative peak limit)."""
    s = duffing_system()
    N, umax = 5, 1.5

    def peak(a, b):
        t = np.linspace(0, 2 * np.pi, 4000)
        u = sum(a[k] * np.cos((k + 1) * t) + b[k] * np.sin((k + 1) * t)
                for k in range(len(a)))
        return float(np.max(np.abs(u)))

    r0 = solve_suppression_sdp(s['c0'], s['c_exist'], s['d_exist'], s['alpha'][:N],
                               s['beta'][:N], N, P_max=s['Pmax'],
                               objective='max_margin', both_branches=True)
    r1 = solve_suppression_sdp(s['c0'], s['c_exist'], s['d_exist'], s['alpha'][:N],
                               s['beta'][:N], N, P_max=s['Pmax'],
                               objective='max_margin', both_branches=True, peak_max=umax)
    out = {'N': N, 'Pmax': s['Pmax'], 'u_max': umax,
           'tau_meansquare': round(float(r0['margin']), 4),
           'peak_meansquare': round(peak(r0['a'], r0['b']), 4),
           'tau_peak': round(float(r1['margin']), 4),
           'peak_realized': round(peak(r1['a'], r1['b']), 4)}
    with open(os.path.join(RESULTS_DIR, 'peak_demo.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"wrote peak_demo.json: {out}")
    return out


def main():
    cases = (compute(duffing_system()), compute(josephson_system()), compute_helmholtz())
    for case in cases:
        path = RESULTS_DIR / (case["name"] + ".json")
        path.write_text(json.dumps(case, indent=2))
    compute_peak_demo()


if __name__ == "__main__":
    main()
