"""
Phase 4 dynamical validation suite.

  4.1 Josephson Poincare sections and largest Lyapunov exponent
      (uncontrolled, N=1, N=3 two-branch designs).
  4.2 Helmholtz safe-basin integrity (escape diagnostics) and Poincare
      sections for non-escaping responses.
  4.3 Margin-meaning experiment: scale the existing forcing by kappa >= 1 and
      compare the first-order critical kappa (band fails, analytic) with the
      simulated critical kappa (chaos or escape returns), across designs.
  4.4 Duffing epsilon sweep: suppression persists over a range of eps.

Scalar outputs go to results/validation.json; figures are regenerated here.
Escape systems use escape diagnostics, not Lyapunov exponents.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import (duffing_c0, duffing_kernels_analytical,
                     josephson_c0, josephson_kernels_analytical,
                     helmholtz_c0, helmholtz_kernels_analytical)
from sdp_solver import solve_suppression_sdp, verified_margins
from dynamics import (duffing_rhs_with_variational, pendulum_rhs_with_variational,
                      helmholtz_rhs, pendulum_rhs, compute_poincare_section,
                      compute_lyapunov_exponent)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'figures')
RESULTS = os.path.join(ROOT, 'results')
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'lines.linewidth': 1.2})
C = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ----------------------------------------------------------------------
# Case-study data, designs, and scaled Melnikov coefficients
# ----------------------------------------------------------------------
def _kernels(fn, Om, n=5):
    al = np.zeros(n); be = np.zeros(n)
    for k in range(n):
        al[k], be[k] = fn(k + 1, Om)
    return al, be


def duffing():
    al, be = _kernels(duffing_kernels_analytical, 1.0)
    gam = [1.20, 0.60, 0.32]
    return dict(name='duffing', c0=duffing_c0(0.80), al=al, be=be,
                c_ex=np.zeros(3), d_ex=np.array([gam[k] * be[k] for k in range(3)]),
                exist_ab=[(gam[k], 0.0) for k in range(3)], Pmax=2.40,
                eps=0.25, delta=0.80, both=True)


def josephson():
    al, be = _kernels(josephson_kernels_analytical, 1.0)
    gc = [0.30, 0.25, 0.20]; gs = [0.20, 0.15, 0.10]
    return dict(name='josephson', c0=josephson_c0(0.10, 0.0), al=al, be=be,
                c_ex=np.array([gc[k] * al[k] for k in range(3)]),
                d_ex=np.array([gs[k] * al[k] for k in range(3)]),
                exist_ab=[(gc[k], gs[k]) for k in range(3)], Pmax=0.15,
                eps=0.10, delta=0.10, both=True)


def helmholtz():
    al, be = _kernels(helmholtz_kernels_analytical, 1.0)
    gam = [0.10, 0.30, 0.15]
    return dict(name='helmholtz', c0=helmholtz_c0(0.10), al=al, be=be,
                c_ex=np.array([-gam[k] * be[k] for k in range(3)]), d_ex=np.zeros(3),
                exist_ab=[(0.0, gam[k]) for k in range(3)], Pmax=0.06,
                eps=0.10, delta=0.10, both=False)


def design(s, N):
    """Two-branch (or single-branch) max-margin control coefficients (a, b)."""
    r = solve_suppression_sdp(s['c0'], s['c_ex'], s['d_ex'], s['al'][:N], s['be'][:N],
                              N, P_max=s['Pmax'], objective='max_margin',
                              both_branches=s['both'])
    return np.asarray(r['a']), np.asarray(r['b'])


def forcing_coeffs(s, a, b, N, kappa=1.0):
    """Existing forcing scaled by kappa, plus the fixed control design."""
    co = []
    for k in range(3):
        ak, bk = s['exist_ab'][k]
        ak *= kappa; bk *= kappa
        if a is not None and k < N:
            ak += a[k]; bk += b[k]
        co.append((ak, bk))
    return co


def scaled_margin(s, a, b, N, kappa):
    """Certified branch margin of a fixed design when the existing forcing is
    scaled by kappa (analytic, first order)."""
    K = 3
    c_f = np.zeros(K); d_f = np.zeros(K)
    for k in range(K):
        c_f[k] = kappa * s['c_ex'][k]
        d_f[k] = kappa * s['d_ex'][k]
        if a is not None and k < N:
            c_f[k] += a[k] * s['al'][k] - b[k] * s['be'][k]
            d_f[k] += b[k] * s['al'][k] + a[k] * s['be'][k]
    mR, mL = verified_margins(s['c0'], c_f, d_f)
    return min(mR, mL) if s['both'] else mR


def critical_kappa_analytic(s, a, b, N):
    """Smallest kappa >= 1 at which the certified margin reaches zero."""
    lo, hi = 1.0, 1.0
    if scaled_margin(s, a, b, N, 1.0) <= 0:
        return 1.0
    while scaled_margin(s, a, b, N, hi) > 0 and hi < 100:
        hi *= 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if scaled_margin(s, a, b, N, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Escape diagnostics (Helmholtz)
# ----------------------------------------------------------------------
def escapes(fc, ic, eps, delta, Om=1.0, horizon=120, phi_esc=1.5):
    """True if phi crosses the escape threshold within the horizon."""
    T = 2 * np.pi / Om

    def ev(t, z):
        return z[0] - phi_esc
    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(lambda t, z: helmholtz_rhs(t, z, eps, delta, fc, Om),
                    [0, horizon * T], ic, method='DOP853', rtol=1e-6, atol=1e-6,
                    events=ev, max_step=T)
    return len(sol.t_events[0]) > 0 or np.max(np.abs(sol.y[0])) > phi_esc


def safe_fraction(s, a, b, N, kappa, grid=(25, 25), extent=(-0.6, 0.9, -0.6, 0.6)):
    fc = forcing_coeffs(s, a, b, N, kappa)
    x = np.linspace(extent[0], extent[1], grid[0])
    y = np.linspace(extent[2], extent[3], grid[1])
    safe = 0; total = 0
    mask = np.zeros((grid[1], grid[0]), dtype=bool)
    for j, y0 in enumerate(y):
        for i, x0 in enumerate(x):
            esc = escapes(fc, [x0, y0], s['eps'], s['delta'], phi_esc=1.5)
            mask[j, i] = not esc
            safe += (not esc); total += 1
    return safe / total, mask, x, y


def lyap_duffing(s, fc, e):
    l, _, _ = compute_lyapunov_exponent(
        duffing_rhs_with_variational, (e, s['delta'], fc, 1.0), [1.0, 0.0], 1.0,
        n_transient=300, n_compute=900)
    return float(l)


def lyap_pendulum(s, fc, n_compute=1200):
    l, _, _ = compute_lyapunov_exponent(
        pendulum_rhs_with_variational, (s['eps'], s['delta'], 0.0, fc, 1.0),
        [0.0, 0.0], 1.0, n_transient=400, n_compute=n_compute)
    return float(l)


# ----------------------------------------------------------------------
# 4.4 Duffing epsilon sweep
# ----------------------------------------------------------------------
def task_duffing_eps(out):
    print('[4.4] Duffing epsilon sweep')
    s = duffing(); a, b = design(s, 5)
    eps_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    unc, ctrl = [], []
    for e in eps_list:
        unc.append(lyap_duffing(s, forcing_coeffs(s, None, None, 0), e))
        ctrl.append(lyap_duffing(s, forcing_coeffs(s, a, b, 5), e))
        print(f'   eps={e}: unc={unc[-1]:.3f} ctrl={ctrl[-1]:.3f}')
    out['duffing_eps'] = {'eps': eps_list, 'lambda_unc': unc, 'lambda_ctrl': ctrl}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color='gray', lw=0.5)
    ax.plot(eps_list, unc, 'o-', color=C[3], label='Uncontrolled')
    ax.plot(eps_list, ctrl, 's-', color=C[2], label='Controlled ($N=5$)')
    ax.axvline(0.25, color='gray', ls=':', lw=0.8, label=r'$\epsilon=0.25$ (nominal)')
    ax.set_xlabel(r'Perturbation strength $\epsilon$')
    ax.set_ylabel(r'Largest Lyapunov exponent $\lambda_{\max}$')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_duffing_eps.pdf')); plt.close(fig)


# ----------------------------------------------------------------------
# 4.1 Josephson Poincare + Lyapunov (nominal)
# ----------------------------------------------------------------------
def task_josephson(out):
    print('[4.1] Josephson Poincare + Lyapunov')
    s = josephson()
    a1, b1 = design(s, 1); a3, b3 = design(s, 3)
    configs = [('Uncontrolled', None, None, 0, C[3]),
               ('Controlled ($N=1$)', a1, b1, 1, C[0]),
               ('Controlled ($N=3$)', a3, b3, 3, C[1])]
    lam = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (lbl, a, b, N, col) in zip(axes, configs):
        fc = forcing_coeffs(s, a, b, N)
        lam[lbl] = lyap_pendulum(s, fc)
        allpx, allpy = [], []
        for ic in [[0.0, 0.0], [0.5, 0.3], [-0.5, -0.3], [1.0, 0.5], [-1.0, 0.5]]:
            px, py = compute_poincare_section(
                pendulum_rhs, (s['eps'], s['delta'], 0.0, fc, 1.0), ic, 1.0, s['eps'],
                n_transient=300, n_collect=600, rtol=1e-10, atol=1e-10)
            allpx.extend(px); allpy.extend(py)
        ax.scatter(allpx, allpy, s=0.5, color=col, alpha=0.5, rasterized=True)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel(r'$\phi$'); ax.set_ylabel(r'$\dot{\phi}$')
        ax.text(0.05, 0.95, rf'$\lambda_{{\max}}={lam[lbl]:.3f}$', transform=ax.transAxes,
                va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        print(f'   {lbl}: lambda={lam[lbl]:.4f}')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig7_josephson_validation.pdf')); plt.close(fig)
    out['josephson'] = {'lambda_nominal': lam}


# ----------------------------------------------------------------------
# 4.2 Helmholtz safe-basin integrity
# ----------------------------------------------------------------------
def task_helmholtz_basin(out):
    print('[4.2] Helmholtz safe-basin integrity')
    s = helmholtz(); a2, b2 = design(s, 2)
    kappas = [1.0, 2.5, 4.0, 5.5]
    frac_unc, frac_ctrl = [], []
    for kp in kappas:
        fu, _, _, _ = safe_fraction(s, None, None, 0, kp, grid=(13, 13))
        fc, _, _, _ = safe_fraction(s, a2, b2, 2, kp, grid=(13, 13))
        frac_unc.append(fu); frac_ctrl.append(fc)
        print(f'   kappa={kp}: safe unc={fu:.3f} ctrl={fc:.3f}')
    # pick the stress level of maximum contrast for the basin picture
    k_star = kappas[int(np.argmax(np.array(frac_ctrl) - np.array(frac_unc)))]
    _, mu, x, y = safe_fraction(s, None, None, 0, k_star, grid=(30, 30))
    _, mc, _, _ = safe_fraction(s, a2, b2, 2, k_star, grid=(30, 30))
    out['helmholtz_basin'] = {'kappa': kappas, 'safe_unc': frac_unc,
                              'safe_ctrl': frac_ctrl, 'k_star': k_star}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, mask, lbl in [(axes[0], mu, f'Uncontrolled ($\\kappa={k_star:g}$)'),
                          (axes[1], mc, f'Controlled $N=2$ ($\\kappa={k_star:g}$)')]:
        ax.imshow(mask, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                  aspect='auto', cmap='Greens', vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=10); ax.set_xlabel(r'$\phi$'); ax.set_ylabel(r'$\dot{\phi}$')
        ax.axvline(1.0, color='red', lw=0.8, ls='--')
    axes[2].plot(kappas, frac_unc, 'o-', color=C[3], label='Uncontrolled')
    axes[2].plot(kappas, frac_ctrl, 's-', color=C[2], label='Controlled ($N=2$)')
    axes[2].set_xlabel(r'Forcing scale $\kappa$'); axes[2].set_ylabel('Safe-basin fraction')
    axes[2].legend(fontsize=8); axes[2].set_title('Basin integrity', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig10_helmholtz_basin.pdf')); plt.close(fig)


# ----------------------------------------------------------------------
# 4.3 Margin-meaning experiment
# ----------------------------------------------------------------------
def sim_kappa_lyap(s, a, b, N, lyap_fn, thr=0.02, kmax=3.0, step=0.15):
    prev_k, prev_l = 1.0, None
    for kp in np.arange(1.0, kmax + 1e-9, step):
        l = lyap_fn(s, forcing_coeffs(s, a, b, N, kp))
        if l > thr:
            if prev_l is not None and prev_l <= thr:
                return prev_k + (thr - prev_l) / (l - prev_l) * (kp - prev_k)
            return float(kp)
        prev_k, prev_l = float(kp), l
    return None


def sim_kappa_escape(s, a, b, N, drop=0.85, kmax=6.0, step=0.5):
    """Critical kappa where the safe-basin fraction erodes to `drop` times its
    value at kappa=1 (relative erosion, since the grid rectangle is larger than
    the safe basin)."""
    f1, _, _, _ = safe_fraction(s, a, b, N, 1.0, grid=(13, 13))
    thr = drop * f1
    prev_k, prev_f = 1.0, f1
    for kp in np.arange(1.5, kmax + 1e-9, step):
        f, _, _, _ = safe_fraction(s, a, b, N, kp, grid=(13, 13))
        if f < thr:
            return prev_k + (prev_f - thr) / max(prev_f - f, 1e-9) * (kp - prev_k)
        prev_k, prev_f = float(kp), f
    return None


def task_margin_meaning(out):
    print('[4.3] Margin-meaning experiment')
    data = []
    for build, Ns in ((duffing, [1, 2, 3, 5]), (josephson, [1, 3, 5]),
                      (helmholtz, [1, 2, 3])):
        s = build()
        for N in Ns:
            a, b = design(s, N)
            tau = scaled_margin(s, a, b, N, 1.0)
            k1 = critical_kappa_analytic(s, a, b, N)
            if s['name'] == 'duffing':
                ks = sim_kappa_lyap(s, a, b, N, lambda ss, fc: lyap_duffing(ss, fc, ss['eps']))
            elif s['name'] == 'josephson':
                ks = sim_kappa_lyap(s, a, b, N, lambda ss, fc: lyap_pendulum(ss, fc, 800), kmax=3.5)
            else:
                ks = sim_kappa_escape(s, a, b, N)
            data.append({'system': s['name'], 'N': N, 'tau': round(tau, 4),
                         'kappa_1st': round(k1, 3),
                         'kappa_sim': round(ks, 3) if ks else None})
            print(f"   {s['name']} N={N}: tau={tau:.4f} k1={k1:.3f} ksim={ks}")
    out['margin_meaning'] = data

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    marks = {'duffing': ('o', C[0], 'MEMS/Duffing'),
             'josephson': ('s', C[1], 'Josephson'),
             'helmholtz': ('^', C[2], 'Helmholtz')}
    for name, (mk, col, lab) in marks.items():
        pts = [d for d in data if d['system'] == name]
        taus = [d['tau'] for d in pts]
        ax.plot(taus, [d['kappa_1st'] for d in pts], mk + '-', color=col, label=lab + ' (first order)')
        ks = [(d['tau'], d['kappa_sim']) for d in pts if d['kappa_sim']]
        if ks:
            ax.plot([t for t, _ in ks], [k for _, k in ks], mk + '--', color=col, alpha=0.6,
                    markerfacecolor='white', label=lab + ' (simulated)')
    ax.set_xlabel(r'Certified margin $\tau$')
    ax.set_ylabel(r'Critical forcing scale $\kappa_{\mathrm{crit}}$')
    ax.axhline(1.0, color='gray', lw=0.4)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig11_margin_meaning.pdf')); plt.close(fig)


def main():
    out = {}
    path = os.path.join(RESULTS, 'validation.json')

    def save():
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
    task_duffing_eps(out); save()
    task_josephson(out); save()
    task_margin_meaning(out); save()
    task_helmholtz_basin(out); save()
    print('DONE; wrote', path)


if __name__ == '__main__':
    main()
