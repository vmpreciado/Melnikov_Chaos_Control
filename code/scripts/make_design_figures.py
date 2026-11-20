"""
Regenerate the SDP design and power-margin figures under the two-branch
certificate: fig1 (Duffing design), fig1b (Duffing power-margin),
fig4 (Duffing performance scaling), fig6 (Josephson design),
fig8 (Josephson power-margin).

Melnikov panels show the two-sided band of eq. (band); power-margin panels
plot the two-branch margin tau_both(P). Dynamical-validation figures
(Poincare, manifolds) are left for the Phase 4 revalidation.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import (duffing_c0, duffing_kernels_analytical,
                     josephson_c0, josephson_kernels_analytical,
                     helmholtz_c0, helmholtz_kernels_analytical)
from sdp_solver import solve_suppression_sdp, evaluate_melnikov

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
                     'text.usetex': False, 'font.family': 'serif',
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = plt.rcParams['axes.prop_cycle'].by_key()['color']
C_UNC, C_N1, C_N3, C_N5 = C[3], C[0], C[1], C[2]
THETA = np.linspace(0, 2 * np.pi, 1000)


def duffing_data():
    delta, Om = 0.80, 1.0
    gamma = [1.20, 0.60, 0.32]
    Nmax = 10
    al = np.zeros(Nmax); be = np.zeros(Nmax)
    for k in range(Nmax):
        al[k], be[k] = duffing_kernels_analytical(k + 1, Om)
    c_ex = np.zeros(3)
    d_ex = np.array([gamma[k] * be[k] for k in range(3)])
    return dict(c0=duffing_c0(delta), al=al, be=be, c_ex=c_ex, d_ex=d_ex,
                Pmax=2.40, Om=Om, gamma=gamma, ftype='cos')


def josephson_data():
    ad, Om = 0.10, 1.0
    gc = [0.30, 0.25, 0.20]; gs = [0.20, 0.15, 0.10]
    Nmax = 10
    al = np.zeros(Nmax); be = np.zeros(Nmax)
    for k in range(Nmax):
        al[k], be[k] = josephson_kernels_analytical(k + 1, Om)
    c_ex = np.array([gc[k] * al[k] for k in range(3)])
    d_ex = np.array([gs[k] * al[k] for k in range(3)])
    return dict(c0=josephson_c0(ad, 0.0), al=al, be=be, c_ex=c_ex, d_ex=d_ex,
                Pmax=0.15, Om=Om)


def solve(d, N, objective, Pmax=None, tau_min=0.01, both=True):
    return solve_suppression_sdp(
        d['c0'], d['c_ex'], d['d_ex'], d['al'][:N], d['be'][:N], N,
        P_max=(d['Pmax'] if Pmax is None else Pmax), objective=objective,
        tau_min=tau_min, both_branches=both)


def M_unc(d):
    return evaluate_melnikov(THETA, d['c0'], d['c_ex'], d['d_ex'])


def draw_band(ax, c0, tau):
    """Dashed edges of the band tau <= -M <= 2|c0| - tau."""
    ax.axhline(-tau, color='gray', lw=0.8, ls='--', alpha=0.8)
    ax.axhline(-(2 * abs(c0) - tau), color='gray', lw=0.8, ls='--', alpha=0.8)


def melnikov_axis(ax):
    ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.set_xlim([0, 2 * np.pi]); ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])


def power_margin(ax, d, Ns, Pgrid, Pmax):
    for N, col, lab in Ns:
        margins = []
        for P in Pgrid:
            r = solve(d, N, 'max_margin', Pmax=P)
            m = r['margin'] if r['margin'] is not None else 0.0
            mL = r['margin_L'] if r['margin_L'] is not None else 0.0
            margins.append(max(min(m, mL), 0.0))
        ax.plot(Pgrid, margins, color=col, label=lab)
    ax.axvline(Pmax, color='gray', ls='--', lw=0.8, label=f'Budget ({Pmax})')
    ax.axhline(abs(d['c0']), color='black', ls=':', lw=0.8,
               label=rf'$|c_0|={abs(d["c0"]):.4f}$')
    ax.axhline(0, color='gray', lw=0.3)
    ax.set_xlabel(r'Total control power $P$')
    ax.set_ylabel(r'Two-branch margin $\tau_{\mathrm{both}}$')
    ax.legend(fontsize=8)


def waveform(ax, res, Om):
    a, b = np.array(res['a']), np.array(res['b'])
    t = np.linspace(0, 2 * np.pi / Om, 500)
    u = sum(a[k] * np.cos((k + 1) * Om * t) + b[k] * np.sin((k + 1) * Om * t)
            for k in range(len(a)))
    ax.plot(t, u, color=C_N5)
    ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$u(t)$'); ax.set_xlim([t[0], t[-1]])


def panel_tag(ax, s):
    ax.text(-0.12, 1.05, s, transform=ax.transAxes, fontsize=12, fontweight='bold')


def fig_duffing():
    d = duffing_data()
    tau_min = solve(d, 1, 'max_margin')['margin']

    # fig1: min-power Melnikov band + N=5 waveform
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(THETA, M_unc(d), color=C_UNC, label='Uncontrolled', lw=1.5)
    ax[0].axhline(0, color='gray', lw=0.5)
    res5 = None
    for N, col in [(1, C_N1), (3, C_N3), (5, C_N5)]:
        r = solve(d, N, 'min_power', tau_min=tau_min)
        M = evaluate_melnikov(THETA, d['c0'], r['c_fourier'], r['d_fourier'])
        ax[0].plot(THETA, M, color=col, label=f'Controlled ($N={N}$)')
        if N == 5:
            res5 = r
    draw_band(ax[0], d['c0'], tau_min)
    melnikov_axis(ax[0]); ax[0].legend(fontsize=8, loc='upper right'); panel_tag(ax[0], '(a)')
    waveform(ax[1], res5, d['Om']); panel_tag(ax[1], '(b)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_duffing_design.pdf')); plt.close(fig)

    # fig1b: power-margin
    fig, ax = plt.subplots(figsize=(6, 4))
    power_margin(ax, d, [(1, C_N1, '$N=1$'), (3, C_N3, '$N=3$'), (5, C_N5, '$N=5$')],
                 np.linspace(0.05, 3.0, 30), d['Pmax'])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1b_duffing_power_margin.pdf')); plt.close(fig)

    # fig4: performance scaling
    Ns = [1, 2, 3, 5, 7]
    P = [solve(d, N, 'min_power', tau_min=tau_min)['energy'] for N in Ns]
    T = [min(solve(d, N, 'max_margin')['margin'], solve(d, N, 'max_margin')['margin_L']) for N in Ns]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(Ns, P, 'o-', color=C[0]); ax[0].set_xlabel(r'Number of harmonics $N$')
    ax[0].set_ylabel(r'Minimum control power $P^*$'); ax[0].set_xticks(Ns); panel_tag(ax[0], '(a)')
    ax[1].plot(Ns, T, 's-', color=C[2]); ax[1].set_xlabel(r'Number of harmonics $N$')
    ax[1].set_ylabel(r'Maximum margin $\tau_{\mathrm{both}}$'); ax[1].set_xticks(Ns); panel_tag(ax[1], '(b)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_duffing_performance.pdf')); plt.close(fig)


def fig_josephson():
    d = josephson_data()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(THETA, M_unc(d), color=C_UNC, label='Uncontrolled', lw=1.5)
    ax[0].axhline(0, color='gray', lw=0.5)
    res5 = None
    taus = {}
    for N, col in [(1, C_N1), (3, C_N3), (5, C_N5)]:
        r = solve(d, N, 'max_margin')
        M = evaluate_melnikov(THETA, d['c0'], r['c_fourier'], r['d_fourier'])
        ax[0].plot(THETA, M, color=col, label=f'Controlled ($N={N}$)')
        taus[N] = min(r['margin'], r['margin_L'])
        if N == 5:
            res5 = r
    draw_band(ax[0], d['c0'], taus[1])
    melnikov_axis(ax[0]); ax[0].legend(fontsize=8); panel_tag(ax[0], '(a)')
    waveform(ax[1], res5, d['Om']); panel_tag(ax[1], '(b)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig6_josephson_design.pdf')); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    power_margin(ax, d, [(1, C_N1, '$N=1$'), (3, C_N3, '$N=3$'), (5, C_N5, '$N=5$')],
                 np.linspace(0.01, 0.5, 25), d['Pmax'])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig8_josephson_power_margin.pdf')); plt.close(fig)


def helmholtz_data():
    delta, Om = 0.10, 1.0
    gamma = [0.10, 0.30, 0.15]
    Nmax = 10
    al = np.zeros(Nmax); be = np.zeros(Nmax)
    for k in range(Nmax):
        al[k], be[k] = helmholtz_kernels_analytical(k + 1, Om)
    c_ex = np.array([-gamma[k] * be[k] for k in range(3)])
    d_ex = np.zeros(3)
    return dict(c0=helmholtz_c0(delta), al=al, be=be, c_ex=c_ex, d_ex=d_ex,
                Pmax=0.06, Om=Om)


def fig_helmholtz():
    d = helmholtz_data()
    tau_target = (2.0 / 3.0) * abs(d['c0'])   # 0.08

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # (a) Melnikov functions: uncontrolled + max-margin N=1,2,3
    a = ax[0]
    a.plot(THETA, M_unc(d), color=C_UNC, label='Uncontrolled', lw=1.5)
    a.axhline(0, color='gray', lw=0.5)
    for N, col in [(1, C_N1), (2, C[4]), (3, C_N3)]:
        r = solve(d, N, 'max_margin', both=False)
        M = evaluate_melnikov(THETA, d['c0'], r['c_fourier'], r['d_fourier'])
        a.plot(THETA, M, color=col, label=f'Controlled ($N={N}$)')
    a.axhline(-tau_target, color='gray', lw=0.8, ls='--', alpha=0.8,
              label=r'$-\tau_{\min}=-0.08$')
    melnikov_axis(a); a.legend(fontsize=8, loc='upper right'); panel_tag(a, '(a)')

    # (b) single-branch power-margin: N=1 saturates below the target
    a = ax[1]
    Pgrid = np.linspace(0.002, 0.20, 40)
    for N, col, lab in [(1, C_N1, '$N=1$'), (2, C[4], '$N=2$'), (3, C_N3, '$N=3$')]:
        margins = []
        for P in Pgrid:
            r = solve(d, N, 'max_margin', Pmax=P, both=False)
            margins.append(max(r['margin'], 0.0) if r['margin'] is not None else 0.0)
        a.plot(Pgrid, margins, color=col, label=lab)
    a.axvline(d['Pmax'], color='gray', ls='--', lw=0.8, label=f'Budget ({d["Pmax"]})')
    a.axhline(tau_target, color='red', ls='--', lw=0.8, label=r'Target $\tau_{\min}=0.08$')
    a.axhline(abs(d['c0']), color='black', ls=':', lw=0.8, label=rf'$|c_0|={abs(d["c0"]):.2f}$')
    a.set_xlabel(r'Total control power $P$'); a.set_ylabel(r'Suppression margin $\tau$')
    a.legend(fontsize=8); panel_tag(a, '(b)')

    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig9_helmholtz.pdf')); plt.close(fig)


if __name__ == "__main__":
    fig_duffing()
    fig_josephson()
    fig_helmholtz()
    print("regenerated fig1, fig1b, fig4, fig6, fig8, fig9 (two-branch + Helmholtz)")
