"""
Master figure generation script for Melnikov chaos control paper.

Generates all figures and data tables for the Nonlinear Dynamics submission.
All numerical values in the paper come from this script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
import sys
import time

# Use colorblind-friendly palette
plt.style.use('seaborn-v0_8-colorblind')

# Figure style
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'text.usetex': False,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import (
    duffing_c0, duffing_kernels_analytical,
    josephson_c0, josephson_kernels_analytical,
    ship_c0, ship_kernels_analytical,
)
from sdp_solver import solve_suppression_sdp, evaluate_melnikov
from dynamics import (
    duffing_rhs, duffing_rhs_with_variational,
    pendulum_rhs, pendulum_rhs_with_variational,
    ship_rhs, ship_rhs_with_variational,
    compute_poincare_section, compute_lyapunov_exponent,
    compute_manifolds_duffing,
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Color scheme
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
C_UNCONTROLLED = COLORS[3]  # red-ish
C_N1 = COLORS[0]  # blue
C_N3 = COLORS[1]  # orange
C_N5 = COLORS[2]  # green
C_N7 = COLORS[4]  # purple

ALL_RESULTS = {}


# =============================================================================
# MEMS / Duffing parameters
# =============================================================================
DUFFING_PARAMS = {
    'delta': 0.80,       # damping
    'Omega': 1.0,        # forcing frequency
    # Multiharmonic existing perturbation: gamma_k * cos(k*Omega*t) for k=1,2,3
    # Cosine forcing produces cross-well dynamics that are directly observable.
    'gamma_exist': [1.20, 0.60, 0.32],  # amplitudes at harmonics 1, 2, 3
    'forcing_type': 'cos',  # cosine forcing
    'eps': 0.25,         # perturbation parameter (small-eps regime for Melnikov validity)
    'P_max': 2.40,       # power budget (scaled for small-eps regime)
}


def get_duffing_kernels(N, Omega):
    """Get alpha, beta arrays for Duffing."""
    alpha = np.zeros(N)
    beta = np.zeros(N)
    for k in range(N):
        alpha[k], beta[k] = duffing_kernels_analytical(k + 1, Omega)
    return alpha, beta


def get_duffing_existing(gamma_list, alpha_arr, beta_arr, forcing_type='cos'):
    """Get existing perturbation Fourier coefficients for Duffing.
    For cos forcing: a_k=gamma_k, b_k=0
      c_k = gamma_k*alpha_k = 0 (alpha_k=0 for Duffing), d_k = gamma_k*beta_k
    For sin forcing: a_k=0, b_k=gamma_k
      c_k = -gamma_k*beta_k, d_k = 0
    """
    N_exist = len(gamma_list)
    c_exist = np.zeros(N_exist)
    d_exist = np.zeros(N_exist)
    for k in range(N_exist):
        if forcing_type == 'cos':
            c_exist[k] = gamma_list[k] * alpha_arr[k]  # = 0 for Duffing
            d_exist[k] = gamma_list[k] * beta_arr[k]
        else:  # sin
            c_exist[k] = -gamma_list[k] * beta_arr[k]
            d_exist[k] = gamma_list[k] * alpha_arr[k]  # = 0 for Duffing
    return c_exist, d_exist


def get_duffing_forcing_coeffs(a_ctrl, b_ctrl, gamma_exist_list, N_ctrl, forcing_type='cos'):
    """
    Build forcing_coeffs list for dynamics simulation.
    Total forcing: existing (multiharmonic) + control.
    """
    N_exist = len(gamma_exist_list)
    N = max(N_exist, N_ctrl)
    coeffs = []
    for k in range(N):
        ak = 0.0
        bk = 0.0
        if k < N_exist:
            if forcing_type == 'cos':
                ak += gamma_exist_list[k]
            else:
                bk += gamma_exist_list[k]
        if k < N_ctrl and a_ctrl is not None:
            ak += a_ctrl[k]
            bk += b_ctrl[k]
        coeffs.append((ak, bk))
    return coeffs


# =============================================================================
# Josephson parameters
# =============================================================================
JOSEPHSON_PARAMS = {
    'alpha_damp': 0.1,   # damping
    'I_dc': 0.0,         # DC bias
    'Omega': 1.0,
    # Multiharmonic microwave drive: mixed cos + sin at harmonics 1, 2, 3
    'gamma_exist_cos': [0.30, 0.25, 0.20],  # cosine amplitudes
    'gamma_exist_sin': [0.20, 0.15, 0.10],  # sine amplitudes
    'eps': 0.1,
    'P_max': 0.15,       # tight microwave power budget
}


def get_josephson_kernels(N, Omega):
    alpha = np.zeros(N)
    beta = np.zeros(N)
    for k in range(N):
        alpha[k], beta[k] = josephson_kernels_analytical(k + 1, Omega)
    return alpha, beta


def get_josephson_existing(gamma_list_or_cos, alpha_arr, gamma_sin=None):
    """Existing perturbation Fourier coefficients for Josephson.

    For pendulum: beta_k = 0, only alpha_k != 0.
    - cos(kOmega t) with amplitude gamma_k^c:  c_k += gamma_k^c * alpha_k
    - sin(kOmega t) with amplitude gamma_k^s:  d_k += gamma_k^s * alpha_k

    Supports two calling conventions:
      get_josephson_existing(gamma_list, alpha)         -- pure cosine (old format)
      get_josephson_existing(gamma_cos, alpha, gamma_sin) -- mixed cos+sin
    """
    gamma_cos = gamma_list_or_cos
    N_exist = len(gamma_cos)
    c_exist = np.zeros(N_exist)
    d_exist = np.zeros(N_exist)
    for k in range(N_exist):
        c_exist[k] = gamma_cos[k] * alpha_arr[k]
    if gamma_sin is not None:
        for k in range(len(gamma_sin)):
            d_exist[k] += gamma_sin[k] * alpha_arr[k]
    return c_exist, d_exist


# =============================================================================
# Ship roll parameters
# =============================================================================
SHIP_PARAMS = {
    'delta': 0.08,       # damping ratio
    'alpha_s': 0.5,      # softening nonlinearity from GZ curve
    'Omega': 1.0,
    'gamma_exist': [0.20, 0.15, 0.08],  # wave excitation at harmonics 1, 2, 3 (sine)
    'eps': 0.1,
    'P_max': 0.05,       # limited fin stabilizer authority (below full-cancellation cost ~0.069)
}


def get_ship_kernels(N, Omega, alpha_s):
    alpha = np.zeros(N)
    beta = np.zeros(N)
    for k in range(N):
        alpha[k], beta[k] = ship_kernels_analytical(k + 1, Omega, alpha_s)
    return alpha, beta


# =============================================================================
# Section 5: MEMS/Duffing figures
# =============================================================================

def compute_duffing_sdp_results():
    """Run all SDP computations for Duffing case study."""
    print("\n" + "=" * 70)
    print("COMPUTING DUFFING/MEMS SDP RESULTS")
    print("=" * 70)

    p = DUFFING_PARAMS
    c0 = duffing_c0(p['delta'])

    results = {}
    N_values = [1, 2, 3, 5, 7, 10]
    N_max = max(N_values)
    alpha_all, beta_all = get_duffing_kernels(N_max, p['Omega'])
    c_exist, d_exist = get_duffing_existing(p['gamma_exist'], alpha_all, beta_all, p.get('forcing_type', 'cos'))

    # Uncontrolled Melnikov
    theta = np.linspace(0, 2 * np.pi, 1000)
    M_uncontrolled = evaluate_melnikov(theta, c0, c_exist, d_exist)
    results['uncontrolled'] = {
        'M_theta': M_uncontrolled.tolist(),
        'theta': theta.tolist(),
        'min_M': float(np.min(M_uncontrolled)),
        'max_M': float(np.max(M_uncontrolled)),
        'c0': c0,
    }
    print(f"  Uncontrolled M: [{np.min(M_uncontrolled):.4f}, {np.max(M_uncontrolled):.4f}]")

    for N in N_values:
        print(f"\n  --- N = {N} ---")
        alpha_n = alpha_all[:N]
        beta_n = beta_all[:N]

        # Max-margin
        res_mm = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_n, beta=beta_n, N_ctrl=N,
            P_max=p['P_max'], objective='max_margin',
            suppress_to_negative=True,
        )
        print(f"    Max-margin: status={res_mm['status']}, margin={res_mm['margin']:.6f}, energy={res_mm['energy']:.6f}, time={res_mm['solve_time']:.3f}s")

        # Min-energy (target margin = 0.05)
        res_me = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_n, beta=beta_n, N_ctrl=N,
            P_max=np.inf, objective='min_energy', tau_min=0.05,
            suppress_to_negative=True,
        )
        if res_me['status'] in ['optimal', 'optimal_inaccurate']:
            print(f"    Min-energy: energy={res_me['energy']:.6f}, margin={res_me['margin']:.6f}")
        else:
            print(f"    Min-energy: {res_me['status']}")

        results[f'N{N}'] = {
            'max_margin': {
                'status': res_mm['status'],
                'margin': float(res_mm['margin']) if res_mm['margin'] is not None else None,
                'energy': float(res_mm['energy']) if res_mm['energy'] is not None else None,
                'A': res_mm['A'].tolist() if res_mm['A'] is not None else None,
                'phi': res_mm['phi'].tolist() if res_mm['phi'] is not None else None,
                'a': res_mm['a'].tolist() if res_mm['a'] is not None else None,
                'b': res_mm['b'].tolist() if res_mm['b'] is not None else None,
                'M_theta': res_mm['M_theta'].tolist() if res_mm['M_theta'] is not None else None,
                'solve_time': res_mm['solve_time'],
            },
            'min_energy': {
                'status': res_me['status'],
                'margin': float(res_me['margin']) if res_me['margin'] is not None else None,
                'energy': float(res_me['energy']) if res_me['energy'] is not None else None,
                'A': res_me['A'].tolist() if res_me['A'] is not None else None,
                'solve_time': res_me['solve_time'],
            },
        }

    return results


def generate_duffing_figures(sdp_results):
    """Generate Figures 1-5 for Duffing/MEMS case study."""
    print("\n  Generating Duffing figures...")
    p = DUFFING_PARAMS
    theta = np.array(sdp_results['uncontrolled']['theta'])

    # --- Figure 1: Design figure (min-power M(theta) + control waveform) ---
    # Re-solve min-energy SDPs at matched margin tau_min = tau*_{N=1}
    c0 = duffing_c0(p['delta'])
    N_max_local = 5
    alpha_all_f1, beta_all_f1 = get_duffing_kernels(N_max_local, p['Omega'])
    c_exist_f1, d_exist_f1 = get_duffing_existing(
        p['gamma_exist'], alpha_all_f1, beta_all_f1, p.get('forcing_type', 'cos')
    )
    tau_min_raw = sdp_results['N1']['max_margin']['margin']
    # Tiny slack so the N=1 min-energy problem is strictly feasible (boundary of N=1 max-margin)
    tau_min = tau_min_raw * (1 - 1e-4)
    print(f"    Re-solving min-power SDPs at matched margin tau_min = {tau_min:.4f}")

    min_power_results = {}
    for N in [1, 3, 5]:
        res = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist_f1, d_exist=d_exist_f1,
            alpha=alpha_all_f1[:N], beta=beta_all_f1[:N], N_ctrl=N,
            P_max=np.inf, objective='min_energy', tau_min=tau_min,
            suppress_to_negative=True,
        )
        min_power_results[N] = res
        print(f"      N={N}: status={res['status']}, energy={res['energy']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): M(theta) for min-power at matched margin
    ax = axes[0]
    M_unc = np.array(sdp_results['uncontrolled']['M_theta'])
    ax.plot(theta, M_unc, color=C_UNCONTROLLED, label='Uncontrolled', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    for N, color in [(1, C_N1), (3, C_N3), (5, C_N5)]:
        res = min_power_results[N]
        if res['M_theta'] is not None:
            ax.plot(theta, np.array(res['M_theta']), color=color,
                    label=f'Controlled (N={N})', linewidth=1.2)

    ax.axhline(-tau_min, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.legend(fontsize=8, loc='upper right')
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Panel (b): Control waveform for N=5 min-power design
    ax = axes[1]
    res5 = min_power_results[5]
    if res5['a'] is not None:
        a_vals = np.array(res5['a'])
        b_vals = np.array(res5['b'])
        t_plot = np.linspace(0, 2 * np.pi / p['Omega'], 500)
        u_ctrl = np.zeros_like(t_plot)
        for k in range(len(a_vals)):
            u_ctrl += a_vals[k] * np.cos((k + 1) * p['Omega'] * t_plot) + \
                      b_vals[k] * np.sin((k + 1) * p['Omega'] * t_plot)
        ax.plot(t_plot, u_ctrl, color=C_N5, linewidth=1.2)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$u(t)$')
        ax.set_xlim([t_plot[0], t_plot[-1]])
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_duffing_design.pdf'))
    plt.close(fig)
    print("    Figure 1 saved.")

    # --- Figure 2: Poincare sections (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    configs = [
        ('Uncontrolled', None, C_UNCONTROLLED),
        ('Controlled (N=1)', 'N1', C_N1),
        ('Controlled (N=3)', 'N3', C_N3),
        ('Controlled (N=5)', 'N5', C_N5),
    ]

    gamma_list = p['gamma_exist']
    ft = p.get('forcing_type', 'cos')
    eps_dyn = p['eps']

    # Compute Lyapunov for uncontrolled to annotate
    forcing_unc = get_duffing_forcing_coeffs(None, None, gamma_list, 0, ft)
    try:
        lam_unc, _, _ = compute_lyapunov_exponent(
            duffing_rhs_with_variational,
            (eps_dyn, p['delta'], forcing_unc, p['Omega']),
            ic=[1.0, 0.0], Omega=p['Omega'],
            n_transient=500, n_compute=3000,
            rtol=1e-10, atol=1e-10,
        )
    except Exception:
        lam_unc = float('nan')

    for idx, (title, key, color) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        if key is None:
            forcing = forcing_unc
        else:
            if sdp_results[key]['max_margin']['a'] is not None:
                a_vals = np.array(sdp_results[key]['max_margin']['a'])
                b_vals = np.array(sdp_results[key]['max_margin']['b'])
                N_ctrl = len(a_vals)
                forcing = get_duffing_forcing_coeffs(a_vals, b_vals, gamma_list, N_ctrl, ft)
            else:
                forcing = forcing_unc

        try:
            if key is None:
                # Use grid of ICs for uncontrolled to show full chaotic attractor
                all_px, all_py = [], []
                ics = [[1.0, 0.0], [-1.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.0, 1.0]]
                for ic in ics:
                    px_i, py_i = compute_poincare_section(
                        duffing_rhs,
                        (eps_dyn, p['delta'], forcing, p['Omega']),
                        ic=ic, Omega=p['Omega'], eps=eps_dyn,
                        n_transient=500, n_collect=2000,
                        rtol=1e-11, atol=1e-11,
                    )
                    all_px.extend(px_i)
                    all_py.extend(py_i)
                px, py = np.array(all_px), np.array(all_py)
            else:
                px, py = compute_poincare_section(
                    duffing_rhs,
                    (eps_dyn, p['delta'], forcing, p['Omega']),
                    ic=[1.0, 0.0], Omega=p['Omega'], eps=eps_dyn,
                    n_transient=500, n_collect=2000,
                    rtol=1e-11, atol=1e-11,
                )
            ax.scatter(px, py, s=0.3, color=color, alpha=0.5, rasterized=True)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center', fontsize=7)

        # Compute and annotate lambda_max
        try:
            if key is None:
                lam = lam_unc
            else:
                lam, _, _ = compute_lyapunov_exponent(
                    duffing_rhs_with_variational,
                    (eps_dyn, p['delta'], forcing, p['Omega']),
                    ic=[1.0, 0.0], Omega=p['Omega'],
                    n_transient=500, n_compute=2000,
                )
            if not np.isnan(lam):
                ax.text(0.05, 0.95, f'$\\lambda_{{\\max}} = {lam:.3f}$',
                        transform=ax.transAxes, fontsize=8, va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
        except Exception:
            pass

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\dot{x}$')
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=10)

    # Force all panels to use the same axis limits as panel (a) (uncontrolled)
    ax_a = axes[0][0]
    xlim_a = ax_a.get_xlim()
    ylim_a = ax_a.get_ylim()
    for i in range(2):
        for j in range(2):
            axes[i][j].set_xlim(xlim_a)
            axes[i][j].set_ylim(ylim_a)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_duffing_poincare.pdf'))
    plt.close(fig)
    print("    Figure 2 saved.")

    # --- Figure 3: Manifolds (3 panels: uncontrolled, controlled, distance) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ft = p.get('forcing_type', 'cos')
    forcing_unc_manifold = get_duffing_forcing_coeffs(None, None, p['gamma_exist'], 0, ft)
    manifold_data = {}

    for idx, (title, forcing_fn) in enumerate([
        ('Uncontrolled', forcing_unc_manifold),
        ('Controlled (N=5)', None),
    ]):
        ax = axes[idx]
        if forcing_fn is None and 'N5' in sdp_results and sdp_results['N5']['max_margin']['a'] is not None:
            a5 = np.array(sdp_results['N5']['max_margin']['a'])
            b5 = np.array(sdp_results['N5']['max_margin']['b'])
            forcing_fn = get_duffing_forcing_coeffs(a5, b5, p['gamma_exist'], 5, ft)

        if forcing_fn is not None:
            try:
                (Wu_x, Wu_y), (Ws_x, Ws_y) = compute_manifolds_duffing(
                    p['eps'], p['delta'], forcing_fn, p['Omega'],
                    n_points=100, T_integrate=6.0, saddle_delta=1e-5,
                )
                ax.scatter(Wu_x, Wu_y, s=0.1, color=COLORS[0], alpha=0.3, label=r'$W^u$', rasterized=True)
                ax.scatter(Ws_x, Ws_y, s=0.1, color=COLORS[3], alpha=0.3, label=r'$W^s$', rasterized=True)
                ax.legend(fontsize=8, markerscale=10)
                ax.set_xlim([-2.5, 2.5])
                ax.set_ylim([-1.5, 1.5])
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, ha='center')

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\dot{x}$')
        ax.set_title(title, fontsize=10)
        ax.text(-0.12, 1.05, ['(a)', '(b)'][idx], transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Panel (c): Melnikov function comparison showing suppressed sign changes
    ax = axes[2]
    theta_m = np.linspace(0, 2*np.pi, 500)
    M_unc = np.array(sdp_results['uncontrolled']['M_theta'])
    theta_m_full = np.array(sdp_results['uncontrolled']['theta'])
    ax.plot(theta_m_full, M_unc, color=C_UNCONTROLLED, label='Uncontrolled', linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    # Min-power N=5 at matched margin (consistent with Figure 1)
    tau_min_raw_f3 = sdp_results['N1']['max_margin']['margin']
    tau_min_f3 = tau_min_raw_f3 * (1 - 1e-4)
    alpha_f3, beta_f3 = get_duffing_kernels(5, p['Omega'])
    c_exist_f3, d_exist_f3 = get_duffing_existing(
        p['gamma_exist'], alpha_f3, beta_f3, p.get('forcing_type', 'cos')
    )
    res_mp5 = solve_suppression_sdp(
        c0_total=duffing_c0(p['delta']), c_exist=c_exist_f3, d_exist=d_exist_f3,
        alpha=alpha_f3, beta=beta_f3, N_ctrl=5,
        P_max=np.inf, objective='min_energy', tau_min=tau_min_f3,
        suppress_to_negative=True,
    )
    if res_mp5['M_theta'] is not None:
        ax.plot(theta_m_full, np.array(res_mp5['M_theta']), color=C_N5,
                label='Controlled (N=5)', linewidth=1.2)
        ax.axhline(-tau_min_f3, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.legend(fontsize=8)
    ax.set_title('Melnikov function', fontsize=10)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_duffing_manifolds.pdf'))
    plt.close(fig)
    print("    Figure 3 saved.")

    # --- Figure 4: Performance (energy vs N, margin vs N) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    N_values = [1, 2, 3, 5, 7, 10]

    # Panel (a): Min energy vs N
    ax = axes[0]
    energies = []
    N_plot = []
    for N in N_values:
        key = f'N{N}'
        if key in sdp_results and sdp_results[key]['min_energy']['energy'] is not None:
            energies.append(sdp_results[key]['min_energy']['energy'])
            N_plot.append(N)
    if energies:
        ax.plot(N_plot, energies, 'o-', color=COLORS[0], markersize=6)
        ax.set_xlabel('Number of harmonics $N$')
        ax.set_ylabel(r'Minimum control power $P^*$')
        ax.set_xticks(N_plot)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Panel (b): Max margin vs N (fixed power)
    ax = axes[1]
    margins = []
    N_plot2 = []
    for N in N_values:
        key = f'N{N}'
        if key in sdp_results and sdp_results[key]['max_margin']['margin'] is not None:
            margins.append(sdp_results[key]['max_margin']['margin'])
            N_plot2.append(N)
    if margins:
        ax.plot(N_plot2, margins, 's-', color=COLORS[2], markersize=6)
        ax.set_xlabel('Number of harmonics $N$')
        ax.set_ylabel(r'Maximum margin $\tau^*$')
        ax.set_xticks(N_plot2)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_duffing_performance.pdf'))
    plt.close(fig)
    print("    Figure 4 saved.")

    # --- Figure 5: Analytical Melnikov prediction for the uncontrolled system ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    eps = p['eps']
    M_ana = np.array(sdp_results['uncontrolled']['M_theta'])

    # Panel (a): Melnikov function with sign-change regions highlighted
    ax = axes[0]
    ax.plot(theta, M_ana, color=COLORS[0], linewidth=1.5)
    ax.fill_between(theta, M_ana, 0, where=M_ana > 0, alpha=0.2, color='red')
    ax.fill_between(theta, M_ana, 0, where=M_ana < 0, alpha=0.2, color='blue')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.set_title('Uncontrolled Melnikov function', fontsize=10)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Panel (b): Controlled N=1,3,5 with margin annotation
    ax = axes[1]
    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        key = f'N{N}'
        if key in sdp_results and sdp_results[key]['max_margin']['M_theta'] is not None:
            M_ctrl = np.array(sdp_results[key]['max_margin']['M_theta'])
            ax.plot(theta, M_ctrl, color=color, label=f'{label}', linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.set_title('Controlled Melnikov functions', fontsize=10)
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_duffing_validation.pdf'))
    plt.close(fig)
    print("    Figure 5 saved.")

    # --- Figure 1b: Duffing power-margin trade-off ---
    print("    Generating Duffing power-margin trade-off figure...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    P_values = np.linspace(0.05, 3.0, 30)
    N_max_d = 5
    alpha_all_d, beta_all_d = get_duffing_kernels(N_max_d, p['Omega'])
    c_exist_d, d_exist_d = get_duffing_existing(
        p['gamma_exist'], alpha_all_d, beta_all_d, p.get('forcing_type', 'cos')
    )
    c0_d = duffing_c0(p['delta'])

    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        margins = []
        for P in P_values:
            res = solve_suppression_sdp(
                c0_total=c0_d, c_exist=c_exist_d, d_exist=d_exist_d,
                alpha=alpha_all_d[:N], beta=beta_all_d[:N], N_ctrl=N,
                P_max=P, objective='max_margin',
                suppress_to_negative=True,
            )
            margins.append(max(res['margin'], 0.0) if res['margin'] is not None else 0.0)
        ax.plot(P_values, margins, color=color, label=label, linewidth=1.2)

    ax.axvline(p['P_max'], color='gray', linestyle='--', linewidth=0.8,
               label=f'Power budget ({p["P_max"]})')
    ax.axhline(abs(c0_d), color='black', linestyle=':', linewidth=0.8,
               label=rf'$|c_0|={abs(c0_d):.4f}$')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.set_xlabel(r'Total control power $P$')
    ax.set_ylabel(r'Suppression margin $\tau^*$')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1b_duffing_power_margin.pdf'))
    plt.close(fig)
    print("    Figure 1b (Duffing power-margin) saved.")


# =============================================================================
# Section 6: Josephson junction figures
# =============================================================================

def compute_josephson_sdp_results():
    """Run all SDP computations for Josephson case study."""
    print("\n" + "=" * 70)
    print("COMPUTING JOSEPHSON JUNCTION SDP RESULTS")
    print("=" * 70)

    p = JOSEPHSON_PARAMS
    c0 = josephson_c0(p['alpha_damp'], p['I_dc'])

    results = {}
    N_values = [1, 2, 3, 5, 7, 10]
    N_max = max(N_values)
    alpha_all, beta_all = get_josephson_kernels(N_max, p['Omega'])

    # Multiharmonic existing perturbation (mixed cos + sin)
    gamma_cos = p.get('gamma_exist_cos', p.get('gamma_exist', []))
    gamma_sin = p.get('gamma_exist_sin', None)
    c_exist, d_exist = get_josephson_existing(gamma_cos, alpha_all, gamma_sin)

    theta = np.linspace(0, 2 * np.pi, 1000)
    M_uncontrolled = evaluate_melnikov(theta, c0, c_exist, d_exist)
    results['uncontrolled'] = {
        'M_theta': M_uncontrolled.tolist(),
        'theta': theta.tolist(),
        'min_M': float(np.min(M_uncontrolled)),
        'max_M': float(np.max(M_uncontrolled)),
        'c0': c0,
    }
    print(f"  Uncontrolled M: [{np.min(M_uncontrolled):.4f}, {np.max(M_uncontrolled):.4f}]")

    for N in N_values:
        print(f"\n  --- N = {N} ---")
        alpha_n = alpha_all[:N]
        beta_n = beta_all[:N]

        # Max-margin
        res_mm = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_n, beta=beta_n, N_ctrl=N,
            P_max=p['P_max'], objective='max_margin',
            suppress_to_negative=True,
        )
        if res_mm['margin'] is not None:
            print(f"    Max-margin: margin={res_mm['margin']:.6f}, energy={res_mm['energy']:.6f}, time={res_mm['solve_time']:.3f}s")
        else:
            print(f"    Max-margin: {res_mm['status']}")

        # Min-energy
        res_me = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_n, beta=beta_n, N_ctrl=N,
            P_max=np.inf, objective='min_energy', tau_min=0.05,
            suppress_to_negative=True,
        )
        if res_me['status'] in ['optimal', 'optimal_inaccurate']:
            print(f"    Min-energy: energy={res_me['energy']:.6f}")
        else:
            print(f"    Min-energy: {res_me['status']}")

        results[f'N{N}'] = {
            'max_margin': {
                'status': res_mm['status'],
                'margin': float(res_mm['margin']) if res_mm['margin'] is not None else None,
                'energy': float(res_mm['energy']) if res_mm['energy'] is not None else None,
                'A': res_mm['A'].tolist() if res_mm['A'] is not None else None,
                'phi': res_mm['phi'].tolist() if res_mm['phi'] is not None else None,
                'a': res_mm['a'].tolist() if res_mm['a'] is not None else None,
                'b': res_mm['b'].tolist() if res_mm['b'] is not None else None,
                'M_theta': res_mm['M_theta'].tolist() if res_mm['M_theta'] is not None else None,
                'solve_time': res_mm['solve_time'],
            },
            'min_energy': {
                'status': res_me['status'],
                'margin': float(res_me['margin']) if res_me['margin'] is not None else None,
                'energy': float(res_me['energy']) if res_me['energy'] is not None else None,
                'A': res_me['A'].tolist() if res_me['A'] is not None else None,
                'solve_time': res_me['solve_time'],
            },
        }

    return results


def generate_josephson_figures(sdp_results):
    """Generate Figures 6-8 for Josephson case study."""
    print("\n  Generating Josephson figures...")
    p = JOSEPHSON_PARAMS
    theta = np.array(sdp_results['uncontrolled']['theta'])

    # --- Figure 6: Design figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    M_unc = np.array(sdp_results['uncontrolled']['M_theta'])
    ax.plot(theta, M_unc, color=C_UNCONTROLLED, label='Uncontrolled', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)

    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        key = f'N{N}'
        if key in sdp_results and sdp_results[key]['max_margin']['M_theta'] is not None:
            M_ctrl = np.array(sdp_results[key]['max_margin']['M_theta'])
            ax.plot(theta, M_ctrl, color=color, label=f'Controlled ({label})', linewidth=1.2)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Panel (b): Control waveform N=5
    ax = axes[1]
    if 'N5' in sdp_results and sdp_results['N5']['max_margin']['a'] is not None:
        a_vals = np.array(sdp_results['N5']['max_margin']['a'])
        b_vals = np.array(sdp_results['N5']['max_margin']['b'])
        t_plot = np.linspace(0, 2 * np.pi / p['Omega'], 500)
        u_ctrl = np.zeros_like(t_plot)
        for k in range(len(a_vals)):
            u_ctrl += a_vals[k] * np.cos((k + 1) * p['Omega'] * t_plot) + \
                      b_vals[k] * np.sin((k + 1) * p['Omega'] * t_plot)
        ax.plot(t_plot, u_ctrl, color=C_N5, linewidth=1.2)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$u(t)$')
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_josephson_design.pdf'))
    plt.close(fig)
    print("    Figure 6 saved.")

    # Figure 7 (Josephson Poincare) removed: uncontrolled system at eps=0.1
    # shows a stable periodic attractor, not chaotic scatter.
    print("    Figure 7 (Josephson Poincare) skipped — removed from paper.")

    # --- Figure 8: Margin vs power ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    P_values = np.linspace(0.01, 0.5, 20)
    N_max_j = 10
    alpha_all, beta_all = get_josephson_kernels(N_max_j, p['Omega'])
    gamma_cos = p.get('gamma_exist_cos', p.get('gamma_exist', []))
    gamma_sin = p.get('gamma_exist_sin', None)
    c_exist, d_exist = get_josephson_existing(gamma_cos, alpha_all, gamma_sin)
    c0 = josephson_c0(p['alpha_damp'], p['I_dc'])

    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        margins = []
        for P in P_values:
            res = solve_suppression_sdp(
                c0_total=c0, c_exist=c_exist, d_exist=d_exist,
                alpha=alpha_all[:N], beta=beta_all[:N], N_ctrl=N,
                P_max=P, objective='max_margin',
                suppress_to_negative=True,
            )
            margins.append(max(res['margin'], 0.0) if res['margin'] is not None else 0.0)
        ax.plot(P_values, margins, color=color, label=label, linewidth=1.2)

    ax.axvline(p['P_max'], color='gray', linestyle='--', linewidth=0.8, label=f'Power budget ({p["P_max"]})')
    ax.set_xlabel(r'Total control power $P$')
    ax.set_ylabel(r'Suppression margin $\tau^*$')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig8_josephson_power_margin.pdf'))
    plt.close(fig)
    print("    Figure 8 saved.")


# =============================================================================
# Section 7: Ship roll figures
# =============================================================================

def compute_ship_sdp_results():
    """Run SDP computations for ship roll case study."""
    print("\n" + "=" * 70)
    print("COMPUTING SHIP ROLL SDP RESULTS")
    print("=" * 70)

    p = SHIP_PARAMS
    c0 = ship_c0(p['delta'], p['alpha_s'])

    results = {}
    N_values = [1, 3, 5]
    N_max = max(N_values)
    alpha_all, beta_all = get_ship_kernels(N_max, p['Omega'], p['alpha_s'])

    # Multiharmonic wave excitation (sine forcing, same structure as Duffing)
    gamma_list = p['gamma_exist']
    N_exist = len(gamma_list)
    c_exist = np.zeros(N_exist)
    d_exist = np.zeros(N_exist)
    for k in range(N_exist):
        c_exist[k] = -gamma_list[k] * beta_all[k]

    theta = np.linspace(0, 2 * np.pi, 1000)
    M_uncontrolled = evaluate_melnikov(theta, c0, c_exist, d_exist)
    results['uncontrolled'] = {
        'M_theta': M_uncontrolled.tolist(),
        'theta': theta.tolist(),
        'min_M': float(np.min(M_uncontrolled)),
        'max_M': float(np.max(M_uncontrolled)),
        'c0': c0,
    }
    print(f"  Uncontrolled M: [{np.min(M_uncontrolled):.4f}, {np.max(M_uncontrolled):.4f}]")

    for N in N_values:
        print(f"\n  --- N = {N} ---")
        alpha_n = alpha_all[:N]
        beta_n = beta_all[:N]

        res_mm = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_n, beta=beta_n, N_ctrl=N,
            P_max=p['P_max'], objective='max_margin',
            suppress_to_negative=True,
        )
        if res_mm['margin'] is not None:
            print(f"    Max-margin: margin={res_mm['margin']:.6f}, energy={res_mm['energy']:.6f}")
        else:
            print(f"    Max-margin: {res_mm['status']}")

        res_me = solve_suppression_sdp(
            c0_total=c0, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_n, beta=beta_n, N_ctrl=N,
            P_max=np.inf, objective='min_energy', tau_min=0.02,
            suppress_to_negative=True,
        )
        if res_me['status'] in ['optimal', 'optimal_inaccurate']:
            print(f"    Min-energy: energy={res_me['energy']:.6f}")

        results[f'N{N}'] = {
            'max_margin': {
                'status': res_mm['status'],
                'margin': float(res_mm['margin']) if res_mm['margin'] is not None else None,
                'energy': float(res_mm['energy']) if res_mm['energy'] is not None else None,
                'A': res_mm['A'].tolist() if res_mm['A'] is not None else None,
                'M_theta': res_mm['M_theta'].tolist() if res_mm['M_theta'] is not None else None,
                'solve_time': res_mm['solve_time'],
            },
            'min_energy': {
                'status': res_me['status'],
                'energy': float(res_me['energy']) if res_me['energy'] is not None else None,
                'solve_time': res_me['solve_time'],
            },
        }

    return results


def generate_ship_figures(sdp_results):
    """Generate Figure 9 for ship roll."""
    print("\n  Generating ship roll figures...")
    theta = np.array(sdp_results['uncontrolled']['theta'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) M(theta)
    ax = axes[0]
    M_unc = np.array(sdp_results['uncontrolled']['M_theta'])
    ax.plot(theta, M_unc, color=C_UNCONTROLLED, label='Uncontrolled', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)

    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        key = f'N{N}'
        if key in sdp_results and sdp_results[key]['max_margin']['M_theta'] is not None:
            M_ctrl = np.array(sdp_results[key]['max_margin']['M_theta'])
            ax.plot(theta, M_ctrl, color=color, label=f'Controlled ({label})', linewidth=1.2)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax.set_xlim([0, 2 * np.pi])
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    p = SHIP_PARAMS

    # (b) Energy comparison
    ax = axes[1]
    N_vals = [1, 3, 5]
    energies_single = []
    energies_multi = []
    for N in N_vals:
        key = f'N{N}'
        if key in sdp_results and sdp_results[key]['min_energy']['energy'] is not None:
            energies_multi.append(sdp_results[key]['min_energy']['energy'])
        else:
            energies_multi.append(np.nan)

    if 'N1' in sdp_results and sdp_results['N1']['min_energy']['energy'] is not None:
        e1 = sdp_results['N1']['min_energy']['energy']
        energies_single = [e1] * len(N_vals)
    else:
        energies_single = [np.nan] * len(N_vals)

    x_pos = np.arange(len(N_vals))
    width = 0.35
    ax.bar(x_pos - width/2, energies_single, width, label='Single harmonic', color=C_N1)
    ax.bar(x_pos + width/2, energies_multi, width, label='Multiharmonic', color=C_N5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'N={n}' for n in N_vals])
    ax.set_ylabel(r'Energy $E^*$')
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig9_ship_combined.pdf'))
    plt.close(fig)
    print("    Figure 9 saved.")


def ship_rhs_wrapper(t, z, eps, delta, alpha_s, forcing_coeffs, Omega):
    """Wrapper for ship roll RHS."""
    from dynamics import ship_rhs
    return ship_rhs(t, z, eps, delta, alpha_s, forcing_coeffs, Omega)


# =============================================================================
# Section 8: Robustness figures
# =============================================================================

def compute_robustness_results():
    """Compute robustness analysis data."""
    print("\n" + "=" * 70)
    print("COMPUTING ROBUSTNESS ANALYSIS")
    print("=" * 70)

    p = DUFFING_PARAMS
    results = {}

    N_max = 10
    alpha_all, beta_all = get_duffing_kernels(N_max, p['Omega'])
    c_exist, d_exist = get_duffing_existing(p['gamma_exist'], alpha_all, beta_all, p.get('forcing_type', 'cos'))
    c0_nom = duffing_c0(p['delta'])

    ft = p.get('forcing_type', 'cos')

    # First, get optimal designs at nominal parameters
    designs = {}
    for N in [1, 3, 5]:
        res = solve_suppression_sdp(
            c0_total=c0_nom, c_exist=c_exist, d_exist=d_exist,
            alpha=alpha_all[:N], beta=beta_all[:N], N_ctrl=N,
            P_max=p['P_max'], objective='max_margin',
            suppress_to_negative=True,
        )
        if res['a'] is not None:
            designs[N] = {'a': res['a'], 'b': res['b']}

    # (a) Damping mismatch sweep
    print("\n  Damping mismatch sweep...")
    delta_range = np.linspace(0.9 * p['delta'], 1.1 * p['delta'], 21)
    damping_results = {1: [], 3: [], 5: []}
    for delta_test in delta_range:
        c0_test = duffing_c0(delta_test)
        alpha_test, beta_test = get_duffing_kernels(N_max, p['Omega'])
        c_exist_test, d_exist_test = get_duffing_existing(p['gamma_exist'], alpha_test, beta_test, p.get('forcing_type', 'cos'))

        for N in [1, 3, 5]:
            if N not in designs:
                damping_results[N].append(np.nan)
                continue
            a_d, b_d = designs[N]['a'], designs[N]['b']
            N_exist = len(c_exist_test)
            N_total = max(N_exist, N)
            c_f = np.zeros(N_total)
            d_f = np.zeros(N_total)
            for k in range(N_exist):
                c_f[k] += c_exist_test[k]
                d_f[k] += d_exist_test[k]
            for k in range(N):
                c_f[k] += a_d[k] * alpha_test[k] - b_d[k] * beta_test[k]
                d_f[k] += b_d[k] * alpha_test[k] + a_d[k] * beta_test[k]
            theta = np.linspace(0, 2 * np.pi, 500)
            M_vals = evaluate_melnikov(theta, c0_test, c_f, d_f)
            damping_results[N].append(-float(np.max(M_vals)))

    results['damping_mismatch'] = {
        'delta_range': delta_range.tolist(),
        'margins': {str(N): damping_results[N] for N in [1, 3, 5]},
    }

    # (b) Frequency mismatch sweep
    print("  Frequency mismatch sweep...")
    Omega_range = np.linspace(0.9 * p['Omega'], 1.1 * p['Omega'], 21)
    freq_results = {1: [], 3: [], 5: []}
    for Omega_test in Omega_range:
        alpha_test, beta_test = get_duffing_kernels(N_max, Omega_test)
        c_exist_test, d_exist_test = get_duffing_existing(p['gamma_exist'], alpha_test, beta_test, p.get('forcing_type', 'cos'))

        for N in [1, 3, 5]:
            if N not in designs:
                freq_results[N].append(np.nan)
                continue
            a_d, b_d = designs[N]['a'], designs[N]['b']
            N_exist = len(c_exist_test)
            N_total = max(N_exist, N)
            c_f = np.zeros(N_total)
            d_f = np.zeros(N_total)
            for k in range(N_exist):
                c_f[k] += c_exist_test[k]
                d_f[k] += d_exist_test[k]
            for k in range(N):
                c_f[k] += a_d[k] * alpha_test[k] - b_d[k] * beta_test[k]
                d_f[k] += b_d[k] * alpha_test[k] + a_d[k] * beta_test[k]
            theta = np.linspace(0, 2 * np.pi, 500)
            M_vals = evaluate_melnikov(theta, c0_nom, c_f, d_f)
            freq_results[N].append(-float(np.max(M_vals)))

    results['freq_mismatch'] = {
        'Omega_range': Omega_range.tolist(),
        'margins': {str(N): freq_results[N] for N in [1, 3, 5]},
    }

    # (c) eps sweep - BOTH uncontrolled AND controlled
    print("  Epsilon sweep (Lyapunov exponents)...")
    eps_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
    ft = p.get('forcing_type', 'cos')
    lyap_results_ctrl = []
    lyap_results_unc = []
    lyap_results_n1 = []

    for eps_val in eps_range:
        # Uncontrolled
        forcing_unc = get_duffing_forcing_coeffs(None, None, p['gamma_exist'], 0, ft)
        try:
            lam_u, se_u, _ = compute_lyapunov_exponent(
                duffing_rhs_with_variational,
                (eps_val, p['delta'], forcing_unc, p['Omega']),
                ic=[1.0, 0.0], Omega=p['Omega'],
                n_transient=500, n_compute=2000,
                rtol=1e-9, atol=1e-9,
            )
            lyap_results_unc.append({'eps': eps_val, 'lambda_max': float(lam_u), 'se': float(se_u)})
        except Exception:
            lyap_results_unc.append({'eps': eps_val, 'lambda_max': np.nan, 'se': np.nan})

        # Controlled N=5
        if 5 in designs:
            a5, b5 = designs[5]['a'], designs[5]['b']
            forcing_c5 = get_duffing_forcing_coeffs(a5, b5, p['gamma_exist'], 5, ft)
            try:
                lam_c, se_c, _ = compute_lyapunov_exponent(
                    duffing_rhs_with_variational,
                    (eps_val, p['delta'], forcing_c5, p['Omega']),
                    ic=[1.0, 0.0], Omega=p['Omega'],
                    n_transient=500, n_compute=2000,
                    rtol=1e-9, atol=1e-9,
                )
                lyap_results_ctrl.append({'eps': eps_val, 'lambda_max': float(lam_c), 'se': float(se_c)})
            except Exception:
                lyap_results_ctrl.append({'eps': eps_val, 'lambda_max': np.nan, 'se': np.nan})
        else:
            lyap_results_ctrl.append({'eps': eps_val, 'lambda_max': np.nan, 'se': np.nan})

        # Controlled N=1
        if 1 in designs:
            a1, b1 = designs[1]['a'], designs[1]['b']
            forcing_c1 = get_duffing_forcing_coeffs(a1, b1, p['gamma_exist'], 1, ft)
            try:
                lam_1, se_1, _ = compute_lyapunov_exponent(
                    duffing_rhs_with_variational,
                    (eps_val, p['delta'], forcing_c1, p['Omega']),
                    ic=[1.0, 0.0], Omega=p['Omega'],
                    n_transient=500, n_compute=2000,
                    rtol=1e-9, atol=1e-9,
                )
                lyap_results_n1.append({'eps': eps_val, 'lambda_max': float(lam_1), 'se': float(se_1)})
            except Exception:
                lyap_results_n1.append({'eps': eps_val, 'lambda_max': np.nan, 'se': np.nan})
        else:
            lyap_results_n1.append({'eps': eps_val, 'lambda_max': np.nan, 'se': np.nan})

        lu = lyap_results_unc[-1]['lambda_max']
        lc = lyap_results_ctrl[-1]['lambda_max']
        l1 = lyap_results_n1[-1]['lambda_max']
        print(f"    eps={eps_val:.2f}: unc={lu:+.5f}, N=1={l1:+.5f}, N=5={lc:+.5f}")

    results['eps_sweep'] = lyap_results_ctrl  # backward compat
    results['eps_sweep_uncontrolled'] = lyap_results_unc
    results['eps_sweep_n1'] = lyap_results_n1

    # (d) Monte Carlo robustness - extended to ±5%, ±10%, ±15%, ±20%
    print("  Monte Carlo robustness (500 trials)...")
    np.random.seed(42)
    n_trials = 500
    ft = p.get('forcing_type', 'cos')
    mc_results = {}

    if 5 in designs:
        a5, b5 = designs[5]['a'], designs[5]['b']

        for pct in [0.05, 0.10, 0.15, 0.20]:
            margins = []
            for _ in range(n_trials):
                delta_pert = p['delta'] * (1 + pct * (2 * np.random.random() - 1))
                Omega_pert = p['Omega'] * (1 + pct * (2 * np.random.random() - 1))

                c0_pert = duffing_c0(delta_pert)
                alpha_pert, beta_pert = get_duffing_kernels(5, Omega_pert)
                c_exist_pert, d_exist_pert = get_duffing_existing(p['gamma_exist'], alpha_pert, beta_pert, ft)

                c_f = np.zeros(5)
                d_f = np.zeros(5)
                for k in range(len(c_exist_pert)):
                    c_f[k] += c_exist_pert[k]
                    d_f[k] += d_exist_pert[k]
                for k in range(5):
                    c_f[k] += a5[k] * alpha_pert[k] - b5[k] * beta_pert[k]
                    d_f[k] += b5[k] * alpha_pert[k] + a5[k] * beta_pert[k]

                theta = np.linspace(0, 2 * np.pi, 500)
                M_vals = evaluate_melnikov(theta, c0_pert, c_f, d_f)
                margins.append(-float(np.max(M_vals)))

            margins_arr = np.array(margins)
            success = float(np.mean(margins_arr > 0))
            mc_results[f'{int(pct*100)}pct'] = {
                'margins': margins,
                'success': success,
                'min_margin': float(np.min(margins_arr)),
                'mean_margin': float(np.mean(margins_arr)),
                'std_margin': float(np.std(margins_arr)),
            }
            pct_str = f"±{int(pct*100)}%"
            print(f"    {pct_str}: success={success:.1%}, min={np.min(margins_arr):.4f}, mean={np.mean(margins_arr):.4f}, std={np.std(margins_arr):.4f}")

    results['monte_carlo'] = mc_results
    # Backward compatibility
    results['monte_carlo']['margins_5pct'] = mc_results.get('5pct', {}).get('margins', [])
    results['monte_carlo']['margins_10pct'] = mc_results.get('10pct', {}).get('margins', [])
    results['monte_carlo']['success_5pct'] = mc_results.get('5pct', {}).get('success', None)
    results['monte_carlo']['success_10pct'] = mc_results.get('10pct', {}).get('success', None)

    return results


def generate_robustness_figures(rob_results):
    """Generate Figures 10-11 for robustness."""
    print("\n  Generating robustness figures...")

    # --- Figure 10: 2x2 robustness panels ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Damping mismatch
    ax = axes[0][0]
    delta_range = np.array(rob_results['damping_mismatch']['delta_range'])
    delta_pct = (delta_range / DUFFING_PARAMS['delta'] - 1) * 100
    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        margins = rob_results['damping_mismatch']['margins'][str(N)]
        ax.plot(delta_pct, margins, color=color, label=label, linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r'Damping mismatch (\%)')
    ax.set_ylabel(r'Realized margin')
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (b) Frequency mismatch
    ax = axes[0][1]
    Omega_range = np.array(rob_results['freq_mismatch']['Omega_range'])
    Omega_pct = (Omega_range / DUFFING_PARAMS['Omega'] - 1) * 100
    for N, color, label in [(1, C_N1, 'N=1'), (3, C_N3, 'N=3'), (5, C_N5, 'N=5')]:
        margins = rob_results['freq_mismatch']['margins'][str(N)]
        ax.plot(Omega_pct, margins, color=color, label=label, linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r'Frequency mismatch (\%)')
    ax.set_ylabel(r'Realized margin')
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (c) Lyapunov vs eps: uncontrolled, N=1, N=5
    ax = axes[1][0]
    eps_unc = rob_results.get('eps_sweep_uncontrolled', [])
    eps_ctrl = rob_results.get('eps_sweep', [])
    eps_n1 = rob_results.get('eps_sweep_n1', [])

    if eps_unc:
        ev = [d['eps'] for d in eps_unc]
        lv = [d['lambda_max'] for d in eps_unc]
        ax.plot(ev, lv, 'o-', color=C_UNCONTROLLED, markersize=5, label='Uncontrolled')
    if eps_n1:
        ev = [d['eps'] for d in eps_n1]
        lv = [d['lambda_max'] for d in eps_n1]
        ax.plot(ev, lv, 's-', color=C_N1, markersize=4, label='Controlled (N=1)')
    if eps_ctrl:
        ev = [d['eps'] for d in eps_ctrl]
        lv = [d['lambda_max'] for d in eps_ctrl]
        ax.plot(ev, lv, '^-', color=C_N5, markersize=4, label='Controlled (N=5)')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\lambda_{\max}$')
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # (d) Success rate vs mismatch level
    ax = axes[1][1]
    mc = rob_results.get('monte_carlo', {})
    pct_levels = []
    success_rates = []
    mean_margins = []
    for pct_key in ['5pct', '10pct', '15pct', '20pct']:
        if pct_key in mc and isinstance(mc[pct_key], dict):
            pct_val = int(pct_key.replace('pct', ''))
            pct_levels.append(pct_val)
            success_rates.append(mc[pct_key]['success'] * 100)
            mean_margins.append(mc[pct_key]['mean_margin'])

    if pct_levels:
        ax2 = ax.twinx()
        ax.bar(pct_levels, success_rates, width=3, color=COLORS[2], alpha=0.6, label='Success rate')
        ax2.plot(pct_levels, mean_margins, 'o-', color=COLORS[0], markersize=6, label='Mean margin')
        ax.set_xlabel('Mismatch level (%)')
        ax.set_ylabel('Success rate (%)')
        ax2.set_ylabel('Mean margin')
        ax.set_ylim([0, 105])
        ax2.axhline(0, color='gray', linewidth=0.5)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower left')

    ax.text(-0.12, 1.05, '(d)', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig10_robustness.pdf'))
    plt.close(fig)
    print("    Figure 10 saved.")

    # --- Figure 11: Monte Carlo histogram ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    mc5 = rob_results['monte_carlo']['margins_5pct']
    mc10 = rob_results['monte_carlo']['margins_10pct']
    if mc5:
        ax.hist(mc5, bins=30, alpha=0.6, color=COLORS[2], label=r'$\pm 5\%$ mismatch', density=True)
    if mc10:
        ax.hist(mc10, bins=30, alpha=0.6, color=COLORS[3], label=r'$\pm 10\%$ mismatch', density=True)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Realized margin')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    if rob_results['monte_carlo']['success_5pct'] is not None:
        ax.text(0.02, 0.95,
                f"±5%: {rob_results['monte_carlo']['success_5pct']:.0%} success\n"
                f"±10%: {rob_results['monte_carlo']['success_10pct']:.0%} success",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig11_montecarlo.pdf'))
    plt.close(fig)
    print("    Figure 11 saved.")


# =============================================================================
# Graphical abstract
# =============================================================================

def generate_graphical_abstract(duffing_results):
    """Generate graphical abstract for Nonlinear Dynamics."""
    print("\n  Generating graphical abstract...")

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 0.3, 1])

    theta = np.array(duffing_results['uncontrolled']['theta'])

    # Left: uncontrolled
    ax1 = fig.add_subplot(gs[0])
    M_unc = np.array(duffing_results['uncontrolled']['M_theta'])
    ax1.plot(theta, M_unc, color=C_UNCONTROLLED, linewidth=1.5)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.fill_between(theta, M_unc, 0, where=M_unc > 0, alpha=0.2, color='red')
    ax1.fill_between(theta, M_unc, 0, where=M_unc < 0, alpha=0.2, color='blue')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax1.set_title('Uncontrolled (chaos)', fontsize=10)
    ax1.set_xlim([0, 2 * np.pi])

    # Center: arrow
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.axis('off')
    ax_mid.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                    xycoords='axes fraction')
    ax_mid.text(0.5, 0.65, 'SDP\nsynthesis', ha='center', va='center',
                fontsize=10, fontweight='bold', transform=ax_mid.transAxes)

    # Right: controlled
    ax2 = fig.add_subplot(gs[2])
    if 'N5' in duffing_results and duffing_results['N5']['max_margin']['M_theta'] is not None:
        M_ctrl = np.array(duffing_results['N5']['max_margin']['M_theta'])
        ax2.plot(theta, M_ctrl, color=C_N5, linewidth=1.5)
        margin = duffing_results['N5']['max_margin']['margin']
        ax2.axhline(-margin, color=C_N5, linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\mathcal{M}(\theta)$')
    ax2.set_title('Controlled (regular)', fontsize=10)
    ax2.set_xlim([0, 2 * np.pi])

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'graphical_abstract.pdf'))
    plt.close(fig)
    print("    Graphical abstract saved.")


# =============================================================================
# Summary table
# =============================================================================

def build_summary_table(duffing_res, josephson_res, ship_res, rob_res):
    """Build the summary performance table."""
    print("\n" + "=" * 70)
    print("SUMMARY PERFORMANCE TABLE")
    print("=" * 70)

    p_d = DUFFING_PARAMS
    p_j = JOSEPHSON_PARAMS

    rows = []
    header = f"{'System':<16} {'N':>3} {'E*':>8} {'tau*':>8} {'lambda_max':>11} {'±5%':>6} {'±10%':>7} {'Solve(s)':>8}"
    print(header)
    print("-" * len(header))

    # Duffing
    for N in [1, 3, 5]:
        key = f'N{N}'
        if key in duffing_res:
            mm = duffing_res[key]['max_margin']
            me = duffing_res[key]['min_energy']
            e_star = f"{me['energy']:.4f}" if me['energy'] is not None else 'N/A'
            tau_star = f"{mm['margin']:.4f}" if mm['margin'] is not None else 'N/A'
            t_solve = f"{mm['solve_time']:.3f}" if mm['solve_time'] else 'N/A'
            row = f"{'MEMS/Duffing':<16} {N:>3} {e_star:>8} {tau_star:>8} {'--':>11} {'--':>6} {'--':>7} {t_solve:>8}"
            print(row)
            rows.append({
                'system': 'MEMS/Duffing', 'N': N,
                'E_star': me['energy'], 'tau_star': mm['margin'],
                'solve_time': mm['solve_time'],
            })

    # Josephson
    for N in [1, 3, 5]:
        key = f'N{N}'
        if key in josephson_res:
            mm = josephson_res[key]['max_margin']
            me = josephson_res[key]['min_energy']
            e_star = f"{me['energy']:.4f}" if me['energy'] is not None else 'N/A'
            tau_star = f"{mm['margin']:.4f}" if mm['margin'] is not None else 'N/A'
            t_solve = f"{mm['solve_time']:.3f}" if mm['solve_time'] else 'N/A'
            row = f"{'Josephson':<16} {N:>3} {e_star:>8} {tau_star:>8} {'--':>11} {'--':>6} {'--':>7} {t_solve:>8}"
            print(row)
            rows.append({
                'system': 'Josephson', 'N': N,
                'E_star': me['energy'], 'tau_star': mm['margin'],
                'solve_time': mm['solve_time'],
            })

    # Ship
    for N in [3, 5]:
        key = f'N{N}'
        if key in ship_res:
            mm = ship_res[key]['max_margin']
            me = ship_res[key]['min_energy']
            e_star = f"{me['energy']:.4f}" if me.get('energy') is not None else 'N/A'
            tau_star = f"{mm['margin']:.4f}" if mm['margin'] is not None else 'N/A'
            t_solve = f"{mm['solve_time']:.3f}" if mm['solve_time'] else 'N/A'
            row = f"{'Ship roll':<16} {N:>3} {e_star:>8} {tau_star:>8} {'--':>11} {'--':>6} {'--':>7} {t_solve:>8}"
            print(row)
            rows.append({
                'system': 'Ship roll', 'N': N,
                'E_star': me.get('energy'), 'tau_star': mm['margin'],
                'solve_time': mm['solve_time'],
            })

    # Add Monte Carlo results
    if rob_res and rob_res['monte_carlo']['success_5pct'] is not None:
        print(f"\nMonte Carlo (N=5 Duffing): ±5% success = {rob_res['monte_carlo']['success_5pct']:.0%}, "
              f"±10% success = {rob_res['monte_carlo']['success_10pct']:.0%}")

    return rows


# =============================================================================
# Main execution
# =============================================================================

def main():
    t_start = time.time()

    # --- Duffing ---
    duffing_sdp = compute_duffing_sdp_results()
    generate_duffing_figures(duffing_sdp)

    # --- Josephson ---
    josephson_sdp = compute_josephson_sdp_results()
    generate_josephson_figures(josephson_sdp)

    # --- Ship roll ---
    ship_sdp = compute_ship_sdp_results()
    generate_ship_figures(ship_sdp)

    # --- Robustness ---
    rob_results = compute_robustness_results()
    generate_robustness_figures(rob_results)

    # --- Graphical abstract ---
    generate_graphical_abstract(duffing_sdp)

    # --- Summary table ---
    summary = build_summary_table(duffing_sdp, josephson_sdp, ship_sdp, rob_results)

    # --- Save all results ---
    all_data = {
        'duffing': duffing_sdp,
        'josephson': josephson_sdp,
        'ship': ship_sdp,
        'robustness': {
            'damping_mismatch': rob_results['damping_mismatch'],
            'freq_mismatch': rob_results['freq_mismatch'],
            'eps_sweep': rob_results['eps_sweep'],
            'monte_carlo_success_5pct': rob_results['monte_carlo']['success_5pct'],
            'monte_carlo_success_10pct': rob_results['monte_carlo']['success_10pct'],
        },
        'summary': summary,
    }

    with open(os.path.join(DATA_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else None)

    # Save parameters
    params = {
        'duffing': DUFFING_PARAMS,
        'josephson': JOSEPHSON_PARAMS,
        'ship': SHIP_PARAMS,
    }
    with open(os.path.join(DATA_DIR, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL COMPUTATIONS COMPLETE in {elapsed:.1f}s")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Data saved to: {DATA_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
