"""
SDP/SOCP synthesis engine for Melnikov-based chaos control.

Solves, via CVXPy with an exact Fejer-Riesz certificate:
  1. Max-margin suppression   (maximize tau s.t. -M(theta) >= tau)
  2. Min-power suppression     (minimize sum A_k^2 s.t. margin >= tau_min)

Two-branch certificate
-----------------------
The twin-well and pendulum saddles each carry two homoclinic branches. With
no DC bias the mirror branch satisfies M_L(theta) = 2 c0 - M(theta), so its
kernels flip sign while c0 is invariant. Sign-definiteness on both branches is
the two-sided band

    tau <= -M(theta) <= 2|c0| - tau,   for all theta,

certified by two Fejer-Riesz blocks with the common constant term |c0| - tau.
Set both_branches=True to impose it.

Conventions (corrected Theorem 1)
---------------------------------
  M(theta) = c0 + sum_k [c_k cos(k theta) + d_k sin(k theta)]
  c_k = a_k alpha_k - b_k beta_k,  d_k = b_k alpha_k + a_k beta_k

Solvers: CLARABEL primary, SCS (eps 1e-9) fallback. Every solve is followed by
a dense-grid feasibility check (40001 points) on both branches.

Decision variables: Cartesian harmonic coefficients (a_k, b_k).
"""

import time
import numpy as np
import cvxpy as cp

GRID_N = 40001
TOL_FEAS = 1e-6


def _total_coeffs(c_exist, d_exist, alpha, beta, a, b, K):
    """Total Melnikov Fourier coefficients (cvxpy expressions), order K."""
    N_exist = len(c_exist)
    N_ctrl = a.shape[0]
    C, D = [], []
    for k in range(K):
        ck = c_exist[k] if k < N_exist else 0.0
        dk = d_exist[k] if k < N_exist else 0.0
        if k < N_ctrl:
            ck = ck + a[k] * alpha[k] - b[k] * beta[k]
            dk = dk + b[k] * alpha[k] + a[k] * beta[k]
        C.append(ck)
        D.append(dk)
    return C, D


def _fejer_riesz_block(t0_expr, A, B, K):
    """Constraints certifying t0 + sum_k (A_k cos + B_k sin) >= 0 for all theta."""
    H = cp.Variable((K + 1, K + 1), hermitian=True)
    cons = [H >> 0,
            cp.real(cp.trace(H)) == t0_expr,
            cp.imag(cp.trace(H)) == 0]
    for k in range(1, K + 1):
        s = sum(H[j, j + k] for j in range(K + 1 - k))
        cons += [cp.real(s) == A[k - 1] / 2.0,
                 cp.imag(s) == -B[k - 1] / 2.0]
    return cons


def _run_solver(prob, prefer='CLARABEL'):
    """CLARABEL primary, SCS (eps 1e-9) fallback."""
    order = ['CLARABEL', 'SCS'] if prefer == 'CLARABEL' else ['SCS', 'CLARABEL']
    last = None
    for name in order:
        try:
            if name == 'SCS':
                prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000, verbose=False)
            else:
                prob.solve(solver=cp.CLARABEL, verbose=False)
            last = prob.status
            if prob.status in ('optimal', 'infeasible'):
                return prob.status
        except Exception as exc:  # noqa: BLE001
            last = f'error:{exc}'
    return last


def evaluate_melnikov(theta_grid, c0, c_fourier, d_fourier):
    """Evaluate M(theta) on a grid."""
    M = np.full_like(theta_grid, float(c0))
    for k in range(len(c_fourier)):
        M += c_fourier[k] * np.cos((k + 1) * theta_grid) + \
            d_fourier[k] * np.sin((k + 1) * theta_grid)
    return M


def verified_margins(c0, c_f, d_f, grid_n=GRID_N):
    """Dense-grid margins on the computed branch and the mirror branch.

    Returns (margin_R, margin_L) where
      margin_R = min_theta (-M(theta)),
      margin_L = min_theta (-M_L(theta)) with M_L = 2 c0 - M.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, grid_n)
    M = evaluate_melnikov(theta, c0, c_f, d_f)
    margin_R = float(np.min(-M))
    margin_L = float(np.min(M - 2.0 * c0))  # -M_L = M - 2 c0
    return margin_R, margin_L


def solve_suppression_sdp(
    c0_total,
    c_exist,
    d_exist,
    alpha,
    beta,
    N_ctrl,
    P_max=np.inf,
    A_max=None,
    objective='max_margin',
    tau_min=0.01,
    suppress_to_negative=True,
    both_branches=False,
    peak_max=None,
):
    """Solve chaos-suppression synthesis with the Fejer-Riesz certificate.

    If both_branches=True, impose the two-sided band on the figure-eight or
    pendulum pair. suppress_to_negative is kept for backward compatibility; the
    two-branch certificate is symmetric in sign and ignores it.
    """
    c_exist = np.asarray(c_exist, dtype=float)
    d_exist = np.asarray(d_exist, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    K = max(len(c_exist), N_ctrl)

    a = cp.Variable(N_ctrl)
    b = cp.Variable(N_ctrl)
    C, D = _total_coeffs(c_exist, d_exist, alpha, beta, a, b, K)

    if objective == 'max_margin':
        tau = cp.Variable()
    else:
        tau = tau_min

    # Computed branch: certify -M(theta) - tau >= 0.
    cons = _fejer_riesz_block(-c0_total - tau, [-c for c in C], [-d for d in D], K)
    # Mirror branch: certify -M_L(theta) - tau = M(theta) - 2 c0 - tau >= 0.
    if both_branches:
        cons += _fejer_riesz_block(-c0_total - tau, C, D, K)

    if np.isfinite(P_max):
        cons.append(cp.sum_squares(a) + cp.sum_squares(b) <= P_max)
    if A_max is not None:
        for k in range(N_ctrl):
            cons.append(cp.norm(cp.hstack([a[k], b[k]]), 2) <= A_max[k])
    # Peak-amplitude constraint |u(t)| <= peak_max, u(t) = sum_k (a_k cos + b_k sin),
    # certified exactly by two Fejer-Riesz blocks on peak_max +/- u(t).
    if peak_max is not None:
        au = [a[k] for k in range(N_ctrl)]
        bu = [b[k] for k in range(N_ctrl)]
        cons += _fejer_riesz_block(peak_max, au, bu, N_ctrl)
        cons += _fejer_riesz_block(peak_max, [-x for x in au], [-x for x in bu], N_ctrl)

    if objective == 'max_margin':
        prob = cp.Problem(cp.Maximize(tau), cons)
    else:
        prob = cp.Problem(cp.Minimize(cp.sum_squares(a) + cp.sum_squares(b)), cons)

    t0 = time.time()
    status = _run_solver(prob, prefer='CLARABEL')
    # Dense-grid feasibility check; re-solve with the fallback on failure.
    a_val = b_val = None
    if status in ('optimal', 'optimal_inaccurate') and a.value is not None:
        a_val = np.asarray(a.value, dtype=float).reshape(-1)
        b_val = np.asarray(b.value, dtype=float).reshape(-1)
        c_f, d_f = _realized_coeffs(c_exist, d_exist, alpha, beta, a_val, b_val, K)
        tau_req = float(tau.value) if objective == 'max_margin' else float(tau_min)
        mR, mL = verified_margins(c0_total, c_f, d_f)
        ok = mR >= tau_req - TOL_FEAS and (not both_branches or mL >= tau_req - TOL_FEAS)
        if not ok:
            _run_solver(prob, prefer='SCS')
            if a.value is not None:
                a_val = np.asarray(a.value, dtype=float).reshape(-1)
                b_val = np.asarray(b.value, dtype=float).reshape(-1)
    solve_time = time.time() - t0

    result = {'status': prob.status, 'solve_time': solve_time,
              'optimal_value': prob.value, 'N_ctrl': N_ctrl,
              'objective': objective, 'both_branches': both_branches}

    if prob.status in ('optimal', 'optimal_inaccurate') and a_val is not None:
        A_val = np.sqrt(a_val**2 + b_val**2)
        phi_val = np.arctan2(b_val, a_val)
        c_f, d_f = _realized_coeffs(c_exist, d_exist, alpha, beta, a_val, b_val, K)
        mR, mL = verified_margins(c0_total, c_f, d_f)
        theta_grid = np.linspace(0.0, 2.0 * np.pi, 1000)
        M_vals = evaluate_melnikov(theta_grid, c0_total, c_f, d_f)
        result.update({
            'a': a_val, 'b': b_val, 'A': A_val, 'phi': phi_val,
            'margin': mR, 'margin_L': mL,
            'energy': float(np.sum(A_val**2)),
            'c_fourier': c_f, 'd_fourier': d_f, 'c0_total': c0_total,
            'M_theta': M_vals, 'theta_grid': theta_grid,
        })
        if objective == 'max_margin':
            result['tau_sdp'] = float(tau.value)
    else:
        result.update({'a': None, 'b': None, 'A': None, 'phi': None,
                       'margin': None, 'margin_L': None, 'energy': None,
                       'c_fourier': None, 'd_fourier': None, 'M_theta': None})
    return result


def _realized_coeffs(c_exist, d_exist, alpha, beta, a_val, b_val, K):
    """Numeric total Fourier coefficients of the realized design."""
    N_exist = len(c_exist)
    N_ctrl = len(a_val)
    c_f = np.zeros(K)
    d_f = np.zeros(K)
    for k in range(K):
        if k < N_exist:
            c_f[k] += c_exist[k]
            d_f[k] += d_exist[k]
        if k < N_ctrl:
            c_f[k] += a_val[k] * alpha[k] - b_val[k] * beta[k]
            d_f[k] += b_val[k] * alpha[k] + a_val[k] * beta[k]
    return c_f, d_f


if __name__ == "__main__":
    from kernels import duffing_c0, duffing_kernels_analytical

    delta, Omega = 0.80, 1.0
    c0 = duffing_c0(delta)
    gamma = [1.20, 0.60, 0.32]
    Nmax = 5
    alpha_arr = np.zeros(Nmax)
    beta_arr = np.zeros(Nmax)
    for k in range(Nmax):
        alpha_arr[k], beta_arr[k] = duffing_kernels_analytical(k + 1, Omega)
    # Cosine existing forcing -> d_k = gamma_k beta_k, c_k = 0.
    c_ex = np.zeros(3)
    d_ex = np.array([gamma[k] * beta_arr[k] for k in range(3)])
    print("Two-branch max-margin (Duffing, P_max=2.40):")
    for N in [1, 2, 3, 5]:
        r = solve_suppression_sdp(c0, c_ex, d_ex, alpha_arr[:N], beta_arr[:N],
                                  N, P_max=2.40, objective='max_margin',
                                  both_branches=True)
        print(f"  N={N}: tau={r['margin']:.4f} mirror={r['margin_L']:.4f} [{r['status']}]")
