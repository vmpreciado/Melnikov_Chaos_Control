"""
SDP/SOCP synthesis engine for Melnikov-based chaos control.

Uses CVXPy to solve:
  1. Max-margin suppression (maximize tau s.t. M(theta) >= tau or -M(theta) >= tau)
  2. Min-energy suppression (minimize energy s.t. margin >= tau_min)
  3. Feasibility

Decision variables: Cartesian coefficients (a_k, b_k).
"""

import numpy as np
import cvxpy as cp
import time


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
):
    """
    Solve chaos suppression via SDP with Fejer-Riesz certificate.

    The total Melnikov function is:
      M(theta) = c0_total + sum_k [(c_k^exist + c_k^ctrl) cos(k theta)
                                 + (d_k^exist + d_k^ctrl) sin(k theta)]
    where c_k^ctrl = a_k * alpha_k - b_k * beta_k (control contribution).

    Suppression means making M(theta) have definite sign.
    If suppress_to_negative=True, we certify -M(theta) >= tau (natural when c0 < 0).
    If suppress_to_negative=False, we certify M(theta) >= tau.

    Parameters
    ----------
    c0_total : float
        Total DC Melnikov coefficient (damping + bias).
    c_exist : np.ndarray, shape (N_total,)
        Existing perturbation's cosine Fourier coefficients (may be longer than N_ctrl).
    d_exist : np.ndarray, shape (N_total,)
        Existing perturbation's sine Fourier coefficients.
    alpha : np.ndarray, shape (N_ctrl,)
        Control response kernels alpha_k for k=1,...,N_ctrl.
    beta : np.ndarray, shape (N_ctrl,)
        Control response kernels beta_k for k=1,...,N_ctrl.
    N_ctrl : int
        Number of control harmonics.
    P_max : float
        Total power budget.
    A_max : np.ndarray or None
        Per-harmonic amplitude bounds.
    objective : str
        'max_margin', 'min_energy', or 'feasibility'.
    tau_min : float
        Minimum margin for min_energy problem.
    suppress_to_negative : bool
        If True, suppress by making -M(theta) >= tau. If False, M(theta) >= tau.
    """
    N_exist = len(c_exist)
    N_total = max(N_exist, N_ctrl)  # total Fourier order

    # Decision variables
    a = cp.Variable(N_ctrl)
    b = cp.Variable(N_ctrl)

    # Build total Fourier coefficients (affine in a, b)
    c_total = np.zeros(N_total)  # will be cp expression
    d_total = np.zeros(N_total)

    # Start with existing
    c_total_list = []
    d_total_list = []
    for k in range(N_total):
        c_k = 0.0
        d_k = 0.0
        if k < N_exist:
            c_k = c_k + c_exist[k]
            d_k = d_k + d_exist[k]
        if k < N_ctrl:
            c_k = c_k + alpha[k] * a[k] - beta[k] * b[k]
            d_k = d_k + alpha[k] * b[k] + beta[k] * a[k]
        c_total_list.append(c_k)
        d_total_list.append(d_k)

    # If suppressing to negative: work with -M(theta) >= tau
    # -M = -c0_total + sum_k [-c_k_total cos(k theta) - d_k_total sin(k theta)]
    if suppress_to_negative:
        eff_c0 = -c0_total
        eff_c = [-c for c in c_total_list]
        eff_d = [-d for d in d_total_list]
    else:
        eff_c0 = c0_total
        eff_c = c_total_list
        eff_d = d_total_list

    # Fejer-Riesz certificate: H >= 0 with anti-diagonal sum constraints
    # We need a (N_total+1) x (N_total+1) Hermitian PSD matrix H
    n = N_total + 1
    H_re = cp.Variable((n, n), symmetric=True)
    H_im = cp.Variable((n, n))

    constraints = []
    constraints.append(H_im == -H_im.T)  # antisymmetric

    # PSD via real block representation
    M_real = cp.bmat([[H_re, -H_im], [H_im, H_re]])
    constraints.append(M_real >> 0)

    # Margin variable
    if objective == 'max_margin':
        tau = cp.Variable()
    elif objective == 'min_energy':
        tau = tau_min
    else:
        tau = 0.0

    # k=0 constraint: sum of diagonal = eff_c0 - tau
    constraints.append(cp.sum(cp.diag(H_re)) == eff_c0 - tau)
    constraints.append(cp.sum(cp.diag(H_im)) == 0)

    # k=1,...,N_total: anti-diagonal sum constraints
    for k in range(1, N_total + 1):
        # Complex Fourier: chat_k = (eff_c[k-1] - i*eff_d[k-1]) / 2
        re_sum = 0
        im_sum = 0
        for j in range(n - k):
            re_sum = re_sum + H_re[j, j + k]
            im_sum = im_sum + H_im[j, j + k]

        constraints.append(re_sum == eff_c[k - 1] / 2.0)
        constraints.append(im_sum == -eff_d[k - 1] / 2.0)

    # Power constraint
    if np.isfinite(P_max):
        constraints.append(cp.sum(cp.square(a) + cp.square(b)) <= P_max)

    # Per-harmonic amplitude bounds
    if A_max is not None:
        for k in range(N_ctrl):
            constraints.append(cp.norm(cp.hstack([a[k], b[k]]), 2) <= A_max[k])

    # Objective
    if objective == 'max_margin':
        prob = cp.Problem(cp.Maximize(tau), constraints)
    elif objective == 'min_energy':
        prob = cp.Problem(cp.Minimize(cp.sum(cp.square(a) + cp.square(b))), constraints)
    else:
        prob = cp.Problem(cp.Minimize(0), constraints)

    # Solve
    t_start = time.time()
    try:
        prob.solve(solver=cp.SCS, max_iters=20000, eps=1e-9, verbose=False)
    except (cp.SolverError, Exception):
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except (cp.SolverError, Exception):
            pass
    solve_time = time.time() - t_start

    result = {
        'status': prob.status,
        'solve_time': solve_time,
        'optimal_value': prob.value,
        'N_ctrl': N_ctrl,
        'objective': objective,
    }

    if prob.status in ['optimal', 'optimal_inaccurate']:
        a_val = np.array(a.value).flatten()
        b_val = np.array(b.value).flatten()
        A_val = np.sqrt(a_val**2 + b_val**2)
        phi_val = np.arctan2(b_val, a_val)

        # Compute actual total Fourier coefficients
        c_f = np.zeros(N_total)
        d_f = np.zeros(N_total)
        for k in range(N_total):
            if k < N_exist:
                c_f[k] += c_exist[k]
                d_f[k] += d_exist[k]
            if k < N_ctrl:
                c_f[k] += a_val[k] * alpha[k] - b_val[k] * beta[k]
                d_f[k] += b_val[k] * alpha[k] + a_val[k] * beta[k]

        # Evaluate M(theta) on grid
        theta_grid = np.linspace(0, 2 * np.pi, 1000)
        M_vals = np.full_like(theta_grid, c0_total)
        for k in range(N_total):
            M_vals += c_f[k] * np.cos((k + 1) * theta_grid) + d_f[k] * np.sin((k + 1) * theta_grid)

        if suppress_to_negative:
            realized_margin = -np.max(M_vals)  # margin for -M >= tau
        else:
            realized_margin = np.min(M_vals)

        result.update({
            'a': a_val,
            'b': b_val,
            'A': A_val,
            'phi': phi_val,
            'margin': realized_margin,
            'energy': float(np.sum(A_val**2)),
            'c_fourier': c_f,
            'd_fourier': d_f,
            'c0_total': c0_total,
            'M_theta': M_vals,
            'theta_grid': theta_grid,
        })
        if objective == 'max_margin':
            result['tau_sdp'] = float(tau.value) if hasattr(tau, 'value') else tau
    else:
        result.update({
            'a': None, 'b': None, 'A': None, 'phi': None,
            'margin': None, 'energy': None,
            'c_fourier': None, 'd_fourier': None,
        })

    return result


def evaluate_melnikov(theta_grid, c0, c_fourier, d_fourier):
    """Evaluate M(theta) on a grid."""
    M = np.full_like(theta_grid, c0)
    for k in range(len(c_fourier)):
        M += c_fourier[k] * np.cos((k + 1) * theta_grid) + d_fourier[k] * np.sin((k + 1) * theta_grid)
    return M


if __name__ == "__main__":
    from kernels import duffing_c0, duffing_kernels_analytical

    print("=" * 70)
    print("SDP SOLVER VERIFICATION")
    print("=" * 70)

    delta = 0.15
    Omega = 1.0
    c0 = duffing_c0(delta)  # -0.2
    gamma_exist = 0.3  # chaos-inducing sine forcing at fundamental

    N_max = 5
    alpha_arr = np.zeros(N_max)
    beta_arr = np.zeros(N_max)
    for k in range(N_max):
        a, b = duffing_kernels_analytical(k + 1, Omega)
        alpha_arr[k] = a
        beta_arr[k] = b

    # Existing perturbation: gamma * sin(Omega*t) at the fundamental
    # In Cartesian: a_1 = 0, b_1 = gamma
    # c_1^exist = 0*alpha_1 - gamma*beta_1 = -gamma*beta_1
    # d_1^exist = gamma*alpha_1 + 0*beta_1 = 0
    c_exist = np.zeros(1)
    d_exist = np.zeros(1)
    c_exist[0] = -gamma_exist * beta_arr[0]
    d_exist[0] = 0.0

    print(f"\nDuffing: delta={delta}, Omega={Omega}, gamma_exist={gamma_exist}")
    print(f"c0 = {c0:.6f}")
    print(f"c_exist[0] = {c_exist[0]:.6f}")

    # Check uncontrolled M(theta)
    theta = np.linspace(0, 2 * np.pi, 500)
    M_uncontrolled = c0 + c_exist[0] * np.cos(theta)
    print(f"Uncontrolled M: min={np.min(M_uncontrolled):.4f}, max={np.max(M_uncontrolled):.4f}")
    print(f"Sign changes: YES (chaos)" if np.min(M_uncontrolled) * np.max(M_uncontrolled) < 0 else "No sign changes")

    # Solve for different N_ctrl
    for N_ctrl in [1, 2, 3, 5]:
        print(f"\n--- N_ctrl = {N_ctrl}, max-margin, P_max=0.5 ---")
        result = solve_suppression_sdp(
            c0_total=c0,
            c_exist=c_exist,
            d_exist=d_exist,
            alpha=alpha_arr[:N_ctrl],
            beta=beta_arr[:N_ctrl],
            N_ctrl=N_ctrl,
            P_max=0.5,
            objective='max_margin',
            suppress_to_negative=True,
        )
        print(f"  Status: {result['status']}")
        if result['margin'] is not None:
            print(f"  Margin: {result['margin']:.6f}")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Amplitudes: {result['A']}")
            print(f"  M range: [{np.min(result['M_theta']):.4f}, {np.max(result['M_theta']):.4f}]")
            print(f"  Solve time: {result['solve_time']:.3f}s")

    # Min-energy test
    print(f"\n--- N_ctrl = 3, min-energy, tau_min=0.05 ---")
    result_e = solve_suppression_sdp(
        c0_total=c0,
        c_exist=c_exist,
        d_exist=d_exist,
        alpha=alpha_arr[:3],
        beta=beta_arr[:3],
        N_ctrl=3,
        P_max=np.inf,
        objective='min_energy',
        tau_min=0.05,
        suppress_to_negative=True,
    )
    print(f"  Status: {result_e['status']}")
    if result_e['margin'] is not None:
        print(f"  Margin: {result_e['margin']:.6f}")
        print(f"  Energy: {result_e['energy']:.6f}")
        print(f"  Amplitudes: {result_e['A']}")
