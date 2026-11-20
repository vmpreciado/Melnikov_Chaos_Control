"""
Simulation and validation engine for Melnikov-based chaos control.

Provides:
  1. Poincare section computation (stroboscopic sampling)
  2. Stable/unstable manifold computation
  3. Largest Lyapunov exponent via variational equations
  4. Melnikov function evaluation
  5. Manifold splitting distance (validation)
"""

import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# System definitions (full non-averaged dynamics)
# =============================================================================

def duffing_rhs(t, z, eps, delta, forcing_coeffs, Omega):
    """
    Full Duffing oscillator: x'' + x - x^3 = eps * (-delta*x' + F(t))
    z = [x, y], F(t) = sum_k (a_k cos(kOmega t) + b_k sin(kOmega t))
    forcing_coeffs: list of (a_k, b_k) tuples for k=1,2,...
    """
    x, y = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)
    dxdt = y
    dydt = x - x**3 + eps * (-delta * y + F)
    return [dxdt, dydt]


def duffing_rhs_with_variational(t, z, eps, delta, forcing_coeffs, Omega):
    """Duffing + tangent linear equations for Lyapunov exponent."""
    x, y, dx, dy = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)

    dxdt = y
    dydt = x - x**3 + eps * (-delta * y + F)

    # Jacobian of the full system
    dfdx = 1 - 3 * x**2
    dfdy = -eps * delta

    ddxdt = dy
    ddydt = dfdx * dx + dfdy * dy

    return [dxdt, dydt, ddxdt, ddydt]


def pendulum_rhs(t, z, eps, alpha_damp, I_dc, forcing_coeffs, Omega):
    """
    Josephson junction / pendulum: phi'' + sin(phi) = eps*(-alpha*phi' + I_dc + F(t))
    z = [phi, dphi]
    """
    phi, dphi = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)
    dphidt = dphi
    ddphidt = -np.sin(phi) + eps * (-alpha_damp * dphi + I_dc + F)
    return [dphidt, ddphidt]


def pendulum_rhs_with_variational(t, z, eps, alpha_damp, I_dc, forcing_coeffs, Omega):
    """Pendulum + tangent linear equations."""
    phi, dphi, dx, dy = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)

    dphidt = dphi
    ddphidt = -np.sin(phi) + eps * (-alpha_damp * dphi + I_dc + F)

    dfdx = -np.cos(phi)
    dfdy = -eps * alpha_damp

    ddxdt = dy
    ddydt = dfdx * dx + dfdy * dy

    return [dphidt, ddphidt, ddxdt, ddydt]


def ship_rhs(t, z, eps, delta, alpha_s, forcing_coeffs, Omega):
    """
    Ship roll: phi'' - phi + alpha_s*phi^3 = eps*(-delta*phi' + F(t))
    Same as Duffing with softening.
    """
    x, y = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)
    dxdt = y
    dydt = x - alpha_s * x**3 + eps * (-delta * y + F)
    return [dxdt, dydt]


def ship_rhs_with_variational(t, z, eps, delta, alpha_s, forcing_coeffs, Omega):
    """Ship roll + tangent linear equations for Lyapunov exponent."""
    x, y, dx, dy = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)

    dxdt = y
    dydt = x - alpha_s * x**3 + eps * (-delta * y + F)

    dfdx = 1 - 3 * alpha_s * x**2
    dfdy = -eps * delta

    ddxdt = dy
    ddydt = dfdx * dx + dfdy * dy

    return [dxdt, dydt, ddxdt, ddydt]


def helmholtz_rhs(t, z, eps, delta, forcing_coeffs, Omega):
    """
    Helmholtz escape oscillator: phi'' + phi - phi^2 = eps*(-delta*phi' + F(t)).
    Center at phi=0, saddle at phi=1; phi -> +inf is escape (capsize).
    """
    x, y = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)
    dxdt = y
    dydt = -x + x**2 + eps * (-delta * y + F)
    return [dxdt, dydt]


def helmholtz_rhs_with_variational(t, z, eps, delta, forcing_coeffs, Omega):
    """Helmholtz oscillator + tangent linear equations for the Lyapunov exponent."""
    x, y, dx, dy = z
    F = 0.0
    for k, (ak, bk) in enumerate(forcing_coeffs, 1):
        F += ak * np.cos(k * Omega * t) + bk * np.sin(k * Omega * t)

    dxdt = y
    dydt = -x + x**2 + eps * (-delta * y + F)

    dfdx = -1 + 2 * x
    dfdy = -eps * delta

    ddxdt = dy
    ddydt = dfdx * dx + dfdy * dy

    return [dxdt, dydt, ddxdt, ddydt]


# =============================================================================
# Poincare section
# =============================================================================

def compute_poincare_section(rhs, rhs_args, ic, Omega, eps,
                              n_transient=200, n_collect=500,
                              rtol=1e-12, atol=1e-12):
    """
    Compute stroboscopic Poincare section.

    Parameters
    ----------
    rhs : callable
        Right-hand side function f(t, z, ...).
    rhs_args : tuple
        Additional arguments to rhs.
    ic : array-like
        Initial condition [x0, y0].
    Omega : float
        Forcing frequency.
    n_transient : int
        Number of forcing periods to discard.
    n_collect : int
        Number of forcing periods to collect.
    """
    T = 2 * np.pi / Omega
    t_total = (n_transient + n_collect) * T
    t_sample = np.arange(n_transient, n_transient + n_collect) * T

    def wrapped_rhs(t, z):
        return rhs(t, z, *rhs_args)

    sol = solve_ivp(wrapped_rhs, [0, t_total], ic,
                    method='DOP853', rtol=rtol, atol=atol,
                    t_eval=t_sample, dense_output=False)

    if sol.status != 0:
        print(f"Warning: integrator status {sol.status}: {sol.message}")

    return sol.y[0], sol.y[1]  # x, y at Poincare section


# =============================================================================
# Manifold computation
# =============================================================================

def compute_manifolds_duffing(eps, delta, forcing_coeffs, Omega,
                               n_points=200, T_integrate=8.0,
                               saddle_delta=1e-6, rtol=1e-11, atol=1e-11):
    """
    Compute W^s and W^u for the Duffing oscillator.
    The saddle is near (0,0) for the perturbed system.
    Eigenvalues of unperturbed linearization at origin: lambda = +1, -1.
    Eigenvectors: (1, 1) for unstable, (1, -1) for stable.
    """
    T_period = 2 * np.pi / Omega

    # Initialize points along unstable eigenvector
    t0_phases = np.linspace(0, T_period, n_points, endpoint=False)

    Wu_x, Wu_y = [], []
    Ws_x, Ws_y = [], []

    def rhs_fwd(t, z):
        return duffing_rhs(t, z, eps, delta, forcing_coeffs, Omega)

    def rhs_bwd(t, z):
        r = duffing_rhs(-t, z, eps, delta, forcing_coeffs, Omega)
        return [-r[0], -r[1]]

    # Unstable manifold: perturb along unstable eigenvector (1, 1)/sqrt(2), integrate forward
    for t0 in t0_phases:
        ic_u = [saddle_delta / np.sqrt(2), saddle_delta / np.sqrt(2)]
        sol = solve_ivp(rhs_fwd, [t0, t0 + T_integrate], ic_u,
                        method='DOP853', rtol=rtol, atol=atol,
                        max_step=0.01)
        Wu_x.extend(sol.y[0])
        Wu_y.extend(sol.y[1])

        # Also negative direction
        ic_u_neg = [-saddle_delta / np.sqrt(2), -saddle_delta / np.sqrt(2)]
        sol = solve_ivp(rhs_fwd, [t0, t0 + T_integrate], ic_u_neg,
                        method='DOP853', rtol=rtol, atol=atol,
                        max_step=0.01)
        Wu_x.extend(sol.y[0])
        Wu_y.extend(sol.y[1])

    # Stable manifold: perturb along stable eigenvector (1, -1)/sqrt(2), integrate backward
    for t0 in t0_phases:
        ic_s = [saddle_delta / np.sqrt(2), -saddle_delta / np.sqrt(2)]
        # Integrate backward = integrate rhs with reversed time
        sol = solve_ivp(lambda t, z: [-v for v in duffing_rhs(2*t0 - t, z, eps, delta, forcing_coeffs, Omega)],
                        [t0, t0 + T_integrate], ic_s,
                        method='DOP853', rtol=rtol, atol=atol,
                        max_step=0.01)
        Ws_x.extend(sol.y[0])
        Ws_y.extend(sol.y[1])

        ic_s_neg = [-saddle_delta / np.sqrt(2), saddle_delta / np.sqrt(2)]
        sol = solve_ivp(lambda t, z: [-v for v in duffing_rhs(2*t0 - t, z, eps, delta, forcing_coeffs, Omega)],
                        [t0, t0 + T_integrate], ic_s_neg,
                        method='DOP853', rtol=rtol, atol=atol,
                        max_step=0.01)
        Ws_x.extend(sol.y[0])
        Ws_y.extend(sol.y[1])

    return (np.array(Wu_x), np.array(Wu_y)), (np.array(Ws_x), np.array(Ws_y))


# =============================================================================
# Lyapunov exponent
# =============================================================================

def compute_lyapunov_exponent(rhs_var, rhs_args, ic, Omega,
                               n_transient=1000, n_compute=5000,
                               rtol=1e-10, atol=1e-10):
    """
    Compute largest Lyapunov exponent via variational equations.

    Parameters
    ----------
    rhs_var : callable
        RHS including variational equations: f(t, [x, y, dx, dy], ...).
    rhs_args : tuple
        Additional arguments.
    ic : array-like
        Initial condition [x0, y0].
    Omega : float
        Forcing frequency.
    n_transient : int
        Periods to discard.
    n_compute : int
        Periods to average over.
    """
    T = 2 * np.pi / Omega

    def wrapped_rhs(t, z):
        return rhs_var(t, z, *rhs_args)

    # Initial tangent vector
    z0 = list(ic) + [1.0, 0.0]

    # Discard transient (renormalize every period to prevent overflow)
    t_current = 0.0
    for i in range(n_transient):
        sol = solve_ivp(wrapped_rhs, [t_current, t_current + T], z0,
                        method='DOP853', rtol=rtol, atol=atol, max_step=T/10)
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            z0 = list(ic) + [1.0, 0.0]
            t_current += T
            continue
        z0 = sol.y[:, -1]
        norm_v = np.sqrt(z0[2]**2 + z0[3]**2)
        if norm_v > 0 and np.isfinite(norm_v):
            z0[2] /= norm_v
            z0[3] /= norm_v
        else:
            z0[2] = 1.0
            z0[3] = 0.0
        t_current += T

    # Compute Lyapunov exponent
    lyap_sum = 0.0
    lyap_values = []
    # t_current is already set from the transient loop

    for i in range(n_compute):
        sol = solve_ivp(wrapped_rhs, [t_current, t_current + T], z0,
                        method='DOP853', rtol=rtol, atol=atol, max_step=T/10)
        if sol.status != 0 or not np.all(np.isfinite(sol.y[:, -1])):
            break
        z0 = sol.y[:, -1]
        norm_v = np.sqrt(z0[2]**2 + z0[3]**2)
        if norm_v == 0 or not np.isfinite(norm_v):
            break
        lyap_sum += np.log(norm_v)
        z0[2] /= norm_v
        z0[3] /= norm_v
        t_current += T
        lyap_values.append(lyap_sum / ((i + 1) * T))

    lyap_max = lyap_sum / (n_compute * T)

    # Estimate standard error from running average
    lyap_arr = np.array(lyap_values)
    if len(lyap_arr) > 100:
        se = np.std(lyap_arr[-500:]) if len(lyap_arr) >= 500 else np.std(lyap_arr[-100:])
    else:
        se = np.nan

    return lyap_max, se, lyap_arr


if __name__ == "__main__":
    print("Dynamics engine loaded. Run generate_all_figures.py for full computations.")
