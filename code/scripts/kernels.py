"""
Analytical response kernels for Melnikov-based chaos control.

Systems:
  1. MEMS / Duffing oscillator
  2. Josephson junction (damped driven pendulum)
  3. Ship roll (softening Duffing)

Each system provides:
  - Homoclinic orbit (x_h(t), y_h(t))
  - Damping/bias coefficient c0
  - Response kernels alpha_k, beta_k such that for control forcing
    with Cartesian coefficients (a_k, b_k):
      c_k = a_k * alpha_k - b_k * beta_k
      d_k = b_k * alpha_k + a_k * beta_k
"""

import numpy as np
from scipy.integrate import quad


# =============================================================================
# MEMS / Duffing oscillator
# =============================================================================
# System: x'' - x + x^3 = eps * (-delta * x' + sum_k [gamma_k cos(kOmega t) + eta_k sin(kOmega t)])
# Potential V(x) = -x^2/2 + x^4/4: saddle at origin, wells at x = +-1
# Homoclinic (upper branch): x_h(t) = sqrt(2) * sech(t), y_h(t) = -sqrt(2) * sech(t) * tanh(t)
# The Melnikov integrand involves omega(f, g) = y_h(t) * g_y
# For damping: g_y = -delta * y_h => contribution = -delta * integral(y_h^2 dt) = -4*delta/3
# For cos forcing: g_y = cos(kOmega t) => c_k involves integral(y_h cos(kOmega t) dt) = 0 (odd*even)
# For sin forcing: g_y = sin(kOmega t) => d_k involves integral(y_h sin(kOmega t) dt) != 0 (odd*odd)
#
# Closed form: integral_{-inf}^{inf} sech(t)*tanh(t)*sin(kOmega*t) dt = pi*k*Omega*sech(pi*k*Omega/2)
# (Gradshteyn & Ryzhik 3.982)
#
# With forcing structure: u_k(t) = a_k cos(kOmega t) + b_k sin(kOmega t)
# The Melnikov becomes:
#   M(theta) = c0 + sum_k [c_k cos(k*theta) + d_k sin(k*theta)]
# where c_k = a_k * alpha_k - b_k * beta_k, d_k = b_k * alpha_k + a_k * beta_k
#
# For Duffing: the integral of y_h cos(kOmega t) = 0 (parity), so alpha_k = 0
# and beta_k = -sqrt(2) * integral(sech(t)*tanh(t)*sin(kOmega t) dt) ...
# Wait, let's be more careful.
#
# y_h(t) = -sqrt(2) sech(t) tanh(t)
# integral(y_h(t) cos(kOmega t) dt) = -sqrt(2) integral(sech(t)tanh(t) cos(kOmega t) dt) = 0
# integral(y_h(t) sin(kOmega t) dt) = -sqrt(2) integral(sech(t)tanh(t) sin(kOmega t) dt)
#                                    = -sqrt(2) * pi*k*Omega * sech(pi*k*Omega/2)
#
# With perturbation structure g = -delta*y + sum_k (a_k cos(kOmega t) + b_k sin(kOmega t)):
# The Melnikov integral for the k-th harmonic cosine term a_k cos(kOmega(t+t0)):
#   = a_k * [integral(y_h cos(kOmega t) dt)*cos(k*theta) - integral(y_h sin(kOmega t) dt)*sin(k*theta)]
# plus similar for the sine term b_k sin(kOmega(t+t0)):
#   = b_k * [integral(y_h sin(kOmega t) dt)*cos(k*theta) + integral(y_h cos(kOmega t) dt)*sin(k*theta)]
#
# Define: I_c_k = integral(y_h(t) cos(kOmega t) dt) = 0
#         I_s_k = integral(y_h(t) sin(kOmega t) dt) = -sqrt(2)*pi*k*Omega*sech(pi*k*Omega/2)
#
# Then: c_k = a_k * I_c_k + b_k * I_s_k = b_k * I_s_k
#        d_k = -a_k * I_s_k + b_k * I_c_k = -a_k * I_s_k
#
# In the framework c_k = a_k*alpha_k - b_k*beta_k, d_k = b_k*alpha_k + a_k*beta_k:
#   alpha_k = I_c_k = 0
#   beta_k = -I_s_k = sqrt(2)*pi*k*Omega*sech(pi*k*Omega/2)
# Check: c_k = a_k*0 - b_k*beta_k = -b_k * sqrt(2)*pi*k*Omega*sech(...)
# But we said c_k = b_k * I_s_k = b_k * (-sqrt(2)*pi*k*Omega*sech(...))
# So c_k = -b_k * sqrt(2)*pi*k*Omega*sech(...) = -b_k*beta_k. Correct.
# d_k = b_k*0 + a_k*beta_k = a_k * sqrt(2)*pi*k*Omega*sech(...)
# But we said d_k = -a_k * I_s_k = a_k * sqrt(2)*pi*k*Omega*sech(...). Correct.

def duffing_homoclinic(t):
    """Upper-branch homoclinic orbit of x'' - x + x^3 = 0."""
    xh = np.sqrt(2) / np.cosh(t)
    yh = -np.sqrt(2) * np.tanh(t) / np.cosh(t)
    return xh, yh


def duffing_c0(delta):
    """Damping coefficient c0 = -delta * integral(y_h^2 dt) = -4*delta/3."""
    return -4.0 * delta / 3.0


def duffing_c0_numerical(delta):
    """Numerical verification of c0."""
    def integrand(t):
        _, yh = duffing_homoclinic(t)
        return -delta * yh**2
    val, err = quad(integrand, -np.inf, np.inf, limit=200)
    return val, err


def duffing_kernels_analytical(k, Omega):
    """
    Analytical response kernels for the Duffing oscillator.

    Returns (alpha_k, beta_k) such that:
      c_k = a_k * alpha_k - b_k * beta_k
      d_k = b_k * alpha_k + a_k * beta_k
    """
    alpha_k = 0.0  # integral(y_h cos(kOmega t) dt) = 0 by parity
    kw = k * Omega
    # beta_k = sqrt(2) * pi * k * Omega * sech(pi*k*Omega/2)
    beta_k = np.sqrt(2) * np.pi * kw / np.cosh(np.pi * kw / 2.0)
    return alpha_k, beta_k


def duffing_kernels_numerical(k, Omega):
    """
    Numerical verification of response kernels via quadrature.
    Returns (I_c_k, I_s_k) = (integral(y_h cos(kOmega t) dt), integral(y_h sin(kOmega t) dt))
    and the corresponding (alpha_k, beta_k).
    """
    kw = k * Omega

    def integrand_cos(t):
        _, yh = duffing_homoclinic(t)
        return yh * np.cos(kw * t)

    def integrand_sin(t):
        _, yh = duffing_homoclinic(t)
        return yh * np.sin(kw * t)

    I_c_k, err_c = quad(integrand_cos, -np.inf, np.inf, limit=200)
    I_s_k, err_s = quad(integrand_sin, -np.inf, np.inf, limit=200)

    alpha_k = I_c_k
    beta_k = -I_s_k

    return alpha_k, beta_k, I_c_k, I_s_k, err_c, err_s


# =============================================================================
# Josephson junction / Pendulum
# =============================================================================
# System: phi'' + sin(phi) = eps * (-alpha * phi' + I_dc + I_ac sin(Omega t) + u(t))
# Homoclinic (upper separatrix): phi_h(t) = 4*arctan(exp(t)) - pi
#   phi_h'(t) = 2*sech(t)... let's verify:
#   phi_h(t) = 4*arctan(e^t) - pi
#   phi_h'(t) = 4*e^t/(1+e^(2t)) = 4/(e^(-t)+e^t) = 2*sech(t)...
#   Wait: 4*e^t/(1+e^(2t)) = 4/(e^(-t)+e^t) = 4/(2*cosh(t)) = 2/cosh(t) = 2*sech(t). No wait.
#   2/cosh(t) = 2*sech(t). Hmm, but standard result for pendulum separatrix is phi_h'(t) = 2*sech(t).
#   Let me recalculate: phi_h(0) = 4*arctan(1) - pi = 4*pi/4 - pi = 0.
#   phi_h(inf) -> 4*arctan(inf) - pi = 4*pi/2 - pi = pi.
#   phi_h(-inf) -> 4*arctan(0) - pi = -pi. Good.
#   phi_h'(0) = 2*sech(0) = 2. And sin(phi_h(0)) = sin(0) = 0.
#   Check EOM: phi_h'' + sin(phi_h) = 0.
#   phi_h' = 2*sech(t), phi_h'' = -2*sech(t)*tanh(t).
#   sin(phi_h(t)): need to compute sin(4*arctan(e^t) - pi) = -sin(4*arctan(e^t))
#   Using double angle: sin(4*arctan(u)) = 2*sin(2*arctan(u))*cos(2*arctan(u))
#   sin(2*arctan(u)) = 2u/(1+u^2), cos(2*arctan(u)) = (1-u^2)/(1+u^2)
#   So sin(4*arctan(u)) = 2 * 2u/(1+u^2) * (1-u^2)/(1+u^2) = 4u(1-u^2)/(1+u^2)^2
#   With u = e^t: sin(phi_h+pi) = 4*e^t*(1-e^(2t))/(1+e^(2t))^2
#   sin(phi_h) = -4*e^t*(1-e^(2t))/(1+e^(2t))^2 = 4*e^t*(e^(2t)-1)/(1+e^(2t))^2
#             = 4*e^t*(e^(2t)-1)/(1+e^(2t))^2
#   phi_h'' = -2*sech(t)*tanh(t) = -2*tanh(t)/cosh(t)
#   We need phi_h'' + sin(phi_h) = 0.
#   = -2*tanh(t)/cosh(t) + 4*e^t*(e^(2t)-1)/(1+e^(2t))^2
#   Simplify: tanh(t)/cosh(t) = sinh(t)/cosh^2(t) = 2*sinh(t)*4/(e^t+e^{-t})^2*...
#   This is getting complicated. Let me just trust the standard result and verify numerically.
#
# Standard result for pendulum separatrix:
#   phi_h(t) = 4*arctan(e^t) - pi (connects -pi to pi)
#   phi_h'(t) = 2*sech(t)
#   c0 from damping: -alpha * integral(phi_h'^2 dt) = -alpha * integral(4*sech^2(t) dt) = -alpha*4*2
#   Wait: integral(sech^2(t) dt from -inf to inf) = [tanh(t)]_{-inf}^{inf} = 1-(-1) = 2
#   So integral(phi_h'^2 dt) = 4 * integral(sech^2(t) dt) = 4*2 = 8
#   c0_damping = -8*alpha
#   Hmm wait, the Melnikov integral uses omega(f, g) where f = (phi', -sin(phi)).
#   For the pendulum: f = (y, -sin(x)), so f_1 = y, f_2 = -sin(x).
#   g = (0, -alpha*y + I_dc + u(t)), so g_1 = 0, g_2 = -alpha*y + I_dc + u(t).
#   omega(f, g) = f_1 * g_2 - f_2 * g_1 = y * (-alpha*y + I_dc + u(t))
#   So: M(t0) = integral(y_h(t) * [-alpha*y_h(t) + I_dc + u(t+t0)] dt)
#   c0 = integral(y_h * (-alpha * y_h + I_dc) dt) = -alpha * integral(y_h^2 dt) + I_dc * integral(y_h dt)
#   integral(y_h^2 dt) = integral(4*sech^2(t) dt) = 8
#   integral(y_h dt) = integral(2*sech(t) dt) = 2*pi  (since integral(sech(t) dt) = pi)
#   So c0 = -8*alpha + 2*pi*I_dc
#
# Response kernels: for u(t) = a_k cos(kOmega t) + b_k sin(kOmega t):
#   I_c_k = integral(y_h(t) * cos(kOmega t) dt) = integral(2*sech(t) * cos(kOmega t) dt)
#         = 2 * pi * sech(pi*kOmega/2)  (standard: integral(sech(t)*cos(at) dt) = pi*sech(pi*a/2))
#   I_s_k = integral(y_h(t) * sin(kOmega t) dt) = integral(2*sech(t) * sin(kOmega t) dt) = 0
#         (sech is even, sin is odd => integrand is odd)
#
#   alpha_k = I_c_k = 2*pi*sech(pi*kOmega/2)
#   beta_k = -I_s_k = 0

def pendulum_homoclinic(t):
    """Upper separatrix of phi'' + sin(phi) = 0."""
    phi_h = 4.0 * np.arctan(np.exp(t)) - np.pi
    dphi_h = 2.0 / np.cosh(t)
    return phi_h, dphi_h


def josephson_c0(alpha, I_dc):
    """
    Damping + DC bias coefficient for Josephson junction.
    c0 = -8*alpha + 2*pi*I_dc
    """
    return -8.0 * alpha + 2.0 * np.pi * I_dc


def josephson_c0_numerical(alpha, I_dc):
    """Numerical verification."""
    def integrand(t):
        _, yh = pendulum_homoclinic(t)
        return yh * (-alpha * yh + I_dc)
    val, err = quad(integrand, -np.inf, np.inf, limit=200)
    return val, err


def josephson_kernels_analytical(k, Omega):
    """
    Response kernels for Josephson junction / pendulum.
    alpha_k = 2*pi*sech(pi*k*Omega/2), beta_k = 0.
    """
    kw = k * Omega
    alpha_k = 2.0 * np.pi / np.cosh(np.pi * kw / 2.0)
    beta_k = 0.0
    return alpha_k, beta_k


def josephson_kernels_numerical(k, Omega):
    """Numerical verification."""
    kw = k * Omega

    def integrand_cos(t):
        _, yh = pendulum_homoclinic(t)
        return yh * np.cos(kw * t)

    def integrand_sin(t):
        _, yh = pendulum_homoclinic(t)
        return yh * np.sin(kw * t)

    I_c_k, err_c = quad(integrand_cos, -np.inf, np.inf, limit=200)
    I_s_k, err_s = quad(integrand_sin, -np.inf, np.inf, limit=200)

    alpha_k = I_c_k
    beta_k = -I_s_k

    return alpha_k, beta_k, I_c_k, I_s_k, err_c, err_s


# =============================================================================
# Ship roll (softening Duffing)
# =============================================================================
# Model: phi'' - phi + alpha_s*phi^3 = eps*(-delta*phi' + F(t))
# With alpha_s > 0 (softening). Potential V(phi) = -phi^2/2 + alpha_s*phi^4/4.
# Saddle at origin, wells at phi = +-1/sqrt(alpha_s).
# Homoclinic: xh(t) = sqrt(2/alpha_s) * sech(t)
# yh(t) = -sqrt(2/alpha_s) * sech(t)*tanh(t)
# Same kernel structure as Duffing up to the scaling factor sqrt(2/alpha_s).
#
# c0 = -delta * integral(y_h^2 dt) = -delta * (2/alpha_s) * integral(sech^2(t)*tanh^2(t) dt)
# integral(sech^2(t)*tanh^2(t) dt from -inf to inf) = 2/3
# So c0 = -delta * (2/alpha_s) * (2/3) = -4*delta/(3*alpha_s)
#
# Response kernels: same parity structure as Duffing.
# integral(y_h(t)*cos(kOmega t) dt) = 0 (odd integrand)
# integral(y_h(t)*sin(kOmega t) dt) = -sqrt(2/alpha_s) * integral(sech(t)*tanh(t)*sin(kOmega t) dt)
#                                    = -sqrt(2/alpha_s) * pi*k*Omega*sech(pi*k*Omega/2)
# alpha_k = 0, beta_k = sqrt(2/alpha_s) * pi*k*Omega*sech(pi*k*Omega/2)

def ship_homoclinic(t, alpha_s):
    """Softening Duffing homoclinic orbit: xh = sqrt(2/alpha_s)*sech(t)."""
    scale = np.sqrt(2.0 / alpha_s)
    xh = scale / np.cosh(t)
    yh = -scale * np.tanh(t) / np.cosh(t)
    return xh, yh


def ship_c0(delta, alpha_s):
    """c0 = -4*delta/(3*alpha_s)."""
    return -4.0 * delta / (3.0 * alpha_s)


def ship_c0_numerical(delta, alpha_s):
    """Numerical verification."""
    def integrand(t):
        _, yh = ship_homoclinic(t, alpha_s)
        return -delta * yh**2
    val, err = quad(integrand, -np.inf, np.inf, limit=200)
    return val, err


def ship_kernels_analytical(k, Omega, alpha_s):
    """Response kernels for ship roll."""
    kw = k * Omega
    alpha_k = 0.0
    beta_k = np.sqrt(2.0 / alpha_s) * np.pi * kw / np.cosh(np.pi * kw / 2.0)
    return alpha_k, beta_k


def ship_kernels_numerical(k, Omega, alpha_s):
    """Numerical verification."""
    kw = k * Omega

    def integrand_cos(t):
        _, yh = ship_homoclinic(t, alpha_s)
        return yh * np.cos(kw * t)

    def integrand_sin(t):
        _, yh = ship_homoclinic(t, alpha_s)
        return yh * np.sin(kw * t)

    I_c_k, err_c = quad(integrand_cos, -np.inf, np.inf, limit=200)
    I_s_k, err_s = quad(integrand_sin, -np.inf, np.inf, limit=200)

    alpha_k = I_c_k
    beta_k = -I_s_k

    return alpha_k, beta_k, I_c_k, I_s_k, err_c, err_s


# =============================================================================
# Helmholtz escape oscillator (ship capsize archetype)
# =============================================================================
# Model: phi'' + phi - phi^2 = eps*(-delta*phi' + sum_k gamma_k sin(k Omega t) + u)
# Center at phi = 0 (upright), saddle at phi = 1 (angle of vanishing stability).
# The saddle carries a single homoclinic loop; the other unstable branch escapes.
# Homoclinic orbit:
#   phi_h(t)  = 1 - (3/2) sech^2(t/2)
#   phi_h'(t) = (3/2) sech^2(t/2) tanh(t/2)   (odd, so alpha_k = 0)
# Damping: c0 = -delta * integral(phi_h'^2 dt) = -(6/5) delta.
# Forcing kernel (unit sine coupling in the velocity equation):
#   beta_k = -integral(phi_h'(t) sin(k Omega t) dt) = -6 pi (k Omega)^2 / sinh(pi k Omega)
# The 1/sinh decay with the extra (k Omega)^2 factor is faster than the 1/cosh
# decay of the twin-well and pendulum kernels.

def helmholtz_homoclinic(t):
    """Homoclinic orbit of phi'' + phi - phi^2 = 0 about the saddle at phi = 1."""
    sech2 = 1.0 / np.cosh(t / 2.0) ** 2
    phi_h = 1.0 - 1.5 * sech2
    dphi_h = 1.5 * sech2 * np.tanh(t / 2.0)
    return phi_h, dphi_h


def helmholtz_c0(delta):
    """Damping coefficient c0 = -delta * integral(phi_h'^2 dt) = -(6/5) delta."""
    return -1.2 * delta


def helmholtz_c0_numerical(delta):
    def integrand(t):
        _, dphi = helmholtz_homoclinic(t)
        return -delta * dphi ** 2
    val, err = quad(integrand, -np.inf, np.inf, limit=400)
    return val, err


def helmholtz_kernels_analytical(k, Omega):
    """alpha_k = 0, beta_k = -6*pi*(k*Omega)^2 / sinh(pi*k*Omega)."""
    kw = k * Omega
    alpha_k = 0.0
    beta_k = -6.0 * np.pi * kw ** 2 / np.sinh(np.pi * kw)
    return alpha_k, beta_k


def helmholtz_kernels_numerical(k, Omega):
    kw = k * Omega

    def integrand_cos(t):
        _, dphi = helmholtz_homoclinic(t)
        return dphi * np.cos(kw * t)

    def integrand_sin(t):
        _, dphi = helmholtz_homoclinic(t)
        return dphi * np.sin(kw * t)

    I_c_k, err_c = quad(integrand_cos, -np.inf, np.inf, limit=400)
    I_s_k, err_s = quad(integrand_sin, -np.inf, np.inf, limit=400)
    alpha_k = I_c_k
    beta_k = -I_s_k
    return alpha_k, beta_k, I_c_k, I_s_k, err_c, err_s


# =============================================================================
# Verification routine
# =============================================================================
def verify_all_kernels():
    """Verify all analytical kernels against numerical quadrature."""
    print("=" * 70)
    print("KERNEL VERIFICATION: Analytical vs Numerical Quadrature")
    print("=" * 70)

    all_ok = True
    tol = 1e-8

    # --- Duffing ---
    print("\n--- MEMS / Duffing Oscillator ---")
    delta = 0.1
    Omega = 1.0

    c0_ana = duffing_c0(delta)
    c0_num, c0_err = duffing_c0_numerical(delta)
    match = abs(c0_ana - c0_num) < tol
    all_ok = all_ok and match
    print(f"  c0: analytical={c0_ana:.10f}, numerical={c0_num:.10f}, err={c0_err:.2e}, match={match}")

    for k in [1, 2, 3, 5, 7, 10]:
        a_ana, b_ana = duffing_kernels_analytical(k, Omega)
        a_num, b_num, Ic, Is, ec, es = duffing_kernels_numerical(k, Omega)
        match_a = abs(a_ana - a_num) < tol
        match_b = abs(b_ana - b_num) < tol
        all_ok = all_ok and match_a and match_b
        print(f"  k={k:2d}: alpha: ana={a_ana:12.8f}, num={a_num:12.8f}, ok={match_a}")
        print(f"         beta:  ana={b_ana:12.8f}, num={b_num:12.8f}, ok={match_b}")

    # --- Josephson ---
    print("\n--- Josephson Junction / Pendulum ---")
    alpha_j = 0.1
    I_dc = 0.0

    c0_ana = josephson_c0(alpha_j, I_dc)
    c0_num, c0_err = josephson_c0_numerical(alpha_j, I_dc)
    match = abs(c0_ana - c0_num) < tol
    all_ok = all_ok and match
    print(f"  c0: analytical={c0_ana:.10f}, numerical={c0_num:.10f}, err={c0_err:.2e}, match={match}")

    for k in [1, 2, 3, 5, 7, 10]:
        a_ana, b_ana = josephson_kernels_analytical(k, Omega)
        a_num, b_num, Ic, Is, ec, es = josephson_kernels_numerical(k, Omega)
        match_a = abs(a_ana - a_num) < tol
        match_b = abs(b_ana - b_num) < tol
        all_ok = all_ok and match_a and match_b
        print(f"  k={k:2d}: alpha: ana={a_ana:12.8f}, num={a_num:12.8f}, ok={match_a}")
        print(f"         beta:  ana={b_ana:12.8f}, num={b_num:12.8f}, ok={match_b}")

    # --- Ship roll ---
    print("\n--- Ship Roll (Softening Duffing) ---")
    delta_s = 0.08
    alpha_s = 0.5

    c0_ana = ship_c0(delta_s, alpha_s)
    c0_num, c0_err = ship_c0_numerical(delta_s, alpha_s)
    match = abs(c0_ana - c0_num) < tol
    all_ok = all_ok and match
    print(f"  c0: analytical={c0_ana:.10f}, numerical={c0_num:.10f}, err={c0_err:.2e}, match={match}")

    for k in [1, 2, 3, 5]:
        a_ana, b_ana = ship_kernels_analytical(k, Omega, alpha_s)
        a_num, b_num, Ic, Is, ec, es = ship_kernels_numerical(k, Omega, alpha_s)
        match_a = abs(a_ana - a_num) < tol
        match_b = abs(b_ana - b_num) < tol
        all_ok = all_ok and match_a and match_b
        print(f"  k={k:2d}: alpha: ana={a_ana:12.8f}, num={a_num:12.8f}, ok={match_a}")
        print(f"         beta:  ana={b_ana:12.8f}, num={b_num:12.8f}, ok={match_b}")

    # --- Helmholtz escape oscillator ---
    print("\n--- Helmholtz Escape Oscillator (capsize archetype) ---")
    delta_h = 0.10
    c0_ana = helmholtz_c0(delta_h)
    c0_num, c0_err = helmholtz_c0_numerical(delta_h)
    match = abs(c0_ana - c0_num) < tol
    all_ok = all_ok and match
    print(f"  c0: analytical={c0_ana:.10f}, numerical={c0_num:.10f}, err={c0_err:.2e}, match={match}")

    for k in [1, 2, 3, 5]:
        a_ana, b_ana = helmholtz_kernels_analytical(k, Omega)
        a_num, b_num, Ic, Is, ec, es = helmholtz_kernels_numerical(k, Omega)
        match_a = abs(a_ana - a_num) < tol
        match_b = abs(b_ana - b_num) < tol
        all_ok = all_ok and match_a and match_b
        print(f"  k={k:2d}: alpha: ana={a_ana:12.8f}, num={a_num:12.8f}, ok={match_a}")
        print(f"         beta:  ana={b_ana:12.8f}, num={b_num:12.8f}, ok={match_b}")

    print(f"\n{'ALL KERNELS VERIFIED' if all_ok else 'SOME KERNELS FAILED'}")
    return all_ok


if __name__ == "__main__":
    verify_all_kernels()
