"""Numerical spot checks accompanying the v11 analytic corrections.

The proofs are in the manuscript; these calculations independently check
their conventions and the Helmholtz obstruction with the actual parameters.
Only the version-11 result directory is written.
"""

from pathlib import Path
import json
import numpy as np

from kernels import helmholtz_kernels_analytical

ROOT = Path(__file__).resolve().parents[1]


def main():
    phases = np.linspace(0, 2 * np.pi, 40001)
    p = np.array([1 + 0.2j, -0.7j, 0.3 - 0.4j])
    v = np.exp(1j * np.outer(np.arange(len(p)), phases))
    gram = np.outer(p.conj(), p)
    lhs = np.abs(p @ v) ** 2
    rhs = np.einsum("it,ij,jt->t", v.conj(), gram, v).real
    gram_error = float(np.max(np.abs(lhs - rhs)))
    assert gram_error < 1e-13

    # An independent coefficient box at one harmonic has support sqrt(2)*rho.
    rho = 1.0
    actual_box_support = float(np.max(rho * (np.abs(np.cos(phases)) + np.abs(np.sin(phases)))))
    assert abs(actual_box_support - np.sqrt(2) * rho) < 1e-12
    radius, phase_error = 0.83, 0.31
    displacement = np.max(np.abs(radius * (np.cos(phases + phase_error) - np.cos(phases))))
    exact_displacement = 2 * radius * abs(np.sin(phase_error / 2))
    assert abs(displacement - exact_displacement) < 1e-8

    # First-harmonic control cannot change the average of M(0) and M(pi).
    beta = np.array([helmholtz_kernels_analytical(k, 1.0)[1] for k in range(1, 4)])
    existing = -np.array([0.10, 0.30, 0.15]) * beta
    c0 = -0.12
    c2, c3 = existing[1:]
    assert c2 >= 2 * abs(c3)
    residual = -c3 * np.cos(phases) + c2 * np.cos(2 * phases) + c3 * np.cos(3 * phases)
    factored = c2 - np.sin(phases) ** 2 * (2 * c2 + 4 * c3 * np.cos(phases))
    factor_error = float(np.max(np.abs(residual - factored)))
    assert factor_error < 1e-14
    upper_margin = abs(c0) - c2
    attained_margin = -float(np.max(c0 + residual))
    assert abs(upper_margin - attained_margin) < 1e-14
    assert upper_margin < 0.08
    b1 = (existing[0] + c3) / beta[0]
    power = float(b1 * b1)
    stored = json.loads((ROOT / "results/helmholtz.json").read_text())
    assert abs(upper_margin - stored["sh_sup_tau"]) < 5.1e-5
    assert abs(power - stored["sh_sup_power"]) < 5.1e-5
    result = {
        "gram_identity_max_absolute_error": gram_error,
        "unit_box_support": actual_box_support,
        "phase_rotation_exact_uniform_bound": exact_displacement,
        "helmholtz": {
            "fixed_cosine_coefficients": existing.tolist(),
            "factorization_max_absolute_error": factor_error,
            "all_power_first_harmonic_margin_bound": float(upper_margin),
            "attained_margin": attained_margin,
            "attaining_controller_a1": 0.0,
            "attaining_controller_b1": float(b1),
            "attaining_power": power,
            "unreachable_target": 0.08,
        },
        "scope": "Numerical checks of displayed analytic identities, not interval certificates.",
    }
    output = ROOT / "results/v11/algebra_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Algebra checks passed; Helmholtz all-power margin bound {upper_margin:.10f} < 0.08")


if __name__ == "__main__":
    main()
