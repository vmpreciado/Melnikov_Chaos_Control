"""Independent high-precision quadrature for the v11 kernel appendix.

Writes only results/v11 and revision_v11; the v10 sources remain unchanged.
The integrals use orbit velocities, not the closed-form Fourier transforms.
"""

from pathlib import Path
import json
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]


def evaluate(system, k, digits, cutoff):
    with mp.workdps(digits):
        w = mp.mpf(k)
        if system == "duffing":
            velocity = lambda t: -mp.sqrt(2) * mp.sech(t) * mp.tanh(t)
            transform = lambda t: -velocity(t) * mp.sin(w * t)
            exact = mp.sqrt(2) * mp.pi * w / mp.cosh(mp.pi * w / 2)
            damping_exact = mp.mpf(4) / 3
            tail = 4 * mp.sqrt(2) * mp.exp(-cutoff)
        elif system == "josephson":
            velocity = lambda t: 2 * mp.sech(t)
            transform = lambda t: velocity(t) * mp.cos(w * t)
            exact = 2 * mp.pi / mp.cosh(mp.pi * w / 2)
            damping_exact = mp.mpf(8)
            tail = 8 * mp.exp(-cutoff)
        elif system == "helmholtz":
            velocity = lambda t: mp.mpf("1.5") * mp.sech(t / 2) ** 2 * mp.tanh(t / 2)
            transform = lambda t: -velocity(t) * mp.sin(w * t)
            exact = -6 * mp.pi * w * w / mp.sinh(mp.pi * w)
            damping_exact = mp.mpf(6) / 5
            tail = 12 * mp.exp(-cutoff)
        else:
            raise ValueError(system)
        bounds = list(range(0, cutoff + 1, 2))
        numerical = 2 * mp.quad(transform, bounds)
        damping = 2 * mp.quad(lambda t: velocity(t) ** 2, bounds)
        return {
            "analytical": mp.nstr(exact, 40),
            "quadrature": mp.nstr(numerical, 40),
            "relative_error": float(abs((numerical - exact) / exact)),
            "absolute_error": float(abs(numerical - exact)),
            "omitted_tail_bound": float(tail),
            "damping_integral": mp.nstr(damping, 40),
            "damping_relative_error": float(abs((damping - damping_exact) / damping_exact)),
        }


def tex_number(value, digits=9):
    x = float(value)
    if 1e-3 <= abs(x) < 100:
        return f"{x:.{digits}f}"
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def main():
    out = {
        "frequency": 1.0,
        "method": "mpmath tanh-sinh quadrature on successive two-unit intervals; parity doubles the positive half-line",
        "working_digits": 45,
        "cutoff": 100,
        "crosscheck_digits": 65,
        "crosscheck_cutoff": 120,
        "note": "Relative errors compare numerical orbit integrals with closed-form nonzero kernels. Zero kernels follow exactly from parity. Tail bounds are analytic; quadrature error estimates are numerical checks, not interval certificates.",
        "rows": [],
    }
    for system in ("duffing", "josephson", "helmholtz"):
        for k in range(1, 11):
            a = evaluate(system, k, 45, 100)
            b = evaluate(system, k, 65, 120)
            with mp.workdps(65):
                difference = abs((mp.mpf(a["quadrature"]) - mp.mpf(b["quadrature"])) / mp.mpf(b["analytical"]))
            row = {"system": system, "k": k, **a, "crosscheck_relative_difference": float(difference),
                   "crosscheck_relative_error": b["relative_error"]}
            if max(row["relative_error"], row["crosscheck_relative_difference"], row["damping_relative_error"]) >= 1e-20:
                raise AssertionError(row)
            out["rows"].append(row)
    out["max_relative_error"] = max(r["relative_error"] for r in out["rows"])
    results = ROOT / "results" / "v11"
    results.mkdir(parents=True, exist_ok=True)
    (results / "kernel_verification.json").write_text(json.dumps(out, indent=2) + "\n")
    labels = {"duffing": "Duffing", "josephson": "Josephson", "helmholtz": "Helmholtz"}
    rows = []
    for row in out["rows"]:
        if row["k"] not in (1, 3, 5, 10):
            continue
        name = labels[row["system"]]
        kernel = r"\kernA_k" if row["system"] == "josephson" else r"\kernB_k"
        rows.append(f"{name} & {row['k']} & ${kernel}$ & ${tex_number(row['analytical'])}$ & "
                    f"${tex_number(row['quadrature'])}$ \\\\")
    (ROOT / "revision_v11" / "tab_kernel_verification.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
    print(f"Verified {len(out['rows'])} nonzero kernels; maximum relative discrepancy {out['max_relative_error']:.2e}")


if __name__ == "__main__":
    main()
