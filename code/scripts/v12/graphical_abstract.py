"""Plot the one-loop Helmholtz comparison using the manuscript sign certificate."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from kernels import helmholtz_kernels_analytical
from sdp_solver import solve_suppression_sdp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figures" / "v12" / "graphical_abstract.pdf")
    output = parser.parse_args().output
    case = json.loads((ROOT / "results" / "helmholtz.json").read_text())
    gamma = np.array(case["gamma"])
    # The case uses unit nondimensional forcing frequency and a sine drive.
    alpha, beta = np.array([helmholtz_kernels_analytical(k, 1.0)
                            for k in range(1, len(gamma) + 1)]).T
    cosine = -gamma * beta
    sine = np.zeros_like(cosine)
    target = case["tau_target"]
    # Averaging M(0) and M(pi) eliminates all tunable fundamental terms.
    ceiling = -case["c0"] - cosine[1]
    if cosine[1] < 2 * abs(cosine[2]):
        raise ValueError("The case no longer satisfies the ceiling attainability condition.")
    arguments = (case["c0"], cosine, sine, alpha[:2], beta[:2], 2)
    minimum = solve_suppression_sdp(*arguments, objective="min_power", tau_min=target)
    if minimum["status"] != "optimal":
        raise ValueError(f"Two-harmonic minimum-power solve: {minimum['status']}")
    target_power = minimum["energy"]
    if (round(target_power, 4) != case["min_power"]["P"]["2"]
            or round(ceiling, 4) != case["sh_sup_tau"]):
        raise ValueError("The independently computed values disagree with the paper.")
    low_power, high_power = 0.015, case["Pmax"]
    if low_power <= case["sh_sup_power"]:
        raise ValueError("The plotted range must start above the ceiling attainment power.")
    powers = np.unique(np.append(np.linspace(low_power, high_power, 64), target_power))
    margins = []
    for power in powers:
        result = solve_suppression_sdp(*arguments, objective="max_margin", P_max=power)
        if result["status"] != "optimal":
            raise ValueError(f"Power-budget solve at {power}: {result['status']}")
        margins.append(result["optimal_value"])
    if not np.isclose(margins[np.searchsorted(powers, target_power)], target, atol=1e-6):
        raise ValueError("The budget sweep does not match the independent minimum-power solve.")

    ink = "#1d2935"
    teal = "#006b61"
    rust = "#944317"
    gray = "#546371"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(6.4, 3.2), facecolor="white")
    fig.text(0.035, 0.93, "Two harmonics reach the target margin",
             fontsize=18.3, fontweight="bold", color=ink, va="top")
    fig.text(0.035, 0.80, "Helmholtz escape oscillator · first-order margin on its homoclinic loop",
             fontsize=11, color=ink, va="top")
    ax = fig.add_axes((0.148, 0.221, 0.810, 0.485))
    ax.set(xlim=(low_power, high_power), ylim=(0.012, 0.114))
    ax.plot(powers, margins, color=teal, lw=2.5, solid_capstyle="round")
    ax.axhline(ceiling, color=rust, lw=2)
    ax.hlines(target, low_power, target_power, colors=gray, linestyles=(0, (3, 2)), lw=1.2)
    ax.vlines(target_power, ax.get_ylim()[0], target, colors=gray,
              linestyles=(0, (3, 2)), lw=1.2)
    ax.scatter(target_power, target, color=teal, edgecolors="white",
               linewidths=1.1, s=50, zorder=4)
    ax.text(low_power + 0.0005, target + 0.0033, "Target",
            color=gray, fontsize=11, va="bottom")
    ax.text(high_power, 0.1035, "Fundamental + second harmonic", color=teal,
            fontsize=11.5, fontweight="bold", ha="right", va="bottom")
    ax.text(high_power, ceiling - 0.0038, "Fundamental only · ceiling at any power",
            color=rust, fontsize=11.2, fontweight="bold", ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax.set_xticks([low_power, target_power, high_power],
                  [f"{low_power:.3f}", f"{target_power:.4f}", f"{high_power:.2f}"])
    ax.set_yticks([ceiling, target], [f"{ceiling:.4f}", f"{target:.2f}"])
    ax.set_xlabel("Normalized control power", color=ink, labelpad=3, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set(color=gray, linewidth=0.7)
    ax.tick_params(axis="both", colors=ink, labelsize=11.5, length=3, width=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "Certified margins on the Helmholtz homoclinic loop",
                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
