"""Version 11 validation figures, with outputs isolated from version 10.

The Josephson sections are recomputed using two-branch max-margin designs.
The forcing-stress figure reuses the preserved numerical scan, reporting its
actual detection thresholds and right-censored Josephson observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
from scipy.integrate import solve_ivp

from kernels import (duffing_c0, duffing_kernels_analytical,
                     josephson_c0, josephson_kernels_analytical)
from sdp_solver import solve_suppression_sdp
from dynamics import (duffing_rhs, duffing_rhs_with_variational,
                      compute_lyapunov_exponent)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "v11"
RESULTS = ROOT / "results" / "v11"
T = 2 * np.pi
COLORS = {"duffing": "#0072B2", "josephson": "#D55E00", "helmholtz": "#009E73"}


def write_json(name, value):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2) + "\n")


def style():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 11, "legend.fontsize": 9,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def josephson_design(N):
    gc = np.array([0.30, 0.25, 0.20])
    gs = np.array([0.20, 0.15, 0.10])
    alpha = np.array([josephson_kernels_analytical(k, 1.0)[0] for k in range(1, 4)])
    beta = np.zeros(3)
    if not N:
        return np.column_stack([gc, gs]), None
    solution = solve_suppression_sdp(
        josephson_c0(0.1, 0.0), gc * alpha, gs * alpha,
        alpha[:N], beta[:N], N, P_max=0.15,
        objective="max_margin", both_branches=True,
    )
    if solution["a"] is None or min(solution["margin"], solution["margin_L"]) <= 0:
        raise RuntimeError(f"Unverified Josephson two-branch design for N={N}")
    forcing = np.column_stack([gc, gs])
    forcing[:N, 0] += solution["a"]
    forcing[:N, 1] += solution["b"]
    controller = {
        "N": N, "a": solution["a"].tolist(), "b": solution["b"].tolist(),
        "power": solution["energy"], "margin_right": solution["margin"],
        "margin_left": solution["margin_L"], "solver_status": solution["status"],
    }
    return forcing, controller


def pendulum_rhs(forcing):
    harmonic_numbers = np.arange(1, len(forcing) + 1)

    def rhs(t, z):
        phase = harmonic_numbers * t
        drive = forcing[:, 0] @ np.cos(phase) + forcing[:, 1] @ np.sin(phase)
        return np.array([z[1], -np.sin(z[0]) + 0.1 * (-0.1 * z[1] + drive)])

    return rhs


def settled_section(rhs, ic, transient=1200, collect=120):
    sample_times = np.arange(transient, transient + collect) * T
    sol = solve_ivp(
        rhs, (0.0, sample_times[-1]), ic, method="DOP853",
        t_eval=sample_times, rtol=1e-11, atol=1e-11, max_step=T / 10,
    )
    if not sol.success or sol.y.shape[1] != collect or not np.isfinite(sol.y).all():
        raise RuntimeError(f"Josephson integration failed: {sol.message}")
    points = sol.y.T
    points[:, 0] = (points[:, 0] + np.pi) % (2 * np.pi) - np.pi
    residual = float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if residual >= 1e-7:
        raise RuntimeError(f"Section has not settled to period one: residual={residual:g}")
    return points, residual, sol.nfev


def floquet(rhs, point):
    def variational(t, z):
        phi = z[2:].reshape(2, 2)
        jac = np.array([[0.0, 1.0], [-np.cos(z[0]), -0.01]])
        return np.concatenate([rhs(t, z[:2]), (jac @ phi).ravel()])

    sol = solve_ivp(
        variational, (0.0, T), np.concatenate([point, np.eye(2).ravel()]),
        method="DOP853", rtol=1e-12, atol=1e-12, max_step=T / 40,
    )
    if not sol.success:
        raise RuntimeError("Josephson Floquet integration failed")
    multipliers = np.linalg.eigvals(sol.y[2:, -1].reshape(2, 2))
    residual = float(np.linalg.norm(sol.y[:2, -1] - point))
    return {
        "multipliers": [[float(v.real), float(v.imag)] for v in multipliers],
        "largest_state_exponent": float(np.log(np.max(np.abs(multipliers))) / T),
        "one_period_residual": residual,
    }


def refresh_josephson():
    initial_conditions = [[0.0, 0.0], [1.0, 0.5], [-1.0, 0.5]]
    configs = [("Uncontrolled", 0, "#777777"),
               ("Controlled ($N=1$)", 1, COLORS["duffing"]),
               ("Controlled ($N=3$)", 3, COLORS["helmholtz"])]
    data = []
    for label, N, color in configs:
        forcing, controller = josephson_design(N)
        rhs = pendulum_rhs(forcing)
        sections, residuals, nfevs = [], [], []
        for ic in initial_conditions:
            points, residual, nfev = settled_section(rhs, ic)
            sections.append(points)
            residuals.append(residual)
            nfevs.append(nfev)
        all_points = np.concatenate(sections)
        center = all_points.mean(axis=0)
        spread = float(np.max(np.linalg.norm(all_points - center, axis=1)))
        if spread >= 1e-7:
            raise RuntimeError(f"Sampled initial conditions did not converge together: {label}")
        record = {
            "label": label, "N": N, "controller": controller,
            "total_forcing_ab": forcing.tolist(),
            "section_points": [p.tolist() for p in sections],
            "section_center": center.tolist(), "maximum_section_radius": spread,
            "successive_point_residuals": residuals, "function_evaluations": nfevs,
            "floquet": floquet(rhs, center), "color": color,
        }
        data.append(record)
        print(f"Josephson {label}: section radius {spread:.3g}; "
              f"Floquet exponent {record['floquet']['largest_state_exponent']:.8f}", flush=True)

    centers = np.array([d["section_center"] for d in data])
    ranges = np.ptp(centers, axis=0)
    padding = np.maximum(0.22 * ranges, [0.04, 0.04])
    limits = [(centers[:, j].min() - padding[j], centers[:, j].max() + padding[j])
              for j in range(2)]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0), sharex=True, sharey=True)
    for i, (ax, record) in enumerate(zip(axes, data)):
        points = np.concatenate([np.asarray(p) for p in record["section_points"]])
        ax.scatter(points[:, 0], points[:, 1], s=26, color=record["color"], zorder=3)
        ax.set_title(record["label"])
        ax.set_xlabel(r"$\phi$ (rad)")
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(alpha=0.18)
        ax.text(0.04, 0.94, r"$\lambda_{\mathrm{F}}=-0.005$", transform=ax.transAxes,
                va="top", fontsize=10)
        ax.text(0.02, 1.10, f"({chr(97 + i)})", transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel(r"$\dot\phi$ (rad / nondimensional time)")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES / "josephson_validation.pdf")
    fig.savefig(FIGURES / "josephson_validation.png", dpi=180)
    plt.close(fig)
    write_json("josephson_validation.json", {
        "methods": {
            "equation": "phi'' + sin(phi) = eps*(-alpha_d*phi' + F(t))",
            "epsilon": 0.1, "alpha_d": 0.1, "Omega": 1.0, "I_dc": 0.0,
            "initial_conditions": initial_conditions, "method": "DOP853",
            "rtol": 1e-11, "atol": 1e-11, "max_step": "T/10",
            "discarded_periods": 1200, "retained_stroboscopic_points_per_ic": 120,
            "sampling_phase": "t = integer*T; phi wrapped to [-pi,pi)",
            "settling_check": "successive-point norm and pooled radius both below 1e-7",
            "floquet": {
                "description": "State monodromy over one forcing period at the settled orbit",
                "method": "DOP853", "rtol": 1e-12, "atol": 1e-12,
                "max_step": "T/40", "exponent": "log(max(abs(multiplier)))/T",
                "interpretation": "State Floquet exponent, excluding the forcing phase",
            },
        },
        "cases": data,
    })


def refresh_duffing():
    """Recompute all four Poincare panels with the current two-branch designs."""
    gc = np.array([1.20, 0.60, 0.32])
    data = []
    for N in (0, 1, 3, 5):
        K = max(N, 3)
        forcing = np.zeros((K, 2))
        forcing[:3, 0] = gc
        controller = None
        if N:
            alpha = np.zeros(K)
            beta = np.array([duffing_kernels_analytical(k, 1.0)[1] for k in range(1, K + 1)])
            solution = solve_suppression_sdp(
                duffing_c0(0.80), np.zeros(3), gc * beta[:3],
                alpha[:N], beta[:N], N, P_max=2.40,
                objective="max_margin", both_branches=True,
            )
            if solution["a"] is None or min(solution["margin"], solution["margin_L"]) <= 0:
                raise RuntimeError(f"Unverified Duffing two-branch design for N={N}")
            forcing[:N, 0] += solution["a"]
            forcing[:N, 1] += solution["b"]
            controller = {
                "N": N, "a": solution["a"].tolist(), "b": solution["b"].tolist(),
                "power": solution["energy"], "margin_right": solution["margin"],
                "margin_left": solution["margin_L"], "solver_status": solution["status"],
            }
        initial_conditions = [[1.0, 0.0]] if N else [[1.0, 0.0], [-1.0, 0.0], [0.5, 0.5]]
        points = []
        for ic in initial_conditions:
            times = np.arange(500, 2500) * T
            sol = solve_ivp(
                lambda t, z: duffing_rhs(t, z, 0.25, 0.80, forcing, 1.0),
                (0.0, times[-1]), ic, t_eval=times, method="DOP853",
                rtol=1e-11, atol=1e-11, max_step=T / 10,
            )
            if not sol.success or sol.y.shape[1] != 2000:
                raise RuntimeError(f"Duffing section integration failed for N={N}")
            points.append(sol.y.T)
        exponent, _, history = compute_lyapunov_exponent(
            duffing_rhs_with_variational, (0.25, 0.80, forcing, 1.0),
            [1.0, 0.0], 1.0, n_transient=300, n_compute=900,
            rtol=1e-10, atol=1e-10,
        )
        if len(history) != 900 or not np.isfinite(exponent):
            raise RuntimeError(f"Duffing Lyapunov computation failed for N={N}")
        record = {
            "N": N, "label": "Uncontrolled" if not N else f"Controlled ($N={N}$)",
            "controller": controller, "total_forcing_ab": forcing.tolist(),
            "initial_conditions": initial_conditions,
            "section_points": [p.tolist() for p in points],
            "largest_state_lyapunov_exponent": float(exponent),
            "running_exponent_tail_range": [float(min(history[-100:])), float(max(history[-100:]))],
        }
        data.append(record)
        print(f"Duffing N={N}: largest state exponent {exponent:.6f}; "
              f"{len(initial_conditions) * 2000} section points", flush=True)

    all_points = np.concatenate([np.concatenate(d["section_points"]) for d in data])
    padding = np.maximum(np.ptp(all_points, axis=0) * 0.08, [0.05, 0.05])
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True, sharey=True)
    colors = ["#777777", COLORS["duffing"], COLORS["helmholtz"], COLORS["josephson"]]
    for i, (ax, record, color) in enumerate(zip(axes.ravel(), data, colors)):
        points = np.concatenate(record["section_points"])
        ax.scatter(points[:, 0], points[:, 1], s=1.0 if not record["N"] else 17,
                   color=color, alpha=0.55 if not record["N"] else 1,
                   rasterized=not bool(record["N"]))
        ax.set_title(record["label"])
        ax.set_xlim(all_points[:, 0].min() - padding[0], all_points[:, 0].max() + padding[0])
        ax.set_ylim(all_points[:, 1].min() - padding[1], all_points[:, 1].max() + padding[1])
        ax.text(0.04, 0.93, rf"$\lambda_{{\max}}={record['largest_state_lyapunov_exponent']:+.3f}$",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.text(0.01, 1.10, f"({chr(97 + i)})", transform=ax.transAxes, fontweight="bold")
        ax.grid(alpha=0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\dot{x}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$")
    figure.tight_layout(w_pad=1.6, h_pad=1.7)
    figure.savefig(FIGURES / "duffing_poincare.pdf")
    figure.savefig(FIGURES / "duffing_poincare.png", dpi=180)
    plt.close(figure)
    write_json("duffing_poincare.json", {
        "methods": {
            "epsilon": 0.25, "delta": 0.80, "Omega": 1.0,
            "design": "max-margin with both_branches=True and Pmax=2.40",
            "poincare": {"method": "DOP853", "rtol": 1e-11, "atol": 1e-11,
                         "max_step": "T/10", "discarded_periods": 500,
                         "retained_points_per_ic": 2000, "sampling_phase": "t=integer*T"},
            "lyapunov": {"method": "DOP853 state plus variational equation",
                         "rtol": 1e-10, "atol": 1e-10, "max_step": "T/10",
                         "discarded_periods": 300, "averaged_periods": 900,
                         "renormalization_interval": "T", "initial_condition": [1.0, 0.0]},
            "limitations": "Attractor diagnostics for the recorded initial conditions; not global absence of chaos",
        },
        "cases": data,
    })


def replot_margin():
    source = ROOT / "results" / "validation.json"
    original = json.loads(source.read_text())
    observations = original["margin_meaning"]
    scan = np.arange(1.0, 3.5 + 1e-9, 0.15)
    last_josephson_sample = float(round(scan[-1], 2))
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.2))
    titles = [("duffing", "MEMS / Duffing"), ("josephson", "Josephson"),
              ("helmholtz", "Helmholtz")]
    for ax, (system, title) in zip(axes, titles):
        pts = [p for p in observations if p["system"] == system]
        color = COLORS[system]
        ax.plot([p["tau"] for p in pts], [p["kappa_1st"] for p in pts],
                "o-", color=color, markersize=5, lw=1.1)
        finite = [p for p in pts if p["kappa_sim"] is not None]
        ax.plot([p["tau"] for p in finite], [p["kappa_sim"] for p in finite],
                "s--", color=color, markerfacecolor="white", markersize=5, lw=1.1)
        unique_taus = sorted(set(p["tau"] for p in pts))
        for tau in unique_taus:
            same = [p for p in pts if p["tau"] == tau]
            Ns = ",".join(str(p["N"]) for p in same)
            offset = (-8, 12) if system == "duffing" and Ns == "2" else (0, 12)
            if system == "duffing" and Ns == "3,5":
                offset = (8, 29)
            ax.annotate(f"$N={Ns}$", (tau, same[0]["kappa_1st"]),
                        xytext=offset, textcoords="offset points", ha="center", fontsize=10)
        if system == "josephson":
            for tau in unique_taus:
                ax.scatter([tau], [last_josephson_sample], s=50, marker="^",
                           facecolors="white", edgecolors=color, zorder=4)
                ax.annotate("", (tau, last_josephson_sample + 0.24),
                            (tau, last_josephson_sample + 0.03),
                            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1})
            ax.text(0.5, 0.63, "No detection at sampled\n" + r"$\kappa\leq 3.40$",
                    transform=ax.transAxes, ha="center", fontsize=10)
        ax.set_title(title)
        ax.set_xlabel(r"Certified margin $\tau$")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(alpha=0.18)
        ax.margins(x=0.30)
        ax.set_ylim(0.9, 3.95)
    axes[0].set_ylabel(r"Forcing scale $\kappa$")
    for i, ax in enumerate(axes):
        ax.text(0.02, 1.10, f"({chr(97 + i)})", transform=ax.transAxes, fontweight="bold")
    handles = [Line2D([], [], color="#333333", marker="o", label="First-order band closes"),
               Line2D([], [], color="#333333", marker="s", linestyle="--", markerfacecolor="white",
                          label="Simulated detection"),
               Line2D([], [], color="#333333", marker="^", linestyle="none", markerfacecolor="white",
                          label="No detection through sampled range")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
                  ncol=2, frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.4)
    figure.savefig(FIGURES / "margin_meaning.pdf")
    figure.savefig(FIGURES / "margin_meaning.png", dpi=180)
    plt.close(figure)
    write_json("margin_meaning.json", {
        "source": "results/validation.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "new_stress_simulations": False,
        "observations": observations,
        "methods": {
            "lyapunov_detection_threshold": 0.02,
            "duffing_scan": {"start": 1.0, "step": 0.15, "last_sample": 2.95},
            "josephson_scan": {"start": 1.0, "step": 0.15,
                               "last_sample": last_josephson_sample,
                               "null_interpretation": "No exponent above 0.02 at the sampled scales"},
            "finite_kappa_sim": "Linear interpolation of the first sampled detection-threshold crossing",
            "lyapunov_periods": {"duffing": {"discarded": 300, "averaged": 900},
                                 "josephson": {"discarded": 400, "averaged": 800}},
            "helmholtz_detection": "Safe fraction below 85 percent of its kappa=1 value",
            "helmholtz_grid": [13, 13], "helmholtz_scan_step": 0.5,
            "helmholtz_horizon_periods": 120, "helmholtz_escape_threshold": 1.5,
            "helmholtz_initial_condition_rectangle": [-0.6, 0.9, -0.6, 0.6],
            "limitations": [
                "Discrete scans can miss intermediate transitions and periodic windows.",
                "The saved scan does not establish a universal monotone relation between tau and kappa.",
                "The first-order band threshold and nonlinear detection threshold diagnose different events.",
            ],
        },
    })
    print("Replotted preserved stress scan with Josephson non-detections through kappa=3.40", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("josephson", "duffing", "margin", "all"), default="all")
    args = parser.parse_args()
    style()
    if args.only in ("duffing", "all"):
        refresh_duffing()
    if args.only in ("josephson", "all"):
        refresh_josephson()
    if args.only in ("margin", "all"):
        replot_margin()


if __name__ == "__main__":
    main()
