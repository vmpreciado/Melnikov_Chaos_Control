"""Check the saved kernel, manifold, return-map and Lyapunov diagnostics."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require_small(value, bound, label):
    if not math.isfinite(value) or value < 0 or value > bound:
        raise RuntimeError(f"{label} is {value!r}; required 0 <= value <= {bound:g}")


def check_kernel_verification():
    kernels = json.loads((ROOT / "results/v11/kernel_verification.json").read_text())
    expected = {(system, k) for system in ("duffing", "josephson", "helmholtz") for k in range(1, 11)}
    rows = kernels["rows"]
    if len(rows) != 30 or {(row["system"], row["k"]) for row in rows} != expected:
        raise RuntimeError("Kernel verification must contain all 30 system/harmonic combinations")
    # The generator requires 1e-20 agreement; the appendix quotes the same bound.
    for row in rows:
        for key in ("relative_error", "crosscheck_relative_difference", "crosscheck_relative_error",
                    "damping_relative_error"):
            require_small(row[key], 1e-20, f"Kernel {row['system']} k={row['k']} {key}")
        require_small(row["omitted_tail_bound"], 5e-43, "Kernel omitted-tail bound")


def check_manifolds():
    data = json.loads((ROOT / "results/v11/manifolds.json").read_text())
    if set(data["cases"]) != {"uncontrolled", "controlled"}:
        raise RuntimeError("Both manifold cases must be present")
    controller = data["controller"]
    if min(controller["margin_R"], controller["margin_L"]) < controller["tau_target"] - 1e-6:
        raise RuntimeError("Manifold controller does not meet its two-branch target")
    expected_arrays = set()
    with np.load(ROOT / "results/v11/manifold_curves.npz", allow_pickle=False) as curves:
        for name, case in data["cases"].items():
            # Shooting uses 1e-9; convergence bounds below enforce the tighter
            # values reported in the manuscript (generator acceptance is looser).
            for step in case["shooting_continuation"]:
                require_small(step["residual_inf"], 1e-9, "Manifold shooting residual")
            require_small(case["liouville_determinant_relative_error"], 1e-8,
                          "Monodromy Liouville determinant error")
            monodromy = np.asarray(case["monodromy"], dtype=float)
            determinant = np.linalg.det(monodromy)
            exact_determinant = np.exp(-data["eps"] * data["delta"] * 2 * np.pi / data["Omega"])
            require_small(float(abs(determinant / exact_determinant - 1)), 1e-8,
                          "Recomputed monodromy determinant error")
            multipliers = np.linalg.eigvals(monodromy)
            if not np.isfinite(multipliers).all() or not (min(abs(multipliers)) < 1 < max(abs(multipliers))):
                raise RuntimeError("Continued manifold orbit is not a hyperbolic saddle")
            for kind in ("stable", "unstable"):
                convergence = case[kind + "_convergence"]
                require_small(convergence["extra_period_max_distance"], 5e-7,
                              "Manifold rescaled-seed convergence")
                require_small(convergence["tighter_tolerance_max_distance"], 2e-10,
                              "Manifold tighter-tolerance convergence")
                for sign in ("negative", "positive"):
                    key = kind + "_" + sign
                    metrics = case["curves"][key]
                    require_small(metrics["max_visible_chord"], 0.012 + 1e-12,
                                  "Manifold visible chord")
                    require_small(metrics["max_visible_midpoint_error"], 2e-5 + 1e-12,
                                  "Manifold midpoint interpolation error")
                    prefix = name + "_" + key
                    expected_arrays.update((prefix + "_q", prefix + "_xy"))
                    q, xy = curves[prefix + "_q"], curves[prefix + "_xy"]
                    if q.shape != (metrics["points"],) or xy.shape != (metrics["points"], 2):
                        raise RuntimeError("Manifold NPZ dimensions do not match the metadata")
                    if not np.isfinite(q).all() or not np.isfinite(xy).all():
                        raise RuntimeError("Manifold NPZ contains non-finite values")
            hits = case["transverse_crossings_in_traced_segments"]
            if name == "uncontrolled" and len(hits) < 3:
                raise RuntimeError("The three reported uncontrolled transverse crossings are missing")
            if name == "controlled" and hits:
                raise RuntimeError("The controlled traced segments now contain crossings")
            for hit in hits:
                require_small(hit["residual"], 1e-10, "Manifold crossing residual")
                if not math.isfinite(hit["tangent_angle_degrees"]) or hit["tangent_angle_degrees"] <= 30:
                    raise RuntimeError("Manifold crossing angle no longer meets the reported 30-degree bound")
                extra_period = hit["extra_period_check"]
                require_small(extra_period["residual"], 2e-6,
                              "Crossing extra-period residual")
                require_small(extra_period["point_displacement"], 2e-6,
                              "Crossing extra-period displacement")
        if set(curves.files) != expected_arrays:
            raise RuntimeError("Unexpected or missing arrays in the saved manifold NPZ")


def check_diagnostics():
    required = [
        "figures/v11/duffing_poincare.pdf", "figures/v11/josephson_validation.pdf",
        "figures/v11/margin_meaning.pdf", "figures/v11/fig_duffing_manifolds.pdf",
        "results/v11/duffing_poincare.json", "results/v11/josephson_validation.json",
        "results/v11/margin_meaning.json", "results/v11/manifolds.json",
        "results/v11/manifold_curves.npz",
        "results/v11/algebra_checks.json",
        "results/v11/kernel_verification.json", "revision_v11/tab_kernel_verification.tex",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Missing v11 artifacts (use --refresh-diagnostics): " + ", ".join(missing))
    check_kernel_verification()
    check_manifolds()
    josephson = json.loads((ROOT / "results/v11/josephson_validation.json").read_text())
    if sorted(case["N"] for case in josephson["cases"]) != [0, 1, 3]:
        raise RuntimeError("Missing or duplicate Josephson cases")
    for case in josephson["cases"]:
        sections = np.asarray(case["section_points"], dtype=float)
        if sections.shape != (3, 120, 2) or not np.isfinite(sections).all():
            raise RuntimeError("Josephson section samples missing or non-finite")
        points = sections.reshape(-1, 2)
        radius = float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
        step_residual = float(np.max(np.linalg.norm(np.diff(sections, axis=1), axis=2)))
        for value in (radius, step_residual, case["maximum_section_radius"],
                      *case["successive_point_residuals"], case["floquet"]["one_period_residual"]):
            require_small(value, 1e-7, "Josephson settling/Floquet residual")
        exponent = case["floquet"]["largest_state_exponent"]
        if not math.isfinite(exponent) or exponent >= 0:
            raise RuntimeError("Josephson periodic orbit is not linearly stable")
        multipliers = np.array([complex(*value) for value in case["floquet"]["multipliers"]])
        if not np.isfinite(multipliers).all() or not np.all(np.abs(multipliers) < 1):
            raise RuntimeError("Josephson Floquet multipliers are not stable")
        computed_exponent = float(np.log(np.max(np.abs(multipliers))) / (2 * np.pi))
        require_small(abs(computed_exponent - exponent), 1e-10, "Josephson Floquet exponent consistency")
    duffing = json.loads((ROOT / "results/v11/duffing_poincare.json").read_text())
    if sorted(case["N"] for case in duffing["cases"]) != [0, 1, 3, 5]:
        raise RuntimeError("Missing or duplicate Duffing cases")
    for case in duffing["cases"]:
        sections = np.asarray(case["section_points"], dtype=float)
        expected_shape = (1 if case["N"] else 3, 2000, 2)
        if sections.shape != expected_shape or not np.isfinite(sections).all():
            raise RuntimeError("Duffing section samples missing or non-finite")
        exponent = case["largest_state_lyapunov_exponent"]
        if not math.isfinite(exponent) or (exponent >= 0 if case["N"] else exponent <= 0):
            raise RuntimeError("Duffing Lyapunov sign does not support the reported behavior")
        if case["N"] and (case["controller"] is None or
                          min(case["controller"]["margin_right"], case["controller"]["margin_left"]) <= 0):
            raise RuntimeError("Duffing current two-branch controller verification failed")
    stress = json.loads((ROOT / "results/v11/margin_meaning.json").read_text())
    source = ROOT / stress["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != stress["source_sha256"]:
        raise RuntimeError("Preserved stress-scan source does not match v11 provenance")
    print("Version-11 diagnostics and stress-data provenance checks passed.", flush=True)


if __name__ == "__main__":
    check_diagnostics()
