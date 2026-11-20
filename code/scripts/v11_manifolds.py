"""Stroboscopic Duffing invariant manifolds for manuscript v11.

This file and its outputs are independent of the frozen v10 artifacts.
The hyperbolic T-periodic orbit is obtained by continuation/shooting; its
monodromy eigendirections seed forward/backward iterates in the SAME t=0
section. Numerical curves are finite segments, not a proof of global absence
of intersections. Run with ``python3 scripts/v11_manifolds.py``; use
``--replot`` to redraw the saved JSON/NPZ data without recomputing numerics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
PERIOD = 2 * np.pi
EPS = 0.25
DELTA = 0.8
EXISTING = np.array([1.2, 0.6, 0.32, 0.0, 0.0])
RTOL = 2e-12
ATOL = 2e-13
SADDLE_EXCLUSION_RADIUS = 1e-6
MIN_CROSSING_ANGLE_DEGREES = 1.0
CROSSING_RESIDUAL_TOLERANCE = 1e-8


def controller():
    """Min-power two-branch N=5 controller at stored matched margin."""
    from sdp_solver import solve_suppression_sdp

    k = np.arange(1, 6)
    beta = np.sqrt(2) * np.pi * k / np.cosh(np.pi * k / 2)
    stored = json.loads((ROOT / "results/duffing.json").read_text())
    tau = stored["tau_min_matched"]
    res = solve_suppression_sdp(
        -4 * DELTA / 3, np.zeros(3), EXISTING[:3] * beta[:3],
        np.zeros(5), beta, 5, objective="min_power", tau_min=tau,
        both_branches=True,
    )
    if res["a"] is None or min(res["margin"], res["margin_L"]) < tau - 1e-6:
        raise RuntimeError("Controller solve did not meet both branch margins")
    return {
        "a": np.asarray(res["a"]), "b": np.asarray(res["b"]),
        "tau_target": tau, "margin_R": float(res["margin"]),
        "margin_L": float(res["margin_L"]), "power": float(res["energy"]),
        "source": "results/duffing.json stored matched margin (four decimals)",
    }


def rhs(t, z, ac, bs, eps=EPS, variational=False):
    forcing = np.dot(ac, np.cos(np.arange(1, 6) * t)) + np.dot(
        bs, np.sin(np.arange(1, 6) * t)
    )
    x, y = z[:2]
    out = [y, x - x**3 + eps * (-DELTA * y + forcing)]
    if variational:
        j = np.array([[0, 1], [1 - 3 * x*x, -eps * DELTA]])
        out.extend((j @ z[2:].reshape(2, 2)).ravel())
    return np.asarray(out)


def flow(z, ac, bs, duration=PERIOD, eps=EPS, variational=False,
         rtol=RTOL, atol=ATOL):
    initial = np.r_[z, np.eye(2).ravel()] if variational else z
    sol = solve_ivp(
        lambda t, zz: rhs(t, zz, ac, bs, eps, variational),
        (0.0, duration), initial, method="DOP853", rtol=rtol,
        atol=atol, max_step=PERIOD / 80,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    end = sol.y[:, -1]
    return (end[:2], end[2:].reshape(2, 2)) if variational else end


def saddle(ac, bs):
    z = np.zeros(2)
    log = []
    prev_eps = 0.0
    for eps in np.linspace(0.0, EPS, 26):
        if eps > 0:
            if prev_eps > 0:
                z = z * eps / prev_eps
            else:
                k = np.arange(1, 6)
                z = -eps * np.array([np.sum(ac / (1 + k*k)),
                                     np.sum(k * bs / (1 + k*k))])
        cache = {}

        def evaluate(zz):
            key = tuple(zz)
            if key not in cache:
                end, mat = flow(zz, ac, bs, eps=eps, variational=True)
                cache[key] = (end - zz, mat - np.eye(2))
            return cache[key]

        sol = root(lambda zz: evaluate(zz)[0], z,
                   jac=lambda zz: evaluate(zz)[1], tol=1e-11)
        z = sol.x
        residual = float(np.linalg.norm(evaluate(z)[0], np.inf))
        if residual > 1e-9:
            raise RuntimeError(f"Periodic-orbit shooting residual {residual} at eps={eps}")
        log.append({"eps": float(eps), "residual_inf": residual})
        prev_eps = eps
    end, mat = flow(z, ac, bs, variational=True)
    val, vec = np.linalg.eig(mat)
    order = np.argsort(np.abs(val))
    val, vec = np.real(val[order]), np.real(vec[:, order])
    vec /= np.linalg.norm(vec, axis=0)
    for k in range(2):
        if vec[0, k] < 0:
            vec[:, k] *= -1
    if not abs(val[0]) < 1 < abs(val[1]):
        raise RuntimeError(f"Continued orbit is not a saddle: {val}")
    return z, val, vec, mat, log


def local_series(p, ac, bs, lam, v):
    """Cubic invariant-manifold parameterization W(s) near the fixed point.

    Integrate exact first/second/third flow variational equations.  Solve
    P(W(s))=W(lam*s) coefficient by coefficient, with W'(0)=v.
    """
    def extended(t, state):
        z = state[:2]
        a = state[2:6].reshape(2, 2)
        b = state[6:14].reshape(2, 2, 2)
        c = state[14:].reshape(2, 2, 2, 2)
        j = np.array([[0, 1], [1 - 3*z[0]**2, -EPS*DELTA]])
        da = j @ a
        db = np.einsum("ij,jab->iab", j, b)
        db[1] -= 6*z[0]*np.einsum("a,b->ab", a[0], a[0])
        dc = np.einsum("ij,jabc->iabc", j, c)
        dc[1] -= 6*z[0]*(np.einsum("ab,c->abc", b[0], a[0])
                           + np.einsum("ac,b->abc", b[0], a[0])
                           + np.einsum("bc,a->abc", b[0], a[0]))
        dc[1] -= 6*np.einsum("a,b,c->abc", a[0], a[0], a[0])
        return np.r_[rhs(t, z, ac, bs), da.ravel(), db.ravel(), dc.ravel()]

    sol = solve_ivp(extended, (0, PERIOD), np.r_[p, np.eye(2).ravel(), np.zeros(24)],
                    method="DOP853", rtol=RTOL, atol=ATOL, max_step=PERIOD/100)
    if not sol.success:
        raise RuntimeError(sol.message)
    a = sol.y[2:6, -1].reshape(2, 2)
    b = sol.y[6:14, -1].reshape(2, 2, 2)
    c = sol.y[14:, -1].reshape(2, 2, 2, 2)
    w2 = np.linalg.solve(lam**2*np.eye(2)-a,
                         0.5*np.einsum("iab,a,b->i", b, v, v))
    w3 = np.linalg.solve(lam**3*np.eye(2)-a,
                         np.einsum("iab,a,b->i", b, v, w2)
                         + np.einsum("iabc,a,b,c->i", c, v, v, v)/6)
    return np.column_stack([v, w2, w3])


def map_many(initial, ac, bs, periods, rtol=RTOL, atol=ATOL):
    """Vectorized independent flows, all ending in the t=0 mod T section."""
    initial = np.asarray(initial)
    k = np.arange(1, 6)

    def many_rhs(t, flat):
        z = flat.reshape(-1, 2)
        force = np.dot(ac, np.cos(k*t)) + np.dot(bs, np.sin(k*t))
        result = np.empty_like(z)
        result[:, 0] = z[:, 1]
        result[:, 1] = z[:, 0] - z[:, 0]**3 + EPS*(-DELTA*z[:, 1]+force)
        return result.ravel()

    sol = solve_ivp(many_rhs, (0, periods*PERIOD), initial.ravel(),
                    method="DOP853", rtol=rtol, atol=atol, max_step=PERIOD/100)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1].reshape(-1, 2)


def seed(p, series, q):
    q = np.asarray(q)
    return p + np.column_stack([q, q*q, q*q*q]) @ series.T


def trace(p, series, ac, bs, periods, qmax, sign, chord=0.012, bend=2e-5):
    """Adaptively sample one parameterized finite manifold branch.

    Refinement requires short image chords and a midpoint interpolation error
    below `bend`; no interpolation is mistaken for a computed trajectory.
    """
    q = np.linspace(0, sign*qmax, 65)
    pts = map_many(seed(p, series, q), ac, bs, periods)
    for level in range(16):
        mids = (q[:-1]+q[1:])/2
        mp = map_many(seed(p, series, mids), ac, bs, periods)
        length = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        error = np.linalg.norm(mp-(pts[:-1]+pts[1:])/2, axis=1)
        # Only the visible region is used in the final figure.  Retain a
        # generous halo so no unrefined segment can jump through the panel.
        nearby = (np.linalg.norm(pts[:-1], axis=1) < 3.5) | (
            np.linalg.norm(pts[1:], axis=1) < 3.5) | (np.linalg.norm(mp, axis=1) < 3.5)
        split = nearby & ((length > chord) | (error > bend))
        if not np.any(split):
            return q, pts, {
                "points": int(len(q)), "refinement_levels": level,
                "max_visible_chord": float(np.max(length[nearby])),
                "max_visible_midpoint_error": float(np.max(error[nearby])),
            }
        idx = np.flatnonzero(split)
        qnew = np.r_[q, mids[idx]]
        pnew = np.vstack([pts, mp[idx]])
        sort = np.argsort(sign*qnew)
        q, pts = qnew[sort], pnew[sort]
        if len(q) > 30000:
            raise RuntimeError("Manifold tracing exceeded point budget")
    raise RuntimeError("Manifold refinement did not converge")


def cross2(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def intersections(unstable, stable, p, ac, bs, su, ss):
    """Locate polygon crossings, then solve equality of the actual flow maps."""
    qu, pu = unstable
    qs, ps = stable
    au, ast = pu[:-1], ps[:-1]
    ru, rs = np.diff(pu, axis=0), np.diff(ps, axis=0)
    diff = ast[None, :, :] - au[:, None, :]
    den = cross2(ru[:, None, :], rs[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        tu = cross2(diff, rs[None, :, :]) / den
        ts = cross2(diff, ru[:, None, :]) / den
    hit = np.isfinite(tu) & (tu > 0) & (tu < 1) & (ts > 0) & (ts < 1)
    found = []
    for i, j in np.argwhere(hit):
        guess = np.array([qu[i]+tu[i, j]*(qu[i+1]-qu[i]),
                          qs[j]+ts[i, j]*(qs[j+1]-qs[j])])
        scale = np.array([abs(qu[-1]), abs(qs[-1])])

        def evaluate(uv, jac=False):
            q = uv*scale
            u, mu = flow(seed(p, su, [q[0]])[0], ac, bs,
                         duration=2*PERIOD, variational=True)
            s, ms = flow(seed(p, ss, [q[1]])[0], ac, bs,
                         duration=-2*PERIOD, variational=True)
            if jac:
                du = mu @ (su @ np.array([1, 2*q[0], 3*q[0]**2]))
                ds = ms @ (ss @ np.array([1, 2*q[1], 3*q[1]**2]))
                return np.column_stack([du*scale[0], -ds*scale[1]])
            return u-s

        sol = root(evaluate, guess/scale, jac=lambda uv: evaluate(uv, True), tol=1e-10)
        qstar = sol.x*scale
        if not np.all(np.isfinite(qstar)):
            continue
        point = flow(seed(p, su, [qstar[0]])[0], ac, bs, duration=2*PERIOD)
        residual = float(np.linalg.norm(evaluate(sol.x)))
        if (not np.isfinite(residual) or residual > CROSSING_RESIDUAL_TOLERANCE
                or np.linalg.norm(point-p) < SADDLE_EXCLUSION_RADIUS):
            continue
        if not (min(qu) <= qstar[0] <= max(qu) and min(qs) <= qstar[1] <= max(qs)):
            continue
        jmat = evaluate(sol.x, True)
        tangent_norms = np.linalg.norm(jmat, axis=0)
        if not np.all(np.isfinite(jmat)) or np.any(tangent_norms < 1e-12):
            continue
        sine_angle = abs(np.linalg.det(jmat))/np.prod(tangent_norms)
        if not np.isfinite(sine_angle) or sine_angle < np.sin(np.deg2rad(MIN_CROSSING_ANGLE_DEGREES)):
            continue
        if any(np.linalg.norm(point-np.array(old["point"])) < 1e-4 for old in found):
            continue
        found.append({"point": point.tolist(), "q_unstable": float(qstar[0]),
                      "q_stable": float(qstar[1]), "residual": residual,
                      "tangent_angle_degrees": float(np.degrees(np.arcsin(min(sine_angle, 1.0))))})
    return found


def crossing_crosscheck(hit, p, ac, bs, su, ss, multipliers):
    """Re-solve a crossing after one extra forward/backward period.

    Floquet scaling identifies the same local manifold parameters. Backward
    iteration magnifies shooting and floating-point errors, so record the
    residual independently rather than implying exact interval verification.
    """
    scale = np.array([0.002, 1e-5])
    original_q = np.array([hit["q_unstable"], hit["q_stable"]])

    def evaluate(uv, jac=False):
        q = uv*scale
        u_q = q[0]/multipliers[1]
        s_q = q[1]*multipliers[0]
        u, mu = flow(seed(p, su, [u_q])[0], ac, bs,
                     duration=3*PERIOD, variational=True)
        s, ms = flow(seed(p, ss, [s_q])[0], ac, bs,
                     duration=-3*PERIOD, variational=True)
        if jac:
            du = mu @ (su @ np.array([1, 2*u_q, 3*u_q**2]))
            ds = ms @ (ss @ np.array([1, 2*s_q, 3*s_q**2]))
            return np.column_stack([du*scale[0]/multipliers[1],
                                    -ds*scale[1]*multipliers[0]])
        return u-s

    sol = root(evaluate, original_q/scale, jac=lambda uv: evaluate(uv, True), tol=1e-9)
    adjusted_q = sol.x*scale
    point = flow(seed(p, su, [adjusted_q[0]/multipliers[1]])[0], ac, bs,
                 duration=3*PERIOD)
    residual = float(np.linalg.norm(evaluate(sol.x)))
    displacement = float(np.linalg.norm(point-np.array(hit["point"])))
    jmat = evaluate(sol.x, True)
    sine = abs(np.linalg.det(jmat))/np.prod(np.linalg.norm(jmat, axis=0))
    angle = float(np.degrees(np.arcsin(min(sine, 1.0))))
    if (not np.all(np.isfinite([residual, displacement, angle]))
            or residual > 2e-6 or displacement > 2e-6
            or angle < MIN_CROSSING_ANGLE_DEGREES):
        raise RuntimeError("Extra-period crossing check did not converge consistently")
    return {"extra_periods": 1, "root_solver_success": bool(sol.success),
            "residual": residual, "point_displacement": displacement,
            "tangent_angle_degrees": angle,
            "point": point.tolist(),
            "acceptance_max_residual_and_displacement": 2e-6}


def draw_figure(result, arrays, figdir):
    """Shared publication rendering for fresh computations and --replot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for case_idx, name in enumerate(["uncontrolled", "controlled"]):
        ax = axes[case_idx]
        case = result["cases"][name]
        for kind, color, style in [("unstable", "#0072B2", "-"),
                                    ("stable", "#D55E00", "--")]:
            for sign in ["negative", "positive"]:
                pts = arrays[f"{name}_{kind}_{sign}_xy"]
                label = r"$W^u$" if kind == "unstable" else r"$W^s$"
                ax.plot(pts[:, 0], pts[:, 1], style, color=color, lw=1.25,
                        label=label if sign == "negative" else None)
        for hit in case["transverse_crossings_in_traced_segments"]:
            ax.plot(*hit["point"], "o", ms=4.8, mfc="white", mec="black", mew=0.9, zorder=5)
        ax.plot(*case["periodic_point"], "ks", ms=4, zorder=6, label="Saddle periodic orbit")
        ax.set_title("(a) Uncontrolled" if case_idx == 0 else r"(b) $N=5$ min-power control")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18, lw=0.5)
    axes[0].set_ylabel(r"$\dot{x}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig_duffing_manifolds.pdf", bbox_inches="tight")
    fig.savefig(figdir / "fig_duffing_manifolds.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--replot", action="store_true",
                      help="redraw existing JSON/NPZ results without recomputing")
    args = parser.parse_args()
    figdir = ROOT / "figures/v11"
    outdir = ROOT / "results/v11"
    if args.replot:
        result = json.loads((outdir / "manifolds.json").read_text())
        with np.load(outdir / "manifold_curves.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        draw_figure(result, arrays, figdir)
        print("redrew", figdir / "fig_duffing_manifolds.pdf", flush=True)
        return
    ctl = controller()
    if not args.probe:
        figdir.mkdir(parents=True, exist_ok=True)
        outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "x''-x+x^3=eps*(-delta*x'+F(t))",
        "eps": EPS, "delta": DELTA, "Omega": 1.0, "section": "t=0 mod 2*pi",
        "controller": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in ctl.items()},
        "method": {
            "periodic_orbit": "shooting with analytic variational Jacobian; epsilon continuation in steps 0.01",
            "local_seeds": "cubic invariant-manifold parameterization from first/second/third flow derivatives",
            "tracing": "P^2(W_u(q)) and P^-2(W_s(q)), both signs of q, all in the same stroboscopic section",
            "integrator": "DOP853", "rtol": RTOL, "atol": ATOL,
            "shooting_max_step": float(PERIOD/80),
            "tracing_max_step": float(PERIOD/100), "unstable_qmax": 0.002,
            "stable_qmax": 1e-5, "maximum_chord": 0.012,
            "maximum_midpoint_interpolation_error": 2e-5,
            "crossing_acceptance": {
                "saddle_exclusion_radius": SADDLE_EXCLUSION_RADIUS,
                "maximum_root_residual": CROSSING_RESIDUAL_TOLERANCE,
                "minimum_tangent_angle_degrees": MIN_CROSSING_ANGLE_DEGREES,
                "minimum_scaled_tangent_norm": 1e-12,
                "requires_finite_parameters_residual_and_tangents": True,
            },
            "convergence": "compare a third period with the seed divided/multiplied by the Floquet multiplier; repeat at tighter ODE tolerances",
            "scope": "finite numerical manifold segments; zero detected crossings does not prove global separation",
        },
        "cases": {},
    }
    arrays = {}
    for name, ac, bs in [
        ("uncontrolled", EXISTING, np.zeros(5)),
        ("controlled", EXISTING + ctl["a"], ctl["b"]),
    ]:
        p, val, vec, monodromy, log = saddle(ac, bs)
        print(name, "saddle", p, "multipliers", val,
              "residual", log[-1]["residual_inf"], flush=True)
        if args.probe:
            continue
        record = {
            "periodic_point": p.tolist(), "floquet_multipliers": val.tolist(),
            "floquet_directions_columns": vec.tolist(), "monodromy": monodromy.tolist(),
            "shooting_continuation": log, "forcing_a": ac.tolist(), "forcing_b": bs.tolist(),
            "liouville_determinant_relative_error": float(abs(np.linalg.det(monodromy)
                / np.exp(-EPS*DELTA*PERIOD)-1)), "curves": {},
        }
        curves = {"unstable": [], "stable": []}
        series = {}
        for kind, eig_idx, periods, qmax in [
            ("unstable", 1, 2, 0.002),
            ("stable", 0, -2, 1e-5),
        ]:
            local = local_series(p, ac, bs, val[eig_idx], vec[:, eig_idx])
            series[kind] = local
            record[kind+"_local_series_columns"] = local.tolist()
            qcheck = np.r_[-np.geomspace(qmax/1000, qmax, 60),
                           np.geomspace(qmax/1000, qmax, 60)]
            reference = map_many(seed(p, local, qcheck), ac, bs, periods)
            qnext = qcheck/val[eig_idx] if periods > 0 else qcheck*val[eig_idx]
            extra = map_many(seed(p, local, qnext), ac, bs, periods+np.sign(periods))
            finer = map_many(seed(p, local, qcheck), ac, bs, periods,
                             rtol=3e-13, atol=3e-14)
            invariant_error = float(np.max(np.linalg.norm(reference-extra, axis=1)))
            tolerance_error = float(np.max(np.linalg.norm(reference-finer, axis=1)))
            if invariant_error > 2e-5 or tolerance_error > 1e-7:
                raise RuntimeError(f"{name} {kind} convergence checks failed")
            record[kind+"_convergence"] = {
                "sample_points": int(len(qcheck)),
                "extra_period_max_distance": invariant_error,
                "tighter_tolerance_max_distance": tolerance_error,
            }
            for sign in [-1, 1]:
                q, pts, metrics = trace(p, local, ac, bs, periods, qmax, sign)
                curves[kind].append((q, pts))
                key = f"{kind}_{'negative' if sign < 0 else 'positive'}"
                record["curves"][key] = metrics
                arrays[name+"_"+key+"_q"] = q
                arrays[name+"_"+key+"_xy"] = pts
        hits = []
        for unstable in curves["unstable"]:
            for stable in curves["stable"]:
                hits.extend(intersections(unstable, stable, p, ac, bs,
                                          series["unstable"], series["stable"]))
        for hit in hits:
            hit["extra_period_check"] = crossing_crosscheck(
                hit, p, ac, bs, series["unstable"], series["stable"], val)
        record["transverse_crossings_in_traced_segments"] = hits
        record["crossing_scope"] = "numerical roots on the finite segments; no global absence claim"
        if name == "uncontrolled" and not hits:
            raise RuntimeError("Expected uncontrolled transverse crossings were not located")
        result["cases"][name] = record
        print(name, "crossings", len(hits), "convergence",
              record["unstable_convergence"], record["stable_convergence"], flush=True)
    if args.probe:
        print("controller", ctl, flush=True)
        return
    np.savez_compressed(outdir / "manifold_curves.npz", **arrays)
    (outdir / "manifolds.json").write_text(json.dumps(result, indent=2)+"\n")
    draw_figure(result, arrays, figdir)
    print("wrote", figdir / "fig_duffing_manifolds.pdf", flush=True)


if __name__ == "__main__":
    main()
