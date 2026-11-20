"""Check, recompute, and optionally regenerate the material in Online Resource 1."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
DESIGNS = ("duffing", "josephson", "helmholtz", "peak_demo", "lenci_rega")


def verify_distribution():
    manifest = json.loads((ROOT / "sha256.json").read_text())
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed distribution file: {name}")
    print(f"Verified {len(manifest)} distributed files.", flush=True)


def compare_values(observed, expected, label):
    if isinstance(expected, dict):
        if observed.keys() != expected.keys():
            raise ValueError(f"Different fields at {label}")
        for key, value in expected.items():
            # These values are elapsed wall-clock seconds on the original machine.
            if key not in ("solve_min", "solve_max"):
                compare_values(observed[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError(f"Different lengths at {label}")
        for index, value in enumerate(expected):
            compare_values(observed[index], value, f"{label}[{index}]")
    elif type(expected) in (int, float) and type(observed) in (int, float):
        # Most optimization results are stored to four decimal places.
        if not math.isclose(observed, expected, rel_tol=0, abs_tol=1.01e-4):
            raise ValueError(f"{label}: computed {observed}, archived {expected}")
    elif observed != expected:
        raise ValueError(f"{label}: computed {observed!r}, archived {expected!r}")


def reproduce(work, args):
    for folder in ("results", "figures", "scripts", "revision_v11"):
        shutil.copytree(ROOT / folder, work / folder)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")

    def run(script, *options):
        subprocess.run([sys.executable, str(work / "scripts" / script), *options],
                       cwd=work, env=env, check=True)

    run("check_diagnostics.py")
    for name in DESIGNS:
        (work / "results" / f"{name}.json").unlink()
    run("compute_results.py")
    run("lenci_rega_benchmark.py")
    for name in DESIGNS:
        path = Path("results", name + ".json")
        compare_values(json.loads((work / path).read_text()),
                       json.loads((ROOT / path).read_text()), name)
    print("All five optimization and benchmark data files reproduce to the reporting tolerance.", flush=True)
    run("v11_algebra_checks.py")

    if args.scans:
        (work / "results/validation.json").unlink()
        run("validation.py")
    if args.figures:
        run("make_design_figures.py")
    if args.diagnostics:
        run("v11_kernel_verification.py")
        run("v11_manifolds.py")
        run("v11_validation_figures.py")
    else:
        run("v11_validation_figures.py", "--only", "margin")

    abstract = work / "figures/v12/graphical_abstract.pdf"
    abstract.unlink(missing_ok=True)
    run("v12/graphical_abstract.py")
    if not abstract.is_file() or abstract.stat().st_size == 0:
        raise RuntimeError("Graphical abstract was not regenerated")
    run("inject_numbers.py")
    run("check_diagnostics.py")
    print("Tables, graphical abstract, and scientific checks complete.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Keep calculations in a new directory.")
    parser.add_argument("--figures", action="store_true", help="Re-solve all design-figure power grids.")
    parser.add_argument("--diagnostics", action="store_true", help="Recompute quadratures, manifolds and sections.")
    parser.add_argument("--scans", action="store_true", help="Recompute the long epsilon, forcing-stress and escape sweeps.")
    args = parser.parse_args()
    verify_distribution()
    if args.output:
        work = args.output.resolve()
        work.mkdir(parents=True, exist_ok=False)
        reproduce(work, args)
        print(f"Reproduced files: {work}", flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="chaos-control-") as temp:
            reproduce(Path(temp), args)


if __name__ == "__main__":
    main()
