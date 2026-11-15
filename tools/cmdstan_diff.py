#!/usr/bin/env python3
"""Compares the fixture models' log densities and gradients with CmdStan's `log_prob` method.

Usage (from the repository root):

    python3 tools/cmdstan_diff.py CMDSTAN_DIR WORK_DIR

CMDSTAN_DIR is a CmdStan 2.40.0 installation and WORK_DIR a scratch directory for the CmdStan
executables. Each model in `crates/stanc-rs-fixtures/models` is compiled by CmdStan and run with
the data in `tools/cmdstan_cases.json` at a fixed unconstrained point, with and without the
Jacobian adjustment. The same point is evaluated by the stanc-rs build of the model
(`cargo run -p stanc-rs-fixtures --example log_prob`). Both compute the log density up to a
constant, so the two must agree.
"""
import concurrent.futures as cf
import csv
import json
import math
import os
import shutil
import subprocess
import sys

TOL = 1e-8
# Looser gradient tolerances, each with its reason. The values from stanc-rs were checked
# against exact derivatives in the fixture tests.
GRADIENT_TOL = {
    # Truncated normals: Stan Math's normal_lcdf and normal_lccdf derivatives are rational
    # approximations with errors near 1e-6, while stanmath-rs differentiates exactly.
    "vector_truncate": 1e-5,
}


def main():
    cmdstan, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = os.path.join(repo, "crates", "stanc-rs-fixtures", "models")
    cases = json.load(open(os.path.join(repo, "tools", "cmdstan_cases.json")))
    os.makedirs(work, exist_ok=True)
    subprocess.run(
        ["cargo", "build", "-q", "--release", "-p", "stanc-rs-fixtures", "--example", "log_prob"],
        cwd=repo, check=True,
    )
    driver = os.path.join(repo, "target", "release", "examples", "log_prob")
    for name, d in cases.items():
        json.dump(d, open(os.path.join(work, f"{name}.data.json"), "w"))
        shutil.copy(os.path.join(models, f"{name}.stan"), work)

    def build(name):
        r = subprocess.run(["make", "-s", os.path.join(work, name)], cwd=cmdstan,
                           capture_output=True, text=True)
        return name, r.returncode, r.stderr[-2000:]

    with cf.ThreadPoolExecutor(4) as ex:
        built = list(ex.map(build, cases))
    failures = 0
    skipped = 0
    for name, code, err in built:
        if code != 0:
            print(f"{name}: CmdStan build failed\n{err}")
            failures += 1
            continue
        data = os.path.join(work, f"{name}.data.json")
        n = int(subprocess.run([driver, name, data], capture_output=True, text=True,
                               check=True).stdout)
        if n == 0:
            print(f"{name}: no parameters, skipped")
            skipped += 1
            continue
        theta = [round(0.6 * math.sin(1.7 * i + 0.3), 6) for i in range(n)]
        ours = json.loads(subprocess.run(
            [driver, name, data, ",".join(map(str, theta))],
            capture_output=True, text=True, check=True).stdout)
        u = os.path.join(work, f"{name}.u.json")
        json.dump({"params_r": theta}, open(u, "w"))
        bad = []
        for key, jac in [("jacobian", 1), ("no_jacobian", 0)]:
            out = os.path.join(work, f"{name}.{key}.csv")
            r = subprocess.run(
                [os.path.join(work, name), "log_prob", f"unconstrained_params={u}",
                 f"jacobian={jac}", "data", f"file={data}", "output", f"file={out}", "sig_figs=18"],
                capture_output=True, text=True)
            if r.returncode != 0:
                bad.append(f"CmdStan log_prob failed: {r.stdout[-500:]}{r.stderr[-500:]}")
                continue
            rows = [row for row in csv.reader(l for l in open(out) if not l.startswith("#"))]
            ref = [float(v) for v in rows[1]]
            got = [ours[key]["lp"]] + ours[key]["gradient"]
            if len(ref) != len(got):
                bad.append(f"{key}: {len(ref)} CmdStan values, {len(got)} stanc-rs values")
                continue
            for i, (a, b) in enumerate(zip(ref, got)):
                tol = TOL if i == 0 else GRADIENT_TOL.get(name, TOL)
                if abs(a - b) > tol * max(1.0, abs(a)):
                    what = "lp" if i == 0 else rows[0][i]
                    bad.append(f"{key} {what}: CmdStan {a!r}, stanc-rs {b!r}")
        if bad:
            failures += 1
            print(f"{name}:\n  " + "\n  ".join(bad))
        else:
            print(f"{name}: agrees ({n} parameters)")
    compared = len(cases) - skipped
    print(f"{compared - failures} of {compared} models with parameters agree with CmdStan")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
