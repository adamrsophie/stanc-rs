#!/usr/bin/env python3
"""Compares the warnings of stanc-rs and stanc3 on stanc3's `test/integration/good` programs.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/warning_diff.py STANC3_DIR STANC [OUT_FILE] [-- extra flags]

STANC3_DIR is a checkout of stanc3 and STANC a stanc3 binary of the same version. Reports how
many programs get warnings from either compiler and on how many the warning text is identical.
With OUT_FILE, both texts of every mismatch are written there. Flags after `--` are passed to
both compilers (for example `--warn-pedantic`).
"""
import concurrent.futures as cf
import os
import subprocess
import sys

args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[: args.index("--")] if "--" in args else args
rs_extra = [x for e in extra for x in e.split("=", 1)]
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(os.path.abspath(args[0]), "test", "integration")
st = os.path.abspath(args[1])
out = args[2] if len(args) > 2 else None
rs = os.path.join(repo, "target", "release", "stanc-rs")
good = sorted(
    os.path.relpath(os.path.join(d, f), root)
    for d, _, fs in os.walk(os.path.join(root, "good"))
    for f in fs
    if f.endswith((".stan", ".stanfunctions"))
)


def one(p):
    inc = f"included,{os.path.dirname(p)}"
    a = subprocess.run([st, f"--include-paths={inc}", "--o=/dev/null", *extra, p], cwd=root,
                       capture_output=True, text=True)
    b = subprocess.run([rs, "check", p, "--include-paths", inc, *rs_extra], cwd=root,
                       capture_output=True, text=True)
    return p, a.stderr.strip(), b.stderr.strip()


with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, good))
warned = [r for r in res if r[1] or r[2]]
same = sum(1 for _, a, b in warned if a == b)
if out:
    with open(out, "w") as f:
        for p, a, b in warned:
            if a != b:
                f.write(f"=== {p}\n--- stanc3\n{a}\n--- stanc-rs\n{b}\n")
print(f"{len(warned)} programs with warnings from either compiler; identical text {same}")
