#!/usr/bin/env python3
"""Compares the MIR of stanc-rs and stanc3 (`--debug-mir-pretty`) on stanc3's
`test/integration/good` programs.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/mir_diff.py STANC3_DIR STANC [OUT_FILE] [-- extra flags]

STANC3_DIR is a checkout of stanc3 and STANC a stanc3 binary of the same version. Reports on
how many programs the printed MIR is identical. With OUT_FILE, both texts of every mismatch are
written there. Flags after `--` are passed to both compilers; a `--debug-*-mir-pretty` flag
among them replaces `--debug-mir-pretty`, to compare another stage of the MIR.
"""
import concurrent.futures as cf
import os
import subprocess
import sys

args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[: args.index("--")] if "--" in args else args
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(os.path.abspath(args[0]), "test", "integration")
st = os.path.abspath(args[1])
out = args[2] if len(args) > 2 else None
rs = os.path.join(repo, "target", "release", "stanc-rs")
stage = next((f for f in extra if f.startswith("--debug-") and f.endswith("mir-pretty")),
             "--debug-mir-pretty")
extra = [f for f in extra if f != stage]
good = sorted(
    os.path.relpath(os.path.join(d, f), root)
    for d, _, fs in os.walk(os.path.join(root, "good"))
    for f in fs
    if f.endswith(".stan")
)


def one(p):
    inc = f"included,{os.path.dirname(p)}"
    a = subprocess.run([st, f"--include-paths={inc}", "--o=/dev/null", stage,
                        *extra, p], cwd=root, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    b = subprocess.run([rs, "check", p, "--include-paths", inc, stage, *extra],
                       cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return p, a.returncode == 0, a.stdout.strip(), b.stdout.strip()


with cf.ThreadPoolExecutor(8) as ex:
    res = [r for r in ex.map(one, good) if r[1]]
same = sum(1 for _, _, a, b in res if a == b)
if out:
    with open(out, "w") as f:
        for p, _, a, b in res:
            if a != b:
                f.write(f"=== {p}\n--- stanc3\n{a}\n--- stanc-rs\n{b}\n")
print(f"{len(res)} programs; identical MIR {same}")
