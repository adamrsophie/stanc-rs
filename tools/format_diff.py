#!/usr/bin/env python3
"""Compares `stanc-rs format` with `stanc --auto-format` on stanc3's `test/integration/good` programs.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/format_diff.py STANC3_DIR STANC [OUT_DIR] [-- extra flags]

STANC3_DIR is a checkout of stanc3 and STANC a stanc3 binary of the same version (CmdStan ships
one as `bin/stanc`). Flags after `--` are passed to both compilers (for example
`--max-line-length=60`). With OUT_DIR, a unified diff of every mismatch is written there.
"""
import concurrent.futures as cf
import difflib
import os
import subprocess
import sys

args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[: args.index("--")] if "--" in args else args
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(os.path.abspath(args[0]), "test", "integration")
st = os.path.abspath(args[1])
out = os.path.abspath(args[2]) if len(args) > 2 else None
rs = os.path.join(repo, "target", "release", "stanc-rs")
good = sorted(
    os.path.relpath(os.path.join(d, f), root)
    for d, _, fs in os.walk(os.path.join(root, "good"))
    for f in fs
    if f.endswith((".stan", ".stanfunctions"))
)
rs_extra = [x for e in extra for x in e.split("=", 1)]


def one(p):
    inc = f"included,{os.path.dirname(p)}"
    a = subprocess.run([st, "--auto-format", f"--include-paths={inc}", *extra, p],
                       cwd=root, capture_output=True, text=True)
    b = subprocess.run([rs, "format", p, "--include-paths", inc, *rs_extra],
                       cwd=root, capture_output=True, text=True)
    if a.returncode != 0:
        return p, None if b.returncode != 0 else "stanc3 failed", a, b
    if b.returncode != 0:
        lines = b.stderr.strip().splitlines() or [""]
        return p, "stanc-rs failed: " + lines[-1][:150], a, b
    return p, None if a.stdout == b.stdout else "differs", a, b


with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, good))
bad = [(p, e, a, b) for p, e, a, b in res if e]
both = sum(1 for _, _, a, _ in res if a.returncode == 0)
if out:
    os.makedirs(out, exist_ok=True)
for p, e, a, b in bad:
    print(p, "|", e)
    if out and e == "differs":
        d = difflib.unified_diff(a.stdout.splitlines(True), b.stdout.splitlines(True),
                                 "stanc3", "stanc-rs")
        with open(os.path.join(out, p.replace("/", "__") + ".diff"), "w") as f:
            f.writelines(d)
same = sum(1 for _, e, a, _ in res if a.returncode == 0 and e is None)
print(f"format identical to stanc3 on {same} of {both} programs stanc3 formats "
      f"({len(good)} good programs)")
