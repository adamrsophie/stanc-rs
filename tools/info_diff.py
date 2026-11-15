#!/usr/bin/env python3
"""Compares `stanc-rs info` with `stanc --info` on stanc3's `test/integration/good` programs.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/info_diff.py STANC3_DIR STANC

STANC3_DIR is a checkout of stanc3 and STANC a stanc3 binary of the same version (CmdStan ships
one as `bin/stanc`). The JSON documents are compared as values, so layout differences are ignored.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(os.path.abspath(sys.argv[1]), "test", "integration")
rs = os.path.join(repo, "target", "release", "stanc-rs")
st = os.path.abspath(sys.argv[2])
good = sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(os.path.join(root, "good")) for f in fs if f.endswith(".stan"))
def one(p):
    inc = f"included,{os.path.dirname(p)}"
    a = subprocess.run([st, "--info", f"--include-paths={inc}", p], cwd=root, capture_output=True, text=True)
    b = subprocess.run([rs, "info", p, "--include-paths", inc], cwd=root, capture_output=True, text=True)
    if a.returncode != 0:
        return p, "stanc3 failed" if b.returncode == 0 else None
    if b.returncode != 0:
        return p, "stanc-rs failed: " + b.stderr.strip().splitlines()[-1][:150]
    ja, jb = json.loads(a.stdout), json.loads(b.stdout)
    if ja == jb: return p, None
    diff = {k: (ja.get(k), jb.get(k)) for k in set(ja) | set(jb) if ja.get(k) != jb.get(k)}
    return p, json.dumps(diff)[:400]
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, good))
bad = [(p, e) for p, e in res if e]
for p, e in bad: print(p, "|", e)
print(f"info matches stanc3 on {len(good) - len(bad)} of {len(good)}")
