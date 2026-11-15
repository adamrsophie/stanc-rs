#!/usr/bin/env python3
"""Compares the error messages of stanc-rs and stanc3 on stanc3's `test/integration/bad` programs.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/message_diff.py STANC3_DIR STANC [OUT_FILE]

STANC3_DIR is a checkout of stanc3 and STANC a stanc3 binary of the same version. A message has
three parts: the header line with the location, the source excerpt between the dashed lines, and
the explanation after it. Reports how many messages are identical and how many agree in each
part. With OUT_FILE, both texts of every mismatch are written there.
"""
import collections
import concurrent.futures as cf
import os
import subprocess
import sys

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(os.path.abspath(sys.argv[1]), "test", "integration")
st = os.path.abspath(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else None
rs = os.path.join(repo, "target", "release", "stanc-rs")
bad = sorted(
    os.path.relpath(os.path.join(d, f), root)
    for d, _, fs in os.walk(os.path.join(root, "bad"))
    for f in fs
    if f.endswith((".stan", ".stanfunctions"))
)
BAR = "   -------------------------------------------------\n"


def one(p):
    inc = f"included,{os.path.dirname(p)}"
    a = subprocess.run([st, f"--include-paths={inc}", "--o=/dev/null", p], cwd=root,
                       capture_output=True, text=True)
    b = subprocess.run([rs, "check", p, "--include-paths", inc], cwd=root,
                       capture_output=True, text=True)
    return p, a.stderr.strip(), b.stderr.strip()


def parts(t):
    x = t.split(BAR)
    if len(x) < 3:
        return t.split("\n")[0], "", t
    return x[0], x[1], BAR.join(x[2:]).strip()


with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, bad))
c = collections.Counter()
diffs = []
for p, a, b in res:
    pa, pb = parts(a), parts(b)
    h, fr, bo = pa[0] == pb[0], pa[1] == pb[1], pa[2] == pb[2]
    kind = "syntax" if a.startswith("Syntax") else "semantic"
    c["header"] += h
    c["frame"] += fr
    c["body"] += bo
    c["same"] += a == b
    c[kind] += 1
    c[kind + " body"] += bo
    if a != b:
        diffs.append((p, a, b, h, fr, bo))
if out:
    with open(out, "w") as f:
        for p, a, b, h, fr, bo in diffs:
            f.write(f"=== {p} header={h} frame={fr} body={bo}\n--- stanc3\n{a}\n--- stanc-rs\n{b}\n")
print(f"{len(res)} programs: identical {c['same']}; header {c['header']}; excerpt {c['frame']}; "
      f"explanation {c['body']} (syntax {c['syntax body']} of {c['syntax']}, "
      f"semantic {c['semantic body']} of {c['semantic']})")
