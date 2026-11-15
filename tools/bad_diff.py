#!/usr/bin/env python3
"""Compares how stanc-rs and stanc3 reject stanc3's `test/integration/bad` programs.

Usage (from the repository root, after `cargo build --release -p stanc-rs`):

    python3 tools/bad_diff.py STANC3_DIR STANC

STANC3_DIR is a checkout of stanc3 and STANC a stanc3 binary of the same version. Reports how
many programs each compiler rejects, whether stanc-rs ever crashes, and how often the first
line of the two error reports names the same category and line.
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(os.path.abspath(sys.argv[1]), "test", "integration")
    rs = os.path.join(repo, "target", "release", "stanc-rs")
    st = os.path.abspath(sys.argv[2])
    bad = sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(os.path.join(root, "bad"))
        for f in fs
        if f.endswith((".stan", ".stanfunctions"))
    )

    def one(p):
        inc = f"included,{os.path.dirname(p)}"
        a = subprocess.run([st, f"--include-paths={inc}", "--o=/dev/null", p], cwd=root,
                           capture_output=True, text=True)
        b = subprocess.run([rs, "check", p, "--include-paths", inc], cwd=root,
                           capture_output=True, text=True)
        return p, a, b

    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, bad))
    stanc3_rejects = sum(1 for _, a, _ in res if a.returncode != 0)
    both = same_line = same_category = 0
    for p, a, b in res:
        crashed = "panicked" in b.stderr or b.returncode not in (0, 1)
        if crashed:
            print(f"CRASH {p}")
        if a.returncode != 0 and b.returncode == 0:
            print(f"ACCEPTED {p}")
        if a.returncode == 0 or b.returncode == 0:
            continue
        both += 1
        h1 = (a.stderr.strip().splitlines() or [""])[0]
        h2 = (b.stderr.strip().splitlines() or [""])[0]
        if h1.split(" in ")[0] == h2.split(" in ")[0]:
            same_category += 1
        l1, l2 = re.search(r"line (\d+)", h1), re.search(r"line (\d+)", h2)
        if l1 and l2 and l1.group(1) == l2.group(1):
            same_line += 1
    print(f"{len(res)} bad programs; stanc3 rejects {stanc3_rejects}; both reject {both}")
    print(f"same line {same_line}; same category {same_category}")


if __name__ == "__main__":
    main()
